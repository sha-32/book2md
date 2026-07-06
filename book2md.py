#!/usr/bin/env python3
"""
book2md — convert a book project folder (epub / pdf / txt, any 1–3 of them)
into a multi-file, cross-linked Markdown edition split by chapter, with
figures abstracted into an assets/ directory and referenced in place.

Design
------
* Source-agnostic: point it at a folder; it uses whichever formats are present.
* Backbone selection: epub > pdf > txt (richest structure wins). The other
  formats are recorded and linked but not merged (epub-first "unified" mode).
* Common intermediate representation: every extractor emits an ordered list of
  typed Blocks (heading / para / image / list / quote / rule / code). Chapter
  detection, figure handling and Markdown rendering all operate on that stream,
  so behaviour is identical no matter which source produced it.
* No machine TOC is assumed (real books here have empty toc.ncx): chapters are
  detected heuristically from "Chapter <n> <title>" headings in the text, with
  a single-file fallback when nothing is found.

Dependencies: Python stdlib + BeautifulSoup4 + Pillow (optional) + poppler CLIs
(pdftotext, pdfinfo, pdfimages). No network, no pip installs required.
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urldefrag

try:
    import warnings
    from bs4 import BeautifulSoup, NavigableString, Tag
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:  # pragma: no cover
    print("error: BeautifulSoup4 is required (pip install beautifulsoup4)", file=sys.stderr)
    raise


# --------------------------------------------------------------------------- #
# Intermediate representation
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    kind: str                       # heading|para|image|list|quote|rule|code
    text: str = ""                  # rendered inline markdown (para/heading/quote/code)
    level: int = 0                  # heading level, or list ordered flag
    items: list = field(default_factory=list)   # list items (markdown strings)
    src: str = ""                   # image: asset path relative to output root
    alt: str = ""                   # image alt/caption
    ordered: bool = False           # list ordering


@dataclass
class Document:
    title: str
    author: str
    blocks: list = field(default_factory=list)
    backbone: str = ""              # which format produced this
    assets: dict = field(default_factory=dict)  # dest-name -> (bytes | source Path)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def slugify(text: str, maxlen: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return (text[:maxlen].rstrip("-")) or "section"


ORDINALS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def words_to_number(phrase: str):
    """Parse 'Twenty-One' / 'twenty one' / '21' -> 21, else None."""
    phrase = phrase.strip().lower()
    if phrase.isdigit():
        return int(phrase)
    total, current, seen = 0, 0, False
    for part in re.split(r"[\s-]+", phrase):
        if part in ORDINALS:
            val = ORDINALS[part]
            seen = True
            if val >= 20 and current:      # e.g. "twenty" after "one" is invalid
                current += val
            else:
                current += val
        else:
            return None
    return current if seen else None


FRONT_MATTER = ("preface", "foreword", "introduction", "acknowledgment",
                "acknowledgement", "prologue")
BACK_MATTER = ("index", "appendix", "afterword", "glossary", "bibliography",
               "epilogue")


def peel_chapter(text: str):
    """If `text` opens with 'Chapter <ordinal>', return (number, remainder,
    tokens_consumed); else None. Handles 'One', 'Twenty-One', 'twenty one', '21'."""
    # tolerate a leading page number and a little OCR junk before "Chapter"
    # (running headers like '8 Chapter One ...', openers like 'L_» Chapter ...')
    m = re.match(r"(?:\d{1,4}\s+)?[\W_]{0,4}(?:[a-zA-Z]{1,3}[\W_]{1,4})?"
                 r"chapter\b[.:]?\s+(.+)$", text, re.IGNORECASE | re.S)
    if not m:
        return None
    tokens = m.group(1).split()
    num, used, prev = 0, 0, None
    for tok in tokens:
        clean = re.sub(r"[^\w-]", "", tok).lower()
        parts = [p for p in clean.split("-") if p]
        if not (parts and all(p in ORDINALS or p.isdigit() for p in parts)):
            break
        val = words_to_number(clean.replace("-", " "))
        if val is None:
            break
        if used == 0:
            num = val
        elif prev is not None and prev >= 20 and prev % 10 == 0 and val < 10:
            num = prev + val          # "twenty" + "one" -> 21
        else:
            break                     # e.g. "nineteen" + title word "two" -> stop
        prev, used = val, used + 1
    if num == 0:
        return None
    return num, " ".join(tokens[used:]), used


def parse_toc(doc: "Document"):
    """Find the Table-of-Contents block and parse it into [(number, title), ...].

    A ToC block is the one paragraph that lists many 'Chapter <n> <title> <page>'
    entries in ascending order. Returns [] when no such block exists."""
    best, best_entries = None, []
    for b in doc.blocks:
        if b.kind not in ("para", "heading"):
            continue
        if len(re.findall(r"\bchapter\b", b.text, re.IGNORECASE)) < 3:
            continue
        entries = []
        # each entry: "Chapter <stuff> <pageno>" up to the next "Chapter" or end
        for m in re.finditer(
                r"chapter\s+(.+?)\s+(\d+)\s*(?=chapter\b|$)",
                b.text, re.IGNORECASE | re.S):
            chunk = re.sub(r"\s+", " ", m.group(1)).strip()
            pc = peel_chapter("Chapter " + chunk)
            if not pc:
                continue
            num, title, _ = pc
            title = re.split(r"\s+\d+\b", title)[0].strip(" .:-")  # cut at page number
            if title:
                entries.append((num, title))
        # keep the block yielding the longest ascending run
        nums = [n for n, _ in entries]
        if len(entries) > len(best_entries) and nums == sorted(nums):
            best, best_entries = b, entries
    doc._toc_block = best
    # de-duplicate while preserving order (some ToCs repeat)
    seen, out = set(), []
    for num, title in best_entries:
        if num not in seen:
            seen.add(num)
            out.append((num, title))
    return out


def _strip_title_prefix(remainder: str, title: str) -> str:
    """Remove a leading, title-matching prefix from an opener block's body text."""
    rt = remainder.lower().lstrip(" *•.-")
    if title and rt.startswith(title.lower()):
        toks = remainder.split()
        return " ".join(toks[len(title.split()):]).lstrip(" *•.-")
    return remainder


def _is_toc_fragment(text: str) -> bool:
    """True for contents/index-style blocks that list chapters rather than open one.

    Such blocks are dense with structural tokens (section numbers, 'Appendix',
    the word 'Contents', or crowded 'Chapter' labels); real openers are prose."""
    head = text[:160]
    if re.search(r"\bcontents\b", head, re.IGNORECASE):
        return True
    if re.search(r"chapter\b.{0,30}chapter\b", head, re.IGNORECASE):
        return True
    if len(re.findall(r"\bappendix\b", head, re.IGNORECASE)) >= 2:
        return True
    if len(re.findall(r"\b\d+\.\d+\b", head)) >= 3:      # 5.4 5.5 5.6 ... listing
        return True
    return False


def _clean_title(t: str) -> str:
    """Trim a candidate title at the first section number / bare page number.

    Titles legitimately contain ':' and ',' ("Instructions: Language of the
    Machine"), so we do not break on those."""
    t = re.split(r"\s+\d+(?:\.\d+)?\b", t)[0]     # cut at '3.1' / bare page no.
    t = re.split(r"(?<=\.)\s+[A-Z]", t)[0]        # cut at a real sentence break
    return t.strip(" .:-")[:70]


def _infer_title(num: int, frags: list) -> str:
    """Best-effort chapter title from its repeated running-header fragments."""
    # keep only fragments that begin like a title (a letter, not another 'Chapter')
    clean = [f for f in frags
             if re.match(r"[A-Za-z]", f) and not re.match(r"chapter\b", f, re.IGNORECASE)]
    title = _clean_title(_word_lcp(clean))
    if len(title) >= 3 and title.lower() != "chapter":
        return title
    if clean:                                     # LCP too short: use first header
        head = leading_lines(clean[0])
        if head:
            return _clean_title(head[0]) or f"Chapter {num}"
    return f"Chapter {num}"


def _title_ok(t: str) -> bool:
    """A sane title starts with a letter, isn't just 'Chapter', and isn't a
    long run of body text."""
    return bool(t and re.match(r"[A-Za-z]", t)
                and t.lower() != "chapter" and len(t.split()) <= 12)


def _pick_title(num: int, toc_title, header_title: str) -> str:
    """Prefer the ToC title when it is sane; otherwise the running-header title.

    The ToC is authoritative when clean (archive.org 'Contents' page); when it
    is fragmented into junk ('2.8', 'Chapter'), the repeated running headers win."""
    if _title_ok(toc_title):
        return toc_title
    if _title_ok(header_title):
        return header_title
    return (toc_title or header_title or f"Chapter {num}")


def _word_lcp(frags: list) -> str:
    """Longest common leading word-run across running-header fragments.

    Repeated running headers ('Memory Hierarchy Design ...') share the chapter
    title as a prefix, then diverge into page body — so their common prefix is
    the title."""
    if not frags:
        return ""
    tokenised = [f.split() for f in frags if f.split()]
    if not tokenised:
        return ""
    prefix = tokenised[0]
    for toks in tokenised[1:]:
        k = 0
        while k < len(prefix) and k < len(toks) and prefix[k].lower() == toks[k].lower():
            k += 1
        prefix = prefix[:k]
        if not prefix:
            break
    return " ".join(prefix[:12]).strip(" .:-")


def leading_lines(text: str, n: int = 3):
    """First few non-numeric, non-empty lines of a block (page numbers stripped)."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"[\divxlcIVXLC.]+", s):   # bare page/roman numbers
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# EPUB extractor (backbone of choice)
# --------------------------------------------------------------------------- #
def extract_epub(path: Path) -> Document:
    z = zipfile.ZipFile(path)
    names = z.namelist()

    # locate the OPF via container.xml
    opf_name = None
    try:
        container = z.read("META-INF/container.xml").decode("utf-8", "replace")
        m = re.search(r'full-path="([^"]+)"', container)
        if m:
            opf_name = m.group(1)
    except KeyError:
        pass
    if not opf_name:
        opf_name = next((n for n in names if n.lower().endswith(".opf")), None)
    if not opf_name:
        raise ValueError("epub has no OPF package document")

    base = opf_name.rsplit("/", 1)[0] if "/" in opf_name else ""
    opf = BeautifulSoup(z.read(opf_name), "html.parser")

    title = (opf.find("dc:title") or opf.find("title"))
    author = (opf.find("dc:creator") or opf.find("creator"))
    title = title.get_text(strip=True) if title else path.stem
    author = author.get_text(strip=True) if author else ""

    # manifest id -> href
    manifest = {}
    for item in opf.find_all("item"):
        manifest[item.get("id")] = item.get("href")

    def resolve(href: str) -> str:
        href = unquote(urldefrag(href)[0])
        return f"{base}/{href}" if base else href

    # spine order
    spine = [manifest.get(ref.get("idref")) for ref in opf.find_all("itemref")]
    spine = [resolve(h) for h in spine if h]

    doc = Document(title=title, author=author, backbone="epub")
    for href in spine:
        try:
            raw = z.read(href)
        except KeyError:
            continue
        soup = BeautifulSoup(raw, "html.parser")
        body = soup.body or soup
        doc_base = href.rsplit("/", 1)[0] if "/" in href else ""
        _walk_html(body, doc, z, doc_base)
    return doc


INLINE_MAP = {"strong": "**", "b": "**", "em": "*", "i": "*", "code": "`"}


def _render_inline(node) -> str:
    """Recursively render an inline HTML subtree to Markdown text."""
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    if not isinstance(node, Tag):
        return ""
    name = node.name.lower()
    inner = "".join(_render_inline(c) for c in node.children)
    if name in INLINE_MAP:
        marker = INLINE_MAP[name]
        return f"{marker}{inner.strip()}{marker}" if inner.strip() else inner
    if name == "a":
        href = node.get("href", "")
        if href and not href.startswith("#"):
            return f"[{inner.strip()}]({href})"
        return inner
    if name == "br":
        return "  \n"
    if name in ("sup",):
        return f"^{inner}^"
    if name in ("sub",):
        return f"~{inner}~"
    return inner


BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol",
              "blockquote", "pre", "hr", "figure", "div", "table"}


def _register_image(doc: Document, z, doc_base: str, src: str) -> str | None:
    src = unquote(urldefrag(src)[0])
    # resolve relative to the html file's directory within the zip
    parts = (f"{doc_base}/{src}" if doc_base else src)
    resolved = str(Path(parts)).replace("\\", "/")
    # normalise ../ segments
    stack = []
    for seg in resolved.split("/"):
        if seg == "..":
            if stack:
                stack.pop()
        elif seg not in (".", ""):
            stack.append(seg)
    resolved = "/".join(stack)
    try:
        data = z.read(resolved)
    except KeyError:
        return None
    dest = f"fig-{slugify(Path(resolved).stem, 40)}{Path(resolved).suffix.lower()}"
    doc.assets[dest] = data
    return dest


def _walk_html(node, doc: Document, z, doc_base: str):
    for child in getattr(node, "children", []):
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()

        if name in ("script", "style", "head"):
            continue

        if name == "img":
            dest = _register_image(doc, z, doc_base, child.get("src", ""))
            if dest:
                alt = (child.get("alt") or "").strip()
                doc.blocks.append(Block("image", src=dest, alt=alt))
            continue

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            txt = _render_inline(child).strip()
            if txt:
                doc.blocks.append(Block("heading", text=txt, level=int(name[1])))
            continue

        if name == "p":
            txt = _render_inline(child).strip()
            imgs = child.find_all("img")
            if txt:
                doc.blocks.append(Block("para", text=txt))
            for im in imgs:
                dest = _register_image(doc, z, doc_base, im.get("src", ""))
                if dest:
                    doc.blocks.append(Block("image", src=dest, alt=(im.get("alt") or "").strip()))
            continue

        if name in ("ul", "ol"):
            items = [_render_inline(li).strip() for li in child.find_all("li", recursive=False)]
            items = [i for i in items if i]
            if items:
                doc.blocks.append(Block("list", ordered=(name == "ol"), items=items))
            continue

        if name == "blockquote":
            txt = _render_inline(child).strip()
            if txt:
                doc.blocks.append(Block("quote", text=txt))
            continue

        if name == "pre":
            doc.blocks.append(Block("code", text=child.get_text()))
            continue

        if name == "hr":
            doc.blocks.append(Block("rule"))
            continue

        # containers: recurse
        _walk_html(child, doc, z, doc_base)


# --------------------------------------------------------------------------- #
# PDF extractor
# --------------------------------------------------------------------------- #
def _run(cmd) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def extract_pdf(path: Path, workdir: Path) -> Document:
    info = _run(["pdfinfo", str(path)])
    title = _kv(info, "Title") or path.stem
    author = _kv(info, "Author") or ""
    doc = Document(title=title, author=author, backbone="pdf")

    keep = _pdf_figure_indices(path, info)   # global image nums worth keeping
    page_images: dict[int, list[Path]] = {}
    if keep:
        imgdir = workdir / "_pdfimg"
        imgdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdfimages", "-all", "-p", str(path), str(imgdir / "img")],
                       capture_output=True)
        for f in sorted(imgdir.glob("img-*")):
            m = re.match(r"img-(\d+)-(\d+)", f.name)
            if m and int(m.group(2)) in keep:
                page_images.setdefault(int(m.group(1)), []).append(f)

    pages = _run(["pdftotext", str(path), "-"]).split("\f")
    for pageno, ptext in enumerate(pages, start=1):
        for img in page_images.get(pageno, []):
            dest = f"fig-p{pageno:04d}-{img.stem.split('-')[-1]}{img.suffix.lower()}"
            doc.assets[dest] = img.read_bytes()
            doc.blocks.append(Block("image", src=dest, alt=f"Figure (page {pageno})"))
        for para in _reflow_paragraphs(ptext):
            doc.blocks.append(Block("para", text=para))
    return doc


def _pdf_figure_indices(path: Path, info: str) -> set:
    """Return global image indices that are genuine figures.

    Skips jbig2/stencil masks and near-full-page images, and returns an empty
    set for scanned books (where every page is one full-page image), so a
    recoded scan does not explode into thousands of bogus 'figures'."""
    listing = _run(["pdfimages", "-list", str(path)])
    m = re.search(r"Page size:\s*([\d.]+)\s*x\s*([\d.]+)", info)
    npages = int(_kv(info, "Pages") or 0)
    if not m or not npages:
        return set()
    pw_pts, ph_pts = float(m.group(1)), float(m.group(2))

    rows, fullpage_pages = [], set()
    for line in listing.splitlines()[2:]:
        c = line.split()
        if len(c) < 15:
            continue
        try:
            page, num, typ = int(c[0]), int(c[1]), c[2]
            w, h, ppi = int(c[3]), int(c[4]), int(c[12])
        except ValueError:
            continue
        near_full = False
        if ppi > 0:
            fw, fh = pw_pts / 72 * ppi, ph_pts / 72 * ppi
            near_full = w >= 0.8 * fw and h >= 0.8 * fh
        if near_full:
            fullpage_pages.add(page)
        rows.append((page, num, typ, w, h, near_full))

    if len(fullpage_pages) > 0.6 * npages:      # a scan, not a figure-bearing PDF
        return set()
    return {num for (_p, num, typ, w, h, near_full) in rows
            if typ == "image" and not near_full and w >= 100 and h >= 100}


def _kv(info: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", info, re.MULTILINE)
    return m.group(1).strip() if m else ""


# --------------------------------------------------------------------------- #
# TXT extractor (plain text or archive.org "Full text of..." HTML wrapper)
# --------------------------------------------------------------------------- #
def extract_txt(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "<pre" in raw.lower() and "</pre>" in raw.lower():
        m = re.search(r"<pre[^>]*>(.*?)</pre>", raw, re.S | re.I)
        body = _html.unescape(re.sub(r"<[^>]+>", "", m.group(1))) if m else raw
    else:
        body = raw
    doc = Document(title=path.stem, author="", backbone="txt")
    for para in _reflow_paragraphs(body):
        doc.blocks.append(Block("para", text=para))
    return doc


def _reflow_paragraphs(text: str):
    """Blank-line-delimited paragraphs; de-hyphenate and unwrap hard line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # join hyphenated line breaks: "combi-\nnation" and OCR "Corpo¬\nration"
    text = re.sub(r"[¬-]\n\s*", "", text)
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        joined = re.sub(r"\s*\n\s*", " ", p).strip()
        joined = re.sub(r"[ \t]+", " ", joined)
        if joined:
            out.append(joined)
    return out


# --------------------------------------------------------------------------- #
# Chapter detection over the block stream (source-agnostic)
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    kind: str          # front | chapter | back
    number: int | None
    title: str
    start: int         # block index (inclusive)
    end: int = 0       # block index (exclusive)


def detect_sections(doc: Document) -> list[Section]:
    from collections import defaultdict

    toc = parse_toc(doc)                         # authoritative titles if present
    toc_block = getattr(doc, "_toc_block", None)
    toc_titles = {n: t for n, t in toc}

    # 1) Harvest chapter markers: real openers AND repeated running headers.
    markers = []                                 # (block_index, num, fragment)
    frags_by_num = defaultdict(list)
    for i, b in enumerate(doc.blocks):
        if b.kind not in ("para", "heading") or b is toc_block:
            continue
        if _is_toc_fragment(b.text):
            continue
        pc = peel_chapter(b.text)
        if not pc:
            continue
        num, frag, _ = pc
        markers.append((i, num, frag))
        frags_by_num[num].append(frag)

    # 2) Boundaries: first marker of each chapter, in ascending order.
    sections: list[Section] = []
    nextnum = 1
    for i, num, frag in markers:
        if num < nextnum or num > nextnum + 3:   # keep it sequential
            continue
        title = _pick_title(num, toc_titles.get(num),
                            _infer_title(num, frags_by_num[num]))
        doc.blocks[i].text = _strip_title_prefix(frag, title)   # clean opener body
        sections.append(Section("chapter", num, title, i))
        nextnum = num + 1

    # front matter = everything before the first chapter (minus the ToC block)
    if not sections:
        sections.append(Section("front", None, doc.title, 0))
    elif sections[0].start > 0:
        sections.insert(0, Section("front", None, "Front Matter", 0))

    # trailing back matter: Index / Appendix after the last chapter
    if sections:
        last_start = sections[-1].start
        for i in range(last_start + 1, len(doc.blocks)):
            b = doc.blocks[i]
            if b.kind not in ("para", "heading"):
                continue
            head = (leading_lines(b.text) or [""])[0].lower()
            if any(head.startswith(k) for k in BACK_MATTER) and len(head) < 40:
                sections.append(Section("back", None,
                                        head.title().strip(" .:-") or "Index", i))
                break

    for j, s in enumerate(sections):
        s.end = sections[j + 1].start if j + 1 < len(sections) else len(doc.blocks)

    # drop the ToC block from whatever section contains it (README replaces it)
    if toc_block is not None:
        try:
            ti = doc.blocks.index(toc_block)
            doc.blocks[ti] = Block("para", text="")   # emptied; skipped at render
        except ValueError:
            pass
    return sections


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def render_block(b: Block) -> str:
    if b.kind == "heading":
        return f"{'#' * max(2, min(b.level, 6))} {b.text}"
    if b.kind == "para":
        return b.text
    if b.kind == "image":
        alt = b.alt or "Figure"
        return f"![{alt}](assets/{b.src})"
    if b.kind == "list":
        if b.ordered:
            return "\n".join(f"{i}. {it}" for i, it in enumerate(b.items, 1))
        return "\n".join(f"- {it}" for it in b.items)
    if b.kind == "quote":
        return "\n".join(f"> {ln}" for ln in b.text.splitlines())
    if b.kind == "code":
        return f"```\n{b.text.rstrip()}\n```"
    if b.kind == "rule":
        return "---"
    return ""


def write_output(doc: Document, sections: list[Section], sources: dict,
                 out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    assets_dir = out / "assets"
    if doc.assets:
        assets_dir.mkdir(exist_ok=True)
    for name, data in doc.assets.items():
        (assets_dir / name).write_bytes(data if isinstance(data, bytes) else Path(data).read_bytes())

    manifest = {
        "title": doc.title,
        "author": doc.author,
        "backbone": doc.backbone,
        "sources": {k: v.name for k, v in sources.items() if v},
        "figure_count": len(doc.assets),
        "chapters": [],
    }

    files = []
    used_slugs = set()
    seq = 0
    for s in sections:
        if s.end - s.start <= 0:
            continue
        # skip a section whose range contains no renderable content
        body_blocks = [b for b in doc.blocks[s.start:s.end]]
        rendered = [render_block(b) for b in body_blocks]
        rendered = [r for r in rendered if r.strip()]
        if not rendered:
            continue

        label = {"chapter": f"Chapter {s.number}", "front": "", "back": ""}[s.kind]
        heading = f"{label}: {s.title}" if label else s.title

        base = slugify(s.title)
        slug = base
        n = 2
        while slug in used_slugs:
            slug = f"{base}-{n}"; n += 1
        used_slugs.add(slug)
        fname = f"{seq:02d}-{slug}.md"
        seq += 1

        # drop the first heading/para if it merely repeats the chapter title
        md = [f"# {heading}", ""]
        md.extend(_join_rendered(rendered))
        (out / fname).write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")

        files.append((fname, heading, s))
        manifest["chapters"].append(
            {"file": fname, "kind": s.kind, "number": s.number, "title": s.title})

    _write_index(doc, files, sources, out)
    (out / "book.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _join_rendered(rendered: list[str]) -> list[str]:
    lines = []
    for r in rendered:
        lines.append(r)
        lines.append("")
    return lines


def _write_index(doc: Document, files, sources, out: Path):
    lines = [f"# {doc.title}", ""]
    if doc.author:
        lines.append(f"*by {doc.author}*")
        lines.append("")
    src_bits = [f"`{v.name}`" for v in sources.values() if v]
    lines.append(f"Generated from {', '.join(src_bits)} "
                 f"(backbone: **{doc.backbone}**). {len(doc.assets)} figures extracted.")
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    for fname, heading, s in files:
        lines.append(f"- [{heading}]({fname})")
    lines.append("")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def discover_sources(folder: Path) -> dict:
    src = {"epub": None, "pdf": None, "txt": None}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".epub" and not src["epub"]:
            src["epub"] = p
        elif ext == ".pdf" and not src["pdf"]:
            src["pdf"] = p
        elif ext == ".txt" and not src["txt"]:
            src["txt"] = p
    return src


def convert(folder: Path, out: Path | None) -> dict:
    sources = discover_sources(folder)
    if not any(sources.values()):
        raise SystemExit(f"no epub/pdf/txt found in {folder}")
    out = out or (folder / "markdown")
    workdir = out / "_work"
    workdir.mkdir(parents=True, exist_ok=True)

    # backbone priority: epub > pdf > txt
    if sources["epub"]:
        doc = extract_epub(sources["epub"])
    elif sources["pdf"]:
        doc = extract_pdf(sources["pdf"], workdir)
    else:
        doc = extract_txt(sources["txt"])

    # prefer real book title from folder-independent metadata
    if not doc.title or doc.title == sources[doc.backbone].stem:
        doc.title = doc.title or folder.name.replace("-", " ").title()

    sections = detect_sections(doc)
    manifest = write_output(doc, sections, sources, out)
    shutil.rmtree(workdir, ignore_errors=True)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path,
                    help="one or more book-project folders (each with 1–3 of epub/pdf/txt)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output dir (default: <folder>/markdown)")
    args = ap.parse_args(argv)
    for folder in args.folders:
        if not folder.is_dir():
            print(f"skip (not a dir): {folder}", file=sys.stderr)
            continue
        out = args.out if (args.out and len(args.folders) == 1) else None
        print(f"==> {folder}")
        m = convert(folder, out)
        print(f"    backbone={m['backbone']} chapters={len(m['chapters'])} "
              f"figures={m['figure_count']}")


if __name__ == "__main__":
    main()
