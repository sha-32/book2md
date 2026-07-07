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
* Sources are identified by CONTENT, not extension: a '.txt' that is really
  archive.org HTML and a '.pdf' that is a scan are common, so file types cannot
  be trusted as discovered (see sniff_type / discover_sources).
* Common intermediate representation: every extractor emits an ordered list of
  typed Blocks (heading / para / image / list / quote / rule / code). Chapter
  detection, figure handling and Markdown rendering all operate on that stream,
  so behaviour is identical no matter which source produced it.
* Figures are identified by CONTEXT, not by file format: an embedded image is
  not necessarily a figure (these scans embed whole pages, header bands and
  spacers), so a single classifier judges every candidate from document,
  geometric, pictorial and caption context (see filter_figures).
* No machine TOC is assumed (real books here have empty toc.ncx): chapters are
  detected from the "Contents" listing and repeated running headers, covering
  both "Chapter N" and bare-numbered ("1 Title") schemes, with a single-file
  fallback when nothing is found.

Dependencies: Python stdlib + BeautifulSoup4 + Pillow + poppler CLIs
(pdftotext, pdfinfo, pdfimages). No network, no pip installs required.
"""
from __future__ import annotations

import argparse
import hashlib
import html as _html
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
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

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None          # these are book scans; the bomb guard is moot
except Exception:                          # pragma: no cover - Pillow is optional
    Image = None


# --------------------------------------------------------------------------- #
# Intermediate representation
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    kind: str                       # heading|para|image|list|quote|rule|code
    text: str = ""                  # rendered inline markdown (para/heading/quote/code)
    level: int = 0                  # heading level, or list ordered flag
    items: list = field(default_factory=list)   # list items (markdown strings)
    src: str = ""                   # image: asset filename (provisional, then final)
    alt: str = ""                   # image alt/caption
    ordered: bool = False           # list ordering
    data: bytes = b""               # image: raw bytes, held until figure filtering
    w: int = 0                      # image: pixel width  (0 = unknown)
    h: int = 0                      # image: pixel height (0 = unknown)


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
        # keep the block yielding the longest ascending run; require several
        # distinct ascending numbers so a prose page that merely mentions
        # "chapter" a few times is not mistaken for a Table of Contents.
        nums = [n for n, _ in entries]
        if (len(entries) > len(best_entries) and nums == sorted(nums)
                and len(set(nums)) >= 3):
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
    # two chapter *labels* crowded together ("Chapter 1 Chapter 2"), not an opener
    # whose body merely says "... In Chapter 1, we ...".
    if re.search(r"chapter\b\s+\w+\s+chapter\b", head, re.IGNORECASE):
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
# Figure identification — by CONTEXT, not by file format
# --------------------------------------------------------------------------- #
# A JPEG embedded in a book is not necessarily a figure: these scans embed whole
# pages, header bands and spacers as images too, and the file type cannot be
# trusted. So figures are judged from context after the whole document is built:
#   * document context — images whose exact pixel size recurs across a large
#     share of the book are structural (page scans, headers, watermarks), never
#     content figures, which vary in size;
#   * geometric context — an image at (near) the dominant page size is a page;
#     a very wide-and-short or tall-and-thin band is a running header / edge;
#   * pictorial context — a near-blank image carries no figure;
#   * textual context — an adjacent "Figure/Table N …" caption is strong
#     positive evidence and overrides the negative signals above.
CAPTION_RE = re.compile(r"^\s*(figure|fig\.?|table|plate|exhibit|scheme)\s*\d",
                        re.IGNORECASE)
_HAS_TESSERACT = shutil.which("tesseract") is not None


def _img_dims(data: bytes):
    if not (Image and data):
        return (0, 0)
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return (0, 0)


def _ink_fraction(data: bytes) -> float:
    """Fraction of non-white pixels (downsampled). ~0 => blank; higher => inked.
    Returns 1.0 when it cannot tell, so uncertainty never drops an image."""
    if not (Image and data):
        return 1.0
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("L")
            im.thumbnail((160, 160))
            hist = im.histogram()             # 256 luminance bins
            total = sum(hist)
            return (sum(hist[:200]) / total) if total else 1.0
    except Exception:
        return 1.0


def _caption_near(doc: "Document", i: int) -> str | None:
    """Textual context: a 'Figure/Table N …' caption right next to the image."""
    for j in (i + 1, i - 1, i + 2):
        if 0 <= j < len(doc.blocks):
            b = doc.blocks[j]
            if b.kind == "para" and CAPTION_RE.match(b.text):
                return (leading_lines(b.text) or [b.text])[0][:120]
    return None


def _document_prose(doc: "Document") -> str:
    """All body text as a normalised lowercase word-stream, for redundancy tests."""
    words = []
    for b in doc.blocks:
        if b.kind in ("para", "heading", "quote"):
            words.extend(re.findall(r"[a-z]+", b.text.lower()))
    return " ".join(words)


def _ocr_words(data: bytes):
    """OCR an image with tesseract; return its lowercase word list (or [])."""
    if not (Image and data):
        return []
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("L")
            if im.width > 1400:                     # cap work for large scans
                im = im.resize((1400, max(1, int(im.height * 1400 / im.width))))
            buf = io.BytesIO()
            im.save(buf, "PNG")
        out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6"],
                             input=buf.getvalue(), capture_output=True, timeout=60)
        return re.findall(r"[a-z]+", out.stdout.decode("utf-8", "replace").lower())
    except Exception:
        return []


def _is_text_render(data: bytes, ink: float, w: int, h: int, prose: str) -> bool:
    """True if an image is prose rendered as a picture (a 'figure description
    text' block) rather than a genuine figure. Decided by CONTENT: OCR the image
    and check whether whole phrases from it already exist in the document text —
    real diagrams contain scattered labels that never reproduce prose sentences."""
    if not (_HAS_TESSERACT and prose):
        return False
    if ink >= 0.35 or min(w, h) < 200:              # photos / tiny marks: not text
        return False
    words = _ocr_words(data)
    if len(words) < 60:                             # a captioned figure has few words;
        return False                                # only dense blocks are text pages
    grams = [" ".join(words[i:i + 6]) for i in range(len(words) - 5)]
    if not grams:
        return False
    hits = sum(1 for g in grams if g in prose)
    return hits / len(grams) >= 0.5                 # verbatim prose sentences, in bulk


_FIG_CAP = re.compile(r"^(figure|fig\.?|table|plate|scheme)\s+\d", re.IGNORECASE)


def _ocr_line_boxes(gray):
    """OCR (tsv) grouped into text lines: [{box,text,words,h}, …] sorted by y."""
    buf = io.BytesIO()
    gray.save(buf, "PNG")
    out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6", "tsv"],
                         input=buf.getvalue(), capture_output=True, timeout=90)
    lines = {}
    for row in out.stdout.decode("utf-8", "replace").splitlines()[1:]:
        c = row.split("\t")
        if len(c) < 12 or not c[11].strip():
            continue
        try:
            key = (c[2], c[3], c[4])
            l, t, w, h, conf = int(c[6]), int(c[7]), int(c[8]), int(c[9]), float(c[10])
        except ValueError:
            continue
        if conf < 30:
            continue
        d = lines.setdefault(key, {"l": l, "t": t, "r": l + w, "b": t + h, "words": []})
        d["l"], d["t"] = min(d["l"], l), min(d["t"], t)
        d["r"], d["b"] = max(d["r"], l + w), max(d["b"], t + h)
        d["words"].append(c[11].strip())
    return sorted(({"box": (d["l"], d["t"], d["r"], d["b"]), "text": " ".join(d["words"]),
                    "n": len(d["words"]), "h": d["b"] - d["t"]} for d in lines.values()),
                  key=lambda x: x["box"][1])


def _extract_page_figures(data: bytes):
    """A page scan may hold real figures, each marked by a 'Figure N …' caption.
    Crop each figure's graphic (excluding the caption and body text) and return
    [(jpeg_bytes, caption_text), …]. The caption is the indicator AND the crop
    boundary; a pure-text page (no caption line) yields nothing."""
    if not (Image and _HAS_TESSERACT):
        return []
    try:
        rgb = Image.open(io.BytesIO(data)).convert("RGB")
        # Fast, low-res pre-check: a page with no "Figure N" caption line has no
        # figure to crop, so skip the expensive full-resolution pass entirely.
        probe = rgb.convert("L")
        probe.thumbnail((900, 1200))
        pbuf = io.BytesIO()
        probe.save(pbuf, "PNG")
        ptxt = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6"],
                              input=pbuf.getvalue(), capture_output=True,
                              timeout=60).stdout.decode("utf-8", "replace")
        if not any(_FIG_CAP.match(ln.strip()) for ln in ptxt.splitlines()):
            return []

        if rgb.width > 2000:
            rgb = rgb.resize((2000, max(1, int(rgb.height * 2000 / rgb.width))))
        gray = rgb.convert("L")
        W, H = gray.size
        px = gray.load()
        lines = _ocr_line_boxes(gray)
        if not lines:
            return []
        med_w = sorted(ln["box"][2] - ln["box"][0] for ln in lines)[len(lines) // 2]
        lh = max(6, sorted(ln["h"] for ln in lines)[len(lines) // 2])
        is_cap = [bool(_FIG_CAP.match(ln["text"])) for ln in lines]
        is_body = [ln["n"] >= 5 and (ln["box"][2] - ln["box"][0]) > 0.5 * med_w
                   and not is_cap[i] for i, ln in enumerate(lines)]
        # group each caption with its wrapped continuation lines
        caps, i = [], 0
        while i < len(lines):
            if is_cap[i]:
                t0, b0, txt, j = lines[i]["box"][1], lines[i]["box"][3], [lines[i]["text"]], i + 1
                while j < len(lines) and not is_cap[j] and lines[j]["box"][1] - b0 < 0.8 * lh:
                    b0 = lines[j]["box"][3]
                    txt.append(lines[j]["text"])
                    j += 1
                caps.append((t0, b0, " ".join(txt)))
                i = j
            else:
                i += 1
        if not caps:
            return []
        inked = [sum(1 for x in range(0, W, 2) if px[x, y] < 130) > 3 for y in range(H)]
        bodyrow = [False] * H
        for i, ln in enumerate(lines):
            if is_body[i]:
                for y in range(ln["box"][1], min(H, ln["box"][3])):
                    bodyrow[y] = True
        caprow = [False] * H
        for t0, b0, _t in caps:
            for y in range(t0, min(H, b0)):
                caprow[y] = True
        maxgap = int(0.05 * H)
        figures = []
        for ci, (ct0, ct1, captext) in enumerate(caps):
            # a caption hugs its own figure: pick the side with the smaller gap to ink
            gu = next((k for k in range(1, ct0) if inked[ct0 - k] and not caprow[ct0 - k]), H)
            gd = next((k for k in range(1, H - ct1) if inked[ct1 + k] and not caprow[ct1 + k]), H)
            up = gu <= gd
            y, step = (ct0 - 1, -1) if up else (ct1 + 1, 1)
            rows, gap = set(), 0
            while 0 <= y < H:
                if bodyrow[y] or caprow[y]:
                    break
                if inked[y]:
                    rows.add(y)
                    gap = 0
                else:
                    gap += 1
                    if gap > maxgap:
                        break
                y += step
            if len(rows) < lh:
                continue
            y0, y1 = min(rows), max(rows)
            cols = [x for x in range(W) if any(px[x, yy] < 130 for yy in range(y0, y1 + 1, 2))]
            if not cols:
                continue
            x0, x1 = max(0, min(cols) - 12), min(W, max(cols) + 12)
            y0, y1 = max(0, y0 - 8), min(H, y1 + 8)
            if x1 - x0 < 64 or y1 - y0 < 48:
                continue
            buf = io.BytesIO()
            rgb.crop((x0, y0, x1, y1)).save(buf, "JPEG", quality=88)
            figures.append((buf.getvalue(), re.sub(r"\s+", " ", captext).strip()))
        return figures
    except Exception:
        return []


def filter_figures(doc: "Document") -> dict:
    """Keep only genuine figures among candidate image blocks; drop the rest and
    commit survivors to doc.assets. Returns a summary of what was dropped/why."""
    imgs = [(i, b) for i, b in enumerate(doc.blocks) if b.kind == "image"]
    n = len(imgs)
    if not n:
        doc.assets = {}
        return {"kept": 0, "dropped": 0, "reasons": {}}

    dims = [(b.w, b.h) for _, b in imgs if b.w and b.h]
    counts = Counter(dims)
    # dimension clusters that recur too often to be content
    cluster_thresh = max(8, int(0.15 * n))
    freq_dims = {d for d, c in counts.items() if c >= cluster_thresh}
    # dominant page size = largest frequent portrait-ish, ≥0.4 MP cluster
    portrait = [d for d in freq_dims
                if d[1] and 0.5 <= d[0] / d[1] <= 0.95 and d[0] * d[1] >= 400_000]
    page = max(portrait, key=lambda d: d[0] * d[1], default=None)

    reasons = Counter()

    # Pass 1 — content dedup: identical bytes are the same picture emitted twice
    # (a scan artifact), so keep only the first occurrence.
    seen = {}
    dup = set()
    for i, b in imgs:
        digest = hashlib.md5(b.data).digest() if b.data else None
        if digest is not None and digest in seen:
            dup.add(i)
        elif digest is not None:
            seen[digest] = i

    # Pass 2 — geometric / frequency / blank context.
    # A full-page-size image is a page scan, not a discrete figure. But it may
    # *contain* real figures, each marked by a "Figure N …" caption: those are
    # cropped out (graphic only, caption excluded) and the page block is replaced
    # by the crops. A page with no caption line is a pure-text page and is
    # dropped. A caption can still rescue a genuinely sub-page image.
    survivors = []
    page_imgs = []                     # (index, block) of full-page scans
    for i, b in imgs:
        if i in dup:
            reasons["duplicate of another image"] += 1
            continue
        w, h = b.w, b.h
        ar = (w / h) if (w and h) else 1.0
        pagesize = bool(page and w >= 0.85 * page[0] and h >= 0.85 * page[1])
        if pagesize:
            page_imgs.append((i, b))
            continue
        caption = _caption_near(doc, i)
        reason = None
        if max(w, h) and max(w, h) < 64:
            reason = "decorative (tiny)"                       # bullets, icons, rules
        elif not caption and (w, h) in freq_dims:
            reason = "repeated page/structural image"
        elif not caption and (ar >= 3.2 or ar <= 1 / 3.2):
            reason = "header/edge strip"
        elif not caption and _ink_fraction(b.data) < 0.006:
            reason = "blank"
        if reason:
            reasons[reason] += 1
            continue
        survivors.append((i, b, caption))

    # Mine figures out of page scans only when the book's figures appear to be
    # *trapped* in pages (few discrete sub-page figures were found). When a book
    # already yields many sub-page figures, its page scans are redundant page
    # images, so drop them without the (expensive, per-page OCR) crop pass.
    page_figs = {}                     # block index -> [(crop_bytes, caption), …]
    if len(survivors) < 40 and page_imgs:
        for i, b in page_imgs:
            crops = _extract_page_figures(b.data)
            if crops:
                page_figs[i] = crops
            else:
                reasons["full text page (no figure)"] += 1
    else:
        reasons["full-page scan"] += len(page_imgs)

    # Pass 3 — content: drop sub-page images that are prose already present as
    # text ("figure description text" rendered as a picture).
    prose = _document_prose(doc)
    assets, used = {}, set()
    keep_blocks = {}                   # block index -> [Block, …] to emit in place

    def _register(data, alt, stem):
        name = f"fig-{stem}"
        base, k = name, 2
        while name in used:
            name = f"{Path(base).stem}-{k}{Path(base).suffix}"
            k += 1
        used.add(name)
        assets[name] = data
        return name

    for i, b, caption in survivors:
        if _is_text_render(b.data, _ink_fraction(b.data), b.w, b.h, prose):
            reasons["text rendered as image"] += 1
            continue
        if caption:
            b.alt = caption
        b.src = _register(b.data, b.alt, b.src)
        b.data = b""                                          # free the bytes
        keep_blocks[i] = [b]

    imgs_by_idx = {i: b for i, b in imgs}
    cropped = 0
    for i, crops in page_figs.items():
        stem = Path(imgs_by_idx[i].src).stem
        blocks = []
        for k, (data, captext) in enumerate(crops):
            w2, h2 = _img_dims(data)
            name = _register(data, captext, f"{stem}-{k:02d}.jpeg")
            blocks.append(Block("image", src=name, alt=captext, w=w2, h=h2))
            cropped += 1
        keep_blocks[i] = blocks

    new_blocks = []
    for j, b in enumerate(doc.blocks):
        if b.kind == "image":
            new_blocks.extend(keep_blocks.get(j, []))
        else:
            new_blocks.append(b)
    doc.blocks = new_blocks
    doc.assets = assets
    reasons.pop("figures cropped from page scans", None)
    return {"kept": len(assets), "dropped": sum(reasons.values()),
            "reasons": dict(reasons), "cropped_from_pages": cropped,
            "ocr": _HAS_TESSERACT}


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


def _image_block(z, doc_base: str, src: str, alt: str) -> "Block | None":
    """Build a candidate image Block carrying raw bytes + pixel dimensions.

    Nothing is committed to assets here: whether this image is a *genuine
    figure* is decided later, in context, by filter_figures()."""
    src = unquote(urldefrag(src)[0])
    parts = (f"{doc_base}/{src}" if doc_base else src)
    resolved = str(Path(parts)).replace("\\", "/")
    stack = []                                   # normalise ../ segments
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
    ext = Path(resolved).suffix.lower()
    if ext == ".svgz":                           # ".svgz" is often plain SVG mislabelled,
        if data[:2] == b"\x1f\x8b":              # or genuinely gzipped — handle both
            try:
                import gzip
                data = gzip.decompress(data)
            except Exception:
                pass
        ext = ".svg"                             # either way the content is SVG
    w, h = _img_dims(data)
    name = f"{slugify(Path(resolved).stem, 40)}{ext}"
    return Block("image", src=name, alt=alt, data=data, w=w, h=h)


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
            blk = _image_block(z, doc_base, child.get("src", ""), (child.get("alt") or "").strip())
            if blk:
                doc.blocks.append(blk)
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
                blk = _image_block(z, doc_base, im.get("src", ""), (im.get("alt") or "").strip())
                if blk:
                    doc.blocks.append(blk)
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

    # Extract every embedded image except jbig2/stencil masks (which are never
    # standalone figures). Whether each survivor is a genuine figure is decided
    # later, in context, by filter_figures() — the same judge the epub path uses.
    nonmask = _pdf_nonmask_indices(path)
    page_images: dict[int, list[Path]] = {}
    if nonmask:
        imgdir = workdir / "_pdfimg"
        imgdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdfimages", "-all", "-p", str(path), str(imgdir / "img")],
                       capture_output=True)
        for f in sorted(imgdir.glob("img-*")):
            m = re.match(r"img-(\d+)-(\d+)", f.name)
            if m and int(m.group(2)) in nonmask:
                page_images.setdefault(int(m.group(1)), []).append(f)

    pages = _run(["pdftotext", str(path), "-"]).split("\f")
    for pageno, ptext in enumerate(pages, start=1):
        for img in page_images.get(pageno, []):
            data = img.read_bytes()
            w, h = _img_dims(data)
            doc.blocks.append(Block("image", data=data, w=w, h=h,
                                    alt=f"Figure (page {pageno})",
                                    src=f"p{pageno:04d}-{img.stem.split('-')[-1]}{img.suffix.lower()}"))
        for para in _reflow_paragraphs(ptext):
            doc.blocks.append(Block("para", text=para))
    return doc


def _pdf_nonmask_indices(path: Path) -> set:
    """Global image indices from `pdfimages -list` that are not masks/stencils."""
    listing = _run(["pdfimages", "-list", str(path)])
    keep = set()
    for line in listing.splitlines()[2:]:
        c = line.split()
        if len(c) < 3:
            continue
        try:
            num, typ = int(c[1]), c[2]
        except ValueError:
            continue
        if typ == "image":
            keep.add(num)
    return keep


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
    book: int = 0      # sub-book index (for multi-book compilations)
    book_title: str = ""


def detect_by_headings(doc: Document) -> list[Section]:
    """Split a born-digital epub by its own heading structure. Scans have no real
    headings (one stray <h2> at most), so this only fires for structured books —
    e.g. the ECMAScript spec, whose <h1> clauses are '1 Scope', '2 Conformance', …
    Splits at the shallowest heading level that occurs often enough to be the
    document's top level."""
    headings = [(i, b) for i, b in enumerate(doc.blocks)
                if b.kind == "heading" and b.text.strip()]
    if len(headings) < 5:
        return []
    by_level = Counter(b.level for _, b in headings)
    top = next((lvl for lvl in sorted(by_level) if by_level[lvl] >= 3), None)
    if top is None:
        return []
    sections, last = [], None
    for i, b in headings:
        if b.level != top:
            continue
        title = re.sub(r"\s+", " ", b.text).strip()
        if title == last:                       # collapse repeated cover/title pages
            continue
        last = title
        # the heading already carries its own number ("1 Scope"), so keep it in
        # the title and leave `number` unset (no extra "Chapter N" prefix).
        sections.append(Section("chapter", None, title, i))
    return sections if len(sections) >= 3 else []


def detect_sections(doc: Document) -> list[Section]:
    from collections import defaultdict

    # 0) Native heading structure wins when present (born-digital epubs).
    sections = detect_by_headings(doc)
    toc_block = None
    if len(sections) >= 3:
        return _finish_sections(doc, sections, toc_block)

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

    # Compilations reset chapter numbering per sub-book (e.g. YDKJS: six books
    # each with "CHAPTER 1..N"). The single ascending pass above stops at the
    # first reset, so if the openers reset, re-detect book-aware and take it when
    # it covers more chapters. Single-book inputs never reset, so are untouched.
    if any(num == 1 for _i, num, _f in markers if _i > (markers[0][0] if markers else 0)):
        multibook = detect_multibook(doc, markers)
        if len(multibook) > len([s for s in sections if s.kind == "chapter"]):
            sections = multibook

    # Fallback for books that number chapters WITHOUT the word "Chapter"
    # (e.g. SICP: "1 Building Abstractions with Procedures"). Only triggers when
    # the "Chapter N" scheme found nothing, so the common case is untouched.
    if len(sections) < 2:
        numeric = detect_numeric_sections(doc)
        if len(numeric) >= 2:
            sections = numeric

    return _finish_sections(doc, sections, toc_block)


def _finish_sections(doc: Document, sections: list, toc_block) -> list:
    """Add front/back matter, close section ranges, drop the ToC block."""
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


def parse_numeric_toc(doc: Document):
    """Parse a 'Contents' listing that numbers chapters as bare integers.

    Recognises a top-level entry by 'N Title [page] N.1' — a chapter heading
    followed (in the listing) by its first subsection — which keeps bare page
    numbers and equations from being mistaken for chapters. The listing is
    often split across several pages/blocks, so entries are accumulated from
    every table-of-contents-like block. Returns [(number, title), ...]."""
    entry = re.compile(
        r"(?<![.\d])\b([1-9])\s+([A-Z][A-Za-z][A-Za-z ,'&-]{4,55}?)\s+\d{0,4}\s*\1\.1\b")
    found = {}
    for b in doc.blocks:
        if b.kind != "para":
            continue
        # a ToC page is dense with dotted subsection numbers
        if len(re.findall(r"\b\d+\.\d+\b", b.text)) < 2:
            continue
        for m in entry.finditer(b.text):
            found.setdefault(int(m.group(1)), m.group(2).strip(" ,"))
    # keep the sequential run starting at 1
    out, k = [], 1
    while k in found:
        out.append((k, found[k]))
        k += 1
    return out


def detect_numeric_sections(doc: Document) -> list:
    """Locate bare-numbered chapter openers using the numeric Contents map."""
    toc = parse_numeric_toc(doc)
    if not toc:
        return []
    titles = {n: t for n, t in toc}
    sections, expected = [], 0
    for i, b in enumerate(doc.blocks):
        if b.kind not in ("para", "heading") or _is_toc_fragment(b.text):
            continue
        if expected >= len(toc):
            break
        num, title = toc[expected]
        # opener: leading "<num> <first words of title>"
        lead = " ".join(b.text.split()[:6])
        two = " ".join(title.split()[:2])
        if re.match(rf"[\W_]{{0,4}}{num}\b[.\s]+{re.escape(two)}", lead, re.IGNORECASE):
            body = re.sub(rf"^[\W_]{{0,4}}{num}\b[.\s]+", "", b.text)
            b.text = _strip_title_prefix(body, title)
            sections.append(Section("chapter", num, title, i))
            expected += 1
    return sections


# --------------------------------------------------------------------------- #
# Multi-book compilations (chapter numbering resets per sub-book)
# --------------------------------------------------------------------------- #
_STOP = {"a", "an", "the", "of", "and", "to", "in", "for", "is", "or", "by",
         "as", "we", "that", "this", "on", "with", "up", "&"}
_TOC_HEAD = re.compile(r"^\s*[*•\s]*(table of\s+)?contents\b", re.IGNORECASE)


def _is_book_toc(text: str) -> bool:
    """A real sub-book Table of Contents, not just a block mentioning 'Contents':
    it either says 'Table of Contents' or lists many dotted page leaders."""
    if not _TOC_HEAD.match(text):
        return False
    if re.match(r"\s*[*•\s]*table of\s+contents", text, re.IGNORECASE):
        return True
    return len(re.findall(r"\.\s+\d", text)) >= 4


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _trim_connectors(t: str) -> str:
    w = t.split()
    while w and (w[-1].lower().strip(".,&:") in _STOP
                 or not w[-1].strip(".,&:-•")):        # trailing connector or punctuation
        w.pop()
    return " ".join(w)


def _title_from_toc(frag: str, toc_norm: str) -> str | None:
    """Longest word-prefix of an opener that appears verbatim in the book's ToC
    text — a format-agnostic way to recover the clean title."""
    words = frag.split()
    best = 0
    for k in range(1, min(len(words), 9) + 1):
        if _norm(" ".join(words[:k])) in toc_norm:
            best = k
    return _trim_connectors(" ".join(words[:best]).strip(" .:-")) if best else None


def _book_titles(pre_blocks: list) -> list:
    """Sub-book titles from their title pages, stripping the author/publisher
    boilerplate that every title page shares (found as the common tokens)."""
    toks = [t.split() for t in pre_blocks]
    common = set()
    if len(toks) >= 2:
        common = set(w.lower() for w in toks[0])
        for tk in toks[1:]:
            common &= set(w.lower() for w in tk)
        common -= _STOP
    titles = []
    for tk in toks:
        cut = len(tk)
        for j, w in enumerate(tk):
            if w.lower() in common:
                cut = j
                break
        titles.append(" ".join(tk[:cut]).strip(" .:-") or "")
    return titles


def detect_multibook(doc: Document, markers: list) -> list:
    """Detect chapters across a compilation whose numbering resets per sub-book.

    Each sub-book opens with a title page then its own Table of Contents, then
    "CHAPTER 1..N". Titles come from the current book's ToC (accumulated across
    its blocks); book titles come from the title pages."""
    # title page = nearest non-empty para before each ToC head
    toc_heads = [i for i, b in enumerate(doc.blocks)
                 if b.kind == "para" and _is_book_toc(b.text)]
    pre = []
    for ti in toc_heads:
        for j in range(ti - 1, max(ti - 4, -1), -1):
            if doc.blocks[j].kind == "para" and doc.blocks[j].text.strip():
                pre.append(doc.blocks[j].text)
                break
        else:
            pre.append("")
    booktitles = _book_titles(pre)
    head_title = dict(zip(toc_heads, booktitles))
    # Use the ORIGINAL opener fragments harvested before any block text was
    # rewritten by the single-sequence pass, keyed by block index.
    marker_by_idx = {i: (num, frag) for i, num, frag in markers}

    sections, current_toc, in_toc = [], "", False
    book, prev, cur_booktitle = -1, 0, ""
    for i, b in enumerate(doc.blocks):
        if b.kind != "para":
            continue
        if i in head_title:
            current_toc, in_toc = _norm(b.text), True
            cur_booktitle = head_title[i]
            continue
        if i in marker_by_idx:
            num, frag = marker_by_idx[i]
            if num == 1 and prev >= 2:            # reset => new sub-book
                book, prev = book + 1, 1
            elif prev < num <= prev + 2:
                prev = num
            else:
                if in_toc:
                    current_toc += " " + _norm(b.text)
                continue
            if book < 0:
                book = 0
            in_toc = False
            title = (_title_from_toc(frag, current_toc)
                     or _clean_title(frag) or " ".join(frag.split()[:4]))
            doc.blocks[i].text = _strip_title_prefix(frag, title)
            sections.append(Section("chapter", num, title, i,
                                    book=max(book, 0), book_title=cur_booktitle))
        elif in_toc:
            current_toc += " " + _norm(b.text)
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

    multibook = any(s.kind == "chapter" and (s.book > 0 or s.book_title)
                    for s in sections)
    manifest["multibook"] = multibook

    files = []
    used_slugs = set()
    seq = 0
    for s in sections:
        if s.end - s.start <= 0:
            continue
        # skip a section whose range contains no renderable content
        body_blocks = list(doc.blocks[s.start:s.end])
        # a heading-split section opens with the heading that titles it — don't
        # render it twice (it becomes the file's H1).
        if body_blocks and body_blocks[0].kind == "heading" \
                and body_blocks[0].text.strip() == s.title:
            body_blocks = body_blocks[1:]
        rendered = [render_block(b) for b in body_blocks]
        rendered = [r for r in rendered if r.strip()]
        if not rendered:
            continue

        # heading-structured sections carry their own number in the title
        # ("1 Scope"), so only prefix "Chapter N" when we assigned a number.
        label = f"Chapter {s.number}" if (s.kind == "chapter" and s.number is not None) else ""
        heading = f"{label}: {s.title}" if label else s.title

        base = slugify(s.title)
        slug = base
        n = 2
        while slug in used_slugs:
            slug = f"{base}-{n}"; n += 1
        used_slugs.add(slug)
        if multibook and s.kind == "chapter":
            fname = f"{s.book + 1:02d}-{s.number:02d}-{slug}.md"
        else:
            fname = f"{seq:02d}-{slug}.md"
        seq += 1

        md = [f"# {heading}", ""]
        if multibook and s.book_title:
            md = [f"# {heading}", "", f"*{s.book_title}*", ""]
        md.extend(_join_rendered(rendered))
        (out / fname).write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")

        files.append((fname, heading, s))
        manifest["chapters"].append(
            {"file": fname, "kind": s.kind, "number": s.number, "title": s.title,
             "book": s.book, "book_title": s.book_title})

    _write_index(doc, files, sources, out, multibook)
    (out / "book.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _join_rendered(rendered: list[str]) -> list[str]:
    lines = []
    for r in rendered:
        lines.append(r)
        lines.append("")
    return lines


def _write_index(doc: Document, files, sources, out: Path, multibook=False):
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
    if multibook:
        cur = None
        for fname, heading, s in files:
            if s.kind == "chapter" and s.book != cur:
                cur = s.book
                lines.append("")
                lines.append(f"### Book {s.book + 1}"
                             + (f" · {s.book_title}" if s.book_title else ""))
            lines.append(f"- [{heading}]({fname})")
    else:
        for fname, heading, s in files:
            lines.append(f"- [{heading}]({fname})")
    lines.append("")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def sniff_type(path: Path) -> str | None:
    """Identify a source by its CONTENT, not its extension — the sample data
    has a '.txt' that is really archive.org HTML and '.pdf' scans, so the
    discovered file type cannot be trusted. Returns 'epub' | 'pdf' | 'text'."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if head[:5] == b"%PDF-":
        return "pdf"
    if head[:2] == b"PK":                        # a zip — is it an epub?
        try:
            z = zipfile.ZipFile(path)
            names = z.namelist()
            if "mimetype" in names and z.read("mimetype").strip() == b"application/epub+zip":
                return "epub"
            if "META-INF/container.xml" in names or any(x.endswith(".opf") for x in names):
                return "epub"
        except zipfile.BadZipFile:
            pass
        return None                              # some other zip: not a book source
    if b"\x00" in head:                          # binary, unknown => not usable as text
        return None
    return "text"                                # plain text or HTML-wrapped OCR


def discover_sources(folder: Path) -> dict:
    """Pick one epub, one pdf, one text source by sniffing file contents."""
    src = {"epub": None, "pdf": None, "txt": None}
    slot = {"epub": "epub", "pdf": "pdf", "text": "txt"}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        key = slot.get(sniff_type(p))
        if key and not src[key]:
            src[key] = p
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

    figstats = filter_figures(doc)          # keep genuine figures, by context
    sections = detect_sections(doc)
    manifest = write_output(doc, sections, sources, out)
    manifest["figures"] = figstats
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
        fig = m.get("figures", {})
        crop = fig.get("cropped_from_pages", 0)
        print(f"    backbone={m['backbone']} chapters={len(m['chapters'])} "
              f"figures={m['figure_count']} (kept {fig.get('kept', 0)}, "
              f"dropped {fig.get('dropped', 0)} non-figures"
              + (f", {crop} cropped from page scans" if crop else "") + ")")
        for reason, cnt in sorted(fig.get("reasons", {}).items(),
                                  key=lambda kv: -kv[1]):
            print(f"        - {cnt:5} {reason}")


if __name__ == "__main__":
    main()
