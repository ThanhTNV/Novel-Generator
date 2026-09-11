# -*- coding: utf-8 -*-
"""
Chapter files on disk: the front-matter block, the body, and the number.

A saved chapter is a Markdown file with an optional YAML block on top::

    ---
    chapter: 3
    title: "Lời mời từ bóng tối"
    scene_date: "1789-02"
    arc: 1
    summary: "Hai dòng tóm tắt."
    words: 2143
    saved_at: "2026-09-11T10:00:00+00:00"
    ---
    # Chapter 3: Lời mời từ bóng tối

    <body>

Two rules matter more than the format:

* **A file with no block is a chapter, not an error.** Every chapter saved
  before front matter existed starts straight at ``# Chapter N``, and an
  agent listing the manuscript must see all of them, so ``parse_front_matter``
  answers ``({}, whole_text)`` for those.
* **A broken block is also not an error.** Someone hand-edits a summary and
  leaves a quote open; the file must still list and read rather than 500,
  so malformed YAML degrades to the same ``({}, whole_text)``.

Only the body ever reaches the memory index. The block is bookkeeping for
humans and agents, and indexing it would make "saved_at" a retrievable fact.
"""

import math
import re
from typing import Any, Dict, Optional, Tuple

import yaml

CHAPTER_GLOB = "chapter-*.md"

_NUMBER_RE = re.compile(r"^chapter-(\d+)")
# "# Chapter 3: Title" or "# Chapter 3" — the header save_chapter writes.
_CHAPTER_HEADING_RE = re.compile(r"^#\s*Chapter\s+\d+\s*(?::\s*(.*))?$", re.IGNORECASE)


def chapter_number(filename: str) -> Optional[int]:
    """The N in ``chapter-NNN-slug.md``, or None for a file not named that way."""
    m = _NUMBER_RE.match(filename or "")
    return int(m.group(1)) if m else None


def _is_fence(line: str) -> bool:
    # Exactly "---" at column 0 (a CRLF file leaves a "\r" to strip). A YAML
    # block scalar inside the metadata is indented, so an indented "---" can
    # never be mistaken for the close.
    return line.rstrip() == "---"


def parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Split a chapter file into ``(meta, body)``.

    Never raises: no block, an unclosed block, and a block that is not valid
    YAML (or not a mapping) all come back as ``({}, text)`` with the file
    untouched, because every one of those is a chapter someone wants to read.
    """
    if not text:
        return {}, text
    # An editor that saves "UTF-8 with BOM" (Notepad, PowerShell's Out-File)
    # puts U+FEFF before the fence, and ``read_text("utf-8")`` keeps it. Left
    # in, "﻿---" is not a fence: the chapter listed as legacy, its whole
    # block leaked into the body and tail, and its scene_date silently went.
    text = text.lstrip(u"﻿")
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if not _is_fence(lines[0]):
        return {}, text
    for i in range(1, len(lines)):
        if not _is_fence(lines[i]):
            continue
        block = "\n".join(lines[1:i])
        try:
            meta = yaml.safe_load(block)
        except Exception:  # noqa: BLE001 - see below
            # Not only YAMLError. PyYAML builds an unquoted 1789-02-30 with
            # datetime.date and lets its plain ValueError out, "!!timestamp
            # abc" escapes as AttributeError, absurd nesting as RecursionError.
            # One such hand edit 500ed the listing of the whole manuscript.
            return {}, text
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            # "---\nsome text\n---" is a horizontal rule, not metadata.
            return {}, text
        return meta, "\n".join(lines[i + 1:])
    return {}, text


def render_chapter(meta: Dict[str, Any], body: str) -> str:
    """
    The inverse of ``parse_front_matter``: ``parse(render(m, b)) == (m, b)``.

    Keys are written in the order given, not sorted, so the file reads the way
    the schema is documented; Unicode is left as-is so a Vietnamese title is
    legible in the file and not an escape sequence.
    """
    if not meta:
        return body
    dumped = yaml.safe_dump(
        meta, allow_unicode=True, sort_keys=False, default_flow_style=False,
        # A long summary folded across lines is valid YAML, but it is the one
        # field people edit by hand, and a fold in the middle of a sentence
        # invites a fix that breaks the block.
        width=4096,
    )
    return "---\n%s---\n%s" % (dumped, body)


def heading_title(body: str) -> Optional[str]:
    """
    A title recovered from the ``# Chapter N: Title`` line a legacy file starts
    with, or from any first-line heading. None when the body has no heading.
    """
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("#"):
            return None
        m = _CHAPTER_HEADING_RE.match(line)
        if m:
            title = (m.group(1) or "").strip()
            # "# Chapter 2" with no title: the heading text is the best we have.
            return title or line.lstrip("#").strip()
        return line.lstrip("#").strip()
    return None


def scene_date_text(value: Any) -> Optional[str]:
    """
    A scene date as the string the API promises.

    PyYAML turns an unquoted ``1789-02-15`` into a ``datetime.date`` — fine for
    ``history.parse_date``, which accepts both, but the JSON contract says
    string-or-null, and a hand-edited file must not change the type.
    """
    if value is None or value == "":
        return None
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return "%04d-%02d-%02d" % (value.year, value.month, value.day)
    return str(value).strip() or None


def as_int(value: Any) -> Optional[int]:
    """An int from front matter, or None — a bool is not a chapter number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is YAML's ``.inf``: unlike ``.nan``, int() of an
        # infinity is not a ValueError, and one ``words: .inf`` 500ed the
        # chapter list and the state snapshot for the whole novel.
        return None


def as_text(value: Any) -> str:
    """
    A string from front matter; None is "".

    ``str()`` alone would hand the API ``"b'\\xff'"`` for a ``!!binary``
    title and ``"2026-09-11 10:00:00+00:00"`` for an unquoted ``saved_at``
    that PyYAML turned into a datetime, so bytes are decoded and datetimes
    written back in the ISO form the app itself saves.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def json_safe(value: Any) -> Any:
    """
    ``value`` as something the JSON encoder will not refuse.

    Starlette serialises with ``allow_nan=False`` and FastAPI decodes bytes
    as strict UTF-8, so an ``arc: .nan`` or a non-UTF-8 ``!!binary`` scalar
    that PyYAML happily accepted turned a readable chapter into a 500 on
    read. A non-finite float has no honest number to give and becomes None.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return dict((as_text(k), json_safe(v)) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    return as_text(value)


_INT_KEYS = ("chapter", "arc", "words")
_TEXT_KEYS = ("title", "summary", "saved_at")


def clean_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    The parsed block as the JSON the API promises: the contracted keys in
    their contracted types, anything else made encoder-safe, and no key
    added — an absent field stays absent, so a reader can tell "no date"
    from "date: null".
    """
    out = {}
    for key, value in (meta or {}).items():
        if key == "scene_date":
            out[key] = scene_date_text(value)
        elif key in _INT_KEYS:
            out[key] = as_int(value)
        elif key in _TEXT_KEYS:
            out[key] = as_text(value)
        else:
            out[as_text(key)] = json_safe(value)
    return out


def tail_words(text: str, count: int) -> str:
    """
    The last ``count`` words of ``text`` with its paragraph breaks intact.

    Joining ``split()[-n:]`` back with spaces would hand an agent the end of
    the chapter as one run-on line; slicing from the n-th-last word keeps the
    prose exactly as written.
    """
    if count <= 0:
        return ""
    words = list(re.finditer(r"\S+", text))
    if len(words) <= count:
        return text.strip()
    return text[words[-count].start():].rstrip()
