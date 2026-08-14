"""
Segmentation: turn a document into provenance-carrying narrative units.

Zero-Mem keeps original traces as the source of record, so this does NOT create
overlapping sliding-window chunks. It splits along boundaries the author
already wrote — paragraphs and markdown headings — and records where each unit
came from, so neighbouring prose can always be recovered at retrieval time.

Hierarchy (mirrors the paper's turn -> window -> episode):

    segment (paragraph)  ->  scene (run of segments)  ->  document (chapter)
"""

import re
import unicodedata
from typing import List, NamedTuple, Optional

from .text import estimate_tokens

# A sentence ends at .!?… possibly followed by closing quotes/brackets, then
# whitespace, then a new sentence. The old pipeline required the next character
# to match [A-Z], which never matches Vietnamese capitals (Đ, Ư, Ổ, Ánh, Ừ...),
# so Vietnamese text barely split at all. Uppercase is tested with str.isupper()
# instead, which is Unicode-correct for every alphabet.
_SENT_BOUNDARY = re.compile(r'([.!?…]+["\'”’)\]\}]*)(\s+)')
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_OPENERS = set('"\'“‘([{—-•*')


def split_sentences(text: str) -> List[str]:
    """Unicode-aware sentence split. Never breaks mid-sentence."""
    out: List[str] = []
    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if not para:
            continue
        start = 0
        for m in _SENT_BOUNDARY.finditer(para):
            end = m.end(1)
            nxt = para[m.end():m.end() + 1]
            # Only break when the next sentence really starts: an uppercase
            # letter in any script, or an opening quote/bracket/dash.
            if nxt and not (nxt.isupper() or nxt in _OPENERS or nxt.isdigit()):
                continue
            piece = para[start:end].strip()
            if piece:
                out.append(piece)
            start = m.end()
        tail = para[start:].strip()
        if tail:
            out.append(tail)
    return out


class Segment(NamedTuple):
    """One narrative unit, stored verbatim."""
    text: str
    index: int          # position within the document
    scene: int          # scene (window) this segment belongs to
    heading: str        # nearest markdown heading above it, if any
    kind: str           # 'prose' | 'heading' | 'list'


def _is_list_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(re.match(r"^([-*+]\s+|\d+[.)]\s+)", stripped))


def _split_long_block(block: str, max_tokens: int) -> List[str]:
    """
    Fall back to sentence packing only when a single authored paragraph is too
    large to be a useful retrieval unit.
    """
    if estimate_tokens(block) <= max_tokens:
        return [block]
    sentences = split_sentences(block)
    if len(sentences) <= 1:
        return [block]
    out: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for sent in sentences:
        cost = estimate_tokens(sent)
        if current and current_tokens + cost > max_tokens:
            out.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += cost
    if current:
        out.append(" ".join(current))
    return out


def segment_document(
    text: str,
    max_segment_tokens: int = 320,
    scene_size: int = 4,
) -> List[Segment]:
    """
    Split a document into ordered segments with heading provenance.

    Markdown headings start a new scene and are attached to every segment
    beneath them, so a retrieved paragraph always knows which character or
    location section it belongs to.
    """
    segments: List[Segment] = []
    heading = ""
    scene = 0
    since_heading = 0

    for raw_block in _PARAGRAPH_RE.split(text):
        block = raw_block.strip()
        if not block:
            continue

        lines = block.split("\n")
        head_match = _HEADING_RE.match(lines[0].strip())
        if head_match:
            heading = head_match.group(2).strip()
            scene += 1
            since_heading = 0
            body = "\n".join(lines[1:]).strip()
            segments.append(
                Segment(lines[0].strip(), len(segments), scene, heading, "heading")
            )
            if not body:
                continue
            block = body

        # Bullet lists describe one entity per line in this project's context
        # files; keep each bullet addressable instead of fusing the whole list.
        block_lines = [l for l in block.split("\n") if l.strip()]
        if block_lines and all(_is_list_line(l) for l in block_lines):
            for line in block_lines:
                if since_heading and since_heading % scene_size == 0:
                    scene += 1
                segments.append(
                    Segment(line.strip(), len(segments), scene, heading, "list")
                )
                since_heading += 1
            continue

        for piece in _split_long_block(block, max_segment_tokens):
            if since_heading and since_heading % scene_size == 0:
                scene += 1
            segments.append(
                Segment(piece, len(segments), scene, heading, "prose")
            )
            since_heading += 1

    return segments
