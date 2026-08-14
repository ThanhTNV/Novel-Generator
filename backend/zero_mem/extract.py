"""
Non-generative entity extraction — the token-free NER behind the
entity-context graph.

Two sources, in precision order:

1. A **gazetteer** built from the project's own ``context/*.md`` files. The
   novel already declares its cast, places and artefacts as markdown headings
   and bold terms, which makes them the highest-precision entity list obtainable
   and costs nothing to compute.
2. A Unicode proper-noun scanner for names the context files have not caught
   yet, plus typed literals (numbers, dates, chapter references).

Vietnamese matters here: names are multi-syllable and each syllable is
capitalised ("Văn Tâm", "Đồ Lục", "Thầy Cao"), and sentences frequently begin
with accented capitals, so the scanner is Unicode-aware and conservative.
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from .text import FOLDED_STOPWORDS, STOPWORDS, _strip_accents

PERSON = "PERSON"
PLACE = "PLACE"
ITEM = "ITEM"
CONCEPT = "CONCEPT"
NUMBER = "NUMBER"
CHAPTER = "CHAPTER"
PROPER = "PROPER"


class Mention(NamedTuple):
    key: str      # canonical, accent-folded, lowercased
    name: str     # surface form as written
    type: str
    count: int


_HEADING_RE = re.compile(r"^#{2,6}\s+(.+?)\s*$", re.MULTILINE)
# Bold terms only count as entities when they are NOT immediately followed by a
# colon. This project writes attributes as "- **Ngoại hình**: ...", and those
# labels repeat under every character; admitting them would create hub nodes
# that connect the entire cast and smear PageRank across all of it.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*(?!\s*[:：])")
_PAREN_RE = re.compile(r"\s*[(（].*?[)）]\s*$")
_TRAILING_ROLE = re.compile(r"\s*[–—-]\s*.*$")

# Vietnamese honorifics/titles that precede a name; kept with the name because
# "Thầy Cao" is how the text refers to that character.
_TITLES = ("thầy", "cô", "ông", "bà", "anh", "chị", "em", "chú", "bác", "dì", "cậu")

_NUM_RE = re.compile(r"\b\d[\d.,]*\b")
_CHAPTER_RE = re.compile(r"\b(?:chương|chapter|ch\.?)\s*(\d+)\b", re.IGNORECASE)


def _canonical(name: str) -> str:
    return _strip_accents(name.strip().lower())


def _clean_heading(raw: str) -> str:
    """'Văn Tâm (Main Character – 21 tuổi)' -> 'Văn Tâm'."""
    name = _PAREN_RE.sub("", raw).strip()
    name = _TRAILING_ROLE.sub("", name).strip()
    return name.strip(" :–—-")


# Structural headings in a story bible ("Arc 1: ...", "Kết thúc dự kiến",
# "Chủng tộc trí tuệ khác là ai?") organise the document; they are not people,
# places or artefacts, and indexing them as entities only adds noise.
_SECTION_PREFIXES = (
    "nhân vật phụ", "ghi chú", "notes", "arc ", "cấu trúc", "kết thúc",
    "các địa điểm", "tổng quan", "overview", "outline", "structure", "timeline",
)


def _is_section_label(name: str) -> bool:
    low = name.strip().lower()
    if low.endswith("?"):
        return True
    return low.startswith(_SECTION_PREFIXES)


class Gazetteer:
    """Known entities harvested from the project's context documents."""

    def __init__(self) -> None:
        self._by_key: Dict[str, Tuple[str, str]] = {}   # key -> (name, type)
        self._patterns: List[Tuple[re.Pattern, str]] = []

    def add(self, name: str, etype: str) -> None:
        name = name.strip()
        if len(name) < 2:
            return
        key = _canonical(name)
        if not key or key in self._by_key:
            return
        # Skip anything that is purely a stopword/section label.
        if key in STOPWORDS:
            return
        self._by_key[key] = (name, etype)

    def finalize(self) -> None:
        """Compile matchers, longest name first so 'Văn Tâm' beats 'Tâm'."""
        self._patterns = []
        for key in sorted(self._by_key, key=len, reverse=True):
            name, etype = self._by_key[key]
            # Match the accented surface form OR its accent-folded spelling.
            variants = {re.escape(name), re.escape(_strip_accents(name))}
            body = "|".join(sorted(variants, key=len, reverse=True))
            self._patterns.append(
                (re.compile(r"(?<![\w])(?:%s)(?![\w])" % body, re.IGNORECASE | re.UNICODE), key)
            )

    def __len__(self) -> int:
        return len(self._by_key)

    def names(self) -> List[Tuple[str, str, str]]:
        return [(k, v[0], v[1]) for k, v in self._by_key.items()]

    def lookup(self, key: str) -> Optional[Tuple[str, str]]:
        return self._by_key.get(key)

    def find(self, text: str) -> Dict[str, int]:
        """Count gazetteer hits in text. Returns key -> count."""
        found: Dict[str, int] = {}
        for pattern, key in self._patterns:
            n = len(pattern.findall(text))
            if n:
                found[key] = found.get(key, 0) + n
        return found


def _type_for_source(filename: str) -> str:
    low = filename.lower()
    if "character" in low or "nhan-vat" in low or "nhanvat" in low:
        return PERSON
    if "location" in low or "place" in low or "dia-diem" in low:
        return PLACE
    return CONCEPT


def build_gazetteer(
    context_dir: Optional[str] = None,
    extra: Optional[Iterable[Tuple[str, str]]] = None,
) -> Gazetteer:
    """
    Harvest entities from ``context/*.md``: section headings name the entity,
    bold terms inside name its attributes and related artefacts.
    """
    gaz = Gazetteer()
    if context_dir:
        directory = Path(context_dir)
        if directory.exists():
            for path in sorted(directory.glob("**/*.md")):
                etype = _type_for_source(path.name)
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                for raw in _HEADING_RE.findall(text):
                    name = _clean_heading(raw)
                    if not name or _is_section_label(name):
                        continue
                    gaz.add(name, etype)
                for bold in _BOLD_RE.findall(text):
                    cleaned = _clean_heading(bold).strip(" :")
                    # Bold is used for field labels ("Ngoại hình") as well as
                    # proper nouns; keep only capitalised multi-word artefacts.
                    if (
                        2 <= len(cleaned) <= 40
                        and cleaned[:1].isupper()
                        and " " in cleaned
                        and not _is_section_label(cleaned)
                    ):
                        gaz.add(cleaned, ITEM)
    for name, etype in (extra or []):
        gaz.add(name, etype)
    gaz.finalize()
    return gaz


# ---------------------------------------------------------------------------
# Unicode proper-noun scanner (for names not yet in the gazetteer)
# ---------------------------------------------------------------------------

def _is_upper_letter(ch: str) -> bool:
    return ch.isalpha() and ch.isupper()


def _scan_proper_nouns(text: str, max_words: int = 6) -> Dict[str, Tuple[str, int]]:
    """
    Find runs of capitalised words. Conservative on purpose: a run is only kept
    if it is multi-word, or a single word that also appears mid-sentence
    somewhere (so sentence-initial ordinary words are not mistaken for names).
    """
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[^\W_]+", text, re.UNICODE)]
    runs: Dict[str, Tuple[str, int]] = {}
    mid_sentence_caps: Set[str] = set()

    # Which capitalised words appear when NOT at a sentence start? Markdown
    # formatting ("- **Vai trò**:", "## Heading") also starts a "sentence" —
    # without this, every bold field label becomes a fake proper noun.
    boundary_chars = ".!?…:\n*#-•|>\"'“”‘’([{"
    for word, start, _end in tokens:
        if not _is_upper_letter(word[:1]):
            continue
        prefix = text[:start].rstrip()
        if prefix and prefix[-1] not in boundary_chars:
            mid_sentence_caps.add(_canonical(word))

    i = 0
    while i < len(tokens):
        word, start, end = tokens[i]
        if not _is_upper_letter(word[:1]):
            i += 1
            continue
        run_words = [word]
        run_end = end
        j = i + 1
        while j < len(tokens) and len(run_words) < max_words:
            nxt, nstart, nend = tokens[j]
            between = text[run_end:nstart]
            if between != " " or not _is_upper_letter(nxt[:1]):
                break
            run_words.append(nxt)
            run_end = nend
            j += 1

        surface = " ".join(run_words)
        key = _canonical(surface)
        keep = False
        if len(run_words) > 1:
            keep = any(_canonical(w) not in FOLDED_STOPWORDS for w in run_words)
        else:
            low = _canonical(word)
            keep = (
                low not in FOLDED_STOPWORDS
                and len(word) > 1
                and (low in mid_sentence_caps or word.isupper())
            )
        if keep:
            name, count = runs.get(key, (surface, 0))
            runs[key] = (name, count + 1)
        i = j if j > i + 1 else i + 1
    return runs


def extract_entities(
    text: str,
    gazetteer: Optional[Gazetteer] = None,
    scan_proper_nouns: bool = True,
) -> List[Mention]:
    """Deterministic entity extraction. No model, no tokens."""
    found: Dict[str, Mention] = {}

    def push(key: str, name: str, etype: str, count: int) -> None:
        if not key:
            return
        prev = found.get(key)
        if prev is None:
            found[key] = Mention(key, name, etype, count)
        else:
            found[key] = Mention(prev.key, prev.name, prev.type, prev.count + count)

    if gazetteer is not None:
        for key, count in gazetteer.find(text).items():
            entry = gazetteer.lookup(key)
            if entry:
                push(key, entry[0], entry[1], count)

    if scan_proper_nouns:
        for key, (surface, count) in _scan_proper_nouns(text).items():
            if key in found:
                continue
            # Attach a leading Vietnamese title to the name it qualifies.
            push(key, surface, PROPER, count)

    for m in _CHAPTER_RE.finditer(text):
        push("chapter:" + m.group(1), "Chương " + m.group(1), CHAPTER, 1)

    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        if len(raw) >= 2:
            push("num:" + raw, raw, NUMBER, 1)

    return list(found.values())
