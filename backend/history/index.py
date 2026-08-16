# -*- coding: utf-8 -*-
"""
Search and checking over the vetted corpus.

Retrieval here spends zero LLM tokens and performs zero generation: it ranks
records with BM25 over accent-folded Vietnamese and returns them verbatim, or
returns nothing. That is what lets the tool promise it never invents — there is
no code path in which a claim comes from anywhere but a loaded, cited record.

The checker is deliberately conservative. It reports only what it can derive
from dates and the corpus itself, because a guardrail that guesses is a
guardrail that cries wolf, and one that cries wolf gets switched off.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.zero_mem.extract import _canonical
from backend.zero_mem.text import BM25Index, _strip_accents, content_tokens, tokenize

from .records import (
    DateTuple,
    HistoryError,
    HistoryRecord,
    format_date,
    load_records,
    parse_date,
)

# Terms that cannot appear in a scene before the given year without being an
# anachronism. Deliberately small and specific: a long fuzzy list produces
# false positives on ordinary Vietnamese prose.
ANACHRONISMS: List[Tuple[int, Tuple[str, ...], str]] = [
    (1830, (u"súng trường", u"sung truong", u"súng liên thanh", u"súng máy",
            u"sung may", u"đại liên"), u"súng nạp hậu / liên thanh"),
    (1804, (u"đầu máy hơi nước", u"tàu hơi nước", u"tau hoi nuoc",
            u"xe lửa", u"xe lua", u"đường sắt", u"duong sat"), u"động cơ hơi nước"),
    (1876, (u"điện thoại", u"dien thoai", u"điện báo", u"dien bao",
            u"telegraph"), u"viễn thông"),
    (1879, (u"bóng đèn điện", u"đèn điện", u"den dien", u"điện lực"), u"điện khí hóa"),
    (1826, (u"nhiếp ảnh", u"máy ảnh", u"may anh", u"chụp ảnh"), u"nhiếp ảnh"),
    (1885, (u"ô tô", u"xe hơi", u"xe hoi", u"động cơ đốt trong"), u"ô tô"),
    (1903, (u"máy bay", u"may bay", u"phi cơ"), u"hàng không"),
    (1928, (u"kháng sinh", u"penicillin", u"thuốc kháng sinh"), u"kháng sinh"),
    (1945, (u"máy tính", u"may tinh", u"vi tính", u"máy vi tính"), u"máy tính điện tử"),
    (1866, (u"vi khuẩn học", u"thuyết vi trùng"), u"thuyết vi trùng"),
]

_YEAR_IN_TEXT = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


class HistoryIndex(object):
    """A loaded corpus: searchable, and able to check a draft against itself."""

    def __init__(self, records: Sequence[HistoryRecord]):
        self.records: List[HistoryRecord] = list(records)
        self.by_id: Dict[str, HistoryRecord] = dict((r.id, r) for r in self.records)
        self._bm25 = BM25Index()
        self._order: List[str] = []
        for i, record in enumerate(self.records):
            self._order.append(record.id)
            self._bm25.add(i, self._indexable(record))

        # Which entities can anchor a date. A name tied to several dated
        # records cannot: "Quang Trung" spans 1753-1792, so a year written near
        # it proves nothing about which event the sentence means, and checking
        # it against every record he appears in flags correct prose as wrong.
        owners: Dict[str, set] = {}
        for record in self.records:
            if record.start is None:
                continue
            for entity in record.entities:
                key = _canonical(entity)
                if len(key) >= 4:
                    owners.setdefault(key, set()).add(record.id)
        surfaces: Dict[str, str] = {}
        for record in self.records:
            for entity in record.entities:
                surfaces.setdefault(_canonical(entity), entity)
        self._anchors: Dict[str, Tuple[str, str]] = dict(
            (key, (next(iter(ids)), surfaces.get(key, key)))
            for key, ids in owners.items() if len(ids) == 1)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _indexable(record: HistoryRecord) -> str:
        parts = [record.claim, " ".join(record.entities), " ".join(record.tags)]
        if record.note:
            parts.append(record.note)
        if record.divergence_note:
            parts.append(record.divergence_note)
        return "\n".join(p for p in parts if p)

    # -- search -------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 6,
        before: Optional[str] = None,
        after: Optional[str] = None,
        include_divergences: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Rank records for a query. Returns records verbatim, never a synthesis.

        ``before`` is the scene's in-story date: records dated later are
        dropped, so a chapter set in 1789 cannot be handed a fact from 1797 and
        write it as already true.
        """
        before_d = parse_date(before, end=True) if before else None
        after_d = parse_date(after, end=False) if after else None

        tokens = content_tokens(query) or tokenize(query)
        scores = self._bm25.scores(tokens)

        # An entity named in the query is a strong signal that BM25 alone,
        # working on short claims, underweights.
        wanted = set(_canonical(t) for t in re.split(r"[,;]| và | and ", query) if t.strip())
        query_folded = _canonical(query)

        ranked: List[Tuple[float, HistoryRecord]] = []
        for i, record in enumerate(self.records):
            score = scores.get(i, 0.0)
            for entity in record.entities:
                key = _canonical(entity)
                if key and (key in query_folded or key in wanted):
                    score += 2.5
            if score <= 0:
                continue
            if before_d and record.start and record.start > before_d:
                continue
            if after_d and record.end and record.end < after_d:
                continue
            if not include_divergences and record.diverges:
                continue
            ranked.append((score, record))

        ranked.sort(key=lambda pair: (-pair[0], pair[1].start or (9999, 12, 31)))
        return [self._hit(r, s) for s, r in ranked[:limit]]

    def timeline(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 40,
    ) -> List[Dict[str, Any]]:
        """Every record inside a window, in chronological order."""
        lo = parse_date(start, end=False) if start else None
        hi = parse_date(end, end=True) if end else None
        out = []
        for record in self.records:
            if record.start is None:
                continue
            if lo and (record.end or record.start) < lo:
                continue
            if hi and record.start > hi:
                continue
            out.append(record)
        out.sort(key=lambda r: r.start)
        return [self._hit(r, 0.0) for r in out[:limit]]

    @staticmethod
    def _hit(record: HistoryRecord, score: float) -> Dict[str, Any]:
        d = record.to_dict()
        d["score"] = round(score, 3)
        d["citation"] = record.citation()
        return d

    # -- checking -----------------------------------------------------------

    def check(self, text: str, scene_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Check a draft against the corpus.

        Reports only what is derivable without a model: a term that could not
        exist yet, a record whose events postdate the scene, and a year written
        next to a locked claim that the corpus dates differently. Everything
        reported carries the record and its source so the author can judge it.
        """
        findings: List[Dict[str, Any]] = []
        scene = parse_date(scene_date, end=True) if scene_date else None
        # Fold without stripping: offsets in `folded` must line up with `text`,
        # since year positions are measured against the original.
        folded = _strip_accents(text.lower())
        lowered = text.lower()

        findings.extend(self._anachronistic_terms(text, folded, scene))
        premature = self._premature_records(folded, scene)
        findings.extend(premature)
        # A thing that does not exist yet will also read as a date conflict.
        # Reporting both says the same thing twice and makes the panel look
        # noisier than the draft actually is.
        already = set(f["record_id"] for f in premature if f.get("record_id"))
        findings.extend(f for f in self._year_mismatches(text, lowered, folded)
                        if f.get("record_id") not in already)

        severity_rank = {"conflict": 0, "anachronism": 1, "warning": 2}
        findings.sort(key=lambda f: (severity_rank.get(f["kind"], 9), f.get("offset", 0)))
        return {
            "scene_date": format_date(scene) if scene else None,
            "checked_records": len(self.records),
            "findings": findings,
            "ok": not findings,
        }

    def _anachronistic_terms(self, text, folded, scene) -> List[Dict[str, Any]]:
        if scene is None:
            return []
        out = []
        for year, terms, label in ANACHRONISMS:
            if scene[0] >= year:
                continue
            for term in terms:
                key = _canonical(term)
                idx = folded.find(key)
                if idx == -1:
                    continue
                out.append({
                    "kind": "anachronism",
                    "term": term,
                    "offset": idx,
                    "message": u"“%s” (%s) chưa tồn tại vào năm %d; sớm nhất khoảng %d."
                               % (term, label, scene[0], year),
                    "record_id": None,
                    "sources": [],
                    "actions": ["rewrite", "accept_divergence"],
                })
                break
        return out

    def _premature_records(self, folded, scene) -> List[Dict[str, Any]]:
        """
        Something named in a scene set before that thing existed.

        Driven only by a record's explicit ``introduces:`` list. Inferring it
        from ``entities`` reported Phú Xuân as an anachronism because an
        emperor happened to die there, and Quang Trung as one in his own reign:
        appearing in a dated record says nothing about when a person or place
        began. A guardrail is allowed to miss things; it is not allowed to be
        wrong, because the first false positive is the one that gets it muted.
        """
        if scene is None:
            return []
        out = []
        for record in self.records:
            if record.start is None or record.start <= scene or not record.introduces:
                continue
            for name in record.introduces:
                key = _canonical(name)
                if len(key) < 3:
                    continue
                at = folded.find(key)
                if at == -1:
                    continue
                out.append({
                    "kind": "anachronism",
                    "term": name,
                    "offset": at,
                    "message": u"“%s” chỉ xuất hiện từ %s, sau thời điểm của cảnh này."
                               % (name, record.date_label),
                    "record_id": record.id,
                    "claim": record.claim,
                    "sources": list(record.sources),
                    "actions": ["rewrite", "accept_divergence"],
                })
                break
        return out

    def _year_mismatches(self, text, lowered, folded) -> List[Dict[str, Any]]:
        """
        A locked claim named near a year the corpus disagrees with.

        Two things keep this from crying wolf. Only entities unique to one
        dated record can anchor a year, and the year must sit inside a window
        around the mention — otherwise an unrelated date elsewhere in the
        chapter, or a name that spans a whole career, produces a false
        conflict. A guardrail that fires on correct prose gets switched off,
        and takes the real warnings with it.
        """
        out = []
        years = [(m.start(), int(m.group(1))) for m in _YEAR_IN_TEXT.finditer(text)]
        if not years:
            return out

        flagged = set()
        for key, (record_id, surface) in self._anchors.items():
            record = self.by_id[record_id]
            if not record.locked or record.start is None or record_id in flagged:
                continue
            at = folded.find(key)
            if at == -1:
                continue
            near = [(pos, y) for pos, y in years if abs(pos - at) <= 160]
            if not near:
                continue
            first, last = record.start[0], (record.end or record.start)[0]
            if record.ongoing:
                # A thing that began then and continued: only a year BEFORE it
                # started is wrong. An office founded in 1792 is still standing
                # in 1795, and flagging that made the checker unusable.
                if any(y >= first for _pos, y in near):
                    continue
                wrong = max(y for _pos, y in near)
            else:
                if any(first <= y <= last for _pos, y in near):
                    continue
                _pos, wrong = near[0]
            flagged.add(record_id)
            out.append({
                "kind": "conflict",
                "term": surface,
                "offset": at,
                "message": u"Bản thảo đặt “%s” vào năm %d, nhưng ghi chép xác "
                           u"nhận %s." % (surface, wrong, record.date_label),
                "record_id": record.id,
                "claim": record.claim,
                "sources": list(record.sources),
                "actions": ["rewrite", "accept_divergence"],
            })
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_yaml(path: Path):
    import yaml
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HistoryError("%s: invalid YAML — %s" % (path.name, exc))


def load_index(*dirs: Tuple[str, Optional[str]]) -> HistoryIndex:
    """
    Load a corpus from ``(scope, directory)`` pairs, later scopes overriding by id.

    Project-level files hold the shared factual record; a novel's own directory
    is where it declares where it departs from that record.
    """
    documents = []
    for scope, directory in dirs:
        if not directory:
            continue
        base = Path(directory)
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.y*ml")):
            documents.append((scope, path.name, _read_yaml(path)))
    return HistoryIndex(load_records(documents))
