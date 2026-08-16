# -*- coding: utf-8 -*-
"""
Historical records — the vetted corpus the LLM is allowed to cite.

The integrity claim of this module is narrow and worth stating exactly: it
cannot tell you a fact is *true*. What it guarantees is that every claim the
tool can return is traceable to a source the author vetted, and that the tool
never produces a claim that is not in the corpus.

Three rules enforce that, all at load time:

1. **A record without a source is rejected**, not skipped. A corpus that
   silently drops malformed records would let an author believe a fact is
   backed when it is not, which is worse than failing loudly.
2. **Ids are unique per scope.** A novel may deliberately override a
   project-level record by reusing its id; two records fighting for one id
   inside the same file is a mistake.
3. **Dates must parse.** An unparseable date cannot be checked for
   anachronism, so it would silently disable the guardrail for that record.

Alternate history needs one more distinction that a plain fact base does not:

* ``locked``     — the story must not contradict this. Pre-divergence canon.
* ``diverges``   — the novel departs here on purpose. Contradicting it is the
                   *point*, so the checker must never flag it.

Without that split, a checker flags the premise of the book as an error.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CONFIDENCE = ("attested", "probable", "disputed")

# (year, month, day) with month/day defaulting at each end of the span.
DateTuple = Tuple[int, int, int]

_YEAR = re.compile(r"^(-?\d{1,4})$")
_YEAR_MONTH = re.compile(r"^(-?\d{1,4})-(\d{1,2})$")
_FULL = re.compile(r"^(-?\d{1,4})-(\d{1,2})-(\d{1,2})$")

_DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


class HistoryError(ValueError):
    """A corpus problem the author has to fix before the tool can be trusted."""


def parse_date(value: Any, end: bool = False) -> Optional[DateTuple]:
    """
    Parse ``1789``, ``1789-09`` or ``1789-09-16`` into a comparable tuple.

    ``end=True`` widens a partial date to the last instant it could mean, so a
    record dated ``1789`` spans the whole year rather than collapsing to 1 Jan.
    Returns None only for a genuinely absent date.
    """
    if value is None or value == "":
        return None
    # PyYAML turns an unquoted 1789-09-16 into a datetime.date.
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return (value.year, value.month, value.day)
    if isinstance(value, int):
        return (value, 12, 31) if end else (value, 1, 1)

    text = str(value).strip()
    m = _FULL.match(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12) or not (1 <= d <= _DAYS_IN_MONTH[mo - 1]):
            raise HistoryError("impossible date: %r" % text)
        return (y, mo, d)
    m = _YEAR_MONTH.match(text)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if not (1 <= mo <= 12):
            raise HistoryError("impossible month: %r" % text)
        return (y, mo, _DAYS_IN_MONTH[mo - 1]) if end else (y, mo, 1)
    m = _YEAR.match(text)
    if m:
        y = int(m.group(1))
        return (y, 12, 31) if end else (y, 1, 1)
    raise HistoryError("unparseable date %r (use YYYY, YYYY-MM or YYYY-MM-DD)" % text)


def format_date(d: Optional[DateTuple]) -> str:
    if d is None:
        return "?"
    return "%04d-%02d-%02d" % d


class HistoryRecord(object):
    """One vetted historical claim."""

    __slots__ = ("id", "claim", "start", "end", "entities", "tags", "confidence",
                 "sources", "locked", "diverges", "divergence_note", "note",
                 "ongoing", "introduces", "anchors", "scope", "origin")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    # -- derived ------------------------------------------------------------

    @property
    def date_label(self) -> str:
        if self.start is None:
            return "không rõ niên đại"
        if self.end and self.end[0] != self.start[0]:
            return "%d–%d" % (self.start[0], self.end[0])
        if self.start[1] == 1 and self.start[2] == 1 and self.end and self.end[1] == 12:
            return str(self.start[0])
        return format_date(self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "date": self.date_label,
            "date_start": format_date(self.start),
            "date_end": format_date(self.end),
            "entities": list(self.entities),
            "introduces": list(self.introduces),
            "anchors": list(self.anchors),
            "tags": list(self.tags),
            "confidence": self.confidence,
            "sources": list(self.sources),
            "locked": self.locked,
            "ongoing": self.ongoing,
            "diverges": self.diverges,
            "divergence_note": self.divergence_note,
            "note": self.note,
            "scope": self.scope,
            "origin": self.origin,
        }

    def citation(self) -> str:
        """One line the model can copy verbatim without inventing provenance."""
        return u"[%s · %s] %s — nguồn: %s" % (
            self.id, self.date_label, self.claim, "; ".join(self.sources))


def _as_list(value: Any, field: str, rid: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if not isinstance(item, (str, int, float)):
                raise HistoryError("%s: %s must be a list of strings" % (rid, field))
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    raise HistoryError("%s: %s must be a string or list" % (rid, field))


def build_record(raw: Dict[str, Any], scope: str, origin: str) -> HistoryRecord:
    """Validate one mapping into a record, or raise HistoryError explaining why."""
    if not isinstance(raw, dict):
        raise HistoryError("%s: every entry must be a mapping, got %s"
                           % (origin, type(raw).__name__))

    rid = str(raw.get("id") or "").strip()
    if not rid:
        raise HistoryError("%s: every record needs an 'id'" % origin)
    if not re.match(r"^[a-z0-9][a-z0-9._-]{0,63}$", rid):
        raise HistoryError("%s: id %r must be lowercase letters, digits, . _ -" % (origin, rid))

    claim = str(raw.get("claim") or "").strip()
    if not claim:
        raise HistoryError("%s: record %r has no 'claim'" % (origin, rid))

    sources = _as_list(raw.get("sources") or raw.get("source"), "sources", rid)
    if not sources:
        # The load-time refusal that the whole design rests on.
        raise HistoryError(
            "%s: record %r has no source. Every claim the tool can return must "
            "be traceable, so an uncited record is refused rather than loaded."
            % (origin, rid))

    confidence = str(raw.get("confidence") or "attested").strip().lower()
    if confidence not in CONFIDENCE:
        raise HistoryError("%s: record %r has confidence %r; expected one of %s"
                           % (origin, rid, confidence, ", ".join(CONFIDENCE)))

    start = parse_date(raw.get("date"), end=False)
    end_raw = raw.get("until") or raw.get("date_end")
    if end_raw is not None:
        end = parse_date(end_raw, end=True)
    else:
        end = parse_date(raw.get("date"), end=True)
    if start and end and end < start:
        raise HistoryError("%s: record %r ends before it starts" % (origin, rid))

    diverges = bool(raw.get("diverges"))
    note = raw.get("divergence_note")
    if diverges and not (note and str(note).strip()):
        raise HistoryError(
            "%s: record %r is marked diverges but has no divergence_note. The "
            "note is what tells the writer what happens instead." % (origin, rid))

    # An institution founded in 1792 still stands in 1795; a battle fought on
    # one day did not also happen three years later. Treating both as point
    # events made the checker flag every later mention of a policy or office.
    # An explicit end implies a span, so `until` infers ongoing.
    ongoing = raw.get("ongoing")
    ongoing = bool(ongoing) if ongoing is not None else (end_raw is not None)

    return HistoryRecord(
        id=rid,
        claim=claim,
        start=start,
        end=end,
        ongoing=ongoing,
        entities=_as_list(raw.get("entities"), "entities", rid),
        # Things this record brings into existence. Only these can trigger a
        # "does not exist yet" flag, and it is deliberately opt-in: inferring
        # it made the checker report Phú Xuân as an anachronism because an
        # emperor happened to die there. A guardrail may miss; it must not lie.
        introduces=_as_list(raw.get("introduces"), "introduces", rid),
        # Terms that pin THIS event to THIS date — usually where it happened,
        # not who was there. Only these can raise a "wrong year" flag.
        # Inferring them from `entities` flagged a 1792 scene for naming
        # "quân Thanh", because the Qing army appeared in exactly one record
        # and so looked distinctive; armies, dynasties and people persist
        # across events and date nothing. Falls back to `introduces`, since a
        # thing's founding record does pin its own date.
        anchors=_as_list(raw.get("anchors"), "anchors", rid),
        tags=_as_list(raw.get("tags"), "tags", rid),
        confidence=confidence,
        sources=sources,
        # A divergence point is never 'locked': contradicting it is the premise
        # of the book, and flagging it would make the checker unusable.
        locked=bool(raw.get("locked", True)) and not diverges,
        diverges=diverges,
        divergence_note=str(note).strip() if note else "",
        note=str(raw.get("note") or "").strip(),
        scope=scope,
        origin=origin,
    )


def load_records(documents: Iterable[Tuple[str, str, Any]]) -> List[HistoryRecord]:
    """
    Build records from ``(scope, origin, parsed_yaml)`` triples.

    Later documents override earlier ones by id, which is how a novel declares
    its own divergence on top of a shared factual corpus.
    """
    by_id: Dict[str, HistoryRecord] = {}
    for scope, origin, payload in documents:
        if payload is None:
            continue
        entries = payload.get("records") if isinstance(payload, dict) else payload
        if entries is None:
            continue
        if not isinstance(entries, (list, tuple)):
            raise HistoryError("%s: expected a list of records" % origin)

        seen_here = set()
        for raw in entries:
            record = build_record(raw, scope, origin)
            if record.id in seen_here:
                raise HistoryError("%s: duplicate id %r in the same file"
                                   % (origin, record.id))
            seen_here.add(record.id)
            by_id[record.id] = record
    return [by_id[k] for k in sorted(by_id)]
