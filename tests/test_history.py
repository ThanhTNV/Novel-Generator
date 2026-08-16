# -*- coding: utf-8 -*-
"""
The historical guardrail.

The promise this feature makes is narrow and these tests hold it to exactly
that: the tool cannot return a claim that is not in the vetted corpus, an
uncited record is refused rather than loaded, and a scene is never handed a
fact from its own future. What it does NOT promise is that the corpus is true —
that is the author's job, and no test can do it for them.

The alternate-history twist matters throughout: this novel deliberately
contradicts the record, so a checker that flagged the premise would be useless.
"""

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.history import TOOLS, dispatch  # noqa: E402
from backend.history.index import HistoryIndex, load_index  # noqa: E402
from backend.history.records import (  # noqa: E402
    HistoryError,
    build_record,
    load_records,
    parse_date,
)

import yaml  # noqa: E402


CORPUS = textwrap.dedent(u"""
records:
  - id: ngoc-hoi
    claim: "Quang Trung đại phá quân Thanh tại Ngọc Hồi – Đống Đa"
    date: 1789-01-30
    entities: ["Quang Trung", "Ngọc Hồi", "Đống Đa", "quân Thanh"]
    confidence: attested
    sources: ["Hoàng Lê nhất thống chí, hồi 14"]

  - id: chieu-khuyen-nong
    claim: "Ban Chiếu Khuyến nông"
    date: 1789
    entities: ["Chiếu Khuyến nông"]
    confidence: attested
    sources: ["Đại Nam thực lục"]

  - id: vien-sung-chinh
    claim: "Lập Viện Sùng Chính, Nguyễn Thiếp làm viện trưởng"
    date: 1791
    entities: ["Viện Sùng Chính", "Nguyễn Thiếp"]
    introduces: ["Viện Sùng Chính"]
    confidence: attested
    sources: ["La Sơn phu tử, Hoàng Xuân Hãn"]

  - id: quang-trung-death
    claim: "Quang Trung băng hà tại Phú Xuân"
    date: 1792-09-16
    entities: ["Quang Trung", "Phú Xuân"]
    confidence: attested
    sources: ["Đại Nam thực lục"]
""")

DIVERGENCE = textwrap.dedent(u"""
records:
  - id: quang-trung-death
    claim: "Quang Trung băng hà tại Phú Xuân"
    date: 1792-09-16
    entities: ["Quang Trung", "Phú Xuân"]
    confidence: attested
    sources: ["Đại Nam thực lục"]
    diverges: true
    divergence_note: "Trong truyện ông sống sót và trị vì tới 1802."
""")


def _index(*docs):
    return HistoryIndex(load_records(
        [("project", "t%d.yaml" % i, yaml.safe_load(d)) for i, d in enumerate(docs)]))


@pytest.fixture()
def index():
    return _index(CORPUS)


@pytest.fixture()
def alt(index):
    """Corpus plus this novel's divergence declaration."""
    return _index(CORPUS, DIVERGENCE)


# ---------------------------------------------------------------------------
# Load-time integrity — the guardrail that everything else rests on
# ---------------------------------------------------------------------------

def test_a_record_without_a_source_is_refused_not_skipped():
    """
    Silently dropping an uncited record would let an author believe a claim is
    backed when it is not. Refusing loudly is the whole integrity story.
    """
    with pytest.raises(HistoryError) as exc:
        build_record({"id": "x", "claim": "điều gì đó"}, "project", "t.yaml")
    assert "no source" in str(exc.value)


def test_a_record_needs_an_id_and_a_claim():
    for raw in ({"claim": "x", "sources": ["s"]}, {"id": "x", "sources": ["s"]}):
        with pytest.raises(HistoryError):
            build_record(raw, "project", "t.yaml")


def test_unparseable_dates_are_refused():
    """An unparseable date silently disables anachronism checking for a record."""
    with pytest.raises(HistoryError):
        build_record({"id": "x", "claim": "c", "sources": ["s"], "date": "khoảng 1789"},
                     "project", "t.yaml")
    with pytest.raises(HistoryError):
        build_record({"id": "x", "claim": "c", "sources": ["s"], "date": "1789-13-01"},
                     "project", "t.yaml")


def test_a_divergence_must_explain_itself():
    with pytest.raises(HistoryError) as exc:
        build_record({"id": "x", "claim": "c", "sources": ["s"], "diverges": True},
                     "project", "t.yaml")
    assert "divergence_note" in str(exc.value)


def test_duplicate_ids_in_one_file_are_refused():
    doc = yaml.safe_load(u"""
records:
  - {id: a, claim: x, sources: [s]}
  - {id: a, claim: y, sources: [s]}
""")
    with pytest.raises(HistoryError):
        load_records([("project", "t.yaml", doc)])


def test_a_novel_may_override_a_project_record_by_id(alt):
    """That is how a novel declares where it departs from the shared record."""
    record = alt.by_id["quang-trung-death"]
    assert record.diverges is True
    assert record.scope == "novel" or record.divergence_note
    assert len(alt) == len(_index(CORPUS)), "override must not duplicate the record"


@pytest.mark.parametrize("raw,expect_start,expect_end", [
    ("1789", (1789, 1, 1), (1789, 12, 31)),
    ("1789-09", (1789, 9, 1), (1789, 9, 30)),
    ("1789-09-16", (1789, 9, 16), (1789, 9, 16)),
])
def test_partial_dates_widen_to_their_full_span(raw, expect_start, expect_end):
    assert parse_date(raw, end=False) == expect_start
    assert parse_date(raw, end=True) == expect_end


# ---------------------------------------------------------------------------
# Search — closed corpus, nothing invented
# ---------------------------------------------------------------------------

def test_search_returns_records_verbatim_with_sources(index):
    hits = index.search(u"Ngọc Hồi Đống Đa")
    assert hits
    top = hits[0]
    assert top["id"] == "ngoc-hoi"
    assert top["sources"] == [u"Hoàng Lê nhất thống chí, hồi 14"]
    assert top["claim"] == index.by_id["ngoc-hoi"].claim


def test_search_finds_nothing_rather_than_guessing(index):
    """Out-of-corpus queries return nothing at all, never an approximation."""
    assert index.search(u"trận Bạch Đằng năm 938") == []
    assert index.search(u"Napoleon") == []


def test_the_empty_result_tells_the_model_not_to_invent():
    """
    The rendered 'nothing found' is the load-bearing string of the whole
    feature: a model handed silence will supply a plausible fact instead.
    """
    from backend.history import _render
    text = _render([])
    assert u"KHÔNG CÓ GHI CHÉP" in text
    assert u"Đừng suy đoán" in text


def test_scene_date_hides_the_future(index):
    """A chapter set in 1789 must not be handed a fact from 1791."""
    unbounded = [h["id"] for h in index.search(u"Viện Sùng Chính Nguyễn Thiếp")]
    assert "vien-sung-chinh" in unbounded

    bounded = [h["id"] for h in index.search(u"Viện Sùng Chính Nguyễn Thiếp", before="1789")]
    assert "vien-sung-chinh" not in bounded


def test_unaccented_query_finds_accented_records(index):
    """Vietnamese is routinely typed without diacritics."""
    assert [h["id"] for h in index.search(u"Ngoc Hoi Dong Da")] == \
           [h["id"] for h in index.search(u"Ngọc Hồi Đống Đa")]


def test_timeline_is_chronological(index):
    """A year-only record starts on 1 Jan, so it precedes a dated event that year."""
    ids = [h["id"] for h in index.timeline("1789", "1792")]
    assert ids == ["chieu-khuyen-nong", "ngoc-hoi", "vien-sung-chinh", "quang-trung-death"]


def test_rendered_output_always_carries_a_source(index):
    from backend.history import _render
    text = _render(index.search(u"Quang Trung"))
    for line in text.splitlines():
        if line.startswith(u"●"):
            assert u"[" in line
    assert u"nguồn:" in text


def test_divergence_is_flagged_loudly_in_what_the_model_reads(alt):
    """
    Without this the model reads 'Quang Trung băng hà 1792' and faithfully
    writes the emperor's death into a book whose premise is that he lived.
    """
    from backend.history import _render
    text = _render(alt.search(u"Quang Trung băng hà"))
    assert u"RẼ NHÁNH" in text
    assert u"sống sót" in text
    assert u"KHÔNG theo sử liệu gốc" in text


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

def test_anachronistic_technology_is_caught(index):
    result = index.check(u"Quân Thanh nổ súng trường vào đội hình Tây Sơn.", scene_date="1789")
    assert not result["ok"]
    kinds = [f["kind"] for f in result["findings"]]
    assert "anachronism" in kinds
    assert any(u"súng trường" in f["term"] for f in result["findings"])


def test_the_same_technology_is_fine_once_it_exists(index):
    assert index.check(u"Quân lính nổ súng trường.", scene_date="1900")["ok"]


def test_a_scene_cannot_mention_its_own_future(index):
    """1789 scene naming an institution founded in 1791."""
    result = index.check(u"Khải bước vào Viện Sùng Chính giữa mùa xuân.", scene_date="1789")
    assert not result["ok"]
    finding = [f for f in result["findings"] if f.get("record_id") == "vien-sung-chinh"][0]
    assert finding["sources"], "a finding must cite the record it came from"


def test_a_wrong_year_beside_a_locked_claim_is_a_conflict(index):
    result = index.check(
        u"Mùa xuân năm 1787, Quang Trung đại phá quân Thanh tại Ngọc Hồi.",
        scene_date="1790")
    conflicts = [f for f in result["findings"] if f["kind"] == "conflict"]
    assert conflicts
    assert conflicts[0]["record_id"] == "ngoc-hoi"
    assert conflicts[0]["sources"]


def test_the_right_year_is_not_a_conflict(index):
    result = index.check(
        u"Mùa xuân năm 1789, Quang Trung đại phá quân Thanh tại Ngọc Hồi.",
        scene_date="1790")
    assert [f for f in result["findings"] if f["kind"] == "conflict"] == []


def test_contradicting_a_declared_divergence_is_never_flagged(alt):
    """
    The premise of the book is that Quang Trung lives past 1792. A checker that
    flagged that would be switched off within a day, taking the real warnings
    with it.
    """
    draft = u"Năm 1795, Quang Trung vẫn ngự tại Phú Xuân, tay còn cầm bản tấu."
    result = alt.check(draft, scene_date="1795")
    assert [f for f in result["findings"] if f.get("record_id") == "quang-trung-death"] == []


def test_the_same_draft_conflicts_when_the_divergence_is_not_declared(index):
    """Mirror of the test above: without the declaration it IS a conflict."""
    draft = u"Năm 1795, Quang Trung vẫn ngự tại Phú Xuân."
    result = index.check(draft, scene_date="1795")
    assert [f for f in result["findings"] if f.get("record_id") == "quang-trung-death"]


def test_a_person_alive_in_the_scene_is_not_an_anachronism(index):
    """
    Quang Trung appears in records through 1792, but he was alive in 1789.
    Only a name denoting exactly one dated thing can prove non-existence —
    otherwise the checker flags the emperor as an anachronism in his own reign.
    """
    result = index.check(u"Quang Trung đứng trên bậc thềm điện, im lặng rất lâu.",
                         scene_date="1789")
    assert result["ok"], [f["message"] for f in result["findings"]]


def test_a_place_that_outlives_the_scene_is_not_an_anachronism(index):
    result = index.check(u"Gió từ sông thổi vào Phú Xuân.", scene_date="1789")
    assert result["ok"], [f["message"] for f in result["findings"]]


def test_something_that_does_not_exist_yet_is_still_caught(index):
    """The narrowing must not disarm the check it exists for."""
    result = index.check(u"Khải bước vào Viện Sùng Chính.", scene_date="1789")
    assert [f for f in result["findings"] if f["record_id"] == "vien-sung-chinh"]


def test_an_ongoing_institution_may_be_mentioned_after_it_is_founded():
    """An office founded in 1791 still stands in 1795."""
    ongoing = _index(CORPUS, textwrap.dedent(u"""
    records:
      - id: vien-sung-chinh
        claim: "Lập Viện Sùng Chính"
        date: 1791
        ongoing: true
        entities: ["Viện Sùng Chính"]
        introduces: ["Viện Sùng Chính"]
        confidence: attested
        sources: ["La Sơn phu tử"]
    """))
    draft = u"Năm 1795, Khải ghé qua Viện Sùng Chính."
    assert ongoing.check(draft, scene_date="1795")["ok"]
    # ...but a year before it existed is still wrong.
    assert not ongoing.check(u"Năm 1780, Khải ghé qua Viện Sùng Chính.",
                             scene_date="1780")["ok"]


def test_until_implies_ongoing():
    idx = _index(textwrap.dedent(u"""
    records:
      - id: span
        claim: "Chế độ nhất khẩu thông thương"
        date: 1757
        until: 1842
        entities: ["Quảng Châu"]
        confidence: attested
        sources: ["The Canton Trade"]
    """))
    assert idx.by_id["span"].ongoing is True


def test_one_record_is_reported_once(index):
    """
    A thing that does not exist yet also reads as a date conflict. Saying both
    makes the panel look noisier than the draft is.
    """
    result = index.check(u"Năm 1780, Khải bước vào Viện Sùng Chính.", scene_date="1780")
    ids = [f["record_id"] for f in result["findings"] if f.get("record_id")]
    assert ids.count("vien-sung-chinh") == 1


def test_every_finding_is_actionable(index):
    result = index.check(u"Năm 1787 Quang Trung phá quân Thanh ở Ngọc Hồi bằng súng trường.",
                         scene_date="1789")
    assert result["findings"]
    for f in result["findings"]:
        assert f["message"]
        assert "rewrite" in f["actions"] and "accept_divergence" in f["actions"]


def test_a_clean_draft_reports_ok(index):
    result = index.check(u"Gió lạnh thổi qua sân điện. Khải cúi đầu, im lặng.",
                         scene_date="1789")
    assert result["ok"] and result["findings"] == []


def test_checking_without_a_scene_date_still_catches_year_conflicts(index):
    """Date-free checks lose anachronism detection but keep contradiction."""
    result = index.check(u"Năm 1787, Quang Trung đại phá quân Thanh tại Ngọc Hồi.")
    assert [f for f in result["findings"] if f["kind"] == "conflict"]


# ---------------------------------------------------------------------------
# The tool surface
# ---------------------------------------------------------------------------

def test_tool_schemas_are_well_formed():
    names = set()
    for tool in TOOLS:
        assert tool["name"] not in names
        names.add(tool["name"])
        assert tool["description"].strip()
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        for required in schema.get("required", []):
            assert required in schema["properties"], required
    assert {"search_history", "history_timeline"} <= names


def test_the_tool_description_forbids_invention():
    tool = [t for t in TOOLS if t["name"] == "search_history"][0]
    assert u"không tự bịa" in tool["description"] or u"TUYỆT ĐỐI không" in tool["description"]
    assert "Never invent" in tool["description"]


def test_the_real_project_corpus_loads_and_is_fully_cited():
    """The shipped history/ must satisfy its own rules."""
    index = load_index(("project", str(ROOT / "history")))
    assert len(index) > 0, "the seed corpus should not be empty"
    for record in index.records:
        assert record.sources, record.id
        assert record.claim.strip()
        assert record.confidence in ("attested", "probable", "disputed")
