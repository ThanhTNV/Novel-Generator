# -*- coding: utf-8 -*-
"""
Zero-Mem tests. Each test targets one of the failures that motivated replacing
the ChromaDB pipeline: Vietnamese text not splitting, stale chunks surviving
re-ingest, mis-attributed context, and fragmented evidence.

All tests run with the hashed embedder (no model download, no network).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.zero_mem import (  # noqa: E402
    build_gazetteer,
    extract_entities,
    segment_document,
    split_sentences,
)
from backend.zero_mem.embeddings import HashEmbedder  # noqa: E402
from backend.zero_mem.engine import ZeroMemEngine  # noqa: E402

VI_PARA = (
    "Tâm bước vào phòng thí nghiệm. Đồ Lục nằm im trên bàn gỗ. "
    "Ánh sáng xám phủ khắp căn phòng. Cậu nhặt cuốn sổ lên. "
    "Ở đó có gì đó rất lạ. Ừ, đúng là kỳ quái thật."
)

CHARACTERS_MD = """# Nhân vật

## Văn Tâm (Main Character – 21 tuổi)

- **Vai trò**: Nhân vật chính
- **Ngoại hình**: Cao 1m78, dáng hơi gầy, tóc đen rối nhẹ, mắt một mí sâu.
- **Năng lực**: Sở hữu Đồ Lục, cuốn sổ đen ghi 108 Vật Chất.

## Nguyên Khang (Bạn thân – 21 tuổi)

- **Vai trò**: Deuteragonist, bạn thân
- **Ngoại hình**: Cao 1m75, chắc khỏe, tóc cắt ngắn gọn, da ngăm.

## Thầy Cao (Mentor đã mất)

- **Vai trò**: Mentor, catalyst cho sự kiện chính
- **Ngoại hình**: Khoảng 50 tuổi, dáng gầy, tóc bạc, đeo kính.
- **Trạng thái**: Đã chết trong lúc nghiên cứu Vật Chất mới.
"""

CHAPTER_1 = """# Chương 1

Văn Tâm ngồi trong giảng đường, gục đầu xuống bàn. Cậu mơ về căn nhà trên Mặt Trăng.

Nguyên Khang huých vai cậu. "Dậy đi ông, thầy nhìn kìa."

Buổi chiều hôm đó, hai đứa nghe tin Thầy Cao qua đời trong phòng thí nghiệm."""

CHAPTER_2 = """# Chương 2

Sương xám phủ kín trường. Văn Tâm mở vali di vật của Thầy Cao và thấy Đồ Lục.

Cuốn sổ đen hút lấy tay cậu. Một dòng chữ hiện lên: "Vật Chất thứ nhất: Thủy Ngân Sống."

Từ hôm đó, Văn Tâm không còn là sinh viên bình thường nữa."""


@pytest.fixture()
def ctx_dir(tmp_path):
    d = tmp_path / "context"
    d.mkdir()
    (d / "characters.md").write_text(CHARACTERS_MD, encoding="utf-8")
    return str(d)


@pytest.fixture()
def engine(tmp_path, ctx_dir):
    eng = ZeroMemEngine(
        str(tmp_path / "zm.db"), context_dir=ctx_dir, embedder=HashEmbedder()
    )
    eng.ingest_text(CHARACTERS_MD, source="characters.md", kind="reference", ordinal=0)
    eng.ingest_text(CHAPTER_1, source="chapter-001.md", kind="chapter", chapter=1)
    eng.ingest_text(CHAPTER_2, source="chapter-002.md", kind="chapter", chapter=2)
    yield eng
    eng.store.close()


# ---------------------------------------------------------------------------
# The Vietnamese sentence bug (root cause of giant chunks)
# ---------------------------------------------------------------------------

class TestSegmentation:
    def test_vietnamese_sentences_split(self):
        # The old regex required [A-Z] after punctuation and produced 2 pieces.
        assert len(split_sentences(VI_PARA)) == 6

    def test_character_bible_becomes_addressable_units(self):
        segments = segment_document(CHARACTERS_MD)
        # old pipeline: 2 chunks for the whole file; now every bullet stands alone
        prose = [s for s in segments if s.kind != "heading"]
        assert len(prose) >= 8

    def test_segments_carry_their_heading(self):
        segments = segment_document(CHARACTERS_MD)
        cao = [s for s in segments if "50 tuổi" in s.text]
        assert cao and cao[0].heading.startswith("Thầy Cao")

    def test_scene_changes_at_headings(self):
        segments = segment_document(CHARACTERS_MD)
        scene_of = {}
        for s in segments:
            scene_of.setdefault(s.heading, set()).add(s.scene)
        tam = scene_of["Văn Tâm (Main Character – 21 tuổi)"]
        khang = scene_of["Nguyên Khang (Bạn thân – 21 tuổi)"]
        assert tam.isdisjoint(khang)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_gazetteer_finds_cast_not_field_labels(self, ctx_dir):
        gaz = build_gazetteer(ctx_dir)
        names = set(n for _k, n, _t in gaz.names())
        assert "Văn Tâm" in names
        assert "Thầy Cao" in names
        # attribute labels must NOT become entities (they were hub-node noise)
        assert "Ngoại hình" not in names
        assert "Vai trò" not in names

    def test_accent_folded_matching(self, ctx_dir):
        gaz = build_gazetteer(ctx_dir)
        # query typed without diacritics still grounds the entity
        found = gaz.find("Van Tam cầm cuốn sổ")
        assert any(k == "van tam" for k in found)

    def test_extracts_unknown_proper_nouns(self, ctx_dir):
        gaz = build_gazetteer(ctx_dir)
        ents = extract_entities("Văn Tâm gặp Liễu Thanh Ca ở cổng trường.", gaz)
        names = [e.name for e in ents]
        assert "Văn Tâm" in names
        assert "Liễu Thanh Ca" in names


# ---------------------------------------------------------------------------
# Supersession (root cause of contradictory stale context)
# ---------------------------------------------------------------------------

class TestSupersession:
    def test_reingest_replaces_not_appends(self, engine):
        before = engine.stats()["segments"]
        engine.ingest_text(CHAPTER_2, source="chapter-002.md", kind="chapter", chapter=2)
        assert engine.stats()["segments"] == before

    def test_revised_fact_wins(self, engine):
        revised = CHAPTER_2.replace("Thủy Ngân Sống", "Hỏa Diễm Lam")
        engine.ingest_text(revised, source="chapter-002.md", kind="chapter", chapter=2)
        res = engine.search("Vật Chất thứ nhất là gì?", top_k=8)
        texts = " ".join(ev.trace.text for ev in res["evidence"])
        assert "Hỏa Diễm Lam" in texts
        assert "Thủy Ngân Sống" not in texts  # stale version is gone, not just outranked

    def test_delete_source(self, engine):
        n = engine.delete_source("chapter-001.md")
        assert n > 0
        res = engine.search("giảng đường Mặt Trăng", top_k=5)
        assert all(ev.trace.source != "chapter-001.md" for ev in res["evidence"])


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_character_question_routes_relational(self, engine):
        res = engine.search("Ngoại hình của Văn Tâm như thế nào?")
        assert res["route"] == "relational"
        top = res["evidence"][0]
        assert "1m78" in top.trace.text

    def test_no_cross_character_attribution(self, engine):
        """The bug that mislabelled Thầy Cao's traits as Nguyên Khang's."""
        ctx = engine.build_context("Thầy Cao là ai?", max_tokens=800)
        for block in ctx["context"].split("### "):
            if not block.strip():
                continue
            label, _, body = block.partition("\n")
            if "50 tuổi" in body or "tóc bạc" in body:
                assert "Thầy Cao" in label

    def test_context_blocks_are_contiguous_prose(self, engine):
        ctx = engine.build_context("Văn Tâm tìm thấy Đồ Lục như thế nào?", max_tokens=900)
        assert ctx["context"]
        # every emitted block is attributed
        for block in ctx["context"].split("\n\n"):
            if block.strip():
                assert block.startswith("### ")

    def test_continuity_prefers_latest_chapter(self, engine):
        res = engine.search("Chuyện gì xảy ra gần đây nhất với Văn Tâm?", top_k=4)
        assert res["profile"]["intent"] == "continuity"
        matches = [ev for ev in res["evidence"] if ev.role == "match"]
        chapters = [ev.trace.chapter for ev in matches if ev.trace.kind == "chapter"]
        assert chapters and max(chapters) == 2

    def test_entity_profile(self, engine):
        prof = engine.entity_profile("Đồ Lục")
        assert prof["found"]
        related = [r["name"] for r in prof["related"]]
        assert any("Văn Tâm" in r or "Thầy Cao" in r for r in related)

    def test_diacritic_free_query_still_grounds(self, engine):
        res = engine.search("Ngoai hinh cua Van Tam?")
        assert res["route"] == "relational"
        texts = " ".join(ev.trace.text for ev in res["evidence"][:3])
        assert "1m78" in texts


# ---------------------------------------------------------------------------
# Legacy facade (server.py / CI contract)
# ---------------------------------------------------------------------------

class TestFacade:
    def test_chunk_text_returns_units(self):
        from backend.rag_pipeline import chunk_text
        units = chunk_text(CHARACTERS_MD)
        assert len(units) >= 8
        assert all(isinstance(u, str) for u in units)

    def test_retrieve_hit_shape(self, engine, monkeypatch):
        import backend.rag_pipeline as rp
        monkeypatch.setattr(rp, "_engine", engine)
        hits = rp.retrieve("Văn Tâm", top_k=3)
        assert hits
        h = hits[0]
        assert set(("text", "metadata", "distance")) <= set(h)
        assert "source" in h["metadata"]

    def test_stats_shape(self, engine, monkeypatch):
        import backend.rag_pipeline as rp
        monkeypatch.setattr(rp, "_engine", engine)
        stats = rp.get_collection_stats()
        assert stats["count"] == stats["segments"] > 0
