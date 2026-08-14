# -*- coding: utf-8 -*-
"""
Gemini extractor tests, all against a mocked httpx transport — no network.

Covers: batch request shape, response parsing, config-variant fallback on 400,
entity merge with the local pipeline, relation storage + supersession, the
content-hash cache (re-ingest of unchanged text costs zero API calls), and
graceful degradation when the API is down.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.zero_mem.embeddings import HashEmbedder  # noqa: E402
from backend.zero_mem.engine import ZeroMemEngine  # noqa: E402
from backend.zero_mem.gemini import (  # noqa: E402
    GeminiExtractor,
    GeminiExtractorError,
    canonicalize_type,
)

CHAPTER = """# Chương 3

Văn Tâm trao Đồ Lục cho Nguyên Khang xem trong căn phòng cũ của Thầy Cao.

Liễu Thanh Ca xuất hiện ở cửa, tự nhận là học trò cũ của Thầy Cao."""


def make_transport(handler):
    return httpx.MockTransport(handler)


def gemini_response(segments):
    return {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps({"segments": segments}, ensure_ascii=False)}]}}
        ]
    }


def extractor_with(handler, model="gemini-2.5-flash-lite"):
    client = httpx.Client(transport=make_transport(handler))
    return GeminiExtractor("test-key", model, client=client)


class TestCanonicalizeType:
    def test_maps_onto_port_taxonomy(self):
        assert canonicalize_type("PERSON") == "PERSON"
        assert canonicalize_type("Location") == "PLACE"
        assert canonicalize_type("ARTIFACT") == "ITEM"
        assert canonicalize_type("ORGANIZATION") == "CONCEPT"
        assert canonicalize_type("weird-thing") == "PROPER"


class TestGeminiExtractor:
    def test_batch_request_and_parse(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_response([
                {"index": 0,
                 "entities": [{"name": "Văn Tâm", "type": "PERSON"},
                              {"name": "Đồ Lục", "type": "ARTIFACT"}],
                 "relations": [{"subject": "Văn Tâm", "relation": "OWNS", "object": "Đồ Lục"}]},
                {"index": 1, "entities": [{"name": "Liễu Thanh Ca", "type": "PERSON"}],
                 "relations": [{"subject": "Liễu Thanh Ca", "relation": "STUDENT_OF", "object": "Thầy Cao"}]},
            ]))

        ex = extractor_with(handler)
        results = ex.extract_segments(["Văn Tâm trao Đồ Lục...", "Liễu Thanh Ca xuất hiện..."])

        cfg = seen["body"]["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["responseSchema"]["type"] == "OBJECT"  # 2.5 legacy shape
        assert "thinkingConfig" not in cfg  # lite: thinking already off
        user_text = seen["body"]["contents"][0]["parts"][0]["text"]
        assert "[0]" in user_text and "[1]" in user_text

        assert [m.name for m in results[0].entities] == ["Văn Tâm", "Đồ Lục"]
        assert results[0].entities[1].type == "ITEM"
        assert results[1].relations == [
            {"subject": "Liễu Thanh Ca", "relation": "STUDENT_OF", "object": "Thầy Cao"}]

    @staticmethod
    def _shape_of(cfg):
        if "responseSchema" in cfg:
            return "legacy"
        if "response_format" in cfg:
            return "modern"
        return "bare"

    def test_variant_fallback_on_400_and_caches_winner(self):
        shapes = []

        def handler(request):
            cfg = json.loads(request.content)["generationConfig"]
            shapes.append(self._shape_of(cfg))
            if "responseSchema" in cfg:
                return httpx.Response(400, text="Unknown name responseSchema")
            return httpx.Response(200, json=gemini_response([{"index": 0, "entities": [], "relations": []}]))

        ex = extractor_with(handler)
        ex.extract_segments(["a"])
        ex.extract_segments(["b"])
        assert shapes == ["legacy", "bare", "bare"]  # winner remembered

    def test_unsupported_thinking_field_does_not_poison_all_variants(self):
        """
        Regression: thinking config used to be merged into every variant, so a
        rejected thinking field returned 400 for all of them and extraction
        could never recover. The live API rejects `thinking_level`, which is
        exactly how this bit in production.
        """
        seen = []

        def handler(request):
            cfg = json.loads(request.content)["generationConfig"]
            seen.append(cfg)
            if "thinkingConfig" in cfg:
                return httpx.Response(400, text='Unknown name "thinkingConfig"')
            return httpx.Response(200, json=gemini_response([{"index": 0, "entities": [], "relations": []}]))

        ex = extractor_with(handler, model="gemini-3.1-flash-lite")
        ex.extract_segments(["a"])  # must not raise
        assert any("thinkingConfig" not in c for c in seen), "no thinking-free variant was tried"

    def test_gemini_3_uses_thinking_config_not_thinking_level(self):
        """The documented top-level `thinking_level` returns 400 on v1beta."""
        seen = {}

        def handler(request):
            seen["cfg"] = json.loads(request.content)["generationConfig"]
            return httpx.Response(200, json=gemini_response([{"index": 0, "entities": [], "relations": []}]))

        extractor_with(handler, model="gemini-3.1-flash-lite").extract_segments(["a"])
        assert "thinking_level" not in seen["cfg"]
        assert seen["cfg"]["thinkingConfig"] == {"thinkingLevel": "LOW"}

    def test_pronoun_endpoints_are_dropped(self):
        def handler(request):
            return httpx.Response(200, json=gemini_response([{
                "index": 0,
                "entities": [],
                "relations": [
                    {"subject": "cậu", "relation": "SỞ_HỮU", "object": "Đồ Lục"},
                    {"subject": "Văn Tâm", "relation": "RECEIVES", "object": "Đồ Lục"},
                    {"subject": "Đồ Lục", "relation": "BELONGS_TO", "object": "he"},
                ],
            }]))

        rels = extractor_with(handler).extract_segments(["x"])[0].relations
        assert rels == [{"subject": "Văn Tâm", "relation": "RECEIVES", "object": "Đồ Lục"}]

    def test_non_400_error_raises_without_variant_retry(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(429, text="quota")

        ex = extractor_with(handler)
        with pytest.raises(GeminiExtractorError):
            ex.extract_segments(["a"])
        assert len(calls) == 1

    def test_markdown_fenced_json_tolerated(self):
        def handler(request):
            payload = "```json\n" + json.dumps({"segments": [{"index": 0, "entities": [], "relations": []}]}) + "\n```"
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": payload}]}}]})

        ex = extractor_with(handler)
        assert ex.extract_segments(["a"])[0].relations == []


class TestEngineIntegration:
    def _engine(self, tmp_path, handler):
        return ZeroMemEngine(
            str(tmp_path / "zm.db"),
            embedder=HashEmbedder(),
            extractor=extractor_with(handler),
        )

    @staticmethod
    def _chapter_handler(request):
        body = json.loads(request.content)
        text = body["contents"][0]["parts"][0]["text"]
        segments = []
        for chunk in text.split("\n\n"):
            if not chunk.startswith("["):
                continue
            idx = int(chunk[1:chunk.index("]")])
            ents, rels = [], []
            if "Đồ Lục" in chunk:
                ents = [{"name": "Văn Tâm", "type": "PERSON"}, {"name": "Đồ Lục", "type": "ITEM"}]
                rels = [{"subject": "Văn Tâm", "relation": "OWNS", "object": "Đồ Lục"}]
            if "Liễu Thanh Ca" in chunk:
                ents.append({"name": "Liễu Thanh Ca", "type": "PERSON"})
                rels.append({"subject": "Liễu Thanh Ca", "relation": "STUDENT_OF", "object": "Thầy Cao"})
            segments.append({"index": idx, "entities": ents, "relations": rels})
        return httpx.Response(200, json=gemini_response(segments))

    def test_relations_stored_and_slm_entities_merged(self, tmp_path):
        engine = self._engine(tmp_path, self._chapter_handler)
        engine.ingest_text(CHAPTER, source="chapter-003.md", kind="chapter", chapter=3)

        assert engine.store.relation_count() == 2
        prof = engine.entity_profile("Đồ Lục")
        assert any(r["relation"] == "OWNS" for r in prof["relations"])
        # relations feed the generation context as a "known relations" block
        ctx = engine.build_context("Văn Tâm và Đồ Lục", max_tokens=900)
        assert "Quan hệ đã biết" in ctx["context"]
        assert "—OWNS→" in ctx["context"]
        # SLM-only entity (not in any gazetteer) is indexed and searchable
        res = engine.search("Liễu Thanh Ca là ai?")
        assert res["route"] == "relational"
        assert any("Liễu Thanh Ca" in ev.trace.text for ev in res["evidence"])
        engine.store.close()

    def test_cache_prevents_repeat_api_calls(self, tmp_path):
        calls = []

        def handler(request):
            calls.append(1)
            return self._chapter_handler(request)

        engine = self._engine(tmp_path, handler)
        engine.ingest_text(CHAPTER, source="chapter-003.md", kind="chapter", chapter=3)
        first = len(calls)
        assert first >= 1
        # identical re-ingest: every segment hash is cached -> zero new calls
        engine.ingest_text(CHAPTER, source="chapter-003.md", kind="chapter", chapter=3)
        assert len(calls) == first
        # relations survive the supersession, not duplicated
        assert engine.store.relation_count() == 2
        # editing one paragraph re-extracts only that paragraph
        edited = CHAPTER.replace("ở cửa", "ở cửa sổ")
        engine.ingest_text(edited, source="chapter-003.md", kind="chapter", chapter=3)
        assert len(calls) == first + 1
        engine.store.close()

    def test_api_failure_degrades_to_local_pipeline(self, tmp_path):
        def handler(request):
            return httpx.Response(500, text="boom")

        engine = self._engine(tmp_path, handler)
        n = engine.ingest_text(CHAPTER, source="chapter-003.md", kind="chapter", chapter=3)
        assert n > 0
        assert engine.store.relation_count() == 0
        # pattern NER still extracted the proper nouns
        res = engine.search("Văn Tâm")
        assert res["route"] == "relational"
        engine.store.close()

    def test_stats_report_extractor(self, tmp_path):
        engine = self._engine(tmp_path, self._chapter_handler)
        assert engine.stats()["extractor"] == "gemini-2.5-flash-lite"
        local = ZeroMemEngine(str(tmp_path / "zm2.db"), embedder=HashEmbedder())
        assert local.stats()["extractor"] == "local-gazetteer-ner"
        local.store.close()
        engine.store.close()
