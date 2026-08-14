# -*- coding: utf-8 -*-
"""
Ollama extractor tests — mocked transport, no network.

The interesting surface is the tolerant parser: Ollama's hosted models treat
the JSON-schema `format` parameter as a hint, so real replies arrive fenced,
as bare arrays, and with aliased field names. Every shape below was observed
from gemma4:31b-cloud on 2026-08-14.
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
from backend.zero_mem.ollama_extract import (  # noqa: E402
    OllamaExtractor,
    OllamaExtractorError,
    normalize_payload,
)


def ollama_reply(content):
    return {"model": "gemma4", "message": {"role": "assistant", "content": content}, "done": True}


def extractor_with(handler, model="gemma4:31b-cloud"):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaExtractor(model, client=client)


class TestTolerantParser:
    def test_real_gemma4_shape_is_accepted(self):
        """Fenced bare array with segment/text/label aliases — the live shape."""
        raw = """```json
[
  {"segment": 0,
   "entities": [{"text": "Văn Tâm", "label": "Person"},
                {"text": "Đồ Lục", "label": "Object"}],
   "relations": [{"subject": "Văn Tâm", "relation": "receives", "object": "Đồ Lục"},
                 {"subject": "Đồ Lục", "relation": "is_relic_of", "object": "Thầy Cao"}]},
  {"segment": 1,
   "entities": [{"text": "Nguyên Khang", "label": "Person"}],
   "relations": []}
]
```"""
        out = normalize_payload(raw, 2)
        assert set(out) == {0, 1}
        assert [m.name for m in out[0].entities] == ["Văn Tâm", "Đồ Lục"]
        assert out[0].entities[0].type == "PERSON"
        assert out[0].entities[1].type == "ITEM"      # 'Object' -> ITEM
        # relation labels normalised to SHORT_UPPERCASE_VERB
        assert out[0].relations[0]["relation"] == "RECEIVES"
        assert out[0].relations[1]["relation"] == "IS_RELIC_OF"

    def test_canonical_schema_shape_also_works(self):
        raw = json.dumps({"segments": [
            {"index": 0, "entities": [{"name": "Thầy Cao", "type": "PERSON"}], "relations": []}]})
        out = normalize_payload(raw, 1)
        assert out[0].entities[0].name == "Thầy Cao"

    def test_missing_index_falls_back_to_position(self):
        raw = json.dumps([
            {"entities": [{"name": "A", "type": "PERSON"}], "relations": []},
            {"entities": [{"name": "B", "type": "PERSON"}], "relations": []},
        ])
        out = normalize_payload(raw, 2)
        assert out[0].entities[0].name == "A"
        assert out[1].entities[0].name == "B"

    def test_out_of_range_index_is_dropped(self):
        raw = json.dumps([{"index": 99, "entities": [{"name": "X", "type": "PERSON"}], "relations": []}])
        assert normalize_payload(raw, 2) == {}

    def test_plain_string_entities_are_accepted(self):
        raw = json.dumps([{"index": 0, "entities": ["Văn Tâm"], "relations": []}])
        out = normalize_payload(raw, 1)
        assert out[0].entities[0].name == "Văn Tâm"

    def test_trailing_prose_after_json_is_tolerated(self):
        raw = ('{"segments":[{"index":0,"entities":[{"name":"A","type":"PERSON"}],"relations":[]}]}'
               "\n\nHope this helps!")
        assert normalize_payload(raw, 1)[0].entities[0].name == "A"

    def test_pronoun_relations_dropped(self):
        raw = json.dumps([{"index": 0, "entities": [], "relations": [
            {"subject": "cậu", "relation": "owns", "object": "Đồ Lục"},
            {"subject": "Văn Tâm", "relation": "owns", "object": "Đồ Lục"}]}])
        rels = normalize_payload(raw, 1)[0].relations
        assert len(rels) == 1 and rels[0]["subject"] == "Văn Tâm"

    def test_subject_target_aliases(self):
        raw = json.dumps([{"index": 0, "entities": [], "relations": [
            {"source": "A", "predicate": "knows", "target": "B"}]}])
        rels = normalize_payload(raw, 1)[0].relations
        assert rels == [{"subject": "A", "relation": "KNOWS", "object": "B"}]

    def test_unparseable_output_raises(self):
        with pytest.raises(OllamaExtractorError):
            normalize_payload("I'm sorry, I can't do that.", 1)


class TestOllamaExtractor:
    def test_request_shape(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=ollama_reply(
                json.dumps({"segments": [{"index": 0, "entities": [], "relations": []}]})))

        extractor_with(handler).extract_segments(["một đoạn văn"])
        assert seen["url"].endswith("/api/chat")
        assert seen["body"]["model"] == "gemma4:31b-cloud"
        assert seen["body"]["stream"] is False
        assert seen["body"]["options"]["temperature"] == 0
        assert seen["body"]["format"]["type"] == "object"   # schema still sent as a hint
        assert "[0]" in seen["body"]["messages"][1]["content"]

    def test_batch_indices_are_local_so_positional_fallback_lines_up(self):
        prompts = []

        def handler(request):
            body = json.loads(request.content)
            prompts.append(body["messages"][1]["content"])
            n = body["messages"][1]["content"].count("[")
            return httpx.Response(200, json=ollama_reply(json.dumps(
                [{"entities": [{"name": "E%d" % i, "type": "PERSON"}], "relations": []}
                 for i in range(n)])))

        res = extractor_with(handler).extract_segments(["a", "b", "c"])
        assert prompts[0].startswith("[0]")
        assert [r.entities[0].name for r in res] == ["E0", "E1", "E2"]

    def test_http_error_raises(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        with pytest.raises(OllamaExtractorError):
            extractor_with(handler).extract_segments(["a"])

    def test_engine_degrades_when_ollama_fails(self, tmp_path):
        def handler(request):
            return httpx.Response(503, text="unavailable")

        engine = ZeroMemEngine(
            str(tmp_path / "zm.db"),
            embedder=HashEmbedder(),
            extractor=extractor_with(handler),
        )
        n = engine.ingest_text(
            "# Chương 5\n\nVăn Tâm gặp Nguyên Khang ở thư viện.",
            source="chapter-005.md", kind="chapter", chapter=5)
        assert n > 0                       # ingest still succeeds
        assert engine.store.relation_count() == 0
        assert engine.search("Văn Tâm")["evidence"]   # local NER still indexed it
        engine.store.close()

    def test_engine_stores_ollama_relations(self, tmp_path):
        def handler(request):
            return httpx.Response(200, json=ollama_reply("""```json
[{"segment": 0,
  "entities": [{"text": "Văn Tâm", "label": "Person"}],
  "relations": [{"subject": "Văn Tâm", "relation": "meets", "object": "Nguyên Khang"}]}]
```"""))

        engine = ZeroMemEngine(
            str(tmp_path / "zm.db"),
            embedder=HashEmbedder(),
            extractor=extractor_with(handler),
        )
        engine.ingest_text("# Chương 5\n\nVăn Tâm gặp Nguyên Khang ở thư viện.",
                           source="chapter-005.md", kind="chapter", chapter=5)
        assert engine.store.relation_count() >= 1
        prof = engine.entity_profile("Văn Tâm")
        assert any(r["relation"] == "MEETS" for r in prof["relations"])
        engine.store.close()
