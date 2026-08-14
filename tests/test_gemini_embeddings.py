# -*- coding: utf-8 -*-
"""
Cloud embedding tests (Gemini), all against a mocked httpx transport.

Covers request shape (task types, dimensionality), normalisation, batching,
provider selection, the content-hash cache that keeps re-ingest free, and
degradation when the API is unreachable.
"""

import json
import math
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.zero_mem.embeddings import (  # noqa: E402
    GeminiEmbedder,
    HashEmbedder,
    create_embedder,
)
from backend.zero_mem.engine import ZeroMemEngine  # noqa: E402

CHAPTER = """# Chương 4

Văn Tâm mở Đồ Lục dưới ánh đèn vàng của thư viện cũ.

Sương xám tràn qua khung cửa sổ, đặc quánh như sữa.

Nguyên Khang siết chặt tay cậu, giọng run: "Chạy đi!"
"""


def embedder_with(handler, model="gemini-embedding-001", dims=768):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiEmbedder("test-key", model, dims, client=client)


class TestGeminiEmbedder:
    def test_document_request_shape(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            seen["key"] = request.headers.get("x-goog-api-key")
            return httpx.Response(200, json={"embedding": {"values": [3.0, 4.0]}})

        emb = embedder_with(handler)
        vec = emb.encode(["Văn Tâm mở Đồ Lục"], is_query=False)[0]

        assert "models/gemini-embedding-001:embedContent" in seen["url"]
        assert seen["key"] == "test-key"
        assert seen["body"]["taskType"] == "RETRIEVAL_DOCUMENT"
        assert seen["body"]["outputDimensionality"] == 768
        # unnormalised [3,4] -> unit vector
        assert vec[0] == pytest.approx(0.6)
        assert vec[1] == pytest.approx(0.8)
        assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)

    def test_query_uses_retrieval_query_task_type(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"embedding": {"values": [1.0, 0.0]}})

        embedder_with(handler).encode(["Đồ Lục là gì?"], is_query=True)
        assert seen["body"]["taskType"] == "RETRIEVAL_QUERY"

    def test_batching_and_order(self):
        sizes = []

        def handler(request):
            body = json.loads(request.content)
            assert "batchEmbedContents" in str(request.url)
            reqs = body["requests"]
            sizes.append(len(reqs))
            assert reqs[0]["model"] == "models/gemini-embedding-001"
            return httpx.Response(200, json={
                "embeddings": [{"values": [float(i + 1), 0.0]} for i in range(len(reqs))]})

        vecs = embedder_with(handler).encode(["t%d" % i for i in range(150)])
        assert sizes == [100, 50]
        assert len(vecs) == 150
        assert vecs[0][0] == pytest.approx(1.0)

    def test_vector_space_encodes_model_and_dims(self):
        assert embedder_with(lambda r: None, dims=768).space == "gemini-embedding-001@768"
        assert embedder_with(lambda r: None, dims=1536).space == "gemini-embedding-001@1536"

    def test_long_input_is_capped(self):
        seen = {}

        def handler(request):
            seen["len"] = len(json.loads(request.content)["content"]["parts"][0]["text"])
            return httpx.Response(200, json={"embedding": {"values": [1.0, 0.0]}})

        embedder_with(handler).encode(["x" * 50000])
        assert seen["len"] == 6000

    def test_mismatched_batch_size_raises(self):
        def handler(request):
            return httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0]}]})

        with pytest.raises(RuntimeError):
            embedder_with(handler).encode(["a", "b"])


class TestProviderSelection:
    def test_auto_prefers_gemini_when_key_present(self):
        emb = create_embedder("auto", gemini_api_key="k")
        assert isinstance(emb, GeminiEmbedder)
        assert emb.space == "gemini-embedding-001@768"

    def test_auto_falls_back_to_hash_without_key(self):
        notes = []
        emb = create_embedder("auto", logger=notes.append)
        assert isinstance(emb, HashEmbedder)
        assert any("GEMINI_API_KEY" in n for n in notes)

    def test_explicit_gemini_without_key_degrades(self):
        assert isinstance(create_embedder("gemini"), HashEmbedder)

    def test_hash_never_touches_network(self):
        assert isinstance(create_embedder("hash", gemini_api_key="k"), HashEmbedder)

    def test_no_local_model_is_loaded_by_default(self):
        """Default config must not pull sentence-transformers."""
        from backend.config import settings
        assert settings.embedding_provider == "auto"
        assert "sentence" not in settings.embedding_provider


class TestEmbedCache:
    def _engine(self, tmp_path, calls):
        def handler(request):
            body = json.loads(request.content)
            n = len(body.get("requests", [])) or 1
            calls.append(n)
            if "batchEmbedContents" in str(request.url):
                return httpx.Response(200, json={
                    "embeddings": [{"values": [1.0, float(i)]} for i in range(n)]})
            return httpx.Response(200, json={"embedding": {"values": [1.0, 0.0]}})

        return ZeroMemEngine(
            str(tmp_path / "zm.db"),
            embedder=embedder_with(handler),
        ), calls

    def test_reingest_of_unchanged_text_costs_nothing(self, tmp_path):
        calls = []
        engine, calls = self._engine(tmp_path, calls)
        engine.ingest_text(CHAPTER, source="chapter-004.md", kind="chapter", chapter=4)
        first = sum(calls)
        assert first > 0

        engine.ingest_text(CHAPTER, source="chapter-004.md", kind="chapter", chapter=4)
        assert sum(calls) == first, "identical re-ingest must not re-embed"

        edited = CHAPTER.replace("ánh đèn vàng", "ánh đèn đỏ")
        engine.ingest_text(edited, source="chapter-004.md", kind="chapter", chapter=4)
        assert sum(calls) == first + 1, "only the edited paragraph re-embeds"
        engine.store.close()

    def test_cache_survives_clear(self, tmp_path):
        calls = []
        engine, calls = self._engine(tmp_path, calls)
        engine.ingest_text(CHAPTER, source="chapter-004.md", kind="chapter", chapter=4)
        first = sum(calls)
        engine.clear()
        engine.ingest_text(CHAPTER, source="chapter-004.md", kind="chapter", chapter=4)
        assert sum(calls) == first
        engine.store.close()

    def test_api_failure_does_not_break_ingest(self, tmp_path):
        def handler(request):
            return httpx.Response(503, text="unavailable")

        engine = ZeroMemEngine(str(tmp_path / "zm.db"), embedder=embedder_with(handler))
        n = engine.ingest_text(CHAPTER, source="chapter-004.md", kind="chapter", chapter=4)
        assert n > 0
        # BM25 + entity graph still answer
        res = engine.search("Văn Tâm mở Đồ Lục")
        assert res["evidence"]
        engine.store.close()
