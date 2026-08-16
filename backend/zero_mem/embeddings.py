"""
Pluggable dense embedder — cloud first, no local model required.

Order of preference:
  1. **Google AI Studio** ``gemini-embedding-001`` (multilingual, strong on
     Vietnamese; $0.15/1M tokens with a free tier) — the default whenever a
     Gemini key is present. Nothing is downloaded or run locally.
  2. OpenAI embeddings, if you'd rather use that key.
  3. A built-in hashed n-gram embedding: not a model, just a deterministic
     function, so it needs no download and works offline. Retrieval still
     holds up because BM25 and the entity graph do most of the work.

Optional: sentence-transformers, only if you explicitly ask for it.

Whichever is chosen, vectors are L2-normalised and persisted under a *vector
space* id combining model and dimensionality, so switching models can never
silently compare incomparable vectors.
"""

import json
from typing import Dict, List, Optional, Sequence

import httpx

from .text import EMBED_DIM, hash_embed, l2_normalize

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_EMBED_DIMS = 768
EMBED_BATCH_LIMIT = 100
EMBED_TIMEOUT_S = 60.0


class HashEmbedder(object):
    """Deterministic, dependency-free. Lexical-ish rather than truly semantic."""

    name = "hash"
    # Bumped when the hashing changes shape. Vectors persisted by an older
    # revision live under the previous space id, so they are simply not loaded
    # and get recomputed — never silently compared against the new ones.
    version = "v2"

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.space = "hash-%s@%d" % (self.version, dim)

    def encode(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        return [hash_embed(t, self.dim) for t in texts]


class GeminiEmbedder(object):
    """
    Google AI Studio embeddings. Retrieval is asymmetric, so stored prose is
    embedded as RETRIEVAL_DOCUMENT and queries as RETRIEVAL_QUERY — a real
    quality gain the task-type-less models can't offer.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_EMBED_MODEL,
        dims: int = DEFAULT_EMBED_DIMS,
        client: Optional[httpx.Client] = None,
    ):
        self.api_key = api_key
        self.model_name = model
        self.dim = dims
        self.space = "%s@%d" % (model, dims)
        self._client = client or httpx.Client(timeout=EMBED_TIMEOUT_S)

    # gemini-embedding-001 accepts 2048 input tokens; cap conservatively by
    # characters so a pathological paragraph can't fail the whole batch.
    def _cap(self, text: str) -> str:
        limit = 24000 if "embedding-2" in self.model_name else 6000
        return text[:limit]

    def _supports_task_type(self) -> bool:
        return "embedding-001" in self.model_name

    def _payload(self, text: str, is_query: bool) -> Dict:
        body = {
            "content": {"parts": [{"text": self._cap(text)}]},
            "outputDimensionality": self.dim,
        }
        if self._supports_task_type():
            body["taskType"] = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        return body

    def _post(self, endpoint: str, body: Dict) -> Dict:
        resp = self._client.post(
            "%s/models/%s:%s" % (GEMINI_BASE, self.model_name, endpoint),
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _vector(values) -> List[float]:
        if not values:
            raise RuntimeError("Gemini returned an empty embedding")
        # Below 3072 dims gemini-embedding-001 returns UNNORMALISED vectors.
        return l2_normalize([float(v) for v in values])

    def encode(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        if len(texts) == 1:
            data = self._post("embedContent", self._payload(texts[0], is_query))
            return [self._vector((data.get("embedding") or {}).get("values"))]

        out: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_LIMIT):
            chunk = texts[i:i + EMBED_BATCH_LIMIT]
            body = {"requests": []}
            for t in chunk:
                req = self._payload(t, is_query)
                req["model"] = "models/" + self.model_name
                body["requests"].append(req)
            data = self._post("batchEmbedContents", body)
            embeddings = data.get("embeddings") or []
            if len(embeddings) != len(chunk):
                raise RuntimeError("Gemini batchEmbedContents returned %d of %d vectors"
                                   % (len(embeddings), len(chunk)))
            for e in embeddings:
                out.append(self._vector(e.get("values")))
        return out


class SentenceTransformerEmbedder(object):
    """Wraps sentence-transformers; loaded lazily so import stays cheap."""

    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # noqa: F401 (import check)
        self._model = SentenceTransformer(model_name)
        dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name
        self.dim = dim
        self.space = "%s@%d" % (model_name, dim)

    def encode(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]


class OpenAIEmbedder(object):
    name = "openai"

    # text-embedding-3-* accept an explicit `dimensions`; ada-002 does not.
    _NATIVE_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str, api_key: str, dims: int = 0):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model_name = model_name
        # The space id must be final before the first encode(). The engine
        # reads embedder.space during reload() to load persisted vectors, which
        # happens *before* anything is encoded — a space that only became
        # correct after the first call meant stored vectors were never found
        # and the whole corpus was re-embedded on every process start.
        self.dim = dims or self._NATIVE_DIMS.get(model_name, 1536)
        self._resizable = model_name.startswith("text-embedding-3")
        self.space = "%s@%d" % (model_name, self.dim)

    def encode(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        out: List[List[float]] = []
        batch = list(texts)
        for i in range(0, len(batch), 128):
            kwargs = {"model": self.model_name, "input": batch[i:i + 128]}
            if self._resizable:
                kwargs["dimensions"] = self.dim
            resp = self._client.embeddings.create(**kwargs)
            for item in resp.data:
                out.append(l2_normalize([float(x) for x in item.embedding]))
        if out and len(out[0]) != self.dim:
            # A model returned a width we did not predict. Fail loudly rather
            # than persist vectors under a space id that misdescribes them.
            raise RuntimeError(
                "OpenAI %s returned %d dims, expected %d — set EMBEDDING_DIMS to match."
                % (self.model_name, len(out[0]), self.dim)
            )
        return out


def create_embedder(
    provider: str,
    model_name: str = "",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    dims: int = DEFAULT_EMBED_DIMS,
    logger=None,
):
    """
    Build the configured embedder, degrading to hashing if unavailable.

    provider: 'auto' (Gemini when a key exists, else hash) | 'gemini' |
              'openai' | 'hash' | 'sentence-transformers'
    """
    def note(msg: str) -> None:
        if logger:
            logger(msg)

    provider = (provider or "auto").lower()

    if provider == "auto":
        provider = "gemini" if gemini_api_key else "hash"
        if provider == "hash":
            note("zero-mem: no GEMINI_API_KEY; using hashed embeddings "
                 "(no download, no network). BM25 + the entity graph carry retrieval.")

    if provider in ("hash", "none", "off"):
        return HashEmbedder()

    if provider == "gemini":
        if not gemini_api_key:
            note("zero-mem: EMBEDDING_PROVIDER=gemini but no GEMINI_API_KEY; using hashed embeddings.")
            return HashEmbedder()
        return GeminiEmbedder(
            gemini_api_key, model_name or DEFAULT_GEMINI_EMBED_MODEL, dims
        )

    if provider == "openai":
        if not openai_api_key:
            note("zero-mem: OPENAI_API_KEY missing; falling back to hashed embeddings.")
            return HashEmbedder()
        try:
            return OpenAIEmbedder(model_name or "text-embedding-3-small", openai_api_key, dims)
        except Exception as exc:  # pragma: no cover - depends on env
            note("zero-mem: OpenAI embedder unavailable (%s); using hashed embeddings." % exc)
            return HashEmbedder()

    # sentence-transformers: opt-in only — it downloads a multi-GB local model.
    try:
        return SentenceTransformerEmbedder(model_name or "BAAI/bge-m3")
    except Exception as exc:
        note(
            "zero-mem: sentence-transformers unavailable (%s); using hashed embeddings. "
            "Retrieval still works — BM25 and the entity graph carry it." % exc
        )
        return HashEmbedder()
