"""
Pluggable dense embedder.

Order of preference:
  1. sentence-transformers (the project's existing BAAI/bge-m3, multilingual —
     the right choice for a Vietnamese corpus) when installed;
  2. OpenAI embeddings when configured;
  3. a built-in hashed n-gram embedding that needs no model download.

Whichever is chosen, vectors are L2-normalised and persisted under a *vector
space* id combining model and dimensionality, so switching models can never
silently compare incomparable vectors.
"""

from typing import Dict, List, Optional, Sequence

from .text import EMBED_DIM, hash_embed, l2_normalize


class HashEmbedder(object):
    """Deterministic, dependency-free. Lexical-ish rather than truly semantic."""

    name = "hash"

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.space = "hash@%d" % dim

    def encode(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        return [hash_embed(t, self.dim) for t in texts]


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

    def __init__(self, model_name: str, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dim = 0
        self.space = model_name

    def encode(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        out: List[List[float]] = []
        batch = list(texts)
        for i in range(0, len(batch), 128):
            resp = self._client.embeddings.create(model=self.model_name, input=batch[i:i + 128])
            for item in resp.data:
                out.append(l2_normalize([float(x) for x in item.embedding]))
        if out and not self.dim:
            self.dim = len(out[0])
            self.space = "%s@%d" % (self.model_name, self.dim)
        return out


def create_embedder(
    provider: str,
    model_name: str,
    openai_api_key: str = "",
    logger=None,
):
    """Build the configured embedder, degrading to hashing if unavailable."""
    def note(msg: str) -> None:
        if logger:
            logger(msg)

    provider = (provider or "").lower()
    if provider in ("hash", "none", "off"):
        return HashEmbedder()

    if provider == "openai":
        if not openai_api_key:
            note("zero-mem: OPENAI_API_KEY missing; falling back to hashed embeddings.")
            return HashEmbedder()
        try:
            return OpenAIEmbedder(model_name, openai_api_key)
        except Exception as exc:  # pragma: no cover - depends on env
            note("zero-mem: OpenAI embedder unavailable (%s); using hashed embeddings." % exc)
            return HashEmbedder()

    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:
        note(
            "zero-mem: sentence-transformers unavailable (%s); using hashed embeddings. "
            "Retrieval still works — BM25 and the entity graph carry it — but "
            "cross-wording semantic matches will be weaker." % exc
        )
        return HashEmbedder()
