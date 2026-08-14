"""
Memory pipeline facade — Zero-Mem edition (arXiv:2607.29377).

The ChromaDB chunk-and-hope RAG pipeline that used to live here is replaced by
a Zero-Mem engine: verbatim narrative traces in SQLite, a character/location
entity graph with Personalized PageRank, BM25 + dense dual-view retrieval, and
evidence closure that returns contiguous, correctly-attributed passages.

Why the old pipeline lost context, concretely:

* its sentence splitter required ``[A-Z]`` after punctuation, which never
  matches Vietnamese capitals (Đ, Ư, Ổ, ...), so this Vietnamese novel barely
  split — the whole character bible collapsed into two ~400-word blobs;
* chunks were keyed by ``sha256(text)`` and never superseded, so every chapter
  revision left stale contradictory chunks behind, all still retrievable;
* every generation ran a hard-coded English query ("unresolved conflict
  tension mystery") against the Vietnamese corpus;
* top-k fragments arrived shuffled and unattributed, so retrieved facts were
  routinely blended across characters.

The public functions below keep their old names/signatures so ``server.py``
and CI continue to work unchanged. ``characters``/``locations`` arguments are
accepted but no longer required — entity extraction is automatic now.
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Union

from backend.config import settings
from backend.zero_mem.embeddings import create_embedder
from backend.zero_mem.engine import ZeroMemEngine
from backend.zero_mem.gemini import create_extractor
from backend.zero_mem.segment import segment_document

_engine: Optional[ZeroMemEngine] = None
_engine_lock = threading.Lock()


def _log(msg: str) -> None:
    print(msg)


def get_engine() -> ZeroMemEngine:
    """Process-wide Zero-Mem engine (lazy: embeddings may load a model)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                gemini_key = settings.gemini_api_key or settings.google_api_key
                embedder = create_embedder(
                    provider=settings.embedding_provider,
                    model_name=settings.embedding_model,
                    openai_api_key=settings.openai_api_key,
                    gemini_api_key=gemini_key,
                    dims=settings.embedding_dims,
                    logger=_log,
                )
                extractor = None
                if settings.zero_mem_extractor in ("auto", "gemini"):
                    api_key = gemini_key
                    extractor = create_extractor(
                        api_key, settings.zero_mem_extract_model, logger=_log
                    )
                    if extractor is None and settings.zero_mem_extractor == "gemini":
                        _log("zero-mem: ZERO_MEM_EXTRACTOR=gemini but no GEMINI_API_KEY; using local NER.")
                _engine = ZeroMemEngine(
                    db_path=settings.zero_mem_db,
                    context_dir=settings.context_dir,
                    embedder=embedder,
                    extractor=extractor,
                    logger=_log,
                )
    return _engine


# ---------------------------------------------------------------------------
# Back-compat API (same names the server / CI import)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 0, overlap: int = 0) -> List[str]:
    """
    Compatibility shim: segmentation replaced chunking. Returns the verbatim
    narrative units (paragraph/bullet granularity, Unicode-aware sentences).
    The chunk_size/overlap knobs are obsolete and ignored.
    """
    return [seg.text for seg in segment_document(text) if seg.kind != "heading"]


def ingest_text(
    text: str,
    source: str = "manual",
    chapter: str = "",
    characters: Optional[List[str]] = None,   # kept for API compat; automatic now
    locations: Optional[List[str]] = None,    # kept for API compat; automatic now
    collection_name: str = "novel",           # obsolete; single store
) -> int:
    """Store a document, superseding any previous version of the same source."""
    engine = get_engine()
    chapter_num: Optional[int] = None
    if chapter:
        digits = "".join(c for c in str(chapter) if c.isdigit())
        if digits:
            chapter_num = int(digits)
    kind = "chapter" if chapter_num is not None else "document"
    return engine.ingest_text(text, source=source, kind=kind, chapter=chapter_num)


def ingest_file(
    filepath: Union[str, Path],
    chapter: str = "",
    characters: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    collection_name: str = "novel",
) -> int:
    engine = get_engine()
    chapter_num: Optional[int] = None
    if chapter:
        digits = "".join(c for c in str(chapter) if c.isdigit())
        if digits:
            chapter_num = int(digits)
    return engine.ingest_file(str(filepath), chapter=chapter_num)


def ingest_directory(
    directory: Union[str, Path],
    collection_name: str = "novel",
) -> int:
    engine = get_engine()
    count = engine.ingest_directory(str(directory))
    # Context files define the cast; refresh the gazetteer and re-extract so
    # documents ingested in the same call see the full entity list.
    engine.refresh_gazetteer()
    return count


def retrieve(
    query: str,
    top_k: int = settings.default_top_k,
    collection_name: str = "novel",
    where: Optional[Dict] = None,
    do_rerank: bool = True,
) -> List[Dict]:
    """
    Structured evidence selection. Returns the legacy hit shape
    ({text, metadata, distance}) so /api/search responses stay stable.
    """
    engine = get_engine()
    result = engine.search(query, top_k=top_k)
    hits: List[Dict] = []
    for ev in result["evidence"]:
        t = ev.trace
        hits.append({
            "text": t.text,
            "metadata": {
                "source": t.source,
                "chapter": "" if t.chapter is None else str(t.chapter),
                "heading": t.heading,
                "kind": t.kind,
                "seq": t.seq,
                "role": ev.role,
                "matched_entities": ",".join(ev.matched_entities),
                "route": result["route"],
            },
            "distance": round(1.0 - min(ev.score, 1.0), 4),
        })
    return hits


def get_collection_stats(collection_name: str = "novel") -> Dict:
    stats = get_engine().stats()
    stats["name"] = collection_name
    stats["count"] = stats.get("segments", 0)
    return stats


def clear_collection(collection_name: str = "novel") -> None:
    get_engine().clear()
