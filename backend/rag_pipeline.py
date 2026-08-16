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

from backend import novels
from backend.config import settings
from backend.zero_mem.embeddings import create_embedder
from backend.zero_mem.engine import ZeroMemEngine
from backend.zero_mem.gemini import create_extractor
from backend.zero_mem.segment import segment_document

# One engine per novel. A single process-wide engine was what made every book
# share a gazetteer, an entity graph and a trace store — the reason chapter 12
# of one novel could retrieve a character from another.
_engines: Dict[str, ZeroMemEngine] = {}
_engine_lock = threading.Lock()


def _log(msg: str) -> None:
    print(msg)


def _build_embedder():
    gemini_key = settings.gemini_api_key or settings.google_api_key
    return create_embedder(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        gemini_api_key=gemini_key,
        dims=settings.embedding_dims,
        logger=_log,
    )


def _build_extractor():
    gemini_key = settings.gemini_api_key or settings.google_api_key
    if settings.zero_mem_extractor == "ollama":
        from backend.zero_mem.ollama_extract import create_ollama_extractor
        return create_ollama_extractor(
            settings.zero_mem_ollama_extract_model,
            settings.ollama_base_url,
            logger=_log,
        )
    if settings.zero_mem_extractor in ("auto", "gemini"):
        extractor = create_extractor(
            gemini_key, settings.zero_mem_extract_model, logger=_log
        )
        if extractor is None and settings.zero_mem_extractor == "gemini":
            _log("zero-mem: ZERO_MEM_EXTRACTOR=gemini but no GEMINI_API_KEY; using local NER.")
        return extractor
    return None


def get_engine(novel=None) -> ZeroMemEngine:
    """
    The Zero-Mem engine for one novel, created on first use and cached.

    ``novel`` is a ``novels.Novel``, a slug, or None for the default workspace.
    """
    if not isinstance(novel, novels.Novel):
        novel = novels.resolve_or_default(novel, logger=_log)

    engine = _engines.get(novel.slug)
    if engine is not None:
        return engine

    with _engine_lock:
        engine = _engines.get(novel.slug)
        if engine is None:
            novel.scaffold()
            engine = ZeroMemEngine(
                db_path=str(novel.db_path),
                context_dir=str(novel.context_dir),
                embedder=_build_embedder(),
                extractor=_build_extractor(),
                logger=_log,
            )
            _engines[novel.slug] = engine
        return engine


def release_engine(slug: str) -> None:
    """
    Drop a novel's engine and close its database handle.

    Deleting a workspace on Windows fails while the SQLite file is still open,
    so this runs before the directory is removed.
    """
    with _engine_lock:
        engine = _engines.pop(slug, None)
    if engine is not None:
        try:
            engine.store.close()
        except Exception as exc:
            _log("zero-mem: could not close the store for '%s' (%s)." % (slug, exc))


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


def _chapter_number(chapter: str) -> Optional[int]:
    if not chapter:
        return None
    digits = "".join(c for c in str(chapter) if c.isdigit())
    return int(digits) if digits else None


def ingest_text(
    text: str,
    source: str = "manual",
    chapter: str = "",
    characters: Optional[List[str]] = None,   # kept for API compat; automatic now
    locations: Optional[List[str]] = None,    # kept for API compat; automatic now
    collection_name: str = "novel",           # obsolete; single store
    novel=None,
) -> int:
    """Store a document, superseding any previous version of the same source."""
    engine = get_engine(novel)
    chapter_num = _chapter_number(chapter)
    kind = "chapter" if chapter_num is not None else "document"
    return engine.ingest_text(text, source=source, kind=kind, chapter=chapter_num)


def ingest_file(
    filepath: Union[str, Path],
    chapter: str = "",
    characters: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    collection_name: str = "novel",
    novel=None,
) -> int:
    engine = get_engine(novel)
    return engine.ingest_file(str(filepath), chapter=_chapter_number(chapter))


def ingest_directory(
    directory: Union[str, Path],
    collection_name: str = "novel",
    novel=None,
) -> int:
    engine = get_engine(novel)
    count = engine.ingest_directory(str(directory))
    # Context files define the cast, and the files ingested earliest in this
    # call were extracted against a gazetteer that did not yet contain names
    # declared by the later ones. Rebuild it, then run a second pass so every
    # document sees the full entity list. The second pass costs no API tokens:
    # both the SLM extraction and the embeddings are cached by content hash.
    engine.refresh_gazetteer()
    engine.ingest_directory(str(directory))
    return count


def retrieve(
    query: str,
    top_k: int = settings.default_top_k,
    collection_name: str = "novel",
    where: Optional[Dict] = None,
    do_rerank: bool = True,
    novel=None,
) -> List[Dict]:
    """
    Structured evidence selection. Returns the legacy hit shape
    ({text, metadata, distance}) so /api/search responses stay stable.
    """
    engine = get_engine(novel)
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


def get_collection_stats(collection_name: str = "novel", novel=None) -> Dict:
    stats = get_engine(novel).stats()
    stats["name"] = collection_name
    stats["count"] = stats.get("segments", 0)
    return stats


def clear_collection(collection_name: str = "novel", novel=None) -> None:
    get_engine(novel).clear()
