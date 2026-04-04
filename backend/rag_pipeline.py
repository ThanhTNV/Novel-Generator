"""
RAG Pipeline: embedding, chunking, ingestion, and retrieval via ChromaDB.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings

_client = None  # type: Optional[chromadb.ClientAPI]
_embedding_fn = None


def _get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _get_embedding_function():
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn

    if settings.embedding_provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        _embedding_fn = OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
        )
    else:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
        )
    return _embedding_fn


def get_collection(name: str = "novel") -> chromadb.Collection:
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = settings.default_chunk_size,
    overlap: int = settings.default_chunk_overlap,
) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_text(
    text: str,
    source: str = "unknown",
    chapter: str = "",
    characters: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    collection_name: str = "novel",
) -> int:
    """Chunk and store text in ChromaDB. Returns number of chunks stored."""
    collection = get_collection(collection_name)
    chunks = chunk_text(text)

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        doc_id = _stable_id(chunk)
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "source": source,
            "chapter": chapter,
            "characters": ",".join(characters or []),
            "locations": ",".join(locations or []),
            "chunk_index": i,
        })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return len(ids)


def ingest_file(
    filepath: Union[str, Path],
    chapter: str = "",
    characters: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    collection_name: str = "novel",
) -> int:
    """Read a text/markdown file and ingest its content."""
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")
    return ingest_text(
        text=text,
        source=path.name,
        chapter=chapter,
        characters=characters,
        locations=locations,
        collection_name=collection_name,
    )


def ingest_directory(
    directory: Union[str, Path],
    collection_name: str = "novel",
) -> int:
    """Ingest all .md and .txt files in a directory."""
    dirpath = Path(directory)
    total = 0
    for fpath in sorted(dirpath.glob("**/*")):
        if fpath.suffix in (".md", ".txt") and fpath.is_file():
            total += ingest_file(fpath, collection_name=collection_name)
    return total


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = settings.default_top_k,
    collection_name: str = "novel",
    where: Optional[Dict] = None,
) -> List[Dict]:
    """Semantic search over stored chunks. Returns list of {text, metadata, distance}."""
    collection = get_collection(collection_name)

    if collection.count() == 0:
        return []

    kwargs = {
        "query_texts": [query],
        "n_results": min(top_k, collection.count()),
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return hits


def retrieve_for_characters(characters: List[str], top_k: int = 5) -> List[Dict]:
    """Retrieve passages relevant to specific characters."""
    all_hits = []
    for char in characters:
        hits = retrieve(
            f"{char} appearance personality relationships dialogue",
            top_k=top_k,
        )
        all_hits.extend(hits)
    seen = set()
    deduped = []
    for h in all_hits:
        key = h["text"][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(h)
    return deduped


def retrieve_for_locations(locations: List[str], top_k: int = 5) -> List[Dict]:
    """Retrieve passages relevant to specific locations."""
    all_hits = []
    for loc in locations:
        hits = retrieve(
            f"{loc} geography architecture atmosphere description",
            top_k=top_k,
        )
        all_hits.extend(hits)
    seen = set()
    deduped = []
    for h in all_hits:
        key = h["text"][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(h)
    return deduped


def retrieve_for_plot(query: str = "unresolved conflict tension mystery", top_k: int = 8) -> List[Dict]:
    """Retrieve passages about active plot threads."""
    return retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_collection_stats(collection_name: str = "novel") -> Dict:
    collection = get_collection(collection_name)
    return {
        "name": collection_name,
        "count": collection.count(),
    }


def clear_collection(collection_name: str = "novel") -> None:
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
