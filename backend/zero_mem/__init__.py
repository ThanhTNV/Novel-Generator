"""
Zero-Mem: zero-token memory operations for the novel generator.

Implements arXiv:2607.29377 — original narrative traces stay the source of
record, and retrieval is deterministic structured evidence selection rather
than a generated summary of the past. No LLM call is made to store, index, or
retrieve; the writing model is the only thing that spends tokens.
"""

from .engine import Evidence, QueryProfile, ZeroMemEngine
from .extract import Gazetteer, Mention, build_gazetteer, extract_entities
from .graph import EntityGraph
from .segment import Segment, segment_document, split_sentences
from .store import Trace, TraceStore
from .text import BM25Index, content_tokens, estimate_tokens, tokenize

__all__ = [
    "ZeroMemEngine",
    "Evidence",
    "QueryProfile",
    "EntityGraph",
    "Gazetteer",
    "Mention",
    "build_gazetteer",
    "extract_entities",
    "Segment",
    "segment_document",
    "split_sentences",
    "Trace",
    "TraceStore",
    "BM25Index",
    "content_tokens",
    "estimate_tokens",
    "tokenize",
]
