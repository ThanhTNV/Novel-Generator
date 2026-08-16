# -*- coding: utf-8 -*-
"""
Regression tests for defects found auditing the Zero-Mem port.

Each test names the concrete failure it locks down, so a future refactor that
reintroduces it fails here rather than in production.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.zero_mem.embeddings import HashEmbedder, OpenAIEmbedder  # noqa: E402
from backend.zero_mem.engine import ZeroMemEngine  # noqa: E402
from backend.zero_mem.text import (  # noqa: E402
    FOLDED_STOPWORDS,
    STOPWORDS,
    content_tokens,
    hash_embed,
    tokenize,
)


# ---------------------------------------------------------------------------
# Accent-folded stopwords
# ---------------------------------------------------------------------------

def test_unaccented_query_drops_the_same_stopwords_as_the_accented_one():
    """
    Readers type Vietnamese without diacritics. 'co', 'nguoi' and 'khong' are
    function words either way, but the stoplist only holds the accented
    spellings — so the folded check has to consult FOLDED_STOPWORDS.
    """
    accented = content_tokens(u"Văn Tâm có người không ở đây")
    plain = content_tokens(u"Van Tam co nguoi khong o day")
    assert accented == plain == [u"van", u"tam"]


def test_content_tokens_keeps_real_words():
    """Folding stopwords must not swallow the content words around them."""
    tokens = content_tokens(u"Đồ Lục ghi 108 Vật Chất ở phòng thí nghiệm")
    for expected in (u"luc", u"ghi", u"108", u"vat", u"chat", u"phong", u"nghiem"):
        assert expected in tokens, tokens
    # 'ở' is a function word in both spellings.
    assert u"o" not in tokens


def test_hash_embed_downweights_vietnamese_stopwords():
    """
    tokenize() folds accents before hash_embed sees a token, so testing raw
    STOPWORDS meant no Vietnamese function word was ever downweighted: each
    got full weight plus a full set of character trigrams.
    """
    for word in (u"và", u"của", u"những"):
        folded = tokenize(word)[0]
        assert folded not in STOPWORDS, "precondition: folding escapes the raw set"
        assert folded in FOLDED_STOPWORDS

    # A stopword-only string must sit far from a content-word string.
    stop_vec = hash_embed(u"và của những các một")
    content_vec = hash_embed(u"Văn Tâm Đồ Lục phòng thí nghiệm")
    overlap = sum(a * b for a, b in zip(stop_vec, content_vec))
    assert abs(overlap) < 0.2


# ---------------------------------------------------------------------------
# Vector-space identity
# ---------------------------------------------------------------------------

def test_openai_embedder_space_is_final_before_first_encode():
    """
    The engine reads embedder.space during reload(), before anything is
    encoded. A space that only became correct after the first encode() meant
    persisted vectors were never found and the corpus was re-embedded on
    every process start.
    """
    pytest.importorskip("openai")
    emb = OpenAIEmbedder("text-embedding-3-small", "sk-test")
    assert emb.space == "text-embedding-3-small@1536"
    assert emb.dim == 1536

    sized = OpenAIEmbedder("text-embedding-3-small", "sk-test", dims=512)
    assert sized.space == "text-embedding-3-small@512"


def test_hash_space_is_versioned():
    """Changing the hashing must not silently reuse vectors from the old one."""
    assert HashEmbedder().space.startswith("hash-v")


# ---------------------------------------------------------------------------
# Ordinals
# ---------------------------------------------------------------------------

def _engine(tmp_path):
    return ZeroMemEngine(db_path=str(tmp_path / "t.db"), embedder=HashEmbedder())


def test_reingesting_a_source_keeps_its_ordinal(tmp_path):
    """
    Re-ingest must not move a document in narrative order. Allocating a fresh
    ordinal each pass reshuffled the world bible and walked reference ordinals
    upward until they collided with the narrative band.
    """
    eng = _engine(tmp_path)
    eng.ingest_text(u"Văn Tâm mở cuốn sổ.", source="a.md", kind="reference")
    eng.ingest_text(u"Nguyên Khang đứng ngoài.", source="b.md", kind="reference")

    before = dict((s["name"], s["ordinal"]) for s in eng.store.sources())
    for _ in range(3):
        eng.ingest_text(u"Văn Tâm mở cuốn sổ. Lần nữa.", source="a.md", kind="reference")
    after = dict((s["name"], s["ordinal"]) for s in eng.store.sources())

    assert after == before
    assert after["a.md"] < after["b.md"]


def test_reference_ordinals_never_reach_the_narrative_band(tmp_path):
    eng = _engine(tmp_path)
    for i in range(5):
        eng.ingest_text(u"Đoạn %d." % i, source="ref-%d.md" % i, kind="reference")
    for _ in range(4):
        for i in range(5):
            eng.ingest_text(u"Đoạn %d sửa." % i, source="ref-%d.md" % i, kind="reference")

    eng.ingest_text(u"Chương một bắt đầu.", source="ch1.md", kind="chapter", chapter=1)
    ordinals = dict((s["name"], s["ordinal"]) for s in eng.store.sources())
    assert max(ordinals["ref-%d.md" % i] for i in range(5)) < ordinals["ch1.md"]
