"""
Lexical primitives for Zero-Mem: Unicode-aware tokenisation, BM25, and a
dependency-free hashed embedding used as the dense fallback.

Everything here is deterministic and costs zero LLM tokens.
"""

import math
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Set

# Vietnamese function words + English stopwords. Kept small on purpose: over
# aggressive stopping hurts a narrative corpus where "he/she/they" carry
# referential weight.
STOPWORDS = set("""
a an the and or but if then else when while of at by for with about against
between into through during before after above below to from up down in out on
off over under again further once here there all any both each few more most
other some such no nor not only own same so than too very can will just should
now is are was were be been being have has had do does did doing would could as
until because how where why what which who whom this that these those it its

và là của có ở cho với những các một này đó khi thì mà nhưng nếu vì nên rồi đã
đang sẽ được cũng như từ đến trong ngoài trên dưới ra vào lên xuống rất quá lắm
không chưa chẳng đừng hãy nữa lại còn về theo bởi tại do nơi ai gì sao nào đây
kia ấy nó họ chúng tôi ta bạn mình anh chị em ông bà thầy cô người
""".split())

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _strip_accents_early(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.replace("đ", "d").replace("Đ", "d")


# Accent-folded mirror of STOPWORDS, so canonical (folded) keys can be tested
# too: 'có' folds to 'co', which must also count as a stopword.
FOLDED_STOPWORDS = set(_strip_accents_early(w) for w in STOPWORDS) | STOPWORDS


def _strip_accents(token: str) -> str:
    """Fold Vietnamese diacritics so 'Tâm' and 'Tam' share a lexical key."""
    decomposed = unicodedata.normalize("NFD", token)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    # đ/Đ has no combining form; normalise it explicitly.
    return stripped.replace("đ", "d").replace("Đ", "d")


def tokenize(text: str, fold_accents: bool = True) -> List[str]:
    """Unicode word tokens, lowercased, optionally accent-folded."""
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    if fold_accents:
        tokens = [_strip_accents(t) for t in tokens]
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def content_tokens(text: str) -> List[str]:
    """Tokens carrying meaning: tokenised minus stopwords."""
    out = []
    for raw in _WORD_RE.finditer(text):
        low = raw.group(0).lower()
        if low in STOPWORDS:
            continue
        folded = _strip_accents(low)
        if folded in STOPWORDS:
            continue
        if len(folded) > 1 or folded.isdigit():
            out.append(folded)
    return out


def estimate_tokens(text: str) -> int:
    """
    Approximate LLM tokens. Vietnamese costs far more tokens per character than
    English (diacritics split into multiple BPE units), so scale by how much of
    the text is non-ASCII instead of assuming ~4 chars/token.
    """
    if not text:
        return 0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / len(text)
    # ~3.6 chars/token for plain ASCII prose, ~1.8 for heavily accented text.
    chars_per_token = 3.6 - 1.8 * min(ratio * 4.0, 1.0)
    return max(1, int(len(text) / chars_per_token))


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Index:
    """Incremental in-memory BM25. Supports removal so re-ingest can supersede."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tf: Dict[int, Dict[str, int]] = {}
        self._len: Dict[int, int] = {}
        self._df: Dict[str, int] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._tf)

    def add(self, doc_id: int, text: str) -> None:
        if doc_id in self._tf:
            self.remove(doc_id)
        tokens = tokenize(text)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for term in tf:
            self._df[term] = self._df.get(term, 0) + 1
        self._tf[doc_id] = tf
        self._len[doc_id] = len(tokens)
        self._total_len += len(tokens)

    def remove(self, doc_id: int) -> None:
        tf = self._tf.pop(doc_id, None)
        if tf is None:
            return
        for term in tf:
            remaining = self._df.get(term, 1) - 1
            if remaining <= 0:
                self._df.pop(term, None)
            else:
                self._df[term] = remaining
        self._total_len -= self._len.pop(doc_id, 0)

    def scores(self, query_tokens: Iterable[str]) -> Dict[int, float]:
        n = len(self._tf)
        if n == 0:
            return {}
        avg_len = self._total_len / n if n else 1.0
        terms = set(query_tokens)
        out: Dict[int, float] = {}
        for doc_id, tf in self._tf.items():
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                df = self._df.get(term, 1)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                length = self._len.get(doc_id, 1)
                denom = freq + self.k1 * (1 - self.b + self.b * length / avg_len)
                score += idf * freq * (self.k1 + 1) / denom
            if score > 0:
                out[doc_id] = score
        return out


# ---------------------------------------------------------------------------
# Hashed n-gram embeddings (dense fallback, no model download)
# ---------------------------------------------------------------------------

EMBED_DIM = 384
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def _fnv1a(text: str, seed: int) -> int:
    h = (_FNV_OFFSET ^ seed) & _MASK32
    for ch in text:
        h ^= ord(ch) & 0xFF
        h = (h * _FNV_PRIME) & _MASK32
    return h


def hash_embed(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Signed feature hashing over words, bigrams, and character trigrams."""
    vec = [0.0] * dim
    words = tokenize(text)

    def feed(feature: str, weight: float) -> None:
        idx = _fnv1a(feature, 0x9747B28C) % dim
        sign = 1.0 if _fnv1a(feature, 0x3C6EF372) & 1 else -1.0
        vec[idx] += sign * weight

    for w in words:
        stop = w in STOPWORDS
        feed("w:" + w, 0.2 if stop else 1.0)
        if not stop:
            padded = "^" + w + "$"
            for i in range(len(padded) - 2):
                feed("c:" + padded[i:i + 3], 0.35)
    for i in range(len(words) - 1):
        feed("b:%s_%s" % (words[i], words[i + 1]), 0.5)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else list(vec)


def max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Scale scores to [0, 1] by their maximum, preserving ratios. Min-max would
    floor the weakest present document at zero and erase real signal during
    dual-view fusion.
    """
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0:
        return dict((k, 0.0) for k in scores)
    return dict((k, max(0.0, v) / top) for k, v in scores.items())
