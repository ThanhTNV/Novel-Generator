# -*- coding: utf-8 -*-
"""
Zero-Mem retrieval benchmark — the baseline MIGRATION.md phase 0 is measured
against, and the harness each later phase reports its exit numbers with.

Generates a synthetic Vietnamese corpus at a given chapter count, then times
ingest, reload, end-to-end search, and each component of the query path.

    python scripts/bench_zero_mem.py                 # 20 and 100 chapters
    python scripts/bench_zero_mem.py 20 100 400      # explicit sizes
    python scripts/bench_zero_mem.py --profile 100   # cProfile a search run

Always runs on the hashed embedder: no network, no API tokens, deterministic.
"""

from __future__ import print_function

import argparse
import io
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.zero_mem.embeddings import HashEmbedder  # noqa: E402
from backend.zero_mem.engine import ZeroMemEngine  # noqa: E402
from backend.zero_mem.text import BM25Index, content_tokens, cosine, hash_embed  # noqa: E402

SEED = 7

NAMES = [u"Văn Tâm", u"Nguyên Khang", u"Thầy Cao", u"Đồ Lục", u"Ánh Nguyệt",
         u"Hải Long", u"Minh Nguyệt", u"Lâm Phong", u"Tuyết Vân", u"Bạch Hổ"]
PLACES = [u"phòng thí nghiệm", u"khu rừng", u"thành phố", u"ngọn núi", u"bờ biển"]
VERBS = [u"bước vào", u"nhìn thấy", u"chạy khỏi", u"tìm kiếm", u"trở về", u"nhặt lên"]
NOUNS = [u"cuốn sổ", u"ánh sáng", u"bóng tối", u"tiếng động", u"cánh cửa", u"con dao"]

# Deliberately spans the query shapes retrieval has to handle: accented and
# unaccented Vietnamese, both intents, and a multi-entity relational query.
QUERIES = [
    u"Văn Tâm làm gì ở phòng thí nghiệm",
    u"Van Tam lam gi o phong thi nghiem",
    u"Đồ Lục là gì",
    u"chuyện gì đã xảy ra gần đây",
    u"Nguyên Khang và Thầy Cao",
    u"mô tả ngoại hình Ánh Nguyệt",
]


def _paragraph(rng):
    return "".join(
        u"%s %s %s ở %s. " % (rng.choice(NAMES), rng.choice(VERBS),
                              rng.choice(NOUNS), rng.choice(PLACES))
        for _ in range(rng.randint(3, 6))
    )


def _chapter(rng, n, paragraphs=25):
    body = [u"## Chương %d\n" % n]
    body.extend(_paragraph(rng) for _ in range(paragraphs))
    return "\n\n".join(body)


def build_engine(n_chapters, db_dir):
    """Ingest n_chapters into a fresh engine. Returns (engine, ingest_seconds)."""
    rng = random.Random(SEED)
    engine = ZeroMemEngine(db_path=os.path.join(db_dir, "bench.db"), embedder=HashEmbedder())
    started = time.perf_counter()
    for i in range(1, n_chapters + 1):
        engine.ingest_text(_chapter(rng, i), source="ch-%03d.md" % i, kind="chapter", chapter=i)
    return engine, time.perf_counter() - started


def _time(fn, repeat):
    started = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - started) / repeat


def bench(n_chapters, repeat=3):
    tmp = tempfile.mkdtemp()
    engine = None
    try:
        engine, ingest_s = build_engine(n_chapters, tmp)
        segments = len(engine.traces)
        engine.search(QUERIES[0], top_k=8)          # warm

        search_ms = _time(
            lambda: [engine.search(q, top_k=8) for q in QUERIES], repeat
        ) / len(QUERIES) * 1000

        tokens = content_tokens(QUERIES[0])
        bm25_ms = _time(lambda: engine.bm25.scores(tokens), 20) * 1000

        query_vec = hash_embed(QUERIES[0])
        vectors = list(engine._hash_vectors.values())
        cosine_ms = _time(lambda: [cosine(v, query_vec) for v in vectors], 5) * 1000

        seeds = engine.graph.align_seeds([u"van tam"])
        ppr_ms = _time(lambda: engine.graph.ppr(seeds), 10) * 1000

        embed_us = _time(lambda: hash_embed(_paragraph(random.Random(1))), 200) * 1e6
        reload_ms = _time(engine.reload, 1) * 1000

        print("chapters=%-4d segments=%-6d entities=%-5d"
              % (n_chapters, segments, len(engine.graph.names)))
        print("   ingest total        %8.2f s   (%.1f ms/chapter)"
              % (ingest_s, ingest_s / n_chapters * 1000))
        print("   reload (per ingest) %8.1f ms  <- runs after EVERY ingest: O(N) each, O(N^2) total"
              % reload_ms)
        print("   search end-to-end   %8.1f ms" % search_ms)
        print("     - bm25.scores      %7.1f ms   (full scan of %d docs)" % (bm25_ms, segments))
        print("     - cosine x N       %7.1f ms   (and sim() is computed ~2x per segment)" % cosine_ms)
        print("     - graph.ppr        %7.1f ms" % ppr_ms)
        print("   hash_embed/segment  %8.1f us" % embed_us)
        print()
        return {"chapters": n_chapters, "segments": segments, "ingest_s": ingest_s,
                "reload_ms": reload_ms, "search_ms": search_ms}
    finally:
        if engine is not None:
            try:
                engine.store.close()
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


def profile(n_chapters, top=15):
    import cProfile
    import pstats

    tmp = tempfile.mkdtemp()
    engine = None
    try:
        engine, _ = build_engine(n_chapters, tmp)
        engine.search(QUERIES[0], top_k=8)
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(5):
            for q in QUERIES:
                engine.search(q, top_k=8)
        pr.disable()
        out = io.StringIO()
        pstats.Stats(pr, stream=out).sort_stats("cumulative").print_stats(top)
        print(out.getvalue())
    finally:
        if engine is not None:
            try:
                engine.store.close()
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sizes", nargs="*", type=int, default=[20, 100],
                    help="chapter counts to benchmark (default: 20 100)")
    ap.add_argument("--profile", type=int, metavar="N",
                    help="cProfile a search run at N chapters instead of benchmarking")
    args = ap.parse_args()

    if args.profile:
        profile(args.profile)
        return
    for n in args.sizes:
        bench(n)


if __name__ == "__main__":
    main()
