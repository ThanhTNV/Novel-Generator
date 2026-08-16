# Zero-Mem → Rust: migration plan

Replace the `backend/zero_mem` internals with a Rust extension module built by
[maturin](https://www.maturin.rs/) and bound with [PyO3](https://pyo3.rs/).
FastAPI, the REST surface, `agent.py`, and `api_client.py` stay in Python.

Two decisions frame everything below:

* **Rust core via PyO3**, not a sidecar. The measured cost is numeric inner
  loops, and a process boundary would eat a large share of the win it is
  supposed to deliver.
* **Greenfield rewrite** from the spec (arXiv:2607.29377), not a line-by-line
  port. This is the faster and cleaner path, and it is also the risky one:
  the current engine encodes ~30 Vietnamese-specific behaviours that were each
  found the hard way. [The behavioural contract](#3-the-behavioural-contract)
  exists so the rewrite re-derives them deliberately instead of rediscovering
  them in production.

---

## 1. Measured baseline

Synthetic Vietnamese corpus, hashed embedder, CPython 3.9.7, Windows.
Reproduced with `scripts/bench_zero_mem.py` (see [phase 0](#phase-0--harness-and-baseline)).

| Corpus | Segments | `search()` | `reload()` per ingest | Ingest total |
|---|---|---|---|---|
| 20 chapters | 520 | 86 ms | 875 ms | 9.4 s |
| 100 chapters | 2 600 | **296 ms** | **3 578 ms** | **167 s** |

`cProfile` over 25 searches at 2 600 segments:

| Component | Share of query time | Why |
|---|---|---|
| `text.cosine` | **75.5 %** | 126 040 calls / 25 searches ≈ **5 042 per query** over a 2 600-segment corpus — `sim(seg_id)` is recomputed by `_graph_view` *and* `_hierarchy_view` with no memoisation. 48.5 M generator steps inside `sum()`. |
| `graph.ppr` | 17 % | 20 iterations over Python dicts. |
| `bm25.scores` | 3 % | Full scan of every document per query; cheap today, linear in corpus size. |

Two findings matter more than the language:

1. **Ingest is O(N²).** `ZeroMemEngine.reload()` runs after *every*
   `ingest_text()` and rebuilds all indices from scratch — re-reading every
   trace, re-adding every BM25 document, and re-running `hash_embed` over the
   whole corpus. At 100 chapters that is 3.6 s of rebuild per chapter ingested.
2. **`sim()` is computed twice per segment per query.** Memoising it is a ~2×
   query win available in Python today, before any Rust exists.

Both are algorithmic. Rust makes the constants ~100× smaller; only the redesign
in [§4](#4-algorithmic-changes) removes the quadratic term. The plan does both,
and reports them separately so the gains are attributable.

---

## 2. Target architecture

```
backend/
  server.py          unchanged   FastAPI, REST, SSE
  agent.py           unchanged   prompt assembly
  api_client.py      unchanged   claude / openai / groq / ollama
  config.py          unchanged
  rag_pipeline.py    thin shim   get_engine() -> zeromem_rs.Engine
  zero_mem/
    embeddings.py    stays Python — network I/O, not compute
    gemini.py        stays Python
    ollama_extract.py stays Python

zeromem-rs/                       new crate
  Cargo.toml
  src/
    text.rs      tokenise, accent-fold, stopwords, BM25 (inverted), hash_embed, cosine
    segment.rs   Unicode sentence/paragraph/heading segmentation
    extract.rs   gazetteer + Unicode proper-noun scanner
    graph.rs     entity-context graph, CSR-sparse PPR
    store.rs     rusqlite — same schema, same file
    engine.rs    routing, dual-view fusion, evidence closure, context assembly
    lib.rs       #[pymodule] zeromem_rs
```

**Boundary design.** Cross-language calls are coarse: `ingest`, `search`,
`build_context`, `entity_profile`, `stats`. Per-segment or per-vector calls
never cross. Embeddings arrive as one `&[f32]` slice per batch, not as a list
of Python lists.

**The GIL.** `search()` and `ingest()` release it via `py.allow_threads`, so
FastAPI's thread pool gets real parallelism on retrieval — something the
current pure-Python engine cannot offer at all.

**The database is unchanged.** Same SQLite file, same schema, same
`extract_cache` / `embed_cache` tables keyed by content hash. An existing
`data/zero_mem.db` opens under the Rust engine with no migration, and the
Python engine can still open it. That property is what makes the parity harness
in [§5](#5-acceptance-the-parity-oracle) possible and what makes rollback a
config flag.

**Embeddings stay in Python.** `GeminiEmbedder` and `OpenAIEmbedder` are HTTP
clients; they are latency-bound on the network and there is nothing to win by
moving them. Rust owns the vectors once they arrive, stored packed.

---

## 3. The behavioural contract

A greenfield rewrite will silently drop these unless each is written as a test
first. This is the acceptance spec — **not** a description of the existing
implementation, which is why the known defects are listed separately in §3.8.

### 3.1 Tokenisation and stopwords

1. Word regex is `[^\W_]+` with Unicode semantics. ASCII `\w` is wrong here.
2. Accent folding is NFD → drop combining marks → `đ/Đ → d`. `đ` has no
   combining form and must be handled explicitly.
3. Tokens of length ≤ 1 are dropped unless they are digits.
4. Stopword tests must consult **both** the accented set and its accent-folded
   mirror. Readers type Vietnamese without diacritics; `co`/`khong`/`nguoi` are
   function words in both spellings. (This was a live bug — fixed in `b65ec7d`.)
5. The stoplist stays deliberately small. Aggressive stopping hurts a narrative
   corpus where pronouns carry referential weight.
6. `estimate_tokens` scales chars-per-token from 3.6 (ASCII) to 1.8 by
   non-ASCII ratio. A flat 4-chars/token estimate under-counts Vietnamese badly
   enough to truncate chapters.

### 3.2 Segmentation

7. Sentence boundaries break on `[.!?…]+` plus optional closing quotes, and
   only when the next character `is_uppercase()` **in any script**, or is an
   opener (`"'“‘([{—-•*`), or is a digit. The original `[A-Z]` test never
   matched `Đ/Ư/Ổ` and collapsed the entire character bible into two blobs.
8. Paragraphs split on blank lines. Markdown headings start a new scene and
   attach to every segment beneath them.
9. A block whose lines are *all* list items yields one segment per bullet —
   this project's context files describe one entity per line.
10. Only oversized paragraphs (> 320 tokens) fall back to sentence packing.
    Authored boundaries are preferred over sliding windows.

### 3.3 Entity extraction

11. Gazetteer is harvested from `context/**/*.md`: `##`–`######` headings name
    entities; `**bold**` names an ITEM only when not followed by `:` — the
    project writes attributes as `- **Ngoại hình**: …`, and admitting those
    creates hub nodes that smear PageRank across the whole cast.
12. Bold ITEMs additionally require: 2–40 chars, initial capital, contains a
    space, not a section label.
13. `_clean_heading` strips a trailing parenthetical and a trailing `– role`.
14. Section labels are skipped: anything ending in `?` or starting with
    `nhân vật phụ`, `ghi chú`, `arc `, `kết thúc`, `tổng quan`, … They organise
    the document; they are not entities.
15. Entity type is inferred from filename: `character`/`nhan-vat` → PERSON,
    `location`/`place`/`dia-diem` → PLACE, else CONCEPT.
16. Gazetteer matching tries longest key first, matches the accented surface
    form **or** its folded spelling, with `(?<![\w])…(?![\w])` guards,
    case-insensitive. "Văn Tâm" must beat "Tâm"; "Van Tam" must find "Văn Tâm".
17. The proper-noun scanner keeps a run of capitalised words joined by exactly
    one space. Multi-word runs are kept if any word is not a stopword.
    Single words are kept only if that word also appears mid-sentence somewhere
    (or is ALL-CAPS) — otherwise every sentence-initial ordinary word becomes a
    name.
18. Sentence-start detection treats markdown punctuation (`*#-•|>`) as a
    boundary, or every bold field label becomes a fake proper noun.
19. Chapter references (`chương|chapter|ch.` + digits) and numbers of ≥ 2 chars
    become typed `chapter:N` / `num:N` entities — graph glue, and excluded from
    user-facing "related entities".

### 3.4 Graph and PPR

20. `w(s,e) = c(e,s) / Σ_e' c(e',s)`; PPR with γ = 0.6, ≤ 20 iterations,
    early exit at L1 delta < 1e-6.
21. Seed alignment: exact key match = 1.0; containment (query key ≥ 3 chars)
    = length-ratio × 0.85, so a short alias never outranks the full name.
    Top 8 seeds, L1-normalised.
22. `S_graph(s) = Σ_{e∈s} π(e)` — a plain sum. Weighting by `w(s,e)` as well
    over-rewards entity-sparse segments: a one-entity segment scores `w = 1.0`
    while the passage that actually answers the question splits mass across
    every name it mentions.

### 3.5 Retrieval and fusion

23. Route is `relational` when the query grounds to seed entities, else `local`;
    the route picks which view is primary. `S = ρ·primary + (1-ρ)·secondary`,
    ρ = 0.6.
24. Graph view = `0.75·normalised_ppr + 0.25·sim`. Hierarchy view unit score =
    `bm25 + 2.0·sim`; top 8 documents by best unit, then top 5 scenes by mean
    unit, then `segment = 0.85·unit + 0.15·scene`.
25. Normalisation is **max-normalise, not min-max**. Min-max floors the weakest
    present document at zero and erases real signal during fusion.
26. Calibration: `reference` kind × 1.15; on `continuity` intent, non-reference
    segments × `1 + 0.35·(ordinal/max_ordinal)`.
27. Intent is classified from Vietnamese *and* English hint lists
    (`continuity` / `description` / `general`).
28. Evidence closure pulls `seq ± 1` at 0.45× and the section heading at 0.6×,
    **never crossing a scene boundary** — the adjacent segment of a different
    section describes a different character, and pulling it in mis-attributes
    its facts.

### 3.6 Context assembly

29. Budget is spent on the highest-scoring segments, but they are **emitted in
    narrative order** `(ordinal, source, seq)`, and adjacent runs merge into one
    block. Six shuffled fragments make a novelist model write incoherent prose;
    the same budget as two contiguous attributed passages does not.
30. A run merges only when source **and** heading match and the seq gap ≤ 2. A
    merged block must never blend two characters.
31. Heading segments are excluded from the budget — the title already appears
    in every block label.
32. Block label is `source (chương N) — heading`. Known relation triples for the
    query's entities are appended, capped at 10 lines, only if the budget allows.

### 3.7 Store

33. Re-ingesting a source replaces its previous segments **in one transaction**.
    Content-hash-keyed chunks that are never superseded leave contradictory
    versions of a scene retrievable forever — the original drift bug.
34. Ordinals: reference base 0, narrative base 1000, chapter N → `1000 + N`. An
    **existing source keeps its ordinal** on re-ingest (fixed in `b65ec7d`).
35. `extract_cache` and `embed_cache` are keyed by content hash and survive
    `clear()`, so re-ingesting unchanged prose costs no API tokens. Editing one
    paragraph of a chapter must re-embed exactly one segment.
36. Vector `space` id is `model@dims` and must be final before the first
    `encode()` — the engine reads it during load, before anything is encoded.
    A changed hashing scheme bumps the space id rather than silently comparing
    incomparable vectors.
37. Entities are garbage-collected when no mention references them.

### 3.8 Known defects — fix, do not re-derive

* **Heading grounding is lost past the 4th segment under a heading.** `scene`
  increments every `scene_size = 4` segments, but the heading segment stays in
  the first scene. Verified: a 6-paragraph section puts paragraphs 5–6 in
  scene 2, which contains no heading segment. Two consequences — heading
  entities are not propagated to those segments during ingest, and heading
  evidence closure finds nothing for them. In a long character-bible section the
  later paragraphs lose their entity grounding entirely.
  *Fix:* track heading identity separately from the scene window, e.g. a
  `section_id` that spans scene boundaries, and key closure on that.
* **`reload()` after every ingest** — see [§4](#4-algorithmic-changes).
* **`sim()` recomputed twice per segment per query** — memoise per query.
* **`BM25Index.scores()` scans every document** — invert it.
* **`_relations_for` scans all relations linearly, per entity** — index by
  canonicalised subject/object.
* **`_next_ordinal` bases `document` kind at 1000**, so a `document`-kind source
  ingested before any chapter can collide with chapter 0. Decide the band
  explicitly.

---

## 4. Algorithmic changes

These are the wins the language does not give for free. Each is worth doing on
its own merits and each is independently measurable.

**Incremental indexing.** Ingest touches only the affected source: remove that
source's postings from the inverted index, its rows from the vector matrix, and
its edges from the graph, then insert the new ones. No full rebuild. This is
what turns O(N²) ingest into O(ΔN).

**Inverted BM25.** `term → Vec<(seg_id, tf)>` postings, scoring only documents
that contain a query term instead of all of them.

**Packed vector matrix.** One contiguous `Vec<f32>` of `n_segments × dims` with
a row index, not a map of boxed float lists. Cosine over an L2-normalised matrix
is a dot product; the whole corpus is one matrix–vector multiply, autovectorised
(`chunks_exact(8)` + explicit f32 accumulation, no `unsafe` needed). Query
similarity is computed **once per query for all segments**, so the double
computation disappears by construction.

**CSR-sparse PPR.** Entity↔segment adjacency in compressed sparse row form with
integer-interned keys, iterating over contiguous slices rather than hash maps.

**Interned keys.** Entity keys become `u32` symbols; string hashing leaves the
hot loop entirely.

---

## 5. Acceptance: the parity oracle

Greenfield means there is no line-by-line correspondence to check, so
correctness is established behaviourally against the Python engine, which stays
in the tree until phase 5 and reads the same database file.

```python
# tests/test_parity.py
@pytest.mark.parametrize("query", CORPUS_QUERIES)   # VI accented, VI plain, EN, pronoun follow-ups
def test_ranking_matches_python(query, corpus):
    py = python_engine.search(query, top_k=8)
    rs = rust_engine.search(query, top_k=8)
    assert ids(py) == ids(rs)
    assert scores_close(py, rs, tol=1e-6)
    assert py_context(query) == rs_context(query)   # byte-identical rendered context
```

Selected by `ZERO_MEM_BACKEND=python|rust|compare`. In `compare` the shim runs
both and logs divergence, so a staging deploy surfaces disagreement on the real
corpus before the Python engine is deleted.

Where the rewrite *intends* to differ — the §3.8 defect fixes — parity is
asserted against a hand-written expectation instead, and the Python engine is
marked `xfail` on that case. Every intentional divergence must be one of those;
anything else is a regression.

The existing 89 tests are the floor, not the ceiling: they must pass unmodified
against the Rust engine through `rag_pipeline`'s public functions.

---

## 6. Phases

Each phase ends green and shippable. Sizes assume one engineer.

### Phase 0 — harness and baseline
*~1 day.* Commit `scripts/bench_zero_mem.py` (corpus generator + the profile in
§1) and `tests/test_parity.py` with the Python engine on both sides, so the
harness is proven before it judges anything. Land the two free Python wins —
memoise `sim()` per query, and skip `reload()` when nothing changed — and record
the new baseline. **Exit:** benchmark reproducible in CI; parity harness green.

### Phase 1 — `text.rs`
*~3 days.* Tokenisation, accent folding, stopwords, `estimate_tokens`, inverted
BM25, `hash_embed`, packed cosine. The purest functions and the largest share of
the profile. **Exit:** contract items 1–6 tested; `hash_embed` byte-identical to
Python for a fixture corpus (it is a specified hash, so this is exact);
BM25 scores within 1e-9.

### Phase 2 — `segment.rs` + `extract.rs`
*~4 days.* Segmentation and the gazetteer/proper-noun scanner. Contract items
7–19. The regex-heavy part, and where Unicode correctness bites — Rust's `regex`
crate has no look-around, so item 16's `(?<![\w])…(?![\w])` guards need explicit
boundary checks or the `fancy-regex` crate. **Exit:** identical segment and
mention output on `context/*.md` and a chapter fixture.

### Phase 3 — `graph.rs` + `store.rs`
*~4 days.* CSR PPR and `rusqlite` against the unchanged schema. Contract items
20–22, 33–37. **Exit:** PPR vectors within 1e-9 of Python for shared seeds; an
existing `zero_mem.db` round-trips through both engines byte-identically.

### Phase 4 — `engine.rs` + PyO3 boundary
*~5 days.* Routing, fusion, calibration, closure, context assembly, incremental
indexing. Contract items 23–32. This is where the §3.8 heading fix lands.
**Exit:** full parity suite green; the 89 existing tests pass against
`ZERO_MEM_BACKEND=rust`; benchmark numbers recorded against §1.

### Phase 5 — packaging and cutover
*~3 days.* maturin wheel build in CI (`manylinux` + Windows + macOS), Dockerfile
gains a Rust build stage, `ZERO_MEM_BACKEND` defaults to `rust`. Python engine
stays one release as the documented fallback, then is deleted.

**Total ≈ 4 weeks.** Phases 1–3 are independent once phase 0 lands and can be
parallelised across people.

---

## 7. Projected performance

Extrapolated from the §1 profile at 2 600 segments. These are **targets for the
phase exit criteria to validate**, not measurements.

| | Python (measured) | Rust (projected) | |
|---|---|---|---|
| `search()` | 296 ms | **2–5 ms** | ~60–100× |
| ├ cosine, whole corpus | 223 ms | < 1 ms | packed f32 matrix–vector, computed once |
| ├ PPR | 50 ms | < 1 ms | CSR + interned u32 keys |
| └ BM25 | 9 ms | < 0.5 ms | postings, not full scan |
| Ingest, 100 chapters | 167 s | **< 2 s** | incremental index — *algorithmic, not language* |
| Resident vectors | ~64 MB | ~8 MB | `Vec<f32>` vs boxed Python floats |
| Concurrent retrieval | serialised by GIL | parallel | `allow_threads` |

The ingest row is the one to watch: most of it is available in Python too. If
the schedule slips, phase 0's incremental-index work delivers the largest single
improvement in the table on its own.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| **Greenfield silently drops a Vietnamese behaviour.** The primary risk. | §3 is the spec, written as tests before the corresponding phase starts. `compare` mode on the real corpus before cutover. |
| Rust `regex` lacks look-around (contract item 16). | Explicit boundary checks, or `fancy-regex`. Decide in phase 2, not phase 4. |
| Unicode case/normalisation differs from Python's `str.isupper()` / NFD. | Pin `unicode-normalization` + `unicode-segmentation`; item 7 and item 2 get exhaustive fixture tests across Vietnamese, Latin, and CJK. |
| Float non-determinism between SIMD and scalar summation changes ranking at ties. | Deterministic accumulation order; break ties on `(score, ordinal, seq)` explicitly rather than relying on sort stability. |
| Build complexity for contributors. | Publish wheels from CI; `pip install -e .` keeps working via maturin's PEP 517 backend. Fallback path stays documented while the Python engine remains. |
| Windows/macOS/Linux wheel matrix. | `maturin-action` in CI from phase 1, not phase 5 — cross-platform breakage found early is cheap. |

---

## 9. What is explicitly out of scope

`server.py`, `agent.py`, `api_client.py`, `config.py`, the frontend, and the
embedding/extractor HTTP clients. They are I/O-bound glue, they are not in the
profile, and rewriting them buys nothing this plan is trying to buy.
