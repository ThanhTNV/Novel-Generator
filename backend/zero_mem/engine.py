"""
Zero-Mem engine for the novel generator.

Pipeline (deterministic; no LLM call is made to store, index, or retrieve):

  1. Query profile φ(q): entities, keywords, intent.
  2. Routing: relational (graph-primary) vs local (hierarchy-primary).
  3. Dual-view scoring: PPR over the entity graph + coarse-to-fine hierarchy
     (document -> scene -> segment) fusing BM25 with dense similarity.
  4. Fusion: S(s) = ρ·Ŝ_primary(s) + (1-ρ)·Ŝ_secondary(s), ρ = 0.6.
  5. Evidence closure: pull in the neighbouring segments and the section
     heading, then emit **contiguous spans** rather than isolated fragments.

Step 5 is what most directly fixes the "bad context" symptom: a novelist model
handed six disconnected 400-word chunks writes incoherent prose, while the same
budget spent on two contiguous, correctly-attributed passages does not.
"""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .embeddings import HashEmbedder, create_embedder
from .extract import Gazetteer, Mention, _canonical, build_gazetteer, extract_entities
from .gemini import GeminiExtractorError, SegmentExtraction
from .graph import EntityGraph
from .segment import Segment, segment_document
from .store import Trace, TraceStore
from .text import (
    BM25Index,
    content_tokens,
    cosine,
    estimate_tokens,
    hash_embed,
    max_normalize,
)

RHO = 0.6                 # primary-view weight
NEIGHBOUR_SCORE = 0.45    # relative score given to closure neighbours
CANON_BOOST = 1.15        # context/ files are authoritative reference material
RECENCY_BOOST = 1.35      # latest chapters win for continuity questions

# Reference material occupies ordinals 0..N and narrative starts above it, so
# world-bible entries always sort before chapter one and "later" always means
# further along the story.
REFERENCE_ORDINAL_BASE = 0
NARRATIVE_ORDINAL_BASE = 1000

_CHAPTER_NUM_RE = re.compile(r"(\d+)")

_CONTINUITY_HINTS = (
    "hiện tại", "bây giờ", "gần đây", "mới nhất", "cuối cùng", "sau cùng",
    "chuyện gì", "điều gì", "tiếp theo", "trước đó", "đã xảy ra", "kết thúc",
    "latest", "current", "recently", "so far", "last chapter", "previously",
    "what happened", "next", "continue", "tiếp tục",
)
_DESCRIPTION_HINTS = (
    "ngoại hình", "tính cách", "trông như", "mô tả", "là ai", "là gì",
    "appearance", "personality", "describe", "who is", "what is", "looks like",
)


class Evidence(object):
    __slots__ = ("trace", "score", "role", "matched_entities", "graph_score", "hier_score")

    def __init__(self, trace: Trace, score: float, role: str,
                 matched_entities: Optional[List[str]] = None,
                 graph_score: float = 0.0, hier_score: float = 0.0):
        self.trace = trace
        self.score = score
        self.role = role                      # 'match' | 'neighbour' | 'heading'
        self.matched_entities = matched_entities or []
        self.graph_score = graph_score
        self.hier_score = hier_score

    def to_dict(self) -> Dict[str, Any]:
        d = self.trace.to_dict()
        d.update({
            "score": round(self.score, 4),
            "role": self.role,
            "matched_entities": self.matched_entities,
            "graph_score": round(self.graph_score, 4),
            "hierarchy_score": round(self.hier_score, 4),
        })
        return d


class QueryProfile(object):
    __slots__ = ("raw", "keywords", "entity_keys", "entity_names", "intent")

    def __init__(self, raw, keywords, entity_keys, entity_names, intent):
        self.raw = raw
        self.keywords = keywords
        self.entity_keys = entity_keys
        self.entity_names = entity_names
        self.intent = intent               # 'continuity' | 'description' | 'general'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keywords": self.keywords[:12],
            "entities": self.entity_names,
            "intent": self.intent,
        }


class ZeroMemEngine(object):
    """Owns the substrate, the indices, and retrieval."""

    def __init__(
        self,
        db_path: str,
        context_dir: Optional[str] = None,
        embedder=None,
        extractor=None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.log = logger or (lambda msg: None)
        self.store = TraceStore(db_path)
        self.context_dir = context_dir
        self.embedder = embedder or HashEmbedder()
        # Optional SLM extractor (Gemini). The deterministic gazetteer/pattern
        # pipeline always runs; the SLM adds relations and unlisted entities.
        self.extractor = extractor
        self.gazetteer: Gazetteer = build_gazetteer(context_dir)
        self.graph = EntityGraph()
        self.bm25 = BM25Index()
        self.traces: Dict[int, Trace] = {}
        self._dense: Dict[int, List[float]] = {}
        self._hash_vectors: Dict[int, List[float]] = {}
        self.reload()

    # -- indexing -----------------------------------------------------------

    def reload(self) -> None:
        """Rebuild in-memory indices from the durable substrate."""
        self.graph.clear()
        self.bm25 = BM25Index()
        self.traces = {}
        self._hash_vectors = {}
        for trace in self.store.all_traces():
            self.traces[trace.id] = trace
            self.bm25.add(trace.id, self._indexable_text(trace))
            self._hash_vectors[trace.id] = hash_embed(trace.text)
        entity_meta = dict((k, (n, t)) for k, n, t, _c in self.store.entities())
        for seg_id, key, count in self.store.all_mentions():
            name, etype = entity_meta.get(key, (key, "PROPER"))
            self.graph.add_mention(seg_id, key, name, etype, count)
        self._dense = self.store.load_embeddings(self.embedder.space)
        missing = [t for t in self.traces.values() if t.id not in self._dense]
        if missing:
            self._embed_traces(missing)

    @staticmethod
    def _indexable_text(trace: Trace) -> str:
        """Fold the section heading into the lexical index for the segment."""
        if trace.heading and trace.heading not in trace.text:
            return trace.heading + "\n" + trace.text
        return trace.text

    def _embed_traces(self, traces: Sequence[Trace]) -> None:
        """
        Embed segments, reusing any vector already computed for identical text.

        Superseding a source gives its segments new row ids, so without the
        content-hash cache every re-ingest would re-embed unchanged prose and
        spend API quota for nothing.
        """
        if not traces:
            return
        space = self.embedder.space
        texts = dict((t.id, self._indexable_text(t)) for t in traces)
        hashes = dict(
            (tid, hashlib.sha256(text.encode("utf-8")).hexdigest())
            for tid, text in texts.items()
        )
        cached = self.store.embed_cache_get(list(set(hashes.values())), space)

        pairs: List[Tuple[int, List[float]]] = []
        misses = [t for t in traces if hashes[t.id] not in cached]
        for t in traces:
            vec = cached.get(hashes[t.id])
            if vec is not None:
                self._dense[t.id] = vec
                pairs.append((t.id, vec))

        if misses:
            try:
                vectors = self.embedder.encode([texts[t.id] for t in misses], is_query=False)
            except Exception as exc:
                self.log("zero-mem: embedding failed (%s); relying on BM25 + graph." % exc)
                vectors = []
            fresh: List[Tuple[str, List[float]]] = []
            for trace, vec in zip(misses, vectors):
                self._dense[trace.id] = vec
                pairs.append((trace.id, vec))
                fresh.append((hashes[trace.id], vec))
            if fresh:
                try:
                    self.store.embed_cache_put(fresh, space)
                except Exception as exc:
                    self.log("zero-mem: could not cache embeddings (%s)." % exc)

        try:
            self.store.save_embeddings(pairs, space)
        except Exception as exc:
            self.log("zero-mem: could not persist embeddings (%s)." % exc)

    def refresh_gazetteer(self) -> None:
        self.gazetteer = build_gazetteer(self.context_dir)

    # -- ingestion ----------------------------------------------------------

    def ingest_text(
        self,
        text: str,
        source: str,
        kind: str = "document",
        chapter: Optional[int] = None,
        ordinal: Optional[int] = None,
    ) -> int:
        """
        Store a document, replacing any previous version of the same source.

        Returns the number of segments stored.
        """
        segments = segment_document(text)
        if not segments:
            self.store.delete_source(source)
            self.reload()
            return 0
        mentions = [extract_entities(s.text, self.gazetteer) for s in segments]
        relations = self._slm_enrich(segments, mentions)
        # Heading text describes everything beneath it; attach its entities to
        # the segments in its scene so "Văn Tâm" grounds his own bullet list.
        by_scene: Dict[int, List] = {}
        for seg, ms in zip(segments, mentions):
            if seg.kind == "heading":
                by_scene[seg.scene] = ms
        for seg, ms in zip(segments, mentions):
            if seg.kind != "heading" and seg.scene in by_scene:
                known = set(m.key for m in ms)
                for m in by_scene[seg.scene]:
                    if m.key not in known:
                        ms.append(m._replace(count=1))

        if ordinal is None:
            # Chapter N must land in the narrative band, not at ordinal N —
            # otherwise chapter 1 collides with the second reference file and
            # sorts *before* the rest of the world bible, which both scrambles
            # narrative order and denies the newest chapter its recency boost.
            if chapter is not None:
                ordinal = NARRATIVE_ORDINAL_BASE + chapter
            else:
                # Re-ingesting a source must not move it. Allocating a fresh
                # ordinal on every pass reshuffled the world bible's internal
                # order and walked reference ordinals upward until they
                # collided with the narrative band.
                ordinal = self._existing_ordinal(source)
                if ordinal is None:
                    ordinal = self._next_ordinal(kind)
        self.store.replace_source(
            name=source, segments=segments, mentions_per_segment=mentions,
            kind=kind, chapter=chapter, ordinal=ordinal, timestamp=time.time(),
            relations_per_segment=relations,
        )
        self.reload()
        return len(segments)

    def _slm_enrich(self, segments: Sequence[Segment], mentions: List[List[Mention]]) -> Optional[List[List[Dict[str, str]]]]:
        """
        Run the Gemini extractor over prose segments (cache-aware), merge its
        entities into the local mention lists, and return per-segment relation
        triples. Returns None when no extractor is configured; never raises.
        """
        if self.extractor is None:
            return None
        model = getattr(self.extractor, "model", "gemini")
        # Headings carry no facts; don't spend API tokens on them.
        work = [i for i, s in enumerate(segments) if s.kind != "heading"]
        hashes = dict(
            (i, hashlib.sha256(segments[i].text.encode("utf-8")).hexdigest())
            for i in work
        )
        cached = self.store.cache_get(list(set(hashes.values())), model)
        missing = [i for i in work if hashes[i] not in cached]

        extracted: Dict[int, SegmentExtraction] = {}
        for i in work:
            if hashes[i] in cached:
                try:
                    extracted[i] = SegmentExtraction.from_json(cached[hashes[i]])
                except ValueError:
                    missing.append(i)
        if missing:
            try:
                results = self.extractor.extract_segments([segments[i].text for i in missing])
                to_cache = []
                for i, res in zip(missing, results):
                    extracted[i] = res
                    to_cache.append((hashes[i], res.to_json()))
                self.store.cache_put(to_cache, model)
            except Exception as exc:
                # Extractors are pluggable (Gemini / Ollama / ...), each with
                # its own error type. Whatever goes wrong, the deterministic
                # pipeline still carries the ingest — it must never fail here.
                self.log("zero-mem: SLM extraction failed (%s); gazetteer/pattern NER covers this ingest." % exc)

        relations: List[List[Dict[str, str]]] = [[] for _ in segments]
        for i, res in extracted.items():
            known = set(m.key for m in mentions[i])
            for m in res.entities:
                if m.key not in known:
                    mentions[i].append(m)
                    known.add(m.key)
            relations[i] = res.relations
        return relations

    def _existing_ordinal(self, source: str) -> Optional[int]:
        for s in self.store.sources():
            if s["name"] == source:
                return s["ordinal"]
        return None

    def _next_ordinal(self, kind: str) -> int:
        # Reference material sorts before narrative so chapters keep natural order.
        base = REFERENCE_ORDINAL_BASE if kind == "reference" else NARRATIVE_ORDINAL_BASE
        existing = [s["ordinal"] for s in self.store.sources() if s["kind"] == kind]
        return max(existing) + 1 if existing else base

    def ingest_file(
        self,
        filepath: str,
        kind: Optional[str] = None,
        chapter: Optional[int] = None,
    ) -> int:
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        if kind is None:
            kind = "reference" if self._is_reference(path) else "chapter"
        if chapter is None:
            chapter = self._infer_chapter(path.name)
        return self.ingest_text(text, source=path.name, kind=kind, chapter=chapter)

    def _is_reference(self, path: Path) -> bool:
        if not self.context_dir:
            return False
        try:
            return os.path.commonpath(
                [os.path.abspath(str(path)), os.path.abspath(self.context_dir)]
            ) == os.path.abspath(self.context_dir)
        except ValueError:
            return False

    @staticmethod
    def _infer_chapter(filename: str) -> Optional[int]:
        if "chapter" in filename.lower() or "chuong" in filename.lower():
            m = _CHAPTER_NUM_RE.search(filename)
            if m:
                return int(m.group(1))
        return None

    def ingest_directory(self, directory: str, kind: Optional[str] = None) -> int:
        total = 0
        for path in sorted(Path(directory).glob("**/*")):
            if path.is_file() and path.suffix.lower() in (".md", ".txt"):
                total += self.ingest_file(str(path), kind=kind)
        return total

    def delete_source(self, source: str) -> int:
        n = self.store.delete_source(source)
        self.reload()
        return n

    def clear(self) -> None:
        self.store.clear()
        self._dense = {}
        self.reload()

    # -- query profiling ----------------------------------------------------

    def build_profile(self, query: str) -> QueryProfile:
        mentions = extract_entities(query, self.gazetteer)
        low = query.lower()
        intent = "general"
        if any(h in low for h in _CONTINUITY_HINTS):
            intent = "continuity"
        elif any(h in low for h in _DESCRIPTION_HINTS):
            intent = "description"
        return QueryProfile(
            raw=query,
            keywords=content_tokens(query),
            entity_keys=[m.key for m in mentions],
            entity_names=[m.name for m in mentions],
            intent=intent,
        )

    # -- dual-view retrieval ------------------------------------------------

    def _similarity(self, query: str) -> Callable[[int], float]:
        """Dense similarity in the configured space, hashed space as floor."""
        dense_q = None
        if self._dense:
            try:
                dense_q = self.embedder.encode([query], is_query=True)[0]
            except Exception as exc:
                self.log("zero-mem: query embedding failed (%s); using hashed space." % exc)
        hash_q = hash_embed(query)

        def sim(seg_id: int) -> float:
            if dense_q is not None:
                vec = self._dense.get(seg_id)
                if vec:
                    return max(0.0, cosine(vec, dense_q))
            return max(0.0, cosine(self._hash_vectors.get(seg_id), hash_q))

        return sim

    def _graph_view(self, seeds: Dict[str, float], sim) -> Dict[int, float]:
        if not seeds:
            return {}
        pi = self.graph.ppr(seeds)
        base = max_normalize(self.graph.score_segments(pi))
        return dict((sid, 0.75 * s + 0.25 * sim(sid)) for sid, s in base.items())

    def _hierarchy_view(self, profile: QueryProfile, sim) -> Dict[int, float]:
        """Coarse-to-fine: document -> scene -> segment."""
        tokens = list(profile.keywords)
        for name in profile.entity_names:
            tokens.extend(content_tokens(name))
        lexical = self.bm25.scores(tokens or content_tokens(profile.raw))

        def unit_score(seg_id: int) -> float:
            return lexical.get(seg_id, 0.0) + 2.0 * sim(seg_id)

        # group by source, then by scene
        by_source: Dict[str, List[Trace]] = {}
        for trace in self.traces.values():
            by_source.setdefault(trace.source, []).append(trace)

        doc_scored = []
        for source, traces in by_source.items():
            best = max(unit_score(t.id) for t in traces)
            if best > 0:
                doc_scored.append((best, source, traces))
        doc_scored.sort(key=lambda x: x[0], reverse=True)

        out: Dict[int, float] = {}
        for _best, _source, traces in doc_scored[:8]:
            scenes: Dict[int, List[Trace]] = {}
            for t in traces:
                scenes.setdefault(t.scene, []).append(t)
            scene_scored = []
            for scene, members in scenes.items():
                score = sum(unit_score(t.id) for t in members) / len(members)
                if score > 0:
                    scene_scored.append((score, members))
            scene_scored.sort(key=lambda x: x[0], reverse=True)
            for scene_score, members in scene_scored[:5]:
                for t in members:
                    # A segment inherits part of its scene's score, which keeps
                    # locally-relevant prose together instead of cherry-picking.
                    value = 0.85 * unit_score(t.id) + 0.15 * scene_score
                    if value > 0:
                        out[t.id] = max(out.get(t.id, 0.0), value)
        return out

    def search(
        self,
        query: str,
        top_k: int = 8,
        include_neighbours: bool = True,
        kinds: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        profile = self.build_profile(query)
        seeds = self.graph.align_seeds(profile.entity_keys)
        route = "relational" if seeds and profile.entity_keys else "local"

        sim = self._similarity(query)
        graph_scores = max_normalize(self._graph_view(seeds, sim))
        hier_scores = max_normalize(self._hierarchy_view(profile, sim))

        if route == "relational":
            primary, secondary = graph_scores, hier_scores
        else:
            primary, secondary = hier_scores, graph_scores

        fused: Dict[int, float] = {}
        for seg_id in set(list(primary.keys()) + list(secondary.keys())):
            fused[seg_id] = RHO * primary.get(seg_id, 0.0) + (1 - RHO) * secondary.get(seg_id, 0.0)

        max_ordinal = max([t.ordinal for t in self.traces.values()] or [0])
        calibrated: Dict[int, float] = {}
        for seg_id, score in fused.items():
            trace = self.traces.get(seg_id)
            if trace is None:
                continue
            if kinds and trace.kind not in kinds:
                continue
            value = score
            if trace.kind == "reference":
                value *= CANON_BOOST
            if profile.intent == "continuity" and trace.kind != "reference" and max_ordinal:
                # Later chapters describe the present state of the story.
                value *= 1.0 + (RECENCY_BOOST - 1.0) * (trace.ordinal / float(max_ordinal))
            calibrated[seg_id] = value

        ranked = sorted(calibrated.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        chosen: Dict[int, Evidence] = {}
        seed_keys = set(seeds.keys())

        def make(trace: Trace, score: float, role: str) -> Evidence:
            ents = self.graph.segment_entities.get(trace.id, {})
            matched = [self.graph.names.get(k, (k, ""))[0] for k in ents if k in seed_keys]
            return Evidence(trace, score, role, matched,
                            graph_scores.get(trace.id, 0.0), hier_scores.get(trace.id, 0.0))

        for seg_id, score in ranked:
            chosen[seg_id] = make(self.traces[seg_id], score, "match")

        if include_neighbours:
            by_source_seq: Dict[Tuple[str, int], Trace] = {}
            headings: Dict[Tuple[str, int], Trace] = {}
            for trace in self.traces.values():
                by_source_seq[(trace.source, trace.seq)] = trace
                if trace.seg_kind == "heading":
                    headings[(trace.source, trace.scene)] = trace
            for ev in list(chosen.values()):
                if ev.role != "match":
                    continue
                t = ev.trace
                for delta in (-1, 1):
                    nb = by_source_seq.get((t.source, t.seq + delta))
                    # Never cross a scene boundary: the adjacent segment of a
                    # different section describes a different character/place,
                    # and pulling it in would mis-attribute its facts.
                    if nb is not None and nb.id not in chosen and nb.scene == t.scene:
                        chosen[nb.id] = make(nb, ev.score * NEIGHBOUR_SCORE, "neighbour")
                head = headings.get((t.source, t.scene))
                if head is not None and head.id not in chosen:
                    chosen[head.id] = make(head, ev.score * 0.6, "heading")

        evidence = sorted(chosen.values(), key=lambda e: e.score, reverse=True)
        return {
            "route": route,
            "profile": profile.to_dict(),
            "seed_entities": sorted(seeds.items(), key=lambda kv: kv[1], reverse=True)[:8],
            "evidence": evidence,
        }

    # -- context assembly ---------------------------------------------------

    def build_context(
        self,
        query: str,
        max_tokens: int = 3000,
        top_k: int = 10,
        kinds: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve and render context as contiguous, attributed passages.

        Segments are selected by score but *emitted in narrative order*, and
        runs of adjacent segments are merged into one block, so the model reads
        continuous prose rather than shuffled fragments.
        """
        result = self.search(query, top_k=top_k, kinds=kinds)
        evidence: List[Evidence] = result["evidence"]
        if not evidence:
            return {"context": "", "used": [], "route": result["route"],
                    "profile": result["profile"], "tokens": 0}

        # Spend the budget on the highest-scoring segments first. Heading
        # segments are skipped: the section title already appears in every
        # block label, so their body would only duplicate it.
        budget = max_tokens
        keep: Dict[int, Evidence] = {}
        spent = 0
        for ev in evidence:
            if ev.trace.seg_kind == "heading":
                continue
            cost = estimate_tokens(ev.trace.text) + 8
            if spent + cost > budget:
                continue
            keep[ev.trace.id] = ev
            spent += cost

        # ...but read them back in the order the author wrote them.
        ordered = sorted(
            keep.values(),
            key=lambda e: (e.trace.ordinal, e.trace.source, e.trace.seq),
        )

        blocks: List[str] = []
        used: List[Dict[str, Any]] = []
        run: List[Evidence] = []

        def flush() -> None:
            if not run:
                return
            head = run[0].trace
            label = head.source
            if head.chapter is not None:
                label = "%s (chương %d)" % (label, head.chapter)
            if head.heading:
                label = "%s — %s" % (label, head.heading)
            body = "\n".join(ev.trace.text for ev in run)
            blocks.append("### %s\n%s" % (label, body))
            run[:] = []

        prev: Optional[Trace] = None
        for ev in ordered:
            t = ev.trace
            # Merge only prose that is truly adjacent AND under the same
            # heading — a merged block must never blend two characters.
            contiguous = (
                prev is not None
                and t.source == prev.source
                and t.heading == prev.heading
                and t.seq - prev.seq <= 2
            )
            if not contiguous:
                flush()
            run.append(ev)
            used.append({
                "source": t.source, "chapter": t.chapter, "seq": t.seq,
                "heading": t.heading, "role": ev.role, "score": round(ev.score, 4),
                "text": t.text[:200],
            })
            prev = t
        flush()

        # Known relation triples for the entities this query is about — the
        # SLM extractor's structured facts, cheap insurance against the model
        # contradicting established relationships.
        rel_lines: List[str] = []
        seen_rel: Set[str] = set()
        for ent_name in result["profile"]["entities"]:
            for rel in self._relations_for(ent_name, _canonical(ent_name), limit=8):
                line = "- %s —%s→ %s" % (rel["subject"], rel["relation"], rel["object"])
                if line not in seen_rel:
                    seen_rel.add(line)
                    rel_lines.append(line)
        if rel_lines:
            rel_block = "### Quan hệ đã biết (known relations)\n" + "\n".join(rel_lines[:10])
            cost = estimate_tokens(rel_block)
            if spent + cost <= max_tokens:
                blocks.append(rel_block)
                spent += cost

        return {
            "context": "\n\n".join(blocks),
            "used": used,
            "route": result["route"],
            "profile": result["profile"],
            "tokens": spent,
        }

    # -- introspection ------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        base = self.store.stats()
        base.update({
            "embedder": getattr(self.embedder, "name", "unknown"),
            "vector_space": self.embedder.space,
            "extractor": getattr(self.extractor, "model", None) or "local-gazetteer-ner",
            "relations": self.store.relation_count(),
            "gazetteer_entries": len(self.gazetteer),
            "indexed_segments": len(self.traces),
        })
        return base

    def _relations_for(self, display: str, key: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Triples where the entity is subject or object, accent-folded."""
        wanted = set(k for k in (_canonical(display), key) if k)
        out = []
        for rel in self.store.all_relations():
            if _canonical(rel["subject"]) in wanted or _canonical(rel["object"]) in wanted:
                out.append(rel)
                if len(out) >= limit:
                    break
        return out

    def entity_profile(self, name: str, limit: int = 8) -> Dict[str, Any]:
        """Everything the store knows about one character/place/item."""
        mentions = extract_entities(name, self.gazetteer)
        # The fallback key must be accent-folded like every key in the graph.
        # Lowercasing alone left the diacritics on, so looking up a character
        # the gazetteer does not declare — "Khải", "Tuyên", the short forms
        # prose actually uses — could never match 'khai'/'tuyen' and always
        # reported found=False.
        keys = [m.key for m in mentions] or [_canonical(name)]
        seeds = self.graph.align_seeds(keys)
        if not seeds:
            return {"entity": name, "found": False, "segments": [], "related": []}
        key = max(seeds.items(), key=lambda kv: kv[1])[0]
        seg_ids = self.graph.entity_segments.get(key, {})
        traces = [self.traces[i] for i in seg_ids if i in self.traces]
        traces.sort(key=lambda t: (t.ordinal, t.seq))
        display, etype = self.graph.names.get(key, (name, "PROPER"))
        return {
            "entity": display,
            "type": etype,
            "found": True,
            "mentions": sum(seg_ids.values()),
            "segments": [t.to_dict() for t in traces[-limit:]],
            "relations": self._relations_for(display, key),
            "related": [
                {"name": n, "weight": round(w, 4)}
                for k2, n, w in self.graph.neighbors(key, 14)
                # numbers/chapter markers are graph glue, not "related entities"
                if not k2.startswith(("num:", "chapter:"))
            ][:10],
        }
