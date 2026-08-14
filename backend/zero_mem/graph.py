"""
Entity-context graph  G = (V_s ∪ V_e, E_se ∪ E_ss)

Co-occurrence weight   w(s, e) = c(e, s) / Σ_e' c(e', s)
Personalized PageRank  π_q = (1 - γ) r_q + γ Pᵀ π_q      with γ = 0.6

Propagation walks entity -> segment -> entity, which is what lets a question
about a character reach the passages that answer it even when those passages
share no vocabulary with the question.
"""

from typing import Dict, Iterable, List, Optional, Tuple

GAMMA = 0.6


class EntityGraph(object):
    def __init__(self) -> None:
        # entity key -> {segment id -> raw mention count}
        self.entity_segments: Dict[str, Dict[int, int]] = {}
        self.entity_total: Dict[str, int] = {}
        # segment id -> {entity key -> raw mention count}
        self.segment_entities: Dict[int, Dict[str, int]] = {}
        self.names: Dict[str, Tuple[str, str]] = {}   # key -> (name, type)

    def add_mention(self, segment_id: int, key: str, name: str, etype: str, count: int) -> None:
        self.names.setdefault(key, (name, etype))
        seg_map = self.entity_segments.setdefault(key, {})
        seg_map[segment_id] = seg_map.get(segment_id, 0) + count
        self.entity_total[key] = self.entity_total.get(key, 0) + count
        ent_map = self.segment_entities.setdefault(segment_id, {})
        ent_map[key] = ent_map.get(key, 0) + count

    def clear(self) -> None:
        self.entity_segments.clear()
        self.entity_total.clear()
        self.segment_entities.clear()
        self.names.clear()

    def weight(self, segment_id: int, key: str) -> float:
        """w(s, e) = c(e, s) / Σ_e' c(e', s)"""
        ents = self.segment_entities.get(segment_id)
        if not ents:
            return 0.0
        c = ents.get(key, 0)
        if not c:
            return 0.0
        total = sum(ents.values())
        return c / total if total else 0.0

    # -- retrieval ----------------------------------------------------------

    def align_seeds(
        self,
        query_keys: Iterable[str],
        max_seeds: int = 8,
    ) -> Dict[str, float]:
        """
        Ground the query's entities against the graph. Exact key match scores
        1.0; a containment match ('tâm' inside 'văn tâm') scores by length
        ratio so a short alias does not outrank the full name.
        """
        query_keys = [k for k in query_keys if k]
        if not query_keys:
            return {}
        scored: List[Tuple[str, float]] = []
        for key in self.entity_segments:
            best = 0.0
            for qk in query_keys:
                if qk == key:
                    best = 1.0
                    break
                if len(qk) >= 3 and (qk in key or key in qk):
                    ratio = min(len(qk), len(key)) / float(max(len(qk), len(key)))
                    if ratio * 0.85 > best:
                        best = ratio * 0.85
            if best > 0:
                scored.append((key, best))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        seeds = dict(scored[:max_seeds])
        total = sum(seeds.values())
        if total > 0:
            seeds = dict((k, v / total) for k, v in seeds.items())
        return seeds

    def ppr(self, seeds: Dict[str, float], iterations: int = 20) -> Dict[str, float]:
        """Personalized PageRank over the entity projection of the graph."""
        if not seeds:
            return {}
        pi = dict(seeds)
        for _ in range(iterations):
            # entity mass -> segments, split by that entity's per-segment counts
            seg_mass: Dict[int, float] = {}
            for key, mass in pi.items():
                total = self.entity_total.get(key, 0)
                if not total:
                    continue
                for seg_id, count in self.entity_segments.get(key, {}).items():
                    seg_mass[seg_id] = seg_mass.get(seg_id, 0.0) + mass * count / total
            # segment mass -> entities, split by w(s, e)
            nxt: Dict[str, float] = {}
            for seg_id, mass in seg_mass.items():
                ents = self.segment_entities.get(seg_id)
                if not ents:
                    continue
                total = sum(ents.values())
                if not total:
                    continue
                for key, count in ents.items():
                    nxt[key] = nxt.get(key, 0.0) + mass * count / total
            merged: Dict[str, float] = dict((k, GAMMA * v) for k, v in nxt.items())
            for key, restart in seeds.items():
                merged[key] = merged.get(key, 0.0) + (1 - GAMMA) * restart
            delta = sum(abs(v - pi.get(k, 0.0)) for k, v in merged.items())
            pi = merged
            if delta < 1e-6:
                break
        return pi

    def score_segments(self, pi: Dict[str, float]) -> Dict[int, float]:
        """
        S_graph(s) = Σ_{e ∈ s} π_q(e)

        Activations are already seed-anchored, so summing them is correct here.
        Weighting by w(s, e) as well would over-reward entity-sparse segments: a
        one-entity segment gets w = 1.0 while the passage that actually answers
        the question splits its mass across every name it mentions.
        """
        scores: Dict[int, float] = {}
        for key, activation in pi.items():
            for seg_id in self.entity_segments.get(key, {}):
                scores[seg_id] = scores.get(seg_id, 0.0) + activation
        return scores

    def neighbors(self, key: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Entities co-occurring with `key`, strongest first."""
        acc: Dict[str, float] = {}
        for seg_id in self.entity_segments.get(key, {}):
            w_self = self.weight(seg_id, key)
            for other in self.segment_entities.get(seg_id, {}):
                if other == key:
                    continue
                acc[other] = acc.get(other, 0.0) + w_self * self.weight(seg_id, other)
        ranked = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [(k, self.names.get(k, (k, "PROPER"))[0], w) for k, w in ranked]
