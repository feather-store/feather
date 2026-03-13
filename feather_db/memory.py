"""
MemoryManager — Feather DB v0.6.0
===================================
Higher-order memory operations that compose existing Feather DB primitives.
All methods are stateless — pass the DB instance as the first argument.

Features
--------
  why_retrieved   Score breakdown explaining why a node was returned
  health_report   Orphan nodes, tier distribution, recall histogram
  search_mmr      Semantic search with Maximal Marginal Relevance diversity
  consolidate     Cluster similar nodes and merge into summary nodes
  assign_tiers    Classify nodes as hot / warm / cold by access patterns
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Score formula constants (must mirror scoring.h)
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_HALF_LIFE  = 30.0   # days
_DEFAULT_TIME_WT    = 0.3


class MemoryManager:
    """
    Stateless helper for living-context memory operations.

    Every public method takes `db` (a feather_db.DB instance) as its first
    argument — no coupling to any particular DB path or embedder.
    """

    # ── Score explanation ────────────────────────────────────────────────────

    @staticmethod
    def why_retrieved(
        db,
        node_id: int,
        query_vec: np.ndarray,
        half_life_days: float = _DEFAULT_HALF_LIFE,
        time_weight: float    = _DEFAULT_TIME_WT,
    ) -> dict:
        """
        Explain why `node_id` would be returned for `query_vec`.

        Returns a dict with keys:
          similarity, stickiness, effective_age_days, recency,
          importance, confidence, final_score, recall_count
        """
        meta = db.get_metadata(node_id)
        if meta is None:
            return {"error": f"Node {node_id} not found"}

        # Retrieve stored vector for similarity calculation
        stored = db.get_vector(node_id, "text")
        if len(stored) == 0:
            similarity = 0.0
        else:
            stored_arr = np.array(stored, dtype=np.float32)
            q = np.array(query_vec, dtype=np.float32)
            dist = float(np.linalg.norm(q - stored_arr))
            similarity = 1.0 / (1.0 + dist)

        now         = time.time()
        age_days    = (now - meta.timestamp) / 86400.0 if meta.timestamp > 0 else 0.0
        stickiness  = 1.0 + math.log(1.0 + meta.recall_count)
        eff_age     = age_days / stickiness
        recency     = 0.5 ** (eff_age / half_life_days)
        final_score = ((1.0 - time_weight) * similarity + time_weight * recency) * meta.importance

        return {
            "node_id":           node_id,
            "similarity":        round(similarity, 4),
            "stickiness":        round(stickiness, 3),
            "effective_age_days":round(eff_age, 2),
            "recency":           round(recency, 4),
            "importance":        round(meta.importance, 3),
            "confidence":        round(meta.confidence, 3),
            "final_score":       round(final_score, 4),
            "recall_count":      meta.recall_count,
            "formula":           (
                f"((1-{time_weight}) × {round(similarity,4)} "
                f"+ {time_weight} × {round(recency,4)}) "
                f"× {round(meta.importance,3)} = {round(final_score,4)}"
            ),
        }

    # ── Health report ────────────────────────────────────────────────────────

    @staticmethod
    def health_report(db, modality: str = "text") -> dict:
        """
        Analyse the DB and return a health summary dict.

        Keys:
          total, hot_count, warm_count, cold_count,
          orphan_count, expired_count,
          recall_histogram, avg_importance, avg_confidence
        """
        ids  = db.get_all_ids(modality)
        now  = int(time.time())

        hot_count = warm_count = cold_count = 0
        orphan_count = expired_count = 0
        total_importance = total_confidence = 0.0
        recall_hist = {"0": 0, "1-5": 0, "6-20": 0, "21-100": 0, ">100": 0}

        for nid in ids:
            meta = db.get_metadata(nid)
            if meta is None:
                continue

            rc = meta.recall_count
            if rc == 0:
                cold_count += 1
                recall_hist["0"] += 1
            elif rc <= 5:
                warm_count += 1
                recall_hist["1-5"] += 1
            elif rc <= 20:
                hot_count += 1
                recall_hist["6-20"] += 1
            elif rc <= 100:
                hot_count += 1
                recall_hist["21-100"] += 1
            else:
                hot_count += 1
                recall_hist[">100"] += 1

            out_edges = db.get_edges(nid)
            in_edges  = db.get_incoming(nid)
            if len(out_edges) == 0 and len(in_edges) == 0:
                orphan_count += 1

            if meta.ttl > 0 and now > meta.timestamp + meta.ttl:
                expired_count += 1

            total_importance  += meta.importance
            total_confidence  += meta.confidence

        n = len(ids) or 1
        return {
            "total":           len(ids),
            "hot_count":       hot_count,
            "warm_count":      warm_count,
            "cold_count":      cold_count,
            "orphan_count":    orphan_count,
            "expired_count":   expired_count,
            "recall_histogram": recall_hist,
            "avg_importance":  round(total_importance / n, 3),
            "avg_confidence":  round(total_confidence / n, 3),
        }

    # ── MMR search ───────────────────────────────────────────────────────────

    @staticmethod
    def search_mmr(
        db,
        query_vec: np.ndarray,
        k: int = 10,
        diversity: float = 0.5,
        fetch_k: Optional[int] = None,
        modality: str = "text",
        filter=None,
        scoring=None,
    ) -> list:
        """
        Semantic search with Maximal Marginal Relevance post-processing.

        diversity=0.0 → pure similarity ranking (same as db.search)
        diversity=1.0 → maximum diversity (ignores query similarity)
        diversity=0.5 → balanced default

        Returns list of feather_db.SearchResult (same type as db.search()).
        """
        if fetch_k is None:
            fetch_k = min(k * 5, 200)

        # Over-fetch candidates
        candidates = db.search(query_vec, k=fetch_k,
                               filter=filter, scoring=scoring, modality=modality)
        if not candidates:
            return []

        # Retrieve vectors for all candidates
        vecs: dict[int, np.ndarray] = {}
        for r in candidates:
            v = db.get_vector(r.id, modality)
            if len(v) > 0:
                vecs[r.id] = np.array(v, dtype=np.float32)

        q = np.array(query_vec, dtype=np.float32)

        selected: list = []
        selected_vecs: list[np.ndarray] = []
        remaining = list(candidates)

        while remaining and len(selected) < k:
            best_idx  = -1
            best_score = -1e9

            for i, r in enumerate(remaining):
                sim_q = r.score  # already computed by db.search

                # Max similarity to already-selected nodes
                if selected_vecs and r.id in vecs:
                    rv = vecs[r.id]
                    sim_sel = max(
                        float(np.dot(rv, sv) / (np.linalg.norm(rv) * np.linalg.norm(sv) + 1e-9))
                        for sv in selected_vecs
                    )
                else:
                    sim_sel = 0.0

                mmr = (1.0 - diversity) * sim_q - diversity * sim_sel

                if mmr > best_score:
                    best_score = mmr
                    best_idx   = i

            if best_idx == -1:
                break

            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            if chosen.id in vecs:
                selected_vecs.append(vecs[chosen.id])

        return selected

    # ── Tier assignment ──────────────────────────────────────────────────────

    @staticmethod
    def assign_tiers(
        db,
        modality: str = "text",
        hot_recall_pct:  float = 0.10,
        warm_recall_pct: float = 0.30,
        write_back: bool = True,
    ) -> dict[int, str]:
        """
        Classify all nodes as 'hot', 'warm', or 'cold' and (optionally)
        write the tier back as an attribute on each node.

        Returns {node_id: tier_str}.
        """
        ids = db.get_all_ids(modality)
        if not ids:
            return {}

        # Score per node: 40% recall_count rank + 60% recency rank
        now = time.time()
        scores: list[tuple[int, float]] = []
        for nid in ids:
            meta = db.get_metadata(nid)
            if meta is None:
                continue
            recency_score = 1.0 / (1.0 + (now - meta.last_recalled_at) / 86400.0) \
                            if meta.last_recalled_at > 0 else 0.0
            scores.append((nid, 0.4 * meta.recall_count + 0.6 * recency_score * 100))

        scores.sort(key=lambda x: x[1], reverse=True)
        n = len(scores)
        hot_n  = max(1, int(n * hot_recall_pct))
        warm_n = max(1, int(n * warm_recall_pct))

        tiers: dict[int, str] = {}
        for rank, (nid, _) in enumerate(scores):
            if rank < hot_n:
                tier = "hot"
            elif rank < hot_n + warm_n:
                tier = "warm"
            else:
                tier = "cold"
            tiers[nid] = tier

        if write_back:
            for nid, tier in tiers.items():
                meta = db.get_metadata(nid)
                if meta is None:
                    continue
                meta.set_attribute("tier", tier)
                db.update_metadata(nid, meta)

        return tiers

    # ── Memory consolidation ─────────────────────────────────────────────────

    @staticmethod
    def consolidate(
        db,
        namespace: str,
        since_hours: float = 24.0,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2,
        llm_fn: Optional[Callable[[str], str]] = None,
        modality: str = "text",
    ) -> list[int]:
        """
        Cluster similar nodes in `namespace` added within `since_hours` and
        merge each cluster into a single summary node.

        Steps:
          1. Fetch matching nodes and their vectors
          2. Greedy union-find clustering by cosine similarity
          3. For each cluster >= min_cluster_size:
             - Generate summary (llm_fn or concat first 200 chars)
             - Create a new FACT node in the same namespace
             - Link each original to new node with 'consolidated_into'
             - Lower original nodes' importance to 0.3

        Returns list of new consolidated node IDs.
        """
        try:
            import feather_db as _fdb
        except ImportError:
            raise ImportError("feather_db must be installed")

        cutoff = int(time.time()) - int(since_hours * 3600)

        # Gather candidate nodes
        all_ids = db.get_all_ids(modality)
        candidates: list[tuple[int, np.ndarray, object]] = []  # (id, vec, meta)

        for nid in all_ids:
            meta = db.get_metadata(nid)
            if meta is None:
                continue
            if meta.namespace_id != namespace:
                continue
            if meta.timestamp < cutoff:
                continue
            if meta.importance <= 0.01:   # already forgotten/consolidated
                continue
            vec = db.get_vector(nid, modality)
            if len(vec) == 0:
                continue
            candidates.append((nid, np.array(vec, dtype=np.float32), meta))

        if len(candidates) < min_cluster_size:
            return []

        # Greedy clustering
        n = len(candidates)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = candidates[i][1], candidates[j][1]
                cos = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-9))
                if cos >= similarity_threshold:
                    union(i, j)

        # Group by cluster root
        clusters: dict[int, list[int]] = {}
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(idx)

        new_ids: list[int] = []

        for root, member_indices in clusters.items():
            if len(member_indices) < min_cluster_size:
                continue

            members = [candidates[i] for i in member_indices]

            # Generate summary
            texts = [m[2].content for m in members]
            if llm_fn is not None:
                summary = llm_fn("\n---\n".join(texts))
            else:
                summary = " | ".join(t[:200] for t in texts)

            # Compute centroid vector
            centroid = np.mean([m[1] for m in members], axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            # Create new consolidated node
            import hashlib
            id_src = f"consolidation:{namespace}:{root}:{int(time.time())}"
            new_id = int.from_bytes(
                hashlib.sha256(id_src.encode()).digest()[:7], "little"
            ) % (2 ** 50)

            new_meta = _fdb.Metadata()
            new_meta.timestamp    = int(time.time())
            new_meta.importance   = max(m[2].importance for m in members)
            new_meta.confidence   = sum(m[2].confidence for m in members) / len(members)
            new_meta.type         = _fdb.ContextType.FACT
            new_meta.source       = "consolidation"
            new_meta.content      = summary
            new_meta.namespace_id = namespace
            new_meta.entity_id    = members[0][2].entity_id
            new_meta.set_attribute("entity_type",    "consolidated_memory")
            new_meta.set_attribute("member_count",   str(len(members)))
            new_meta.set_attribute("consolidation_ts", str(int(time.time())))

            db.add(id=new_id, vec=centroid, meta=new_meta, modality=modality)

            # Link originals → new node + reduce importance
            for nid, _, orig_meta in members:
                db.link(from_id=nid, to_id=new_id,
                        rel_type="consolidated_into", weight=1.0)
                orig_meta.importance = 0.3
                db.update_metadata(nid, orig_meta)

            new_ids.append(new_id)

        return new_ids
