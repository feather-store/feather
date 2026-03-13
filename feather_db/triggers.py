"""
Triggers & Contradiction Detector — Feather DB v0.6.0
=======================================================
Proactive memory: fire callbacks when new nodes match registered patterns,
and auto-detect contradictions when a new node is semantically opposed to
an existing high-importance node.

Usage
-----
    from feather_db.triggers import WatchManager, ContradictionDetector

    wm = WatchManager()
    wm.watch(db,
             query_text="competitor launched new FD product",
             threshold=0.75,
             callback=lambda nid, sim: print(f"⚡ Match: node {nid} (sim={sim:.3f})"),
             embed_fn=my_embed)

    # After every db.add(), fire matching triggers:
    wm.check_triggers(db, new_node_id=42, embed_fn=my_embed)

    # Contradiction detector (standalone):
    cd = ContradictionDetector()
    conflicts = cd.check(db, new_node_id=42, embed_fn=my_embed)
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np


class WatchManager:
    """
    Registry of semantic watches over a Feather DB.

    A watch fires when a newly added node has cosine-similarity ≥ threshold
    to the watch's query vector.
    """

    def __init__(self) -> None:
        self._watches: list[dict] = []

    # ── Register ─────────────────────────────────────────────────────────────

    def watch(
        self,
        db,
        query_text: str,
        threshold: float,
        callback: Callable[[int, float], None],
        embed_fn: Callable[[str], np.ndarray],
        watch_id: Optional[str] = None,
    ) -> str:
        """
        Register a semantic watch.

        Parameters
        ----------
        db          Feather DB instance (used to compute the initial query vec)
        query_text  Natural language description of what to watch for
        threshold   Cosine similarity threshold [0-1] to fire the callback
        callback    Function(node_id: int, similarity: float) called on match
        embed_fn    Text → np.ndarray embedder (same as used for ingestion)
        watch_id    Optional stable ID for this watch (auto-generated if None)

        Returns the watch_id.
        """
        vec = embed_fn(query_text)
        wid = watch_id or f"watch_{len(self._watches)}_{int(time.time())}"
        self._watches.append({
            "id":         wid,
            "query_text": query_text,
            "query_vec":  np.array(vec, dtype=np.float32),
            "threshold":  threshold,
            "callback":   callback,
        })
        return wid

    def remove_watch(self, watch_id: str) -> bool:
        """Remove a registered watch by ID. Returns True if found."""
        before = len(self._watches)
        self._watches = [w for w in self._watches if w["id"] != watch_id]
        return len(self._watches) < before

    def list_watches(self) -> list[dict]:
        """Return registered watch metadata (without vectors)."""
        return [{"id": w["id"], "query_text": w["query_text"],
                 "threshold": w["threshold"]} for w in self._watches]

    # ── Check ────────────────────────────────────────────────────────────────

    def check_triggers(
        self,
        db,
        new_node_id: int,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        new_vec: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """
        Check all registered watches against the newly added node.
        Fires matching callbacks and returns a list of match records.

        Provide either `embed_fn` (used to re-embed the node's content)
        or `new_vec` directly (preferred — avoids an extra API call).
        """
        if not self._watches:
            return []

        # Get the vector for the new node
        if new_vec is not None:
            vec = np.array(new_vec, dtype=np.float32)
        else:
            raw = db.get_vector(new_node_id, "text")
            if len(raw) == 0:
                return []
            vec = np.array(raw, dtype=np.float32)

        norm_vec = np.linalg.norm(vec)

        matches: list[dict] = []
        for watch in self._watches:
            qv   = watch["query_vec"]
            norm_q = np.linalg.norm(qv)
            if norm_vec < 1e-9 or norm_q < 1e-9:
                continue
            sim = float(np.dot(vec, qv) / (norm_vec * norm_q))
            if sim >= watch["threshold"]:
                try:
                    watch["callback"](new_node_id, sim)
                except Exception:
                    pass
                matches.append({
                    "watch_id":   watch["id"],
                    "node_id":    new_node_id,
                    "similarity": round(sim, 4),
                    "query_text": watch["query_text"],
                })

        return matches


class ContradictionDetector:
    """
    Detects when a newly added node semantically contradicts existing nodes.

    Detection heuristic:
      - Find top-k similar nodes to the new node
      - If similarity > threshold AND sources differ → flag as potential contradiction
      - Auto-link with 'contradicts' edge (bidirectional)
      - Returns list of contradicting node IDs found
    """

    def check(
        self,
        db,
        new_node_id: int,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        new_vec: Optional[np.ndarray] = None,
        threshold: float = 0.90,
        top_k: int = 5,
        auto_link: bool = True,
    ) -> list[int]:
        """
        Check for contradictions against the newly added node.

        Returns list of existing node IDs that may contradict the new node.
        If auto_link=True, creates 'contradicts' edges for each pair found.

        Note: similarity threshold of 0.90 means the nodes are very similar
        in topic/meaning. Whether they actually contradict requires domain
        logic — this class handles the structural detection.
        """
        # Get new node vector and metadata
        if new_vec is not None:
            query_vec = np.array(new_vec, dtype=np.float32)
        else:
            raw = db.get_vector(new_node_id, "text")
            if len(raw) == 0:
                return []
            query_vec = np.array(raw, dtype=np.float32)

        new_meta = db.get_metadata(new_node_id)
        if new_meta is None:
            return []

        # Search for highly similar nodes
        results = db.search(query_vec, k=top_k + 1)

        contradictions: list[int] = []
        for r in results:
            if r.id == new_node_id:
                continue
            # Only flag if above threshold
            if r.score < threshold:
                continue
            # Source must differ (same source updating its own record isn't a contradiction)
            if r.metadata.source == new_meta.source:
                continue
            # Skip if already linked with contradicts
            existing_rels = {e.rel_type for e in new_meta.edges if e.target_id == r.id}
            if "contradicts" in existing_rels:
                continue

            contradictions.append(r.id)

            if auto_link:
                db.link(from_id=new_node_id, to_id=r.id,
                        rel_type="contradicts", weight=round(r.score, 3))

        return contradictions

    def scan_all(
        self,
        db,
        modality: str = "text",
        threshold: float = 0.90,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Scan the entire DB for potential contradictions (expensive — use sparingly).
        Returns list of {node_a, node_b, similarity} dicts.
        """
        ids    = db.get_all_ids(modality)
        found: list[dict] = []
        seen:  set[tuple[int, int]] = set()

        for nid in ids:
            raw = db.get_vector(nid, modality)
            if len(raw) == 0:
                continue
            q    = np.array(raw, dtype=np.float32)
            meta = db.get_metadata(nid)
            if meta is None:
                continue

            results = db.search(q, k=top_k + 1, modality=modality)
            for r in results:
                if r.id == nid:
                    continue
                if r.score < threshold:
                    continue
                pair = (min(nid, r.id), max(nid, r.id))
                if pair in seen:
                    continue
                if r.metadata.source == meta.source:
                    continue
                seen.add(pair)
                found.append({
                    "node_a":     nid,
                    "node_b":     r.id,
                    "similarity": round(r.score, 4),
                    "source_a":   meta.source,
                    "source_b":   r.metadata.source,
                })

        return found
