"""
EpisodeManager — Feather DB v0.6.0
====================================
Episodic memory: group ordered sequences of nodes into named episodes.

An episode is a first-class citizen stored in the DB itself:
  - A header node (entity_id="__episode__") represents the episode
  - Member nodes are linked via 'episode_contains' edges
  - Nodes carry an 'episode_id' attribute for fast reverse-lookup
  - Episodes can be open (accepting new members) or closed (sealed)

Usage
-----
    from feather_db.episodes import EpisodeManager

    em = EpisodeManager()

    # Begin a new episode
    hid = em.begin_episode(db, "q1_budget_campaign",
                            "Q1 campaign around Union Budget 2026",
                            embed_fn=embed)

    # Add nodes to the episode
    em.add_to_episode(db, node_id=1001, episode_id="q1_budget_campaign")
    em.add_to_episode(db, node_id=9001, episode_id="q1_budget_campaign")

    # Retrieve ordered nodes
    nodes = em.get_episode(db, "q1_budget_campaign")

    # Seal the episode
    em.close_episode(db, "q1_budget_campaign")
"""

from __future__ import annotations

import hashlib
import time
from typing import Callable, Optional

import numpy as np


def _episode_header_id(episode_id: str) -> int:
    """Deterministic uint64 from episode_id string. Stable across processes."""
    digest = hashlib.sha256(f"__ep__{episode_id}".encode()).digest()[:8]
    return int.from_bytes(digest, "little") % (2 ** 50)


class EpisodeManager:
    """
    Stateless episodic memory manager.

    All episode state lives in the Feather DB — no in-process state is held.
    """

    # ── Create ───────────────────────────────────────────────────────────────

    def begin_episode(
        self,
        db,
        episode_id: str,
        description: str,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        namespace: str = "episodes",
        importance: float = 0.9,
    ) -> int:
        """
        Create an episode header node. Returns the header node ID.

        The header node:
          - entity_id = "__episode__"
          - attribute episode_id = episode_id
          - attribute episode_status = "open"
        """
        try:
            import feather_db as _fdb
        except ImportError:
            raise ImportError("feather_db must be installed")

        hid = _episode_header_id(episode_id)

        # Check if already exists
        existing = db.get_metadata(hid)
        if existing is not None:
            return hid

        if embed_fn is not None:
            vec = np.array(embed_fn(description), dtype=np.float32)
        else:
            vec = np.zeros(db.dim("text"), dtype=np.float32)

        meta = _fdb.Metadata()
        meta.timestamp    = int(time.time())
        meta.importance   = importance
        meta.type         = _fdb.ContextType.EVENT
        meta.source       = "episode_manager"
        meta.content      = description
        meta.namespace_id = namespace
        meta.entity_id    = "__episode__"
        meta.set_attribute("episode_id",     episode_id)
        meta.set_attribute("episode_status", "open")
        meta.set_attribute("episode_ts",     str(int(time.time())))

        db.add(id=hid, vec=vec, meta=meta, modality="text")
        return hid

    # ── Add member ───────────────────────────────────────────────────────────

    def add_to_episode(
        self,
        db,
        node_id: int,
        episode_id: str,
        weight: float = 1.0,
    ) -> None:
        """
        Tag `node_id` as a member of `episode_id` and link from the header.
        """
        meta = db.get_metadata(node_id)
        if meta is None:
            raise ValueError(f"Node {node_id} not found in DB")

        meta.set_attribute("episode_id", episode_id)
        db.update_metadata(node_id, meta)

        hid = _episode_header_id(episode_id)
        db.link(from_id=hid, to_id=node_id,
                rel_type="episode_contains", weight=weight)

    # ── Retrieve ────────────────────────────────────────────────────────────

    def get_episode(
        self,
        db,
        episode_id: str,
    ) -> list[dict]:
        """
        Return all member nodes in chronological order.

        Each item: {id, timestamp, content, entity_type, importance, metadata}
        """
        hid   = _episode_header_id(episode_id)
        edges = db.get_edges(hid)

        members: list[dict] = []
        for e in edges:
            if e.rel_type != "episode_contains":
                continue
            meta = db.get_metadata(e.target_id)
            if meta is None:
                continue
            members.append({
                "id":          e.target_id,
                "timestamp":   meta.timestamp,
                "content":     meta.content,
                "entity_type": meta.get_attribute("entity_type"),
                "importance":  round(meta.importance, 3),
                "metadata":    meta,
                "edge_weight": round(e.weight, 3),
            })

        members.sort(key=lambda x: x["timestamp"])
        return members

    # ── Close ────────────────────────────────────────────────────────────────

    def close_episode(
        self,
        db,
        episode_id: str,
    ) -> dict:
        """
        Seal the episode:
          - Mark episode_status = "closed"
          - Add 'episode_end' edge from last node back to header
          - Returns episode summary dict
        """
        hid    = _episode_header_id(episode_id)
        header = db.get_metadata(hid)
        if header is None:
            raise ValueError(f"Episode '{episode_id}' not found")

        members = self.get_episode(db, episode_id)
        if members:
            last_id = members[-1]["id"]
            db.link(from_id=last_id, to_id=hid,
                    rel_type="episode_end", weight=1.0)

        header.set_attribute("episode_status", "closed")
        header.set_attribute("episode_end_ts", str(int(time.time())))
        header.set_attribute("member_count",   str(len(members)))
        db.update_metadata(hid, header)

        return {
            "episode_id":   episode_id,
            "header_id":    hid,
            "member_count": len(members),
            "status":       "closed",
            "first_ts":     members[0]["timestamp"] if members else None,
            "last_ts":      members[-1]["timestamp"] if members else None,
        }

    # ── List episodes ────────────────────────────────────────────────────────

    def list_episodes(self, db, namespace: str = "episodes") -> list[dict]:
        """
        Return all episodes in the DB (header nodes by namespace).
        """
        ids = db.get_all_ids("text")
        episodes: list[dict] = []

        for nid in ids:
            meta = db.get_metadata(nid)
            if meta is None:
                continue
            if meta.entity_id != "__episode__":
                continue
            if namespace and meta.namespace_id != namespace:
                continue
            episodes.append({
                "episode_id":  meta.get_attribute("episode_id"),
                "header_id":   nid,
                "description": meta.content,
                "status":      meta.get_attribute("episode_status"),
                "timestamp":   meta.timestamp,
                "member_count": meta.get_attribute("member_count") or "?",
            })

        episodes.sort(key=lambda x: x["timestamp"], reverse=True)
        return episodes
