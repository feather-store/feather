"""Schema-aware hierarchy helpers for vertical-aware reasoning.

Phase 9.1 Week 3. The default hierarchy encodes the canonical Marketing
attribution chain:

    Brand → Channel → Campaign → AdSet → Ad → Creative

Used by the Cloud reasoner to:
- Walk up: "why did this creative perform?" → enclosing ad/adset/campaign
- Walk down: "show me all creatives under the Q2 Nike campaign"
- Aggregate KPIs at the right level (campaign-level CTR vs ad-level CTR)

OSS ships the in-memory implementation; Cloud will back it with the
entity graph already in Feather, but the public API stays the same.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional


# Default Marketing hierarchy. Cloud verticals can extend or replace via
# Hierarchy(levels=[...]) during initialisation.
MARKETING_HIERARCHY = [
    "Brand",
    "Channel",
    "Campaign",
    "AdSet",
    "Ad",
    "Creative",
]


@dataclass
class HierarchyNode:
    """One node in a hierarchy. canonical_id is stable; parent_id refers
    to another node's canonical_id (or None for roots)."""
    kind: str
    canonical_id: str
    name: str = ""
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Hierarchy:
    """In-memory parent-child index over canonical entities.

    Single-tenant in OSS. Cloud will swap for a DB-backed implementation
    keyed by namespace_id; the API stays identical so the IngestPipeline
    + reasoner remain ignorant of storage backend.
    """

    def __init__(self, levels: Optional[list[str]] = None):
        self.levels = levels or list(MARKETING_HIERARCHY)
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"hierarchy levels must be unique: {self.levels}")
        self._level_idx = {k: i for i, k in enumerate(self.levels)}
        self._nodes: dict[str, HierarchyNode] = {}
        self._children: dict[str, list[str]] = {}

    # ── Mutation ───────────────────────────────────────────────────

    def add(self, node: HierarchyNode) -> None:
        """Add a node. Raises ValueError if its kind is not a valid
        level for this hierarchy."""
        if node.kind not in self._level_idx:
            raise ValueError(
                f"unknown level: {node.kind!r}; hierarchy = {self.levels}"
            )
        if (node.parent_id and node.parent_id in self._nodes):
            parent = self._nodes[node.parent_id]
            if self._level_idx[parent.kind] >= self._level_idx[node.kind]:
                raise ValueError(
                    f"parent {parent.canonical_id} ({parent.kind}) is not "
                    f"strictly above {node.canonical_id} ({node.kind}) in "
                    f"the hierarchy"
                )
        self._nodes[node.canonical_id] = node
        if node.parent_id:
            self._children.setdefault(node.parent_id, []).append(
                node.canonical_id
            )

    def add_many(self, nodes: Iterable[HierarchyNode]) -> None:
        for n in nodes:
            self.add(n)

    # ── Lookups ────────────────────────────────────────────────────

    def get(self, canonical_id: str) -> Optional[HierarchyNode]:
        return self._nodes.get(canonical_id)

    def parent(self, canonical_id: str) -> Optional[HierarchyNode]:
        node = self._nodes.get(canonical_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def ancestors(self, canonical_id: str) -> list[HierarchyNode]:
        """Walk from this node UP to the root. Excludes the node itself.
        Returns [] if the node is unknown or already a root."""
        out: list[HierarchyNode] = []
        cur = self.parent(canonical_id)
        seen = {canonical_id}
        while cur is not None and cur.canonical_id not in seen:
            out.append(cur)
            seen.add(cur.canonical_id)
            cur = self.parent(cur.canonical_id)
        return out

    def children(self, canonical_id: str) -> list[HierarchyNode]:
        return [self._nodes[c]
                for c in self._children.get(canonical_id, [])
                if c in self._nodes]

    def descendants(self, canonical_id: str) -> list[HierarchyNode]:
        """All transitive descendants in BFS order."""
        out: list[HierarchyNode] = []
        queue = list(self._children.get(canonical_id, []))
        seen = {canonical_id}
        while queue:
            cid = queue.pop(0)
            if cid in seen:
                continue
            seen.add(cid)
            node = self._nodes.get(cid)
            if not node:
                continue
            out.append(node)
            queue.extend(self._children.get(cid, []))
        return out

    def level_of(self, canonical_id: str) -> Optional[str]:
        node = self._nodes.get(canonical_id)
        return node.kind if node else None

    def level_index(self, kind: str) -> Optional[int]:
        return self._level_idx.get(kind)

    def is_descendant_of(self, descendant_id: str,
                         ancestor_id: str) -> bool:
        return any(n.canonical_id == ancestor_id
                   for n in self.ancestors(descendant_id))

    def common_ancestor(self, a_id: str, b_id: str
                        ) -> Optional[HierarchyNode]:
        """Lowest common ancestor of two nodes (or None if disjoint)."""
        a_chain = [a_id] + [n.canonical_id for n in self.ancestors(a_id)]
        b_chain_set = {b_id} | {n.canonical_id for n in self.ancestors(b_id)}
        for cid in a_chain:
            if cid in b_chain_set:
                return self._nodes.get(cid)
        return None

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, canonical_id: str) -> bool:
        return canonical_id in self._nodes
