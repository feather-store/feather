"""QueryPlanner — turns a natural-language query into a retrieval plan.

Phase 9.1 Week 3 SKELETON. The default planner emits a single-step plan
(plain hybrid_search top-K). Subclasses (or LLM-backed planners shipped
in Phase 11) can produce multi-step plans with attribute filters,
hierarchy walks, reranking, and graph expansion.

Why a planner instead of just calling db.search? Three reasons:
1. **Composable retrieval** — at scale, queries like "compare CTR
   across Q2 and Q3 campaigns" need attribute filtering + hierarchy
   walking, not just top-K.
2. **Cost control** — the planner can choose to stop at vector search
   if the query is simple, or invoke reranking only for hard queries.
3. **Auditability** — every PlanResult carries the plan that produced
   it, so users can inspect why a particular fact surfaced.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlanStep:
    """One step in a retrieval plan.

    kind:
        'hybrid_search'  — vector + BM25 top-K with optional filter
        'vector_search'  — pure vector top-K
        'attribute_scan' — filter-only (no embedding)
        'expand_graph'   — N-hop walk from current results
        'rerank'         — apply a reranker over current results
        'hierarchy_walk' — walk up/down the entity hierarchy
    """
    kind: str
    params: dict = field(default_factory=dict)
    rationale: str = ""


@dataclass
class QueryPlan:
    """An ordered list of steps. Earlier steps populate the working
    set; later steps refine."""
    steps: list[PlanStep] = field(default_factory=list)
    rationale: str = ""

    def __len__(self) -> int:
        return len(self.steps)


class QueryPlanner:
    """Default planner: single-step hybrid_search.

    Args:
        db:         feather_db.DB instance (planner may inspect schema /
                    namespaces but does not execute searches itself).
        provider:   optional LLMProvider for plan generation. None →
                    default heuristic planner.
        default_k:  top-K used when the caller does not specify.
    """

    name = "default_planner"
    version = "0.1.0"

    def __init__(self, db, *, provider=None, default_k: int = 10):
        self._db = db
        self._provider = provider
        self._default_k = default_k

    def plan(self, query: str, *,
             context: Optional[dict] = None) -> QueryPlan:
        """Produce a retrieval plan for `query`.

        Default behavior: one hybrid_search step with default_k.
        Override in subclasses to emit multi-step plans.
        """
        ctx = context or {}
        k = ctx.get("k", self._default_k)
        modality = ctx.get("modality", "text")
        params = {"k": k, "modality": modality}
        if ctx.get("filter") is not None:
            params["filter"] = ctx["filter"]
        return QueryPlan(
            steps=[PlanStep(
                kind="hybrid_search",
                params=params,
                rationale="default single-step hybrid retrieval",
            )],
            rationale="default planner: single-step hybrid_search",
        )
