"""Stage 5 of the Phase 9 pipeline — query-time reasoner.

A QueryPlanner turns a natural-language query into a QueryPlan; a
PlanExecutor runs the plan against a Feather DB and returns a
PlanResult that carries both the evidence and the plan that produced
it.

Skeleton in Phase 9.1 Week 3. Full implementation (LLM-backed planner,
multi-step execution, reranking, hierarchy walks) lands in Phase 11.
"""
from .planner import QueryPlanner, QueryPlan, PlanStep
from .executor import PlanExecutor, PlanResult, SUPPORTED_KINDS

__all__ = [
    "QueryPlanner",
    "QueryPlan",
    "PlanStep",
    "PlanExecutor",
    "PlanResult",
    "SUPPORTED_KINDS",
]
