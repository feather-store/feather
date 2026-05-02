"""PlanExecutor — runs a QueryPlan against a Feather DB.

Phase 9.1 Week 3 SKELETON. Currently handles hybrid_search and
vector_search step kinds; expand_graph / rerank / hierarchy_walk land
in Phase 11. Unimplemented step kinds are skipped with a logged
warning so partial plans still produce a result.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .planner import QueryPlan


@dataclass
class PlanResult:
    """Output of plan execution.

    results carries SearchResult-like objects (whatever the underlying
    DB returned). plan is preserved so the caller can audit which steps
    produced which evidence; warnings collects step-level issues
    (unsupported step kind, search failure) so a partial plan still
    surfaces useful output.
    """
    results: list = field(default_factory=list)
    plan: Optional[QueryPlan] = None
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    step_traces: list[dict] = field(default_factory=list)


SUPPORTED_KINDS = {"hybrid_search", "vector_search"}


class PlanExecutor:
    """Run a QueryPlan against a feather_db.DB.

    Args:
        db:        feather_db.DB instance.
        embedder:  required for hybrid_search and vector_search steps;
                   anything with .embed(text) -> np.ndarray.
    """

    name = "default_executor"
    version = "0.1.0"

    def __init__(self, db, *, embedder=None):
        self._db = db
        self._embedder = embedder

    def execute(self, plan: QueryPlan, query: str) -> PlanResult:
        t0 = time.perf_counter()
        results: list[Any] = []
        warnings: list[str] = []
        traces: list[dict] = []

        for i, step in enumerate(plan.steps):
            t_step = time.perf_counter()
            n_before = len(results)
            try:
                if step.kind == "hybrid_search":
                    results = self._do_hybrid(step, query)
                elif step.kind == "vector_search":
                    results = self._do_vector(step, query)
                elif step.kind in {"expand_graph", "rerank",
                                   "hierarchy_walk", "attribute_scan"}:
                    warnings.append(
                        f"step {i} ({step.kind}) not yet implemented; "
                        f"skipping"
                    )
                else:
                    warnings.append(
                        f"step {i} unknown kind {step.kind!r}; skipping"
                    )
            except Exception as e:
                warnings.append(
                    f"step {i} ({step.kind}) failed: "
                    f"{type(e).__name__}: {str(e)[:120]}"
                )
            traces.append({
                "step": i,
                "kind": step.kind,
                "results_before": n_before,
                "results_after": len(results),
                "elapsed_seconds": time.perf_counter() - t_step,
            })

        return PlanResult(
            results=results,
            plan=plan,
            elapsed_seconds=time.perf_counter() - t0,
            warnings=warnings,
            step_traces=traces,
        )

    # ── Step handlers ──────────────────────────────────────────────

    def _do_hybrid(self, step, query: str) -> list:
        if self._embedder is None:
            raise RuntimeError("hybrid_search step requires an embedder")
        qvec = self._embedder.embed(query)
        params = step.params or {}
        k = params.get("k", 10)
        modality = params.get("modality", "text")
        flt = params.get("filter")
        try:
            if flt is not None:
                return self._db.hybrid_search(qvec, query, k=k,
                                              modality=modality, filter=flt)
            return self._db.hybrid_search(qvec, query, k=k,
                                          modality=modality)
        except Exception:
            # Fall back to pure vector if hybrid path is unavailable
            if flt is not None:
                return self._db.search(qvec, k=k, modality=modality,
                                       filter=flt)
            return self._db.search(qvec, k=k, modality=modality)

    def _do_vector(self, step, query: str) -> list:
        if self._embedder is None:
            raise RuntimeError("vector_search step requires an embedder")
        qvec = self._embedder.embed(query)
        params = step.params or {}
        k = params.get("k", 10)
        modality = params.get("modality", "text")
        flt = params.get("filter")
        if flt is not None:
            return self._db.search(qvec, k=k, modality=modality, filter=flt)
        return self._db.search(qvec, k=k, modality=modality)
