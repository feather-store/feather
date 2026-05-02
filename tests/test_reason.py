"""Tests for the feather_db.reason skeleton."""
from __future__ import annotations

from feather_db.reason import (
    QueryPlanner, QueryPlan, PlanStep,
    PlanExecutor, PlanResult, SUPPORTED_KINDS,
)


class FakeDB:
    """Stand-in for feather_db.DB capturing search invocations."""
    def __init__(self):
        self.search_calls: list[dict] = []
        self.hybrid_calls: list[dict] = []

    def search(self, qvec, k=10, modality="text", filter=None):
        self.search_calls.append({"k": k, "modality": modality,
                                  "filter": filter})
        return [{"id": i, "score": 1.0 / (1 + i)} for i in range(k)]

    def hybrid_search(self, qvec, query, k=10, modality="text",
                      filter=None):
        self.hybrid_calls.append({"q": query, "k": k,
                                  "modality": modality, "filter": filter})
        return [{"id": i, "score": 1.0 / (1 + i)} for i in range(k)]


class FakeEmbedder:
    def __init__(self, dim=8):
        self.dim = dim
        self.calls: list[str] = []
    def embed(self, text):
        self.calls.append(text)
        return [0.1] * self.dim


# ── QueryPlanner ─────────────────────────────────────────────────────

def test_default_plan_is_single_hybrid_step():
    db = FakeDB()
    planner = QueryPlanner(db)
    plan = planner.plan("when did the Q2 campaign start?")
    assert isinstance(plan, QueryPlan)
    assert len(plan) == 1
    assert plan.steps[0].kind == "hybrid_search"
    assert plan.steps[0].params["k"] == 10


def test_planner_passes_through_context():
    planner = QueryPlanner(FakeDB(), default_k=5)
    plan = planner.plan("x", context={"k": 25, "modality": "visual"})
    assert plan.steps[0].params["k"] == 25
    assert plan.steps[0].params["modality"] == "visual"


def test_planner_default_k_used_when_not_in_context():
    planner = QueryPlanner(FakeDB(), default_k=7)
    plan = planner.plan("x")
    assert plan.steps[0].params["k"] == 7


# ── PlanExecutor ─────────────────────────────────────────────────────

def test_executor_runs_hybrid_search_step():
    db = FakeDB()
    emb = FakeEmbedder()
    executor = PlanExecutor(db, embedder=emb)
    plan = QueryPlan(steps=[PlanStep(kind="hybrid_search",
                                      params={"k": 5})])
    result = executor.execute(plan, "test query")
    assert isinstance(result, PlanResult)
    assert len(result.results) == 5
    assert len(db.hybrid_calls) == 1
    assert db.hybrid_calls[0]["q"] == "test query"
    assert emb.calls == ["test query"]


def test_executor_runs_vector_search_step():
    db = FakeDB()
    emb = FakeEmbedder()
    executor = PlanExecutor(db, embedder=emb)
    plan = QueryPlan(steps=[PlanStep(kind="vector_search",
                                      params={"k": 3})])
    result = executor.execute(plan, "x")
    assert len(result.results) == 3
    assert len(db.search_calls) == 1
    assert len(db.hybrid_calls) == 0


def test_executor_warns_on_unimplemented_step():
    executor = PlanExecutor(FakeDB(), embedder=FakeEmbedder())
    plan = QueryPlan(steps=[
        PlanStep(kind="hybrid_search", params={"k": 1}),
        PlanStep(kind="rerank", params={}),
        PlanStep(kind="expand_graph", params={}),
    ])
    result = executor.execute(plan, "x")
    assert any("rerank" in w for w in result.warnings)
    assert any("expand_graph" in w for w in result.warnings)
    # First step still produced results
    assert len(result.results) == 1


def test_executor_warns_on_unknown_kind():
    executor = PlanExecutor(FakeDB(), embedder=FakeEmbedder())
    plan = QueryPlan(steps=[PlanStep(kind="quantum_dance")])
    result = executor.execute(plan, "x")
    assert any("quantum_dance" in w for w in result.warnings)


def test_executor_step_traces_populated():
    executor = PlanExecutor(FakeDB(), embedder=FakeEmbedder())
    plan = QueryPlan(steps=[
        PlanStep(kind="hybrid_search", params={"k": 4}),
        PlanStep(kind="rerank"),
    ])
    result = executor.execute(plan, "x")
    assert len(result.step_traces) == 2
    assert result.step_traces[0]["kind"] == "hybrid_search"
    assert result.step_traces[0]["results_after"] == 4
    assert result.step_traces[1]["kind"] == "rerank"


def test_executor_hybrid_falls_back_to_vector_on_error():
    """When db.hybrid_search raises, executor falls back to db.search."""
    class BrokenHybridDB(FakeDB):
        def hybrid_search(self, *a, **k):
            raise RuntimeError("BM25 unavailable")
    db = BrokenHybridDB()
    executor = PlanExecutor(db, embedder=FakeEmbedder())
    plan = QueryPlan(steps=[PlanStep(kind="hybrid_search",
                                      params={"k": 2})])
    result = executor.execute(plan, "x")
    assert len(result.results) == 2  # via search() fallback
    assert len(db.search_calls) == 1


def test_executor_no_embedder_raises_on_search_step():
    executor = PlanExecutor(FakeDB(), embedder=None)
    plan = QueryPlan(steps=[PlanStep(kind="vector_search")])
    result = executor.execute(plan, "x")
    # Step failure is captured as a warning, not raised
    assert any("requires an embedder" in w for w in result.warnings)


def test_supported_kinds_is_a_set():
    assert "hybrid_search" in SUPPORTED_KINDS
    assert "vector_search" in SUPPORTED_KINDS
