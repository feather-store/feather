"""Metric helpers: latency percentiles, recall@k, NDCG."""
from __future__ import annotations
import math
import statistics
from typing import Iterable, Sequence


def latency_stats(samples_ms: Sequence[float]) -> dict:
    if not samples_ms:
        return {"count": 0}
    s = sorted(samples_ms)
    n = len(s)

    def p(q: float) -> float:
        idx = min(n - 1, max(0, int(math.ceil(q * n)) - 1))
        return s[idx]

    return {
        "count": n,
        "mean_ms": statistics.fmean(s),
        "p50_ms": p(0.50),
        "p95_ms": p(0.95),
        "p99_ms": p(0.99),
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def recall_at_k(predicted: Iterable[Sequence[int]], truth: Iterable[Sequence[int]], k: int) -> float:
    """Mean recall@k over a set of queries."""
    hits, total = 0, 0
    for pred, true in zip(predicted, truth):
        pset = set(pred[:k])
        tset = set(true[:k])
        if not tset:
            continue
        hits += len(pset & tset) / len(tset)
        total += 1
    return hits / total if total else 0.0


def ndcg_at_k(predicted: Sequence[int], relevant: dict[int, float], k: int) -> float:
    """NDCG@k. `relevant` maps doc_id -> gain (0 if irrelevant)."""
    def dcg(ids: Sequence[int]) -> float:
        return sum(
            relevant.get(did, 0.0) / math.log2(i + 2)
            for i, did in enumerate(ids[:k])
        )

    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg(predicted) / idcg


def qps(count: int, wall_seconds: float) -> float:
    return count / wall_seconds if wall_seconds > 0 else 0.0
