"""Correctness: HNSW recall@k vs brute-force ground truth.

These tests pin the recall floor we promise at the default ef. If the
floor regresses (e.g. a binding change silently changes the default ef,
or the distance function breaks), CI fails.

Baselines measured on 2026-04-25 against v0.8.0 (ef default = 50):
    n=1k,  dim=128  -> recall@10 ≈ 0.99
    n=5k,  dim=128  -> recall@10 ≈ 0.995
    n=10k, dim=128  -> recall@10 ≈ 0.99

Floors are set below measured values to avoid seed flakiness.
"""
from __future__ import annotations
import numpy as np
import pytest

import feather_db


def _clustered(n: int, dim: int, n_clusters: int = 64, seed: int = 42):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n_clusters, dim), dtype=np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
    assign = rng.integers(0, n_clusters, size=n)
    noise = 0.15 * rng.standard_normal((n, dim), dtype=np.float32)
    base = c[assign] + noise
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
    return base.astype(np.float32), c


def _brute_force_topk(base: np.ndarray, q: np.ndarray, k: int) -> list[int]:
    d = np.sum((base - q) ** 2, axis=1)
    idx = np.argpartition(d, k)[:k]
    return sorted(idx.tolist(), key=lambda i: d[i])


def _build_db(tmp_path_feather: str, base: np.ndarray) -> feather_db.DB:
    db = feather_db.DB.open(tmp_path_feather, dim=base.shape[1])
    for i, v in enumerate(base):
        m = feather_db.Metadata()
        m.content = f"doc_{i}"
        db.add(id=i + 1, vec=v, meta=m)
    return db


def _recall_at_k(db: feather_db.DB, base: np.ndarray,
                 queries: np.ndarray, k: int) -> float:
    total = 0.0
    for q in queries:
        pred = {r.id - 1 for r in db.search(q, k=k)}
        truth = set(_brute_force_topk(base, q, k))
        total += len(pred & truth) / len(truth)
    return total / len(queries)


@pytest.mark.parametrize("n,dim,floor", [
    (1_000,  128, 0.95),
    (5_000,  128, 0.95),
])
def test_recall_floor(tmp_path_feather, n, dim, floor):
    base, centroids = _clustered(n, dim)
    # 40 random queries from the cluster distribution
    rng = np.random.default_rng(7)
    qassign = rng.integers(0, centroids.shape[0], size=40)
    qnoise = 0.15 * rng.standard_normal((40, dim), dtype=np.float32)
    queries = centroids[qassign] + qnoise
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    db = _build_db(tmp_path_feather, base)
    r = _recall_at_k(db, base, queries, k=10)
    assert r >= floor, f"recall@10={r:.3f} below floor {floor}"


def test_search_returns_ids_not_indices(tmp_path_feather):
    """Regression: db.search must return the ID the caller added,
    not the internal row index."""
    base, _ = _clustered(50, 64)
    db = feather_db.DB.open(tmp_path_feather, dim=64)
    custom_ids = [1000 + i * 7 for i in range(50)]
    for cid, v in zip(custom_ids, base):
        db.add(id=cid, vec=v, meta=feather_db.Metadata())

    res = db.search(base[0], k=5)
    ids = {r.id for r in res}
    assert custom_ids[0] in ids
    assert all(i in custom_ids for i in ids)
