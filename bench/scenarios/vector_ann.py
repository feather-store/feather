"""Vector ANN scenario: build + search latency + recall@k vs brute-force ground truth."""
from __future__ import annotations
import os
import tempfile
import time
import numpy as np

import feather_db
from ..metrics import latency_stats, recall_at_k, qps


def _brute_force_topk(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact L2 nearest neighbors. Returns (n_queries, k) index array."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        d = np.sum((base - q) ** 2, axis=1)
        out[i] = np.argpartition(d, k)[:k][np.argsort(d[np.argpartition(d, k)[:k]])]
    return out


def run(base: np.ndarray, queries: np.ndarray, k: int = 10, ef: int | None = None):
    """Execute the scenario. Returns a metrics dict the runner persists."""
    n, dim = base.shape
    path = tempfile.mktemp(suffix=".feather")
    try:
        db = feather_db.DB.open(path, dim=dim)

        # Build
        t0 = time.perf_counter()
        for i in range(n):
            m = feather_db.Metadata()
            m.content = f"doc_{i}"
            db.add(id=i + 1, vec=base[i], meta=m)
        build_s = time.perf_counter() - t0

        # Apply tuning
        if ef is not None:
            db.set_ef(ef)
        ef_actual = db.get_ef()

        # Save so we can measure on-disk footprint
        db.save()
        file_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0

        # Search + latency
        search_ms: list[float] = []
        preds: list[list[int]] = []
        t0 = time.perf_counter()
        for q in queries:
            s = time.perf_counter()
            res = db.search(q, k=k)
            search_ms.append((time.perf_counter() - s) * 1000)
            preds.append([r.id - 1 for r in res])  # ids are 1-indexed
        search_wall = time.perf_counter() - t0

        # Ground truth via brute force
        truth = _brute_force_topk(base, queries, k)
        recall = recall_at_k(preds, truth.tolist(), k=k)

        stats = latency_stats(search_ms)
        stats.update({
            "build_seconds": build_s,
            "build_ingest_qps": qps(n, build_s),
            "search_qps": qps(len(queries), search_wall),
            f"recall@{k}": recall,
            "file_size_mb": file_mb,
            "bytes_per_vec": (file_mb * 1024 * 1024 / n) if n else 0,
            "ef": ef_actual,
        })
        return stats
    finally:
        if os.path.exists(path):
            os.remove(path)
