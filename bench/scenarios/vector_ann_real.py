"""Vector ANN scenario for datasets that ship pre-computed ground truth.

Same shape as vector_ann.py but uses the dataset's GT instead of running
brute force locally. Reports recall@1 / recall@10 / recall@100 across an
optional ef sweep so we capture the latency–quality curve in one run.
"""
from __future__ import annotations
import os
import tempfile
import time
import numpy as np

import feather_db
from ..metrics import latency_stats, recall_at_k, qps


# Vector IDs are 1-indexed when added; ground truth indexes are 0-based
# row positions of the base array. We store id = row_index + 1 and
# subtract 1 when comparing.
_ID_OFFSET = 1


def run(base: np.ndarray, queries: np.ndarray, gt: np.ndarray,
        k: int = 10, ef: int | list[int] | None = None) -> dict:
    """Build the index, optionally sweep ef, return metrics.

    Args:
        base:    (N, D) float32
        queries: (Q, D) float32
        gt:      (Q, K_gt) int64 — exact top-K_gt neighbors per query
        k:       primary recall k. Must satisfy k <= gt.shape[1].
        ef:      None  -> use DB default (50)
                 int   -> set ef once
                 list  -> sweep, returning per-ef stats under metrics["sweep"]
    """
    n, dim = base.shape
    if k > gt.shape[1]:
        raise ValueError(f"k={k} exceeds ground-truth columns {gt.shape[1]}")

    path = tempfile.mktemp(suffix=".feather")
    try:
        db = feather_db.DB.open(path, dim=dim)

        # Build
        t0 = time.perf_counter()
        for i in range(n):
            db.add(id=i + _ID_OFFSET, vec=base[i], meta=feather_db.Metadata())
        build_s = time.perf_counter() - t0

        db.save()
        file_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0

        # Truth as set per query (exclude -1 sentinels from base subsetting)
        truth_sets = [set(int(x) for x in row[:k] if x >= 0) for row in gt]

        def _eval(ef_value: int | None) -> dict:
            if ef_value is not None:
                db.set_ef(ef_value)
            ef_actual = db.get_ef()

            lats, hits = [], []
            t0 = time.perf_counter()
            for q, truth in zip(queries, truth_sets):
                s = time.perf_counter()
                res = db.search(q, k=k)
                lats.append((time.perf_counter() - s) * 1000)
                pred = {r.id - _ID_OFFSET for r in res}
                if truth:
                    hits.append(len(pred & truth) / len(truth))
            wall = time.perf_counter() - t0

            stats = latency_stats(lats)
            stats[f"recall@{k}"] = float(np.mean(hits)) if hits else 0.0
            stats["search_qps"] = qps(len(queries), wall)
            stats["ef"] = ef_actual
            return stats

        if isinstance(ef, list):
            sweep = {f"ef={e}": _eval(e) for e in ef}
            primary = sweep[next(iter(sweep))]  # first ef as the headline
        else:
            primary = _eval(ef)
            sweep = None

        out = dict(primary)
        out.update({
            "n": n,
            "dim": dim,
            "build_seconds": build_s,
            "build_ingest_qps": qps(n, build_s),
            "file_size_mb": file_mb,
            "bytes_per_vec": (file_mb * 1024 * 1024 / n) if n else 0,
        })
        if sweep is not None:
            out["sweep"] = sweep
        return out
    finally:
        if os.path.exists(path):
            os.remove(path)
