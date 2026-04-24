"""Synthetic clustered vectors — fast, deterministic, no download."""
from __future__ import annotations
import numpy as np


def load_synthetic(n: int = 10_000, dim: int = 128, n_clusters: int = 64,
                   n_queries: int = 200, seed: int = 42):
    """Generate clustered unit vectors + random queries.

    Returns:
        base: (n, dim) float32 unit vectors
        queries: (n_queries, dim) float32 unit vectors
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_clusters, dim), dtype=np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    assign = rng.integers(0, n_clusters, size=n)
    noise = 0.15 * rng.standard_normal((n, dim), dtype=np.float32)
    base = centroids[assign] + noise
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12

    qassign = rng.integers(0, n_clusters, size=n_queries)
    qnoise = 0.15 * rng.standard_normal((n_queries, dim), dtype=np.float32)
    queries = centroids[qassign] + qnoise
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    return base.astype(np.float32), queries.astype(np.float32)
