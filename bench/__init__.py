"""Feather DB benchmark harness.

Runs reproducible benchmarks across scenarios (vector ANN, BM25 keyword,
hybrid RRF, memory benchmarks) and emits JSON + Markdown reports under
bench/results/ and bench/reports/.

Usage:
    python -m bench run vector_ann --dataset synthetic --n 10000 --dim 128
    python -m bench run hybrid     --dataset beir_scifact
    python -m bench report                 # regenerate latest.md
"""
from .runner import BenchRunner, BenchResult
from .metrics import latency_stats, recall_at_k

__all__ = ["BenchRunner", "BenchResult", "latency_stats", "recall_at_k"]
