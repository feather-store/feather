"""Dataset loaders. Each returns (vectors_np, ground_truth_or_None, queries_np_or_None)."""
from .synthetic import load_synthetic

__all__ = ["load_synthetic"]
