"""Pluggable embedders for benchmark scenarios.

Phase 1 ships only the deterministic hash embedder (free, reproducible,
not semantically meaningful — used to validate pipeline plumbing).
Phase 2 will add OpenAI, Gemini, and Voyage real embedders behind the
same interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import hashlib
import numpy as np


class Embedder(ABC):
    dim: int
    name: str

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return float32 (dim,) unit vector."""

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts], axis=0)


class DeterministicEmbedder(Embedder):
    """Hash-based 'embedder'. Same text -> same vector; semantically random.

    Used for Phase 1 plumbing validation. With this embedder, retrieval
    quality is at chance level — that is the EXPECTED behavior, and seeing
    chance-level results is how we confirm the pipeline is wired correctly
    before paying for real embedders.
    """

    def __init__(self, dim: int = 768, seed_salt: str = "feather"):
        self.dim = dim
        self.name = f"deterministic_d{dim}"
        self._salt = seed_salt

    def embed(self, text: str) -> np.ndarray:
        # Use SHA-256 of (salt + text) as 32 bytes of seed material.
        digest = hashlib.sha256((self._salt + text).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def get_embedder(name: str, **kwargs) -> Embedder:
    """Factory. Phase 1: deterministic only. Phase 2 adds openai/gemini."""
    if name == "deterministic":
        return DeterministicEmbedder(**kwargs)
    if name == "openai":
        from .embedders_openai import OpenAIEmbedder  # lazy import
        return OpenAIEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name}")
