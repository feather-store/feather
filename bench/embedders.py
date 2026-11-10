"""Pluggable embedders for benchmark scenarios.

Phase 1 ships only the deterministic hash embedder (free, reproducible,
not semantically meaningful — used to validate pipeline plumbing).
Phase 2 will add OpenAI, Gemini, and Voyage real embedders behind the
same interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import hashlib
import os
import threading
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


class CachedEmbedder(Embedder):
    """A paid embedder behind a disk cache of whole batches.

    A LongMemEval run embeds every haystack again (~$1.9 per 100
    questions with gemini-embedding-001), though the texts never change
    between runs that differ only in k, budget or reader. A batch is keyed
    by the embedder's name and the exact texts, so a changed text, query
    rewrite or model is a miss, never a stale vector. Single embeds
    (follow-up queries an LLM wrote) pass through: they differ per run.
    """

    def __init__(self, inner: Embedder, cache_dir: str):
        self.inner, self.dim, self.name = inner, inner.dim, inner.name
        self._dir = os.path.join(cache_dir, inner.name)
        os.makedirs(self._dir, exist_ok=True)
        self.hits = self.misses = 0

    def embed(self, text: str) -> np.ndarray:
        return self.inner.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        h = hashlib.sha256(self.name.encode())
        for t in texts:
            h.update(hashlib.sha256(t.encode("utf-8")).digest())
        path = os.path.join(self._dir, h.hexdigest()[:32] + ".npy")
        try:
            vecs = np.load(path)
            if vecs.shape == (len(texts), self.dim):
                self.hits += 1
                return vecs
        except (OSError, ValueError):
            pass
        vecs = np.asarray(self.inner.embed_batch(texts), dtype=np.float32)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.npy"
        np.save(tmp, vecs)
        os.replace(tmp, path)            # a reader never sees half a file
        self.misses += 1
        return vecs


def get_embedder(name: str, **kwargs) -> Embedder:
    """Factory. Phase 1: deterministic only. Phase 2 adds openai/gemini."""
    if name == "deterministic":
        return DeterministicEmbedder(**kwargs)
    if name == "openai":
        from .embedders_openai import OpenAIEmbedder  # lazy import
        return OpenAIEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name}")
