"""
Feather DB v0.8.0 — BM25 Hybrid Search Demo

Demonstrates:
  - keyword_search(): pure BM25 over content field
  - hybrid_search(): BM25 + dense vector via Reciprocal Rank Fusion (RRF)

Run:
    source repro_venv/bin/activate
    python examples/bm25_hybrid_search_demo.py
"""

import feather_db
import numpy as np
import time
import tempfile
import os

DIM = 128  # embedding dimension

def embed(text: str) -> np.ndarray:
    """Simulated embedding — replace with sentence-transformers in production."""
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    v = rng.randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def main():
    tmpf = tempfile.mktemp(suffix=".feather")
    db = feather_db.DB.open(tmpf, dim=DIM)

    corpus = [
        "Python is widely used for machine learning and data science pipelines",
        "BM25 is the Okapi BM25 ranking function used in information retrieval",
        "Feather DB is an embedded vector database with BM25 hybrid search",
        "JavaScript and TypeScript are popular for web frontend development",
        "Hybrid search combines BM25 keyword matching with dense vector similarity",
        "Vector databases store embeddings for approximate nearest neighbour search",
        "Reciprocal Rank Fusion merges multiple ranked lists into one ranking",
        "Python pandas and numpy are essential tools for numerical data manipulation",
        "Information retrieval systems use inverted indexes for fast keyword lookup",
        "Machine learning requires large training datasets and GPU compute resources",
    ]

    print("Indexing corpus...")
    for i, text in enumerate(corpus, start=1):
        meta = feather_db.Metadata()
        meta.content = text
        meta.timestamp = int(time.time())
        meta.importance = 1.0
        db.add(id=i, vec=embed(text), meta=meta)
    print(f"  {len(corpus)} documents indexed.\n")

    # ── BM25 keyword search ──────────────────────────────────────────────
    query = "BM25 hybrid search retrieval"
    print(f"keyword_search: '{query}'")
    print("-" * 60)
    for r in db.keyword_search(query, k=5):
        print(f"  id={r.id:2d} bm25={r.score:.3f} | {r.metadata.content[:70]}")
    print()

    # ── Dense vector search (baseline) ──────────────────────────────────
    print(f"vector_search: '{query}'")
    print("-" * 60)
    for r in db.search(embed(query), k=5):
        print(f"  id={r.id:2d} sim={r.score:.4f}  | {r.metadata.content[:70]}")
    print()

    # ── Hybrid search (RRF) ─────────────────────────────────────────────
    print(f"hybrid_search (RRF, rrf_k=60): '{query}'")
    print("-" * 60)
    for r in db.hybrid_search(embed(query), query, k=5, rrf_k=60):
        print(f"  id={r.id:2d} rrf={r.score:.5f} | {r.metadata.content[:70]}")
    print()

    # ── Python-focused query ─────────────────────────────────────────────
    q2 = "Python data science tools"
    print(f"keyword_search: '{q2}'")
    print("-" * 60)
    for r in db.keyword_search(q2, k=5):
        print(f"  id={r.id:2d} bm25={r.score:.3f} | {r.metadata.content[:70]}")

    db.save()
    os.unlink(tmpf)
    print("\nDone.")


if __name__ == "__main__":
    main()
