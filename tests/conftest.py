"""
Shared fixtures for Feather DB v0.7.0 test suite.
"""
import os
import tempfile
import numpy as np
import pytest
import feather_db
from feather_db import DB, ContextEngine, FilterBuilder, RelType


# ---------------------------------------------------------------------------
# Deterministic embedder (no model needed)
# ---------------------------------------------------------------------------

def make_embedder(dim: int = 128):
    """Returns a deterministic embedder of given dimensionality."""
    def embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.random(dim).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    return embed


EMBED = make_embedder(128)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_path_feather(tmp_path):
    """Returns a temp .feather file path (deleted automatically)."""
    return str(tmp_path / "test.feather")


@pytest.fixture
def db(tmp_path_feather):
    """Fresh DB with 128-dim text index."""
    return DB.open(tmp_path_feather, dim=128)


@pytest.fixture
def populated_db(tmp_path_feather):
    """DB with 6 pre-loaded records (text + visual multimodal)."""
    records = [
        (1,  "Onboarding drop-off rate is 42% at step 3",       "product",  "user_retention"),
        (2,  "Users want a one-click deploy option",             "product",  "deployment"),
        (3,  "Competitor released v2 with 50% faster search",   "product",  "competitor"),
        (4,  "Dark mode is the most requested UI feature",       "product",  "ui_requests"),
        (5,  "API latency spikes under 500 concurrent requests", "infra",    "performance"),
        (6,  "LLM context window is the main adoption blocker",  "research", "llm_context"),
    ]
    db = DB.open(tmp_path_feather, dim=128)
    for rid, content, ns, eid in records:
        meta = feather_db.Metadata()
        meta.content = content
        meta.namespace_id = ns
        meta.entity_id = eid
        meta.importance = 0.8
        meta.set_attribute("entity_type", "fact")
        db.add(id=rid, vec=EMBED(content), meta=meta)
        # multimodal visual pocket — pass same meta to preserve metadata
        db.add(id=rid, vec=make_embedder(64)(content), meta=meta, modality="visual")
    db.save()
    return db


@pytest.fixture
def engine(tmp_path_feather):
    """ContextEngine in offline (heuristic) mode."""
    return ContextEngine(
        db_path=tmp_path_feather,
        dim=128,
        provider=None,
        embedder=EMBED,
        namespace="test",
    )
