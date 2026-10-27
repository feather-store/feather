"""All three search paths must agree on what is dead.

`is_dead_meta()` is the single rule — source == "_forgotten", or the
"_deleted" attribute set to "true". `search()` applies it in its
FilterWrapper and `keyword_search()` in `bm25_top_nolock`'s eligible(), but
`hybrid_search()`'s vector-side functor was missing it, so a filtered hybrid
query returned tombstones that the other two paths excluded:

    search()         -> [1]
    keyword_search() -> [1]
    hybrid_search()  -> [1, 2]     <- the tombstone, fused back in by RRF

A `_deleted` record is NOT markDelete'd in HNSW (only forget() does that), so
the graph walk reaches it and nothing else kept it out. These tests pin all
three paths to the same answer.
"""
import numpy as np
import pytest

import feather_db
from feather_db import FilterBuilder

DIM = 8


def _seed(db):
    """id=1 live, id=2 soft-deleted, id=3 forgotten. All match the query."""
    base = np.ones(DIM, dtype=np.float32)
    for rec_id, content, kind in [
        (1, "alpha budget report", "live"),
        (2, "alpha budget tombstone", "deleted"),
        (3, "alpha budget forgotten", "live"),
    ]:
        meta = feather_db.Metadata()
        meta.content = content
        meta.namespace_id = "ns"
        if kind == "deleted":
            meta.set_attribute("_deleted", "true")
        # Near-identical vectors so all three are genuine ANN candidates.
        db.add(id=rec_id, vec=base * (1.0 - rec_id * 0.001), meta=meta)
    db.forget(3)


@pytest.fixture
def seeded(tmp_path):
    db = feather_db.DB.open(str(tmp_path / "dead.feather"), dim=DIM)
    _seed(db)
    yield db
    db.close()


def _ids(results):
    return sorted(r.id for r in results)


def test_three_search_paths_agree_on_dead_records(seeded):
    q = np.ones(DIM, dtype=np.float32)
    f = FilterBuilder().namespace("ns").build()

    vector = _ids(seeded.search(q, k=10, filter=f))
    keyword = _ids(seeded.keyword_search("alpha budget", k=10, filter=f))
    hybrid = _ids(seeded.hybrid_search(q, "alpha budget", k=10, filter=f))

    assert vector == keyword == hybrid == [1], (
        f"paths disagree: search={vector} keyword={keyword} hybrid={hybrid}"
    )


def test_hybrid_search_excludes_soft_deleted(seeded):
    """The specific regression: RRF must not resurrect a _deleted record."""
    q = np.ones(DIM, dtype=np.float32)
    f = FilterBuilder().namespace("ns").build()
    assert 2 not in _ids(seeded.hybrid_search(q, "alpha budget", k=10, filter=f))


def test_hybrid_search_excludes_forgotten(seeded):
    q = np.ones(DIM, dtype=np.float32)
    f = FilterBuilder().namespace("ns").build()
    assert 3 not in _ids(seeded.hybrid_search(q, "alpha budget", k=10, filter=f))


def test_live_record_still_returned(seeded):
    """Guard against 'fixing' this by filtering everything out."""
    q = np.ones(DIM, dtype=np.float32)
    f = FilterBuilder().namespace("ns").build()
    assert _ids(seeded.hybrid_search(q, "alpha budget", k=10, filter=f)) == [1]
