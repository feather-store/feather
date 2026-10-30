"""BM25 doc numbers: re-indexing past the renumbering point keeps results exact.

Postings name documents by a u32 number; every re-index takes a new one and
the numbering is compacted once retired numbers outnumber live ones. A
renumbering that mapped a posting to the wrong document would score it for
words it never had, so compare against an index built fresh on reopen.
"""
import numpy as np

import feather_db

N, ROUNDS = 1000, 70          # 70k re-indexes: past the 65536 threshold


def _meta(text):
    m = feather_db.Metadata()
    m.content = text
    return m


def _ranked(db, q):
    return [(r.id, round(r.score, 5)) for r in db.keyword_search(q, k=20)]


def test_reindex_past_renumbering_matches_fresh_index(tmp_path):
    path = str(tmp_path / "bm25.feather")
    rng = np.random.default_rng(0)
    vecs = rng.random((N, 8), dtype=np.float32)
    ids = list(range(N))
    db = feather_db.DB.open(path, dim=8)
    for r in range(ROUNDS):
        db.add_batch(ids, vecs, [_meta(f"round{r} doc{i} common w{i % 7}") for i in ids])
    # Retire some documents outright, as forget() does.
    for i in range(0, N, 10):
        db.forget(i)

    queries = [f"round{ROUNDS - 1}", "round0", "common", "w3", "doc17", "doc20 w6 common"]
    live = {q: _ranked(db, q) for q in queries}
    assert live["round0"] == []                       # superseded content is gone
    assert {i for i, _ in live[f"round{ROUNDS - 1}"]}.isdisjoint(range(0, N, 10))
    assert live["doc17"][0][0] == 17
    assert live["doc20 w6 common"] == [] or live["doc20 w6 common"][0][0] != 20
    db.close()

    with feather_db.DB.open(path, dim=8, read_only=True) as fresh:
        for q in queries:
            assert _ranked(fresh, q) == live[q], q
