"""The BM25 index is built on first keyword use, not at open.

Writes made before that first use skip the index; the build then reads the
records as they are. Every change made in that window must show up exactly
as it would had the index been live all along.
"""
import numpy as np

import feather_db


def _meta(text):
    m = feather_db.Metadata()
    m.content = text
    return m


def _ids(db, q):
    return sorted(r.id for r in db.keyword_search(q, k=10))


def _seed(path):
    db = feather_db.DB.open(path, dim=4)
    for i, t in enumerate(["alpha report", "beta report", "gamma memo"]):
        db.add(id=i, vec=np.ones(4, dtype=np.float32) * i, meta=_meta(t))
    db.close()


def test_changes_before_first_use_are_indexed(tmp_path):
    path = str(tmp_path / "lazy.feather")
    _seed(path)
    db = feather_db.DB.open(path, dim=4)
    db.add(id=3, vec=np.ones(4, dtype=np.float32), meta=_meta("delta report"))  # new
    db.update_metadata(1, _meta("beta invoice"))                               # changed
    db.forget(2)                                                               # gone
    assert _ids(db, "report") == [0, 3]
    assert _ids(db, "invoice") == [1]
    assert _ids(db, "memo") == []
    db.add(id=4, vec=np.ones(4, dtype=np.float32), meta=_meta("epsilon report"))  # after
    assert _ids(db, "report") == [0, 3, 4]
    hy = db.hybrid_search(np.ones(4, dtype=np.float32), "invoice", k=5)
    assert 1 in [r.id for r in hy]
    db.close()


def test_wal_replay_before_first_use(tmp_path):
    path = str(tmp_path / "wal.feather")
    _seed(path)
    db = feather_db.DB.open(path, dim=4)
    db.update_metadata(0, _meta("alpha invoice"))
    db.close(save=False)                         # leave the change in the WAL only
    with feather_db.DB.open(path, dim=4, read_only=True) as ro:
        assert _ids(ro, "invoice") == [0]
        assert _ids(ro, "report") == [1]


def test_hybrid_builds_it_first(tmp_path):
    path = str(tmp_path / "hy.feather")
    _seed(path)
    with feather_db.DB.open(path, dim=4, read_only=True) as ro:
        r = ro.hybrid_search(np.zeros(4, dtype=np.float32), "gamma", k=1)
        assert r[0].id == 2
