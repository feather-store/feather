"""Filtered search: SIMD exact scan + large-set HNSW route (0.19).

* Indexed filters (namespace / entity / attribute) resolve an exact candidate
  set. Small or very selective sets are scanned exactly with the index's SIMD
  kernel on the stored vectors (no per-candidate copy); only the top-k get
  their metadata copied.
* Large, unselective sets (candidates * selectivity^2 > C * k) go through
  filtered HNSW with ef raised ~1/selectivity, falling back to the exact scan
  if the walk comes up short. FEATHER_PREFILTER_MODE=exact|hnsw forces a route.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import feather_db
from feather_db import DB, FilterBuilder

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _brute_topk(V, ids, q, k):
    d = np.sum((V[ids] - q) ** 2, axis=1)
    order = np.argsort(d, kind="stable")[:k]
    return [int(ids[i]) for i in order]


def test_exact_path_equals_brute_force(tmp_path):
    rng = np.random.default_rng(0)
    n, dim = 4000, 64
    V = rng.standard_normal((n, dim)).astype(np.float32)
    metas = []
    for i in range(n):
        m = feather_db.Metadata()
        m.namespace_id = f"ns{i % 7}"
        m.set_attribute("tier", "gold" if i % 5 == 0 else "std")
        metas.append(m)
    db = DB.open(str(tmp_path / "e.feather"), dim=dim)
    db.add_batch(list(range(n)), V, metas)
    ns3 = np.array([i for i in range(n) if i % 7 == 3])
    both = np.array([i for i in range(n) if i % 7 == 3 and i % 5 == 0])
    for q in rng.standard_normal((25, dim)).astype(np.float32):
        f = FilterBuilder().namespace("ns3").build()
        assert [r.id for r in db.search(q, k=10, filter=f)] == _brute_topk(V, ns3, q, 10)
        f2 = FilterBuilder().namespace("ns3").attribute("tier", "gold").build()
        assert [r.id for r in db.search(q, k=10, filter=f2)] == _brute_topk(V, both, q, 10)
    db.close()


def test_only_returned_hits_carry_metadata_and_salience(tmp_path):
    db = DB.open(str(tmp_path / "m.feather"), dim=8)
    rng = np.random.default_rng(1)
    for i in range(500):
        m = feather_db.Metadata()
        m.namespace_id = "x"
        m.content = f"row {i}"
        db.add(i, rng.random(8).astype(np.float32), m)
    res = db.search(rng.random(8).astype(np.float32), k=5, filter=FilterBuilder().namespace("x").build())
    assert len(res) == 5 and all(r.metadata.content.startswith("row ") for r in res)
    touched = [i for i in range(500) if db.get_metadata(i).recall_count]
    assert sorted(touched) == sorted(r.id for r in res)
    db.close()


def test_query_dim_mismatch_raises(tmp_path):
    db = DB.open(str(tmp_path / "d.feather"), dim=8)
    db.add(1, np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        db.search(np.ones(5, dtype=np.float32), k=1)
    db.close()


def _run(code, **env):
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {REPO!r})\n" + code],
                       capture_output=True, text=True, timeout=600, env={**os.environ, **env})
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


_LARGE = """
import numpy as np, tempfile, os, feather_db
from feather_db import FilterBuilder
rng = np.random.default_rng(2)
n, dim = 30000, 32
V = rng.standard_normal((n, dim)).astype(np.float32)
metas = []
for i in range(n):
    m = feather_db.Metadata(); m.namespace_id = "big" if i % 4 == 0 else ("tiny" if i % 997 == 1 else "other")
    metas.append(m)
db = feather_db.DB.open(os.path.join(tempfile.mkdtemp(), "l.feather"), dim=dim)
db.add_batch(list(range(n)), V, metas)
def brute(ids, q, k):
    d = np.sum((V[ids] - q) ** 2, axis=1); return set(int(ids[i]) for i in np.argsort(d)[:k])
big = np.array([i for i in range(n) if i % 4 == 0]); tiny = np.array([i for i in range(n) if i % 997 == 1 and i % 4 != 0])
rb = rt = 0; full = True
for q in rng.standard_normal((40, dim)).astype(np.float32):
    got = [r.id for r in db.search(q, k=10, filter=FilterBuilder().namespace("big").build(), record_salience=False)]
    full &= len(got) == 10 and all(i % 4 == 0 for i in got)
    rb += len(brute(big, q, 10) & set(got))
    got = [r.id for r in db.search(q, k=10, filter=FilterBuilder().namespace("tiny").build(), record_salience=False)]
    rt += len(brute(tiny, q, 10) & set(got))
print("RECALL_BIG", rb / 400, "RECALL_TINY", rt / 400, "FULL", full)
"""


def test_large_candidate_set_uses_graph_with_good_recall():
    # 7.5k of 30k records (sel 25%): 7500 * 0.25^2 = 469 > 10 * 10 -> HNSW route
    out = _run(_LARGE).split()
    recall_big, recall_tiny, full = float(out[1]), float(out[3]), out[5] == "True"
    assert full, "HNSW route returned fewer than k in-filter results"
    assert recall_big >= 0.9, f"recall@10 on the HNSW route: {recall_big}"
    assert recall_tiny == 1.0, "small candidate set must use the exact scan"


def test_forced_exact_route_is_exact():
    out = _run(_LARGE, FEATHER_PREFILTER_MODE="exact").split()
    assert float(out[1]) == 1.0 and float(out[3]) == 1.0
