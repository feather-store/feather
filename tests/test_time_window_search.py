"""A time window (valid_after/valid_before, timestamp_after/before) is a
range of the engine's time indexes, used as the candidate set: results are
the exact top-k, whatever the window's width, alone or with other filters."""
import random

import numpy as np
import pytest

import feather_db
from feather_db import DB, FilterBuilder

DIM = 16


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = np.random.default_rng(3)
    db = DB.open(str(tmp_path_factory.mktemp("w") / "w.feather"), dim=DIM)
    rows = []
    b = feather_db.WriteBatch()
    for i in range(6000):
        m = feather_db.Metadata()
        m.timestamp = int(1_600_000_000 + rng.integers(0, 10_000_000))
        if rng.random() < 0.7:
            m.valid_at = int(1_500_000_000 + rng.integers(0, 10_000_000))
        m.namespace_id = f"org.u{i % 3}" if i % 7 else ""
        m.entity_id = f"e{i % 5}"
        m.set_attribute("kind", "a" if i % 2 else "b")
        m.key = f"k{i}"
        v = rng.standard_normal(DIM).astype(np.float32)
        b.upsert(m, v)
        rows.append((m.namespace_id, m.timestamp, m.valid_at or m.timestamp, m.entity_id,
                     m.get_attribute("kind"), v))
        if len(b) >= 1000:
            db.write(b)
            b = feather_db.WriteBatch()
    db.write(b)
    ids = {}
    for ns in ("org.u0", "org.u1", "org.u2"):
        for rid in db.ids_in_namespace(ns):
            ids[db.get_metadata(rid).key] = rid
    yield db, rows, rng
    db.close()


def _brute(db, rows, q, k, keep):
    best = []
    for i, (ns, ts, vt, ent, kind, v) in enumerate(rows):
        if keep(ns, ts, vt, ent, kind):
            best.append((float(np.sum((q - v) ** 2)), f"k{i}"))
    best.sort()
    return [key for _, key in best[:k]]


CASES = [
    # (filter builder, python predicate) for windows of many widths
    ("narrow valid, one namespace", lambda f, lo, hi: f.namespace("org.u1").valid_between(lo, hi),
     lambda ns, ts, vt, e, kd, lo, hi: ns == "org.u1" and lo <= vt <= hi),
    ("valid, prefix", lambda f, lo, hi: f.namespace_prefix("org").valid_between(lo, hi),
     lambda ns, ts, vt, e, kd, lo, hi: ns.startswith("org.") and lo <= vt <= hi),
    ("valid, everywhere", lambda f, lo, hi: f.valid_between(lo, hi),
     lambda ns, ts, vt, e, kd, lo, hi: lo <= vt <= hi),
    ("timestamp, one namespace", lambda f, lo, hi: f.namespace("org.u2").after(lo + 100_000_000).before(hi + 100_000_000),
     lambda ns, ts, vt, e, kd, lo, hi: ns == "org.u2" and lo + 100_000_000 <= ts <= hi + 100_000_000),
    ("valid + entity + attribute", lambda f, lo, hi: f.namespace("org.u0").entity("e3").attribute("kind", "a").valid_between(lo, hi),
     lambda ns, ts, vt, e, kd, lo, hi: ns == "org.u0" and e == "e3" and kd == "a" and lo <= vt <= hi),
    ("only an upper bound", lambda f, lo, hi: f.namespace("org.u1").valid_between(before=hi),
     lambda ns, ts, vt, e, kd, lo, hi: ns == "org.u1" and vt <= hi),
]


@pytest.mark.parametrize("name,build,keep", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("width", [0, 2_000, 200_000, 3_000_000, 30_000_000])
def test_a_window_search_is_the_exact_top_k(world, name, build, keep, width):
    db, rows, rng = world
    for _ in range(3):
        lo = int(1_500_000_000 + rng.integers(0, 10_000_000))
        hi = lo + width
        q = rng.standard_normal(DIM).astype(np.float32)
        f = build(FilterBuilder(), lo, hi).build()
        got = [r.metadata.key for r in db.search(q, k=10, filter=f, record_salience=False)]
        want = _brute(db, rows, q, 10, lambda *a: keep(*a, lo, hi))
        assert got == want, (name, width)
