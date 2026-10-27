"""Runtime-dispatched distance kernels (0.19).

Every kernel this CPU supports (scalar, SSE2, AVX2+FMA, AVX-512 / NEON) must
agree with numpy, for any dim including every tail length, and the choice must
not change search results.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from feather_db import core

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIMS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65,
        100, 127, 128, 129, 384, 768, 1000, 1536, 3072]
LEVELS = core._simd_levels()


def test_simd_info_reports_a_level():
    info = core.simd_info()
    assert info["active"] in ("scalar", "sse2", "avx2+fma", "avx512f", "neon")
    assert info["detected"] in LEVELS


@pytest.mark.parametrize("level", LEVELS)
def test_float_l2_matches_numpy(level):
    rng = np.random.default_rng(0)
    for dim in DIMS:
        a = rng.standard_normal(dim).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)
        expect = float(np.sum((a.astype(np.float64) - b) ** 2))
        got = core._l2sqr(a, b, level)
        assert got == pytest.approx(expect, rel=1e-4, abs=1e-5), (level, dim)


@pytest.mark.parametrize("level", LEVELS)
def test_int8_l2_is_exact(level):
    rng = np.random.default_rng(1)
    for dim in DIMS + [8191, 8192, 8193, 16400, 70000]:
        a = rng.integers(-127, 128, dim, dtype=np.int8)
        b = rng.integers(-127, 128, dim, dtype=np.int8)
        expect = int(np.sum((a.astype(np.int64) - b) ** 2))
        assert core._int8_l2sqr(a, b, level) == expect, (level, dim)
    # worst case for lane overflow: every |a_i - b_i| = 254
    n = 1 << 18
    a = np.full(n, 127, dtype=np.int8)
    b = np.full(n, -127, dtype=np.int8)
    assert core._int8_l2sqr(a, b, level) == n * 254 * 254


@pytest.mark.parametrize("level", LEVELS)
def test_float_query_vs_int8_row_matches_numpy(level):
    rng = np.random.default_rng(4)
    for dim in DIMS:
        q = rng.standard_normal(dim).astype(np.float32)
        v = rng.integers(-127, 128, dim, dtype=np.int8)
        scale = np.float32(0.0137)
        expect = float(np.sum((q.astype(np.float64) - v.astype(np.float64) * scale) ** 2))
        got = core._f32_i8_l2sqr(q, v, float(scale), level)
        assert got == pytest.approx(expect, rel=1e-4, abs=1e-5), (level, dim)


def test_int8_filtered_scan_matches_float_scan(tmp_path):
    """The int8 exact scan (asymmetric SIMD kernel) ranks like a float scan of
    the dequantized vectors."""
    import feather_db
    from feather_db import DB, FilterBuilder
    rng = np.random.default_rng(5)
    n, dim, max_abs = 3000, 96, 4.0
    V = rng.uniform(-max_abs, max_abs, (n, dim)).astype(np.float32)
    db = DB.open(str(tmp_path / "i8.feather"), dim=dim)
    db.set_int8_ram("text", max_abs)
    metas = []
    for i in range(n):
        m = feather_db.Metadata(); m.namespace_id = "a" if i % 2 else "b"; metas.append(m)
    db.add_batch(list(range(n)), V, metas)
    scale = max_abs / 127.0
    deq = np.clip(np.round(V / scale), -127, 127) * scale
    b_ids = np.array([i for i in range(n) if i % 2 == 0])
    f = FilterBuilder().namespace("b").build()
    for q in rng.uniform(-max_abs, max_abs, (20, dim)).astype(np.float32):
        d = np.sum((deq[b_ids] - q) ** 2, axis=1)
        expect = [int(b_ids[j]) for j in np.argsort(d, kind="stable")[:10]]
        assert [r.id for r in db.search(q, k=10, filter=f, record_salience=False)] == expect
    db.close()


def test_unsupported_level_is_rejected():
    if "avx512f" in LEVELS or "neon" in LEVELS:
        pytest.skip("CPU supports every x86 level")
    with pytest.raises(ValueError):
        core._l2sqr(np.ones(4, np.float32), np.ones(4, np.float32), "avx512f")


def _search_ids(env_level):
    code = f"""
import sys; sys.path.insert(0, {REPO!r})
import numpy as np, tempfile, os, feather_db
from feather_db import core
rng = np.random.default_rng(3)
p = os.path.join(tempfile.mkdtemp(), "s.feather")
db = feather_db.DB.open(p, dim=100)            # 100: exercises the tail path
metas = []
for i in range(3000):
    m = feather_db.Metadata(); m.namespace_id = "a" if i % 3 else "b"; metas.append(m)
db.add_batch(list(range(3000)), rng.standard_normal((3000, 100)).astype(np.float32), metas)
f = feather_db.FilterBuilder().namespace("b").build()
out = []
for q in rng.standard_normal((20, 100)).astype(np.float32):
    out.append([r.id for r in db.search(q, k=10, filter=f, record_salience=False)])
print(core.simd_info()["active"]); print(out)
db.close(save=False)
"""
    env = {**os.environ}
    if env_level:
        env["FEATHER_SIMD_RUNTIME"] = env_level
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    return lines[0], lines[1]


def test_runtime_cap_and_identical_exact_results():
    """The pre-filtered exact path must return the same ids at every level."""
    active, fast = _search_ids(None)
    scalar_name, slow = _search_ids("scalar")
    assert scalar_name == "scalar"
    assert active == core.simd_info()["active"]
    assert fast == slow
