"""DB.open and every call that can block run without the GIL.

They used to hold it: a slow open, or a read queued behind a writer on the
data lock, froze every Python thread in the process for as long as it took —
an API server's event loop, a heartbeat, a progress bar.

Each test measures how long the blocking call took and how long a pure-Python
ticker thread went without running. With the GIL held the ticker's longest
gap is the whole call; released, it is a scheduler tick. The bound is relative
to the call's own duration, so a slow CI machine does not make it flaky.

Releasing the GIL also took away an accidental guarantee: opens of one path
were never concurrent. The last tests hold the engine to that on its own.
"""
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 64
N = 30000


class Ticker:
    """Counts how long a pure-Python thread goes without getting the GIL."""

    def __init__(self):
        self.max_gap = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(0.001)
            now = time.perf_counter()
            self.max_gap = max(self.max_gap, now - last)
            last = now

    def __enter__(self):
        self._t.start()
        time.sleep(0.02)
        self.max_gap = 0.0
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def _metas(lo, hi):
    out = []
    for i in range(lo, hi):
        m = feather_db.Metadata()
        m.content = f"record {i} about topic {i % 97}"
        m.namespace_id = f"ns{i % 10}"
        out.append(m)
    return out


@pytest.fixture(scope="module")
def big_file(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("gil") / "big.feather")
    with DB.open(path, dim=DIM) as db:
        db.add_batch(list(range(N)), np.random.rand(N, DIM).astype(np.float32),
                     _metas(0, N))
    return path


def test_a_slow_open_does_not_freeze_other_threads(big_file):
    with Ticker() as tick:
        t0 = time.perf_counter()
        db = DB.open(big_file, dim=DIM)
        took = time.perf_counter() - t0
    db.close()
    if took < 0.04:
        pytest.skip(f"open took {took * 1000:.0f} ms: too fast to observe")
    assert tick.max_gap < took / 2, (
        f"a {took * 1000:.0f} ms open stalled other threads "
        f"{tick.max_gap * 1000:.0f} ms")


_TICKER_SRC = """
import threading, time
class Ticker:
    def __init__(self):
        self.max_gap = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(0.001)
            now = time.perf_counter()
            self.max_gap = max(self.max_gap, now - last)
            last = now
    def __enter__(self):
        self._t.start(); time.sleep(0.02); self.max_gap = 0.0
        return self
    def __exit__(self, *exc):
        self._stop.set(); self._t.join()
"""


def _run_scenario(body, **env):
    """Run `body` in a fresh process (env vars the engine reads once, at first
    use). It must print `took max_gap` in seconds."""
    src = f"import sys; sys.path.insert(0, {REPO!r})\n" + _TICKER_SRC + body
    e = dict(os.environ, **env)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, env=e, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    took, gap = map(float, r.stdout.split()[-2:])
    return took, gap


def test_a_read_queued_behind_a_writer_does_not_freeze_other_threads(tmp_path):
    """add_batch holds the data lock exclusively per chunk (it already ran
    without the GIL); a reader waiting for that lock held the GIL and so
    stopped everything else too. Large chunks make the wait measurable."""
    took, gap = _run_scenario(f"""
import numpy as np, feather_db
db = feather_db.DB.open({str(tmp_path / "w.feather")!r}, dim={DIM})
db.add(0, np.random.rand({DIM}).astype(np.float32))
n = 20000
vecs = np.random.rand(n, {DIM}).astype(np.float32)
done = threading.Event(); longest = [0.0]
def reader():
    while not done.is_set():
        t0 = time.perf_counter()
        db.size()
        longest[0] = max(longest[0], time.perf_counter() - t0)
r = threading.Thread(target=reader)
with Ticker() as tick:
    r.start()
    db.add_batch(list(range(1, n + 1)), vecs)
    done.set(); r.join()
db.close()
print(longest[0], tick.max_gap)
""", FEATHER_BATCH_CHUNK="4000")
    if took < 0.03:
        pytest.skip(f"the reader waited only {took * 1000:.0f} ms")
    assert gap < took / 2, (
        f"a read waited {took * 1000:.0f} ms and stalled other threads "
        f"{gap * 1000:.0f} ms")


def test_dropping_the_last_reference_does_not_freeze_other_threads(tmp_path):
    """~DB() of a never-closed writer joins a running compaction and then
    checkpoints. On pybind11 >= 3 that runs without the GIL; older pybind11
    has no way to ask for it, and the build there keeps the old behaviour."""
    path = str(tmp_path / "drop.feather")
    took, gap = _run_scenario(f"""
import numpy as np, feather_db
n = 20000
db = feather_db.DB.open({path!r}, dim={DIM})
db.add_batch(list(range(n)), np.random.rand(n, {DIM}).astype(np.float32))
db.set_auto_compact(0.1)
for i in range(0, n, 4):
    db.forget(i)
compacting = db.is_compacting()
with Ticker() as tick:
    t0 = time.perf_counter()
    del db
    took = time.perf_counter() - t0
print(took if compacting else 0.0, tick.max_gap)
""")
    with DB.open(path, dim=DIM) as db:
        assert db.size() == 15000             # the checkpoint still happened
    if took < 0.04:
        pytest.skip(f"the drop took {took * 1000:.0f} ms: too fast to observe")
    pybind11 = pytest.importorskip("pybind11", reason="cannot tell which pybind11 built this")
    if int(pybind11.__version__.split(".")[0]) < 3:
        pytest.skip("built against pybind11 < 3: destructors keep the GIL")
    assert gap < took / 2, (
        f"a {took * 1000:.0f} ms drop stalled other threads {gap * 1000:.0f} ms")


# ── what the GIL used to serialise ────────────────────────────────────────

def _other_process_can_write(path):
    src = (f"import sys; sys.path.insert(0, {REPO!r})\n"
           f"import feather_db\n"
           f"try:\n"
           f"    feather_db.DB.open({path!r}, dim={DIM}).close(); print('opened')\n"
           f"except RuntimeError as e:\n"
           f"    print('refused')\n")
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip() == "opened"


@pytest.mark.skipif(sys.platform == "win32", reason="the registry is POSIX")
def test_concurrent_writer_opens_of_one_path_all_succeed(tmp_path):
    path = str(tmp_path / "c.feather")
    with DB.open(path, dim=DIM) as db:
        for i in range(50):
            db.add(i, np.random.rand(DIM).astype(np.float32), _metas(i, i + 1)[0])

    handles, errors = [], []
    barrier = threading.Barrier(8)

    def opener():
        barrier.wait()
        try:
            handles.append(DB.open(path, dim=DIM))
        except Exception as e:                 # noqa: BLE001
            errors.append(str(e))

    threads = [threading.Thread(target=opener) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    assert all(h.size() == 50 for h in handles)

    assert not _other_process_can_write(path)  # the flock is really held
    for h in handles:
        h.close(save=False)
    assert _other_process_can_write(path)      # ...and really released


@pytest.mark.skipif(sys.platform == "win32", reason="the registry is POSIX")
def test_opens_racing_closes_never_leak_or_lose_the_lock(tmp_path):
    """A close that dropped the registry entry before unlocking let a racing
    open hit our own flock and admit itself against an entry nobody held the
    lock for — leaving the file unlocked while a handle was open."""
    path = str(tmp_path / "r.feather")
    DB.open(path, dim=DIM).close()
    errors = []

    def churn():
        try:
            for _ in range(40):
                DB.open(path, dim=DIM).close(save=False)
        except Exception as e:                 # noqa: BLE001
            errors.append(str(e))

    threads = [threading.Thread(target=churn) for _ in range(6)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []

    held = DB.open(path, dim=DIM)
    assert not _other_process_can_write(path)
    held.close()
    assert _other_process_can_write(path)
