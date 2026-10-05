"""Writers are not starved by a steady read load (0.19).

libstdc++'s std::shared_mutex prefers readers: with overlapping queries a writer
waited until no reader held the lock at all. Measured before the fix: 8 hybrid-
search readers blocked every write for the whole measurement window. The engine
now uses a phase-fair lock.
"""
import threading
import time

import numpy as np

import feather_db
from feather_db import DB


def test_writer_progresses_under_heavy_hybrid_reads(tmp_path):
    dim = 256
    db = DB.open(str(tmp_path / "fair.feather"), dim=dim)
    rng = np.random.default_rng(0)
    metas = []
    for i in range(20_000):
        m = feather_db.Metadata()
        m.content = f"alpha beta gamma record {i} topic{i % 50}"
        metas.append(m)
    db.add_batch(list(range(20_000)), rng.standard_normal((20_000, dim)).astype(np.float32), metas)
    Q = rng.standard_normal((64, dim)).astype(np.float32)

    stop = threading.Event()

    def reader(w):
        i = w
        while not stop.is_set():
            db.hybrid_search(Q[i % 64], "alpha beta topic7", k=10)
            i += 1

    readers = [threading.Thread(target=reader, args=(w,)) for w in range(8)]
    [t.start() for t in readers]
    time.sleep(0.2)
    writes = 0
    t_end = time.perf_counter() + 3.0
    V = rng.standard_normal((100_000, dim)).astype(np.float32)
    while time.perf_counter() < t_end:
        db.add(1_000_000 + writes, V[writes], feather_db.Metadata())
        writes += 1
    stop.set()
    [t.join() for t in readers]
    # reader-preferring lock: ~0 writes in 3 s. Phase-fair: at least a handful
    # per second even on a slow CI disk (each write also fsyncs).
    assert writes >= 20, f"only {writes} writes in 3s under read load: writers starved"
    db.close()


def test_readers_keep_flowing_while_a_save_holds_the_lock(tmp_path):
    """A checkpoint holds the lock shared for the whole snapshot. With a
    writer queued behind it, a plain fair lock would make every new reader
    wait for the entire save (measured: p99 2.2 s). Checkpoints take the lock
    in long-shared mode, which keeps admitting readers."""
    import os
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = f"""
import sys, threading, time
sys.path.insert(0, {repo!r})
import numpy as np, feather_db
db = feather_db.DB.open({str(tmp_path / 'save.feather')!r}, dim=64)
rng = np.random.default_rng(0)
db.add_batch(list(range(40000)), rng.standard_normal((40000, 64)).astype(np.float32), [])
stop = threading.Event(); lat = []; saves = []
def saver():
    while not stop.is_set():
        t = time.perf_counter(); db.save(); saves.append(time.perf_counter() - t)
def writer(w):
    i = 0
    while not stop.is_set():
        db.add(10**7 + w * 10**6 + i, rng.standard_normal(64).astype(np.float32)); i += 1
def reader():
    q = rng.standard_normal(64).astype(np.float32)
    while not stop.is_set():
        t = time.perf_counter(); db.search(q, k=10, record_salience=False); lat.append(time.perf_counter() - t)
ts = [threading.Thread(target=saver)] + [threading.Thread(target=writer, args=(w,)) for w in range(2)] \\
     + [threading.Thread(target=reader) for _ in range(4)]
[t.start() for t in ts]; time.sleep(4); stop.set(); [t.join() for t in ts]
lat.sort()
print("P99", lat[int(len(lat) * 0.99)], "SAVE", sum(saves) / max(1, len(saves)), "N", len(lat))
db.close(save=False)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
                         env={**os.environ, "FEATHER_WAL_SYNC": "0"})
    assert "P99" in out.stdout, out.stdout + out.stderr
    _, p99, _, save_s, _, n = out.stdout.split()
    p99, save_s = float(p99), float(save_s)
    assert p99 < max(0.25 * save_s, 0.02), f"read p99 {p99:.3f}s vs save {save_s:.3f}s: readers stalled by saves"
