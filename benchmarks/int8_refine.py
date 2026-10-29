"""In-RAM int8: fixed vs trained scale, with and without the float re-rank.

    python benchmarks/int8_refine.py               # 100k x 384, the reference size
    python benchmarks/int8_refine.py --n 20000     # quicker

Embedding-shaped data (clustered, uneven per-dimension spread, a few outlier
dimensions, unit norm). Each configuration is built and queried in its own
process, which memory-maps the data from a file the parent wrote, so its
memory is the engine's alone. Memory is the physical footprint (macOS: what
Activity Monitor shows, compressed pages included; Linux: RSS / VmHWM).

  * build: add_batch + save; peak footprint above the process's baseline
  * query: a fresh process opens the saved file as a writer (vectors on the
    heap, not mapped), runs the queries: footprint, recall@10 against brute
    force, latency p50 / p99. Latency is with the file in the page cache — each
    re-ranked candidate costs a pread, which on a cold cache is a disk read.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CONFIGS = {
    "float32":                 None,
    "int8 clip 1.0, refine 0": (1.0, 0),     # the pre-0.22 default
    "int8 trained, refine 0":  (None, 0),
    "int8 trained, refine 2":  (None, 2),    # the default now
}


def data(n, dim, nq, seed=7):
    rng = np.random.default_rng(seed)
    spread = rng.lognormal(0.0, 0.5, dim).astype(np.float32)
    spread[rng.choice(dim, 4, replace=False)] *= 6.0
    c = rng.standard_normal((max(8, n // 100), dim)).astype(np.float32) * spread
    def rows(m):
        x = c[rng.integers(0, len(c), m)] + rng.standard_normal((m, dim)).astype(np.float32) * spread * 0.6
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return rows(n), rows(nq)


def footprint_mb():
    """(current, peak) physical footprint of this process, in MB."""
    if sys.platform == "darwin":
        import ctypes
        buf = (ctypes.c_uint64 * 64)()                 # rusage_info_v4 is 16 B uuid + u64s
        ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, buf)
        # after the uuid (2 slots): [9] ri_phys_footprint, [30] ri_lifetime_max_phys_footprint
        return buf[9] / 2**20, buf[30] / 2**20
    status = dict(l.split(":", 1) for l in open("/proc/self/status"))
    kb = lambda k: int(status[k].split()[0]) / 1024
    return kb("VmRSS"), kb("VmHWM")


def child_build(path, cfg, n, dim, xfile):
    import feather_db
    X = np.load(xfile, mmap_mode="r")
    base = footprint_mb()[0]
    db = feather_db.DB.open(path, dim=dim)
    if cfg is not None:
        db.set_int8_ram("text", cfg[0], refine=cfg[1])
    t = time.perf_counter()
    for s in range(0, n, 10_000):
        db.add_batch(list(range(s, min(n, s + 10_000))), X[s:s + 10_000])
    build_s = time.perf_counter() - t
    db.save()
    info = db.int8_info("text")
    db.close()
    return {"build_s": build_s, "peak_mb": footprint_mb()[1] - base, "clip": info["clip"]}


def child_query(path, qfile, dim):
    import feather_db
    Q = np.load(qfile)
    base = footprint_mb()[0]
    t = time.perf_counter()
    db = feather_db.DB.open(path, dim=dim)
    open_s = time.perf_counter() - t
    ids, lat = [], []
    for q in Q:
        t = time.perf_counter()
        res = db.search(q, k=10, record_salience=False)
        lat.append((time.perf_counter() - t) * 1e3)
        ids.append([r.id for r in res])
    rss = footprint_mb()[0] - base
    db.close(save=False)
    lat.sort()
    return {"open_s": open_s, "rss_mb": rss, "ids": ids,
            "p50_ms": lat[len(lat) // 2], "p99_ms": lat[int(len(lat) * 0.99)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--child", nargs=2)
    a = ap.parse_args()
    if a.child:
        mode, spec = a.child
        spec = json.loads(spec)
        cfg = spec["cfg"] if spec["cfg"] is None else tuple(spec["cfg"])
        out = child_build(spec["path"], cfg, a.n, a.dim, spec["x"]) if mode == "build" \
            else child_query(spec["path"], spec["q"], a.dim)
        print(json.dumps(out))
        return

    X, Q = data(a.n, a.dim, a.nq)
    d = (Q ** 2).sum(1)[:, None] - 2 * Q @ X.T + (X ** 2).sum(1)[None, :]
    truth = np.argsort(d, axis=1)[:, :10]
    tmp = tempfile.mkdtemp()
    xfile, qfile = os.path.join(tmp, "x.npy"), os.path.join(tmp, "q.npy")
    np.save(xfile, X)
    np.save(qfile, Q)
    del X, d
    run = lambda mode, spec: json.loads(subprocess.check_output(
        [sys.executable, __file__, "--n", str(a.n), "--dim", str(a.dim), "--nq", str(a.nq),
         "--child", mode, json.dumps(spec)], cwd=REPO).decode().strip().splitlines()[-1])
    print(f"{a.n} x {a.dim}, {a.nq} queries, k=10, ef=50\n")
    print(f"{'':26s} {'recall':>7s} {'p50 ms':>7s} {'p99 ms':>7s} {'mem MB':>7s} "
          f"{'peak MB':>8s} {'file MB':>8s} {'open s':>7s} {'clip':>6s}")
    try:
        for name, cfg in CONFIGS.items():
            path = os.path.join(tmp, name.replace(" ", "_").replace(",", "") + ".feather")
            b = run("build", {"path": path, "cfg": cfg, "x": xfile})
            q = run("query", {"path": path, "cfg": cfg, "q": qfile})
            rec = np.mean([len(set(g) & set(t.tolist())) / 10 for g, t in zip(q["ids"], truth)])
            print(f"{name:26s} {rec:7.4f} {q['p50_ms']:7.3f} {q['p99_ms']:7.3f} {q['rss_mb']:7.0f} "
                  f"{b['peak_mb']:8.0f} {os.path.getsize(path) / 2**20:8.1f} {q['open_s']:7.2f} "
                  f"{b['clip']:6.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
