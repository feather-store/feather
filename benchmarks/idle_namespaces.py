"""Memory of a shard: every namespace open (as at startup before 0.22) vs
open on use, closed when idle.

    python benchmarks/idle_namespaces.py                 # 40 tenants x 5k x 256
    python benchmarks/idle_namespaces.py --tenants 100 --n 2000

Builds the tenants' files, then in a fresh process: the footprint with all of
them open, after the idle sweep closes them, and with the few a request
touches reopened. Memory is the physical footprint (macOS: what Activity
Monitor shows; Linux: RSS).
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
sys.path.insert(0, os.path.join(REPO, "feather-api"))


def footprint_mb():
    if sys.platform == "darwin":
        import ctypes
        buf = (ctypes.c_uint64 * 64)()
        ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, buf)
        return buf[9] / 2**20
    status = dict(l.split(":", 1) for l in open("/proc/self/status"))
    return int(status["VmRSS"].split()[0]) / 1024


def child(data_dir, tenants, hot):
    import gc
    os.environ["FEATHER_DATA_DIR"] = data_dir
    from app.db_manager import DBManager
    base = footprint_mb()
    mgr = DBManager(data_dir=data_dir, idle_close_s=0.5, max_open=0)
    start = footprint_mb() - base
    t = time.perf_counter()
    for i in range(tenants):                       # what 0.21's startup did
        mgr.get(f"t{i}", create=False)
    open_all_s = time.perf_counter() - t
    all_open = footprint_mb() - base
    time.sleep(0.6)
    mgr.sweep()
    gc.collect()
    idle = footprint_mb() - base
    t = time.perf_counter()
    for i in range(hot):
        mgr.get(f"t{i}", create=False)
    reopen_ms = (time.perf_counter() - t) / max(1, hot) * 1e3
    some_open = footprint_mb() - base
    mgr.close_all()
    return {"start": start, "all_open": all_open, "idle": idle, "some_open": some_open,
            "open_all_s": open_all_s, "reopen_ms": reopen_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", type=int, default=40)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--child", nargs=3)
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.child[0], int(a.child[1]), int(a.child[2]))))
        return
    import feather_db
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    for i in range(a.tenants):
        with feather_db.DB.open(os.path.join(d, f"t{i}.feather"), dim=a.dim) as db:
            metas = []
            for j in range(a.n):
                m = feather_db.Metadata()
                m.content = f"tenant {i} note {j} " + "x" * 120
                metas.append(m)
            db.add_batch(list(range(a.n)), rng.random((a.n, a.dim), dtype=np.float32), metas)
    disk = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 2**20
    out = json.loads(subprocess.check_output(
        [sys.executable, __file__, "--child", d, str(a.tenants), str(a.hot)]).decode().splitlines()[-1])
    shutil.rmtree(d, ignore_errors=True)
    print(f"{a.tenants} tenants x {a.n} x {a.dim} ({disk:.0f} MB on disk)\n")
    print(f"  shard started (nothing opened):     {out['start']:7.0f} MB")
    print(f"  every tenant open (0.21 startup):   {out['all_open']:7.0f} MB   "
          f"({out['open_all_s']:.1f} s to open them)")
    print(f"  all idle, closed by the sweep:      {out['idle']:7.0f} MB")
    print(f"  {a.hot} hot tenants reopened:           {out['some_open']:7.0f} MB   "
          f"({out['reopen_ms']:.0f} ms per reopen)")
    print(f"\n  tenants per GB, all open: {a.tenants / max(out['all_open'], 1) * 1024:.0f}; "
          f"with {a.hot} hot: {a.tenants / max(out['some_open'], 1) * 1024:.0f}")


if __name__ == "__main__":
    main()
