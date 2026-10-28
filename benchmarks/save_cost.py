"""Save cost and writer stalls: format v9 (whole-file snapshot) vs v10 (append).

    python benchmarks/save_cost.py                # 100k x 384, the reference size
    python benchmarks/save_cost.py --n 20000      # quicker

Builds one DB, then measures the same workload against a copy in each format
(FEATHER_FORMAT=9 keeps writing v9):

  * bytes and wall time per save after a small change (100 adds, 100 updates)
  * add() latency on a writer thread while saves run — the stall a save
    imposes on writers, which is what v10 removes
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def child(path, fmt, n, dim, rounds):
    if fmt == 9:
        os.environ["FEATHER_FORMAT"] = "9"
    import feather_db
    rng = np.random.default_rng(7)
    db = feather_db.DB.open(path, dim=dim)
    db.save()                                   # convert: v10 file -> v9, or no-op
    nxt = n
    saves = []
    for r in range(rounds):
        for _ in range(100):
            m = feather_db.Metadata(); m.content = f"round {r}"
            db.add(nxt, rng.random(dim, dtype=np.float32), m); nxt += 1
        for i in rng.integers(0, n, 100):
            db.update_importance(int(i), float(rng.random()))
        size0 = os.path.getsize(path)
        t = time.perf_counter(); db.save(); dt = time.perf_counter() - t
        written = (db.storage_stats()["last_commit_bytes"] if fmt == 10
                   else os.path.getsize(path))
        saves.append({"ms": dt * 1e3, "bytes": written, "file": os.path.getsize(path),
                      "grew": os.path.getsize(path) - size0})

    # Writer stalls: one thread adds continuously; the main thread saves.
    lat_idle, lat_save = [], []
    phase = {"saving": False}
    stop = threading.Event()

    def writer():
        nonlocal nxt
        v = rng.random(dim, dtype=np.float32)
        while not stop.is_set():
            t = time.perf_counter()
            db.add(nxt, v)
            dt = (time.perf_counter() - t) * 1e3
            (lat_save if phase["saving"] else lat_idle).append(dt)
            nxt += 1

    th = threading.Thread(target=writer)
    th.start()
    time.sleep(1.0)
    for _ in range(rounds):
        phase["saving"] = True
        db.save()
        phase["saving"] = False
        time.sleep(0.3)
    stop.set(); th.join()
    db.close()
    pct = lambda a, q: float(np.percentile(a, q)) if a else 0.0
    print(json.dumps({
        "format": fmt, "saves": saves,
        "add_idle_p99_ms": pct(lat_idle, 99), "add_idle_max_ms": max(lat_idle, default=0),
        "add_during_save_p99_ms": pct(lat_save, 99),
        "add_during_save_max_ms": max(lat_save, default=0),
        "adds_during_save": len(lat_save),
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--child", nargs=2)
    a = ap.parse_args()
    if a.child:
        child(a.child[0], int(a.child[1]), a.n, a.dim, a.rounds)
        return

    import feather_db
    d = tempfile.mkdtemp(prefix="feather_save_cost_")
    base = os.path.join(d, "base.feather")
    print(f"building {a.n} x {a.dim} ...", flush=True)
    t = time.perf_counter()
    db = feather_db.DB.open(base, dim=a.dim)
    rng = np.random.default_rng(1)
    for s in range(0, a.n, 10_000):
        e = min(a.n, s + 10_000)
        metas = []
        for i in range(s, e):
            m = feather_db.Metadata(); m.content = f"record {i}"; metas.append(m)
        db.add_batch(list(range(s, e)), rng.random((e - s, a.dim), dtype=np.float32), metas)
    db.close()
    print(f"  built in {time.perf_counter() - t:.0f}s, {os.path.getsize(base)/1e6:.0f} MB", flush=True)

    results = {}
    for fmt in (9, 10):
        p = os.path.join(d, f"v{fmt}.feather")
        shutil.copy(base, p)
        env = dict(os.environ); env.pop("FEATHER_FORMAT", None)
        out = subprocess.run([sys.executable, __file__, "--n", str(a.n), "--dim", str(a.dim),
                              "--rounds", str(a.rounds), "--child", p, str(fmt)],
                             capture_output=True, text=True, env=env, cwd=REPO)
        if out.returncode:
            print(out.stderr); sys.exit(1)
        results[fmt] = json.loads(out.stdout.strip().splitlines()[-1])

    print(f"\n{a.n} x {a.dim}, save after 100 adds + 100 importance updates "
          f"(median of {a.rounds}):")
    print(f"{'':28}{'v9':>14}{'v10':>14}")
    med = lambda f, k: float(np.median([s[k] for s in results[f]["saves"]]))
    print(f"{'bytes written per save':28}{med(9,'bytes')/1e6:>11.1f} MB{med(10,'bytes')/1e6:>11.2f} MB")
    print(f"{'save wall time':28}{med(9,'ms'):>11.0f} ms{med(10,'ms'):>11.1f} ms")
    print(f"{'file size after rounds':28}{results[9]['saves'][-1]['file']/1e6:>11.1f} MB"
          f"{results[10]['saves'][-1]['file']/1e6:>11.1f} MB")
    print(f"\nadd() latency on a writer thread:")
    for k, label in (("add_idle_p99_ms", "no save running, p99"),
                     ("add_during_save_p99_ms", "while saving, p99"),
                     ("add_during_save_max_ms", "while saving, max")):
        print(f"  {label:26}{results[9][k]:>11.2f} ms{results[10][k]:>11.2f} ms")
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
