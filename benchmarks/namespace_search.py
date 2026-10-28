"""Filtered search inside one namespace: exact scan vs the namespace's graph.

    python benchmarks/namespace_search.py               # 100k x 384
    python benchmarks/namespace_search.py --n 20000

Embedding-shaped data (clustered, unit norm). Tenants: one holding half the
records, one a tenth, one a hundredth, and 1,000 small ones sharing the rest.
For each, the median latency of a k=10 search filtered to that namespace,
first with namespace graphs off (every filtered search scans its namespace),
then with them on and built, and recall@10 against brute force over the
namespace. Unfiltered search is the reference.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feather_db  # noqa: E402


def data(n, dim, nq, seed=0):
    rng = np.random.default_rng(seed)
    spread = rng.lognormal(0.0, 0.5, dim).astype(np.float32)
    c = rng.standard_normal((max(8, n // 100), dim)).astype(np.float32) * spread

    def rows(m):
        x = c[rng.integers(0, len(c), m)] + rng.standard_normal((m, dim)).astype(np.float32) * spread * 0.6
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return rows(n), rows(nq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--nq", type=int, default=300)
    a = ap.parse_args()

    X, Q = data(a.n, a.dim, a.nq)
    tenants = {"half": a.n // 2, "tenth": a.n // 10, "hundredth": a.n // 100}
    ns = [name for name, size in tenants.items() for _ in range(size)]
    ns += [f"t{i % 1000}" for i in range(a.n - len(ns))]
    tenants["one of 1000"] = sum(1 for x in ns if x == "t7")
    name_of = {"one of 1000": "t7"}
    ns_arr = np.array(ns)

    tmp = tempfile.mkdtemp()
    db = feather_db.DB.open(os.path.join(tmp, "ns.feather"), dim=a.dim)
    metas = []
    for x in ns:
        m = feather_db.Metadata()
        m.namespace_id = x
        metas.append(m)
    t = time.perf_counter()
    for s in range(0, a.n, 10_000):
        db.add_batch(list(range(s, min(a.n, s + 10_000))), X[s:s + 10_000], metas[s:s + 10_000])
    print(f"{a.n} x {a.dim}: built in {time.perf_counter() - t:.0f} s\n")

    def median_ms(flt):
        ts = []
        for q in Q:
            t0 = time.perf_counter()
            db.search(q, k=10, filter=flt, record_salience=False)
            ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1e3

    def recall(flt, ids):
        hits = 0
        for q in Q[:50]:
            got = {r.id for r in db.search(q, k=10, filter=flt, record_salience=False)}
            d = ((X[ids] - q) ** 2).sum(1)
            hits += len(got & set(ids[np.argsort(d)[:10]].tolist()))
        return hits / 500

    print(f"unfiltered: {median_ms(None):.3f} ms\n")
    print(f"{'namespace':14s} {'records':>8s} {'scan ms':>8s} {'graph ms':>9s} {'speedup':>8s} "
          f"{'recall':>7s} {'graph build s':>14s}")
    for label, size in tenants.items():
        name = name_of.get(label, label)
        flt = feather_db.FilterBuilder().namespace(name).build()
        ids = np.nonzero(ns_arr == name)[0]
        db.set_namespace_ann_min(0)
        scan = median_ms(flt)
        db.set_namespace_ann_min(4096)
        t = time.perf_counter()
        db.search(Q[0], k=10, filter=flt, record_salience=False)       # queues the build
        db.wait_for_namespace_graphs(3600)
        build = time.perf_counter() - t
        graph = median_ms(flt)
        built = size >= 4096
        print(f"{label:14s} {size:8d} {scan:8.3f} {graph:9.3f} {scan / graph:7.1f}x "
              f"{recall(flt, ids):7.3f} {build if built else 0.0:14.1f}"
              + ("" if built else "   (below 4096: still scanned)"))
    db.close(save=False)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
