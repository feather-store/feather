"""
Search-path benchmark for the runtime SIMD dispatch + filtered-search work.

Runs against whatever feather_db is importable, so the same script measures an
old build and a new one:

    python benchmarks/search_simd_bench.py --n 100000 --dim 768 --out new_768.json

Measures single-thread and 8-thread unfiltered search, and single-thread
filtered search for namespaces of several sizes (small = exact scan,
large = the path that changed most). record_salience=False where supported.
"""
import argparse, json, os, sys, tempfile, threading, time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("FEATHER_BENCH_PATH", HERE))
import feather_db  # noqa: E402
from feather_db import FilterBuilder  # noqa: E402

NS_SIZES = [500, 5_000, 25_000, 60_000]


def pct(lat):
    a = np.asarray(lat) * 1e3
    return {"p50_ms": round(float(np.percentile(a, 50)), 4),
            "p99_ms": round(float(np.percentile(a, 99)), 4),
            "qps": round(len(a) / (a.sum() / 1e3), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--int8", action="store_true", help="store vectors as in-RAM int8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    # clustered data (not uniform noise) so HNSW recall is realistic
    centers = rng.standard_normal((256, a.dim)).astype(np.float32)
    V = (centers[rng.integers(0, 256, a.n)] + 0.3 * rng.standard_normal((a.n, a.dim))).astype(np.float32)
    ns_of = np.full(a.n, "rest", dtype=object)
    start = 0
    for size in NS_SIZES:
        ns_of[start:start + size] = f"ns{size}"
        start += size
    rng.shuffle(ns_of)
    metas = []
    for i in range(a.n):
        m = feather_db.Metadata()
        m.namespace_id = ns_of[i]
        m.content = f"record {i}"
        metas.append(m)

    path = os.path.join(tempfile.mkdtemp(), "b.feather")
    db = feather_db.DB.open(path, dim=a.dim)
    if a.int8:
        db.set_int8_ram("text", float(np.abs(V).max()))
    t0 = time.perf_counter()
    for s in range(0, a.n, 20_000):
        db.add_batch(list(range(s, min(a.n, s + 20_000))), V[s:s + 20_000], metas[s:s + 20_000])
    build_s = time.perf_counter() - t0
    Q = (centers[rng.integers(0, 256, a.queries)] +
         0.3 * rng.standard_normal((a.queries, a.dim))).astype(np.float32)

    kw = {}
    try:
        db.search(Q[0], k=1, record_salience=False)
        kw["record_salience"] = False
    except TypeError:
        pass

    info = getattr(feather_db.core, "simd_info", lambda: {"active": "n/a (pre-0.19)"})()
    R = {"_meta": {"n": a.n, "dim": a.dim, "queries": a.queries, "build_s": round(build_s, 1),
                   "int8": a.int8,
                   "simd": info, "version": feather_db.__version__,
                   "FEATHER_SIMD_RUNTIME": os.environ.get("FEATHER_SIMD_RUNTIME")}}

    for q in Q[:50]:
        db.search(q, k=10, **kw)                      # warm up
    lat = []
    for q in Q:
        t = time.perf_counter(); db.search(q, k=10, **kw); lat.append(time.perf_counter() - t)
    R["unfiltered_1t"] = pct(lat)

    done, count, lock = threading.Event(), [0], threading.Lock()

    def worker(w):
        i, c = w, 0
        while not done.is_set():
            db.search(Q[i % len(Q)], k=10, **kw); i += 8; c += 1
        with lock:
            count[0] += c
    ts = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    t0 = time.perf_counter(); [t.start() for t in ts]; time.sleep(4); done.set(); [t.join() for t in ts]
    R["unfiltered_8t_qps"] = round(count[0] / (time.perf_counter() - t0), 1)

    R["filtered_1t"] = {}
    for size in NS_SIZES:
        f = FilterBuilder().namespace(f"ns{size}").build()
        members = np.where(ns_of == f"ns{size}")[0]
        lat, hits, results = [], 0, []
        for q in Q[:20]:
            db.search(q, k=10, filter=f, **kw)            # warm up
        for q in Q[:200]:                                 # time first ...
            t = time.perf_counter(); results.append(db.search(q, k=10, filter=f, **kw))
            lat.append(time.perf_counter() - t)
        for q, res in zip(Q[:200], results):              # ... then score recall. Computing
            d = np.sum((V[members] - q) ** 2, axis=1)     # truth between timed searches
                                                          # evicted the caches (tens of MB).
            truth = set(int(members[j]) for j in np.argpartition(d, 10)[:10])
            hits += len(truth & {r.id for r in res})
        R["filtered_1t"][f"ns_{size}"] = {**pct(lat), "recall@10": round(hits / 2000, 3)}
    print(json.dumps(R, indent=1))
    with open(a.out, "w") as fh:
        json.dump(R, fh, indent=2)


if __name__ == "__main__":
    main()
