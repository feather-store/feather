"""
Engine contention benchmark (Phase 0 of CONCURRENCY_FIX_PLAN.md).

The existing suites measure each operation on its own, and search scaling with
readers only. This one measures *interference*: how much the latency of a read
grows while writes, saves, compactions or metadata updates run against the same
DB.

Durability mode is process-wide (the engine caches FEATHER_WAL_SYNC on first
use), so `orchestrate` runs each mode in its own subprocess:

    python benchmarks/concurrency_bench.py orchestrate --out benchmarks/runs/concurrency
    python benchmarks/concurrency_bench.py run --suite contention --sync 1 --out x.json

Reads use record_salience=False so they don't mutate what they measure.
Latencies are measured from Python, so they include GIL hand-off time. That is
real overhead for any Python caller, but it is not pure engine time.
"""
import argparse, itertools, json, os, subprocess, sys, tempfile, threading, time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import feather_db as fdb  # noqa: E402

VOCAB = [f"w{i}" for i in range(5000)]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def stats(lat_ms, seconds):
    if not lat_ms:
        return {"ops": 0}
    a = np.asarray(lat_ms)
    return {
        "ops": int(a.size),
        "ops_per_s": round(a.size / seconds, 1),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p90_ms": round(float(np.percentile(a, 90)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "p999_ms": round(float(np.percentile(a, 99.9)), 3),
        "max_ms": round(float(a.max()), 2),
        "over_10ms": int((a > 10).sum()),
        "over_50ms": int((a > 50).sum()),
    }


def make_meta(rng, i, ns_count=20):
    m = fdb.Metadata()
    m.timestamp = int(time.time()) - int(rng.integers(0, 90 * 86400))
    m.importance = float(rng.uniform(0.3, 1.0))
    m.content = " ".join(rng.choice(VOCAB, 12))
    m.namespace_id = f"ns{i % ns_count}"
    m.entity_id = f"e{i % 1000}"
    m.set_attribute("color", ("red", "blue", "green")[i % 3])
    return m


def build_db(n, dim, seed=0):
    path = tempfile.mktemp(suffix=".feather", dir=os.environ.get("BENCH_TMP"))
    db = fdb.DB.open(path, dim=dim)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for s in range(0, n, 10_000):
        e = min(n, s + 10_000)
        ids = np.arange(s + 1, e + 1, dtype=np.uint64)
        vecs = rng.standard_normal((e - s, dim)).astype(np.float32)
        metas = [make_meta(rng, i) for i in range(s, e)]
        db.add_batch(ids, vecs, metas)
    db.save()   # checkpoint: start every scenario from an empty WAL
    return db, path, time.perf_counter() - t0


class Load:
    """Run named worker loops in threads for a fixed window; collect latencies."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.stop = threading.Event()
        self.lat = {}
        self.threads = []
        self.lock = threading.Lock()

    def add(self, name, count, body):
        # body(worker_index) -> callable doing one op; latency is timed here
        for w in range(count):
            op = body(w)

            def loop(op=op, name=name):
                mine = []
                while not self.stop.is_set():
                    t = time.perf_counter()
                    op()
                    mine.append((time.perf_counter() - t) * 1e3)
                with self.lock:
                    self.lat.setdefault(name, []).extend(mine)

            self.threads.append(threading.Thread(target=loop, daemon=True))

    def run(self, during=None):
        for t in self.threads:
            t.start()
        t0 = time.perf_counter()
        extra = during() if during else None     # optional foreground action
        remaining = self.seconds - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
        self.stop.set()
        for t in self.threads:
            t.join()
        el = time.perf_counter() - t0
        return {k: stats(v, el) for k, v in self.lat.items()}, extra, el


def reader_search(db, queries, k=10):
    def body(w):
        it = itertools.cycle(queries[w::7] if len(queries) > 7 else queries)
        return lambda: db.search(next(it), k=k, record_salience=False)
    return body


def reader_hybrid(db, queries, kw, k=10):
    def body(w):
        it = itertools.cycle(range(w, len(queries), 7))
        return lambda: (lambda i: db.hybrid_search(queries[i], kw[i], k=k))(next(it))
    return body


def writer_add(db, dim, next_id, seed):
    def body(w):
        rng = np.random.default_rng(seed + w)
        pool = rng.standard_normal((256, dim)).astype(np.float32)
        metas = [make_meta(rng, i) for i in range(256)]
        c = itertools.count()

        def op():
            i = next(c)
            db.add(id=next(next_id), vec=pool[i % 256], meta=metas[i % 256])
        return op
    return body


def writer_batch(db, dim, next_id, seed, batch=1000):
    def body(w):
        rng = np.random.default_rng(seed + 100 + w)
        vecs = rng.standard_normal((batch, dim)).astype(np.float32)
        metas = [make_meta(rng, i) for i in range(batch)]

        def op():
            ids = np.fromiter((next(next_id) for _ in range(batch)), dtype=np.uint64, count=batch)
            db.add_batch(ids, vecs, metas)
        return op
    return body


def writer_link(db, n, seed):
    def body(w):
        rng = np.random.default_rng(seed + 200 + w)
        pairs = rng.integers(1, n + 1, size=(4096, 2))
        c = itertools.count()

        def op():
            a, b = pairs[next(c) % 4096]
            db.link(int(a), int(b), "related_to", 0.5)
        return op
    return body


# ─────────────────────────────────────────────────────────────────────────────
# suite 1: read/write contention (run once per durability mode)
# ─────────────────────────────────────────────────────────────────────────────
def suite_contention(a):
    R = {}
    db, path, build_s = build_db(a.n, a.dim)
    R["setup"] = {"n": a.n, "dim": a.dim, "build_s": round(build_s, 2)}
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((1000, a.dim)).astype(np.float32)
    KW = [" ".join(rng.choice(VOCAB, 3)) for _ in range(1000)]
    next_id = itertools.count(a.n + 1)
    T = a.seconds

    def scenario(name, readers=0, writers=0, wkind="add", reader="search", during=None):
        L = Load(T)
        if readers:
            L.add("read", readers, reader_search(db, Q) if reader == "search" else reader_hybrid(db, Q, KW))
        if writers:
            body = {"add": lambda: writer_add(db, a.dim, next_id, 7),
                    "batch": lambda: writer_batch(db, a.dim, next_id, 7),
                    "link": lambda: writer_link(db, a.n, 7)}[wkind]()
            L.add("write", writers, body)
        res, extra, el = L.run(during)
        res["window_s"] = round(el, 2)
        if extra is not None:
            res["foreground"] = extra
        R[name] = res
        print(f"  {name:<34} " + "  ".join(
            f"{k}: {v.get('ops_per_s', '-')}/s p50={v.get('p50_ms', '-')} p99={v.get('p99_ms', '-')} max={v.get('max_ms', '-')}"
            for k, v in res.items() if isinstance(v, dict) and "ops" in v), flush=True)

    # readers alone: the reference every mixed run is compared against
    for r in (1, 4, 8):
        scenario(f"readers_only_r{r}", readers=r)
    # writers alone: does write throughput scale with writer threads?
    for w in (1, 4, 16):
        scenario(f"writers_only_add_w{w}", writers=w)
    # the core question: reader latency while single-record writes run
    for w in (1, 4, 16):
        scenario(f"r8_plus_add_w{w}", readers=8, writers=w)
    scenario("r8_plus_link_w4", readers=8, writers=4, wkind="link")
    scenario("r8_plus_add_batch1000_w1", readers=8, writers=1, wkind="batch")
    scenario("hybrid_r8_only", readers=8, reader="hybrid")
    scenario("hybrid_r8_plus_add_w4", readers=8, writers=4, reader="hybrid")

    # save(): readers share the lock with save, writers must wait for it
    def save_loop():
        durs = []
        t_end = time.perf_counter() + T - 0.5
        while time.perf_counter() < t_end:
            t = time.perf_counter(); db.save(); durs.append((time.perf_counter() - t) * 1e3)
        return {"saves": len(durs), "save_ms_mean": round(float(np.mean(durs)), 1),
                "save_ms_max": round(float(np.max(durs)), 1)}
    scenario("r8_plus_add_w2_during_save_loop", readers=8, writers=2, during=save_loop)
    R["final_size"] = db.size()
    db.save()
    for p in (path, path + ".wal"):
        if os.path.exists(p):
            R.setdefault("file_bytes", {})[os.path.basename(p)[-5:]] = os.path.getsize(p)
            os.remove(p)
    return R


# ─────────────────────────────────────────────────────────────────────────────
# suite 2: CPU-bound exclusive sections (run with FEATHER_WAL_SYNC=0 so fsync
# cost doesn't hide them)
# ─────────────────────────────────────────────────────────────────────────────
def suite_exclusive(a):
    R = {}
    rng = np.random.default_rng(9)
    Q = rng.standard_normal((1000, a.dim)).astype(np.float32)

    # ── compaction stall ─────────────────────────────────────────────────────
    db, path, _ = build_db(a.n, a.dim, seed=1)
    for i in range(1, a.n + 1, 10):          # forget 10 %
        db.forget(i)
    L0 = Load(3); L0.add("read", 8, reader_search(db, Q)); base, _, _ = L0.run()

    def do_compact():
        time.sleep(0.5)
        t = time.perf_counter(); removed = db.compact()
        return {"compact_s": round(time.perf_counter() - t, 2), "removed": removed}
    L = Load(0.1)   # window grows to cover the compaction (run() sleeps only the remainder)
    L.add("read", 8, reader_search(db, Q))
    nid = itertools.count(a.n + 1)
    L.add("write", 1, writer_add(db, a.dim, nid, 3))
    res, extra, el = L.run(do_compact)
    R["compaction_stall"] = {"readers_before": base["read"], "during": res, "compaction": extra,
                             "window_s": round(el, 2)}
    print("  compaction_stall", json.dumps(R["compaction_stall"]), flush=True)

    # ── auto-compaction: the forget() that crosses the threshold pays for it ─
    db.set_auto_compact(0.10)
    lat = []
    live = [i for i in range(2, a.n + 1, 5)]   # 20 % > threshold, so it must fire
    t_start = time.perf_counter()
    for i in live:
        t = time.perf_counter(); db.forget(i); lat.append((time.perf_counter() - t) * 1e3)
    if hasattr(db, "wait_for_compaction"):   # 0.19+: compaction runs in the background
        t = time.perf_counter(); db.wait_for_compaction()
        R["auto_compact_background_s"] = round(time.perf_counter() - t, 2)
    R["auto_compact_forget"] = {"forgets": len(lat), "slowest_forget_ms": round(max(lat), 1),
                                "forgets_over_100ms": int(sum(x > 100 for x in lat)),
                                "median_forget_ms": round(float(np.median(lat)), 4),
                                "elapsed_s": round(time.perf_counter() - t_start, 2)}
    print("  auto_compact_forget", R["auto_compact_forget"], flush=True)
    for p in (path, path + ".wal"):
        if os.path.exists(p): os.remove(p)

    # ── update_metadata cost vs. total edge count ────────────────────────────
    R["update_metadata_vs_edges"] = {}
    n_meta = min(a.n, 100_000)
    db, path, _ = build_db(n_meta, 32, seed=2)
    erng = np.random.default_rng(5)
    total = 0
    for target in (10_000, 100_000, 500_000):
        pairs = erng.integers(1, n_meta + 1, size=(target - total, 2))
        for x, y in pairs:
            db.link(int(x), int(y), "related_to", 0.5)
        total = target
        ids = erng.integers(1, n_meta + 1, size=300)
        lat_u, lat_i = [], []
        for rid in ids:
            m = db.get_metadata(int(rid))
            t = time.perf_counter(); db.update_metadata(int(rid), m); lat_u.append((time.perf_counter() - t) * 1e3)
            t = time.perf_counter(); db.update_importance(int(rid), 0.7); lat_i.append((time.perf_counter() - t) * 1e3)
        R["update_metadata_vs_edges"][str(target)] = {
            "update_metadata_ms_mean": round(float(np.mean(lat_u)), 3),
            "update_metadata_ms_p99": round(float(np.percentile(lat_u, 99)), 3),
            "update_importance_ms_mean": round(float(np.mean(lat_i)), 4)}
        print(f"  update_metadata @ {target} edges", R["update_metadata_vs_edges"][str(target)], flush=True)

    # readers while a thread loops update_metadata at 500k edges
    Qs = np.random.default_rng(1).standard_normal((1000, 32)).astype(np.float32)
    L0 = Load(3); L0.add("read", 8, reader_search(db, Qs)); base, _, _ = L0.run()

    def upd_body(w):
        r = np.random.default_rng(77 + w); c = itertools.count()
        ids = r.integers(1, n_meta + 1, size=1024)
        def op():
            rid = int(ids[next(c) % 1024]); db.update_metadata(rid, db.get_metadata(rid))
        return op
    L = Load(a.seconds); L.add("read", 8, reader_search(db, Qs)); L.add("update_metadata", 1, upd_body)
    res, _, _ = L.run()
    R["readers_during_update_metadata_500k_edges"] = {"readers_alone": base["read"], **res}
    print("  readers_during_update_metadata", json.dumps(R["readers_during_update_metadata_500k_edges"]), flush=True)
    for p in (path, path + ".wal"):
        if os.path.exists(p): os.remove(p)
    return R


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "orchestrate"])
    ap.add_argument("--suite", choices=["contention", "exclusive"], default="contention")
    ap.add_argument("--sync", default="1")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        os.environ["FEATHER_WAL_SYNC"] = a.sync      # before the first WAL write
        t0 = time.time()
        R = suite_contention(a) if a.suite == "contention" else suite_exclusive(a)
        R["_meta"] = {"suite": a.suite, "wal_sync": a.sync, "n": a.n, "dim": a.dim,
                      "seconds": a.seconds, "cpus": os.cpu_count(), "wall_s": round(time.time() - t0, 1),
                      "version": fdb.__version__}
        with open(a.out, "w") as f:
            json.dump(R, f, indent=2)
        return

    os.makedirs(a.out, exist_ok=True)
    jobs = [("contention", "1"), ("contention", "0"), ("exclusive", "0")]
    for dims in ((a.dim,),):
        for suite, sync in jobs:
            out = os.path.join(a.out, f"engine_{suite}_sync{sync}_{a.n}x{dims[0]}.json")
            print(f"\n=== {suite}  FEATHER_WAL_SYNC={sync}  {a.n}x{dims[0]} ===", flush=True)
            env = dict(os.environ, FEATHER_WAL_SYNC=sync)
            subprocess.run([sys.executable, __file__, "run", "--suite", suite, "--sync", sync,
                            "--n", str(a.n), "--dim", str(dims[0]), "--seconds", str(a.seconds),
                            "--out", out], env=env, check=True)


if __name__ == "__main__":
    main()
