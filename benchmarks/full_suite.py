"""
Feather DB — end-to-end micro-benchmark of every core operation.

Covers ingestion (add / add_batch / multimodal), vector search (latency,
recall, ef sweep, concurrent QPS), filtered + decay-scored search, BM25 +
hybrid, graph (link / edges / auto_link / context_chain / export), metadata
ops, secondary indexes, persistence (save / cold load / quantized / int8-RAM),
and deletion lifecycle (forget / purge / compact).

    python benchmarks/full_suite.py --n 100000 --dim 128 --out results.json
"""
import argparse, gc, json, os, sys, tempfile, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feather_db import core as fc  # noqa: E402

PROC = psutil.Process()
R = {}


def rss_mb():
    return PROC.memory_info().rss / 2**20


def pct(xs):
    a = np.asarray(xs)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}


def time_each(fn, items):
    lat = []
    t0 = time.perf_counter()
    for it in items:
        s = time.perf_counter(); fn(it); lat.append((time.perf_counter() - s) * 1e3)
    wall = time.perf_counter() - t0
    out = pct(lat); out["ops_per_s"] = len(items) / wall; out["count"] = len(items)
    return out


def rm(p):
    for x in (p, p + ".wal", p + ".tmp"):
        if os.path.exists(x): os.remove(x)


def log(section, key, val):
    R.setdefault(section, {})[key] = val
    if isinstance(val, dict):
        short = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in val.items())
    else:
        short = val
    print(f"[{section}] {key}: {short}", flush=True)


def clustered(rng, n, dim, k=500):
    centers = (rng.standard_normal((k, dim)) * 3.0).astype(np.float32)
    a = rng.integers(0, k, size=n)
    return (centers[a] + rng.standard_normal((n, dim)).astype(np.float32)).astype(np.float32)


def brute_topk(base, qs, k):
    bn = (base ** 2).sum(1)
    out = []
    for i in range(0, len(qs), 64):
        q = qs[i:i + 64]
        d = bn[None, :] - 2 * q @ base.T
        idx = np.argpartition(d, k, axis=1)[:, :k]
        out.extend(idx[j][np.argsort(d[j, idx[j]])] for j in range(len(q)))
    return np.array(out)


def recall(db, qs, gt, k, ef=None, **kw):
    if ef: db.set_ef(ef)
    hits = 0
    lat = []
    for q, g in zip(qs, gt):
        s = time.perf_counter()
        res = db.search(q, k=k, record_salience=False, **kw)
        lat.append((time.perf_counter() - s) * 1e3)
        hits += len(set(r.id for r in res) & set(int(x) for x in g))
    out = pct(lat); out[f"recall@{k}"] = hits / (len(qs) * k); out["qps_1thread"] = 1e3 / out["mean_ms"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--links", type=int, default=50_000)
    ap.add_argument("--out", default="full_suite.json")
    a = ap.parse_args()
    N, DIM, NQ, K = a.n, a.dim, a.queries, 10
    rng = np.random.default_rng(42)
    R["config"] = {"n": N, "dim": DIM, "queries": NQ, "k": K, "cpus": os.cpu_count(),
                   "feather_version": getattr(fc, "__version__", "0.18.2")}
    print(f"== Feather full suite  N={N} dim={DIM} ==", flush=True)

    data = clustered(rng, N, DIM)
    qs = clustered(np.random.default_rng(7), NQ, DIM)
    vocab = [f"w{i}" for i in range(5000)]
    zipf = np.minimum(rng.zipf(1.3, size=(N, 12)) - 1, len(vocab) - 1)
    texts = [" ".join(vocab[j] for j in row) for row in zipf]
    NS = [f"ns{i % 100}" for i in range(N)]          # 100 namespaces -> 1% each
    ENT = [f"e{i % 5000}" for i in range(N)]         # 5000 entities -> 20 recs each
    COLOR = ["red", "green", "blue", "gold"]

    def meta(i):
        m = fc.Metadata()
        m.content = texts[i]; m.namespace_id = NS[i]; m.entity_id = ENT[i]
        m.type = fc.ContextType.FACT; m.importance = float((i % 10) / 10)
        m.timestamp = int(time.time()) - int(i % 365) * 86400
        m.set_attribute("color", COLOR[i % 4])
        return m

    metas = [meta(i) for i in range(N)]
    gc.collect()
    base_rss = rss_mb()

    # ---------------- 1. Ingestion ----------------
    sub = min(N, 20_000)
    p = tempfile.mktemp(suffix=".feather"); db = fc.DB.open(p, dim=DIM)
    t0 = time.perf_counter()
    for i in range(sub): db.add(id=i, vec=data[i])
    dt = time.perf_counter() - t0
    log("ingest", f"add() serial, no metadata ({sub})", {"seconds": dt, "vec_per_s": sub / dt})
    del db; rm(p)

    p = tempfile.mktemp(suffix=".feather"); db = fc.DB.open(p, dim=DIM)
    t0 = time.perf_counter()
    for i in range(sub): db.add(id=i, vec=data[i], meta=metas[i])
    dt = time.perf_counter() - t0
    log("ingest", f"add() serial, full metadata ({sub})", {"seconds": dt, "vec_per_s": sub / dt})
    del db; rm(p)

    path = tempfile.mktemp(suffix=".feather"); db = fc.DB.open(path, dim=DIM)
    t0 = time.perf_counter()
    db.add_batch(list(range(N)), data, metas)
    dt = time.perf_counter() - t0
    log("ingest", f"add_batch() parallel, full metadata ({N})", {"seconds": dt, "vec_per_s": N / dt})
    gc.collect()
    log("memory", "RSS delta after building main index (MB)", rss_mb() - base_rss)

    # multimodal pocket
    img = np.random.default_rng(3).standard_normal((min(N, 10_000), 64)).astype(np.float32)
    t0 = time.perf_counter()
    db.add_batch(list(range(len(img))), img, [metas[i] for i in range(len(img))], "visual")
    dt = time.perf_counter() - t0
    log("ingest", f"add_batch() 2nd modality 'visual' dim=64 ({len(img)})", {"seconds": dt, "vec_per_s": len(img) / dt})

    # ---------------- 2. Vector search ----------------
    gt = brute_topk(data, qs, K)
    for ef in (10, 50, 100, 200, 400):
        log("search", f"top-{K} ef={ef}", recall(db, qs, gt, K, ef=ef))
    db.set_ef(50)
    for k in (1, 50, 100):
        log("search", f"latency k={k} (ef=50)", time_each(lambda q: db.search(q, k=k, record_salience=False), qs[:500]))
    log("search", "k=10 WITH record_salience=True (default)", time_each(lambda q: db.search(q, k=10), qs[:500]))
    sres = db.search(qs[0], k=5, modality="visual")
    log("search", "modality='visual' k=10", time_each(lambda q: db.search(q[:64].copy(), k=10, modality="visual", record_salience=False), qs[:500]))

    # concurrent throughput (GIL released in C++)
    big = np.concatenate([qs] * 4)
    for th in (1, 2, 4, 8, 16, 32):
        if th > (os.cpu_count() or 1): break
        chunks = np.array_split(big, th)
        def work(c):
            for q in c: db.search(q, k=K, record_salience=False)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(th) as ex: list(ex.map(work, chunks))
        dt = time.perf_counter() - t0
        log("search_concurrency", f"{th} threads", {"qps": len(big) / dt})

    # ---------------- 3. Filtered + scored search ----------------
    def filt(**kw):
        f = fc.SearchFilter()
        for k_, v in kw.items(): setattr(f, k_, v)
        return f
    cases = {
        "namespace (1% selectivity)": filt(namespace_id="ns7"),
        "entity (0.02% selectivity)": filt(entity_id="e42"),
        "attribute color=red (25%)": filt(attributes_match={"color": "red"}),
        "importance>=0.9 (10%, scan filter)": filt(importance_gte=0.9),
        "type=FACT (100%)": filt(types=[fc.ContextType.FACT]),
    }
    for name, f in cases.items():
        log("filtered_search", name, time_each(lambda q: db.search(q, k=K, filter=f, record_salience=False), qs[:300]))
    sc = fc.ScoringConfig(30.0, 0.3, 0.0)
    log("filtered_search", "adaptive-decay ScoringConfig(30d, 0.3)", time_each(lambda q: db.search(q, k=K, scoring=sc, record_salience=False), qs[:300]))
    # completeness of pre-filtered results
    f = filt(namespace_id="ns7")
    full = sum(len(db.search(q, k=K, filter=f, record_salience=False)) == K for q in qs[:100])
    log("filtered_search", "namespace filter returns full top-k (fraction)", full / 100)

    # ---------------- 4. BM25 + hybrid ----------------
    kq = [" ".join(vocab[j] for j in np.minimum(rng.zipf(1.3, size=3) - 1, 4999)) for _ in range(500)]
    db.keyword_search(kq[0], k=K)  # build/warm BM25
    log("keyword", "keyword_search BM25 k=10", time_each(lambda s: db.keyword_search(s, k=K), kq))
    log("keyword", "keyword_search + namespace filter", time_each(lambda s: db.keyword_search(s, k=K, filter=filt(namespace_id="ns7")), kq[:300]))
    pairs = list(zip(qs[:500], kq))
    log("keyword", "hybrid_search (BM25+HNSW RRF) k=10", time_each(lambda pq: db.hybrid_search(pq[0], pq[1], k=K), pairs))

    # ---------------- 5. Graph ----------------
    L = a.links
    src = rng.integers(0, N, L); dst = rng.integers(0, N, L)
    t0 = time.perf_counter()
    for s_, d_ in zip(src.tolist(), dst.tolist()): db.link(s_, d_, "related_to", 0.8)
    dt = time.perf_counter() - t0
    log("graph", f"link() x{L}", {"seconds": dt, "links_per_s": L / dt})
    ids = rng.integers(0, N, 5000).tolist()
    log("graph", "get_edges(id)", time_each(db.get_edges, ids))
    log("graph", "get_incoming(id)", time_each(db.get_incoming, ids))
    for hops in (1, 2, 3):
        log("graph", f"context_chain k=5 hops={hops}", time_each(lambda q: db.context_chain(q, k=5, hops=hops), qs[:200]))
    t0 = time.perf_counter(); js = db.export_graph_json("ns7", ""); dt = time.perf_counter() - t0
    log("graph", "export_graph_json(namespace=ns7)", {"seconds": dt, "json_mb": len(js) / 2**20})
    # auto_link on a smaller DB (it's O(N * candidates))
    ap_ = tempfile.mktemp(suffix=".feather"); adb = fc.DB.open(ap_, dim=DIM)
    na = min(N, 20_000); adb.add_batch(list(range(na)), data[:na])
    # sim = 1/(1+L2^2): pick the threshold at the median nearest-neighbour sim so ~half link
    nn = [adb.search(data[i], k=2, record_salience=False)[1] for i in range(200)]
    thr = float(np.median([1.0 / (1.0 + float(((data[i] - data[r.id]) ** 2).sum())) for i, r in zip(range(200), nn)]))
    t0 = time.perf_counter(); created = adb.auto_link("text", thr, "related_to", 15); dt = time.perf_counter() - t0
    log("graph", f"auto_link over {na} records", {"threshold": thr, "seconds": dt, "edges_created": created, "records_per_s": na / dt})
    del adb; rm(ap_)

    # ---------------- 6. Metadata + secondary index ----------------
    log("metadata", "get_metadata(id)", time_each(db.get_metadata, ids))
    log("metadata", "get_vector(id)", time_each(db.get_vector, ids))
    log("metadata", "touch(id)", time_each(db.touch, ids))
    log("metadata", "update_importance(id)", time_each(lambda i: db.update_importance(i, 0.5), ids))
    log("metadata", "update_metadata(id, meta)", time_each(lambda i: db.update_metadata(i, metas[i]), ids[:2000]))
    log("secondary_index", "ids_in_namespace (1k hits)", time_each(db.ids_in_namespace, [f"ns{i}" for i in range(100)] * 5))
    log("secondary_index", "ids_for_entity (20 hits)", time_each(db.ids_for_entity, [f"e{i}" for i in range(2000)]))
    log("secondary_index", "ids_with_attribute color=red (25k hits)", time_each(lambda c: db.ids_with_attribute("color", c), COLOR * 25))
    log("secondary_index", "namespace_size", time_each(db.namespace_size, [f"ns{i}" for i in range(100)] * 20))
    log("secondary_index", "list_namespaces", time_each(lambda _: db.list_namespaces(), range(200)))

    # ---------------- 7. Persistence ----------------
    t0 = time.perf_counter(); db.save(); dt = time.perf_counter() - t0
    size = os.path.getsize(path) / 2**20
    log("persistence", "save() float32 + persisted graph (v9)", {"seconds": dt, "file_mb": size, "bytes_per_record": size * 2**20 / N})
    del db; gc.collect()
    r0 = rss_mb(); t0 = time.perf_counter(); db = fc.DB.open(path, dim=DIM); db.search(qs[0], k=K)
    dt = time.perf_counter() - t0
    log("persistence", "cold open() + first search (persisted graph)", {"seconds": dt, "records_per_s": N / dt})
    log("persistence", "recall after reload (ef=50)", recall(db, qs[:300], gt[:300], K, ef=50)[f"recall@{K}"])

    # quantized on-disk
    qp = tempfile.mktemp(suffix=".feather"); qdb = fc.DB.open(qp, dim=DIM); qdb.set_quantized("text", True)
    qdb.add_batch(list(range(N)), data)
    t0 = time.perf_counter(); qdb.save(); dt = time.perf_counter() - t0
    qsize = os.path.getsize(qp) / 2**20
    del qdb; gc.collect()
    t0 = time.perf_counter(); qdb = fc.DB.open(qp, dim=DIM); qdb.search(qs[0], k=K); dl = time.perf_counter() - t0
    rq = recall(qdb, qs[:300], gt[:300], K, ef=50)
    log("persistence", "on-disk int8 quantized (v7): save / file / load / recall",
        {"save_s": dt, "file_mb": qsize, "load_s": dl, f"recall@{K}": rq[f"recall@{K}"], "p50_ms": rq["p50_ms"]})
    del qdb; rm(qp)

    # int8 in RAM
    ip = tempfile.mktemp(suffix=".feather"); gc.collect(); r0 = rss_mb()
    idb = fc.DB.open(ip, dim=DIM); idb.set_int8_ram("text", float(np.abs(data).max()))
    t0 = time.perf_counter(); idb.add_batch(list(range(N)), data); dt = time.perf_counter() - t0
    int8_rss = rss_mb() - r0
    ri = recall(idb, qs[:300], gt[:300], K, ef=50)
    fp = tempfile.mktemp(suffix=".feather"); gc.collect(); r1 = rss_mb()
    fdb = fc.DB.open(fp, dim=DIM); fdb.add_batch(list(range(N)), data); f32_rss = rss_mb() - r1
    log("persistence", "in-RAM int8 vs float32 (vectors only)",
        {"int8_build_s": dt, "int8_rss_mb": int8_rss, "float32_rss_mb": f32_rss,
         "ram_ratio": f32_rss / max(int8_rss, 1e-9), f"int8_recall@{K}": ri[f"recall@{K}"], "int8_p50_ms": ri["p50_ms"]})
    del idb, fdb; rm(ip); rm(fp)

    # ---------------- 8. Deletion lifecycle ----------------
    kill = rng.choice(N, N // 10, replace=False).tolist()
    t0 = time.perf_counter()
    for i in kill: db.forget(i)
    dt = time.perf_counter() - t0
    log("lifecycle", f"forget() x{len(kill)}", {"seconds": dt, "ops_per_s": len(kill) / dt})
    log("lifecycle", "search latency with 10% tombstones", time_each(lambda q: db.search(q, k=K, record_salience=False), qs[:300]))
    t0 = time.perf_counter(); n_p = db.purge("ns3"); dt = time.perf_counter() - t0
    log("lifecycle", "purge(namespace ns3)", {"seconds": dt, "removed": n_p})
    t0 = time.perf_counter(); n_c = db.compact(); dt = time.perf_counter() - t0
    log("lifecycle", "compact() after 10% forget", {"seconds": dt, "removed": n_c})
    log("lifecycle", "search latency after compact", time_each(lambda q: db.search(q, k=K, record_salience=False), qs[:300]))
    t0 = time.perf_counter(); db.save(); dt = time.perf_counter() - t0
    log("lifecycle", "save() after compact", {"seconds": dt, "file_mb": os.path.getsize(path) / 2**20})
    del db; rm(path)

    with open(a.out, "w") as fh: json.dump(R, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
