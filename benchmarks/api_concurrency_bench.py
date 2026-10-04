"""
Cloud API contention benchmark (Phase 0 of CONCURRENCY_FIX_PLAN.md).

Starts the real feather-api under uvicorn (1 worker, the shipped config) and
measures /search latency on its own, then while other traffic runs:
single-record writes, server-side-embedded ingests (against a fake embedding
provider with a controllable delay), deletes and bulk imports.

Load is generated from several processes, so the client's own GIL isn't the
bottleneck. Search uses track=false.

    python benchmarks/api_concurrency_bench.py --n 20000 --dim 128 --out runs/api.json

The fake provider mimics Ollama's /api/embeddings, so no product code changes.
"""
import argparse, base64, json, multiprocessing as mp, os, random, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import numpy as np

try:
    import psutil
except ImportError:  # optional: server CPU sampling
    psutil = None

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = "bench"
EMBED_DELAY = {"s": 0.0}           # mutable: changed between scenarios
CALLS = {"n": 0, "texts": 0}       # provider calls seen by the fake embedder


# ─────────────────────────────────────────────────────────────────────────────
# fake embedding provider (Ollama-compatible)
# ─────────────────────────────────────────────────────────────────────────────
def start_fake_embedder(port, dim):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            time.sleep(EMBED_DELAY["s"])                      # one delay per HTTP call
            if self.path.endswith("/api/embed"):              # batch endpoint (Ollama >= 0.3)
                texts = req.get("input") or []
                texts = [texts] if isinstance(texts, str) else texts
                CALLS["n"] += 1; CALLS["texts"] += len(texts)
                body = json.dumps({"embeddings": [[random.gauss(0, 1) for _ in range(dim)]
                                                   for _ in texts]}).encode()
            else:                                             # legacy single endpoint
                CALLS["n"] += 1; CALLS["texts"] += 1
                body = json.dumps({"embedding": [random.gauss(0, 1) for _ in range(dim)]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ─────────────────────────────────────────────────────────────────────────────
# load generator (runs in child processes)
# ─────────────────────────────────────────────────────────────────────────────
def _worker(kind, base, dim, n_records, proc_idx, n_threads, seconds, out_q, go):
    rng = np.random.default_rng(1000 + proc_idx)
    Qa = rng.standard_normal((256, dim)).astype(np.float32)
    Q = Qa.tolist()
    QB = [base64.b64encode(v.astype("<f4").tobytes()).decode() for v in Qa]
    lat, codes, lock = [], {}, threading.Lock()

    def thread(t):
        c = httpx.Client(base_url=base, timeout=60)
        mine, my_codes, i = [], {}, 0
        uid = 50_000_000 + proc_idx * 1_000_000 + t * 50_000
        go.wait()
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            i += 1
            try:
                s = time.perf_counter()
                if kind == "search":
                    r = c.post(f"/v1/{NS}/search", json={"vector": Q[i % 256], "k": 10, "track": False})
                elif kind == "search_fast":      # 0.19: binary vector, ids+scores only
                    r = c.post(f"/v1/{NS}/search", json={"vector_b64": QB[i % 256], "k": 10,
                                                          "track": False, "include_metadata": False})
                elif kind == "write":
                    r = c.post(f"/v1/{NS}/vectors", json={"id": uid + i, "vector": Q[i % 256],
                                                           "metadata": {"content": f"write {i} alpha beta"}})
                elif kind == "ingest":
                    r = c.post(f"/v1/{NS}/ingest_text", json={"text": f"ingested note {proc_idx}-{t}-{i} gamma"})
                elif kind == "delete":
                    rid = 1 + ((proc_idx * 7919 + t * 104729 + i * 31) % n_records)
                    r = c.delete(f"/v1/{NS}/records/{rid}")
                elif kind == "import":
                    items = [{"id": uid + i * 1000 + j, "vector": Q[(i + j) % 256],
                              "metadata": {"content": f"bulk {i} {j} delta"}} for j in range(1000)]
                    r = c.post(f"/v1/{NS}/import", json={"items": items})
                mine.append((time.perf_counter() - s) * 1e3)
                my_codes[r.status_code] = my_codes.get(r.status_code, 0) + 1
            except Exception as e:  # noqa: BLE001  (timeouts count as errors)
                my_codes[type(e).__name__] = my_codes.get(type(e).__name__, 0) + 1
        with lock:
            lat.extend(mine)
            for k, v in my_codes.items():
                codes[k] = codes.get(k, 0) + v

    ts = [threading.Thread(target=thread, args=(t,)) for t in range(n_threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    out_q.put((kind, lat, codes))


def stats(lat, seconds):
    if not lat:
        return {"ops": 0}
    a = np.asarray(lat)
    return {"ops": int(a.size), "ops_per_s": round(a.size / seconds, 1),
            "p50_ms": round(float(np.percentile(a, 50)), 2), "p90_ms": round(float(np.percentile(a, 90)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2), "max_ms": round(float(a.max()), 1),
            "over_100ms": int((a > 100).sum()), "over_1s": int((a > 1000).sum())}


def run_scenario(base, dim, n_records, loads, seconds, server_pid):
    """loads: {kind: n_clients}. Clients are spread over processes of <=8 threads."""
    ctx = mp.get_context("fork")
    q, go, procs, pidx = ctx.Queue(), ctx.Event(), [], 0
    for kind, clients in loads.items():
        while clients > 0:
            th = min(8, clients); clients -= th
            p = ctx.Process(target=_worker, args=(kind, base, dim, n_records, pidx, th, seconds, q, go))
            p.start(); procs.append(p); pidx += 1
    time.sleep(1.0)                                   # let clients connect
    sp = psutil.Process(server_pid) if psutil else None
    if sp: sp.cpu_percent(None)
    go.set()
    time.sleep(seconds)
    cpu = sp.cpu_percent(None) if sp else None
    merged = {}
    for _ in procs:
        kind, lat, codes = q.get(timeout=seconds + 120)
        m = merged.setdefault(kind, {"lat": [], "codes": {}})
        m["lat"].extend(lat)
        for k, v in codes.items():
            m["codes"][str(k)] = m["codes"].get(str(k), 0) + v
    for p in procs: p.join()
    out = {k: {**stats(v["lat"], seconds), "status": v["codes"]} for k, v in merged.items()}
    out["server_cpu_pct"] = cpu
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--embed-port", type=int, default=8792)
    ap.add_argument("--slow-embed-ms", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    start_fake_embedder(a.embed_port, a.dim)
    data_dir = tempfile.mkdtemp()
    env = dict(os.environ, FEATHER_DATA_DIR=data_dir, FEATHER_DB_DIM=str(a.dim), FEATHER_DEV_MODE="1",
               FEATHER_EMBED_PROVIDER="ollama", FEATHER_EMBED_MODEL="fake",
               FEATHER_EMBED_BASE_URL=f"http://127.0.0.1:{a.embed_port}", FEATHER_EMBED_DIM=str(a.dim),
               PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("FEATHER_API_KEY", None)
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(a.port),
                            "--workers", "1", "--log-level", "warning"],
                           cwd=os.path.join(HERE, "feather-api"), env=env)
    base = f"http://127.0.0.1:{a.port}"
    R = {"_meta": {"n": a.n, "dim": a.dim, "seconds": a.seconds, "slow_embed_ms": a.slow_embed_ms,
                   "workers": 1, "cpus": os.cpu_count(), "wal_sync": os.environ.get("FEATHER_WAL_SYNC", "1")}}
    try:
        for _ in range(150):
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200: break
            except Exception: time.sleep(0.2)
        rng = np.random.default_rng(0)
        c = httpx.Client(base_url=base, timeout=300)
        t0 = time.perf_counter()
        for s in range(0, a.n, 1000):
            items = [{"id": i + 1, "vector": rng.standard_normal(a.dim).astype(np.float32).tolist(),
                      "metadata": {"content": f"record {i} topic{i % 50}", "namespace_id": f"ns{i % 20}"}}
                     for i in range(s, min(a.n, s + 1000))]
            r = c.post(f"/v1/{NS}/import", json={"items": items}); assert r.status_code < 300, r.text[:300]
        c.post(f"/v1/{NS}/flush")
        R["_meta"]["import_s"] = round(time.perf_counter() - t0, 1)
        print("imported", a.n, "in", R["_meta"]["import_s"], "s", flush=True)

        def sc(name, loads, delay_ms=0.0):
            EMBED_DELAY["s"] = delay_ms / 1000
            c0, t0_ = CALLS["n"], CALLS["texts"]
            res = run_scenario(base, a.dim, a.n, loads, a.seconds, srv.pid)
            res["loads"] = loads
            if delay_ms: res["embed_delay_ms"] = delay_ms
            if CALLS["n"] > c0:
                res["embed_provider_calls"] = CALLS["n"] - c0
                res["embed_texts"] = CALLS["texts"] - t0_
            R[name] = res
            print(f"  {name:<40}", "  ".join(
                f"{k}: {v['ops_per_s']}/s p50={v['p50_ms']} p99={v['p99_ms']} max={v['max_ms']} {v['status']}"
                for k, v in res.items() if isinstance(v, dict) and "ops_per_s" in v),
                f"cpu={res['server_cpu_pct']}", flush=True)

        sc("search_c1", {"search": 1})
        sc("search_c16", {"search": 16})
        sc("search_c64", {"search": 64})
        sc("search_fast_c16", {"search_fast": 16})       # vector_b64 + include_metadata=false
        sc("search_c16_plus_write_c8", {"search": 16, "write": 8})
        sc("search_c16_plus_ingest_c64_fast_embed", {"search": 16, "ingest": 64}, delay_ms=5)
        sc("search_c16_plus_ingest_c64_slow_embed", {"search": 16, "ingest": 64}, delay_ms=a.slow_embed_ms)
        sc("search_c16_plus_delete_c2", {"search": 16, "delete": 2})
        sc("search_c16_plus_import_c1", {"search": 16, "import": 1})
        R["_meta"]["final_stats"] = c.get(f"/v1/namespaces/{NS}/stats").json()
    finally:
        srv.terminate()
        try: srv.wait(30)
        except Exception: srv.kill()
    with open(a.out, "w") as f:
        json.dump(R, f, indent=2)


if __name__ == "__main__":
    main()
