"""
Feather Cloud API (feather-api/) end-to-end HTTP benchmark.

Starts uvicorn on a throwaway data dir, bulk-imports N records over HTTP, then
measures per-endpoint latency (single client) and /search throughput under
concurrent clients. Uses track=false so reads don't mutate what they measure.

    python benchmarks/api_bench.py --n 20000 --dim 128 --out api.json
"""
import argparse, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import httpx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = {}


def pct(xs):
    a = np.asarray(xs)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "req_per_s": 1e3 / float(a.mean())}


def timed(c, n, fn):
    lat = []
    for i in range(n):
        s = time.perf_counter(); r = fn(c, i); lat.append((time.perf_counter() - s) * 1e3)
        assert r.status_code < 300, r.text[:300]
    return pct(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default="api_bench.json")
    a = ap.parse_args()
    N, DIM, NS = a.n, a.dim, "bench"
    data_dir = tempfile.mkdtemp()
    env = dict(os.environ, FEATHER_DATA_DIR=data_dir, FEATHER_DB_DIM=str(DIM), FEATHER_DEV_MODE="1",
               PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("FEATHER_API_KEY", None)
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(a.port),
                            "--log-level", "warning"], cwd=os.path.join(HERE, "feather-api"), env=env)
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(base + "/docs", timeout=1).status_code == 200: break
            except Exception: time.sleep(0.2)
        rng = np.random.default_rng(0)
        V = rng.standard_normal((N, DIM)).astype(np.float32)
        words = [f"w{i}" for i in range(2000)]
        c = httpx.Client(base_url=base, timeout=120)

        # bulk import
        t0 = time.perf_counter()
        for s in range(0, N, 1000):
            items = [{"id": i + 1, "vector": V[i].tolist(),
                      "metadata": {"content": " ".join(rng.choice(words, 10)), "namespace_id": f"ns{i % 20}",
                                   "attributes": {"color": ["red", "blue"][i % 2]}}}
                     for i in range(s, min(N, s + 1000))]
            r = c.post(f"/v1/{NS}/import", json={"items": items})
            assert r.status_code < 300, r.text[:300]
        dt = time.perf_counter() - t0
        R["import"] = {"records": N, "seconds": dt, "records_per_s": N / dt, "batch": 1000}
        print("import", R["import"], flush=True)

        Q = rng.standard_normal((500, DIM)).astype(np.float32).tolist()
        kq = [" ".join(rng.choice(words, 3)) for _ in range(500)]
        ep = {
            "POST /vectors (single add)": lambda c, i: c.post(f"/v1/{NS}/vectors", json={"id": N + 10 + i, "vector": Q[i % 500], "metadata": {"content": "x"}}),
            "POST /search k=10": lambda c, i: c.post(f"/v1/{NS}/search", json={"vector": Q[i % 500], "k": 10, "track": False}),
            "POST /search k=10 track=true": lambda c, i: c.post(f"/v1/{NS}/search", json={"vector": Q[i % 500], "k": 10}),
            "POST /keyword_search k=10": lambda c, i: c.post(f"/v1/{NS}/keyword_search", json={"query": kq[i % 500], "k": 10}),
            "POST /hybrid_search k=10": lambda c, i: c.post(f"/v1/{NS}/hybrid_search", json={"vector": Q[i % 500], "query": kq[i % 500], "k": 10}),
            "GET /records/{id}": lambda c, i: c.get(f"/v1/{NS}/records/{(i * 37) % N + 1}"),
            "POST /records/{id}/link": lambda c, i: c.post(f"/v1/{NS}/records/{(i * 37) % N + 1}/link", json={"to_id": (i * 91) % N + 1}),
            "GET /records/{id}/edges": lambda c, i: c.get(f"/v1/{NS}/records/{(i * 37) % N + 1}/edges"),
            "POST /context_chain k=5 hops=2": lambda c, i: c.post(f"/v1/{NS}/context_chain", json={"vector": Q[i % 500], "k": 5, "hops": 2}),
            "GET /records?limit=100": lambda c, i: c.get(f"/v1/{NS}/records", params={"limit": 100}),
            "GET /namespaces/{ns}/stats": lambda c, i: c.get(f"/v1/namespaces/{NS}/stats"),
        }
        R["endpoints"] = {}
        for name, fn in ep.items():
            n = 100 if "stats" in name or "limit" in name else 300
            R["endpoints"][name] = timed(c, n, fn)
            print(name, {k: round(v, 3) for k, v in R["endpoints"][name].items()}, flush=True)

        t0 = time.perf_counter(); r = c.post(f"/v1/{NS}/save"); R["save_seconds"] = time.perf_counter() - t0

        R["search_concurrency"] = {}
        for th in (1, 4, 16, 32):
            per = 1600 // th
            def work(k):
                with httpx.Client(base_url=base, timeout=120) as cc:
                    for i in range(per):
                        cc.post(f"/v1/{NS}/search", json={"vector": Q[(k * per + i) % 500], "k": 10, "track": False})
            t0 = time.perf_counter()
            with ThreadPoolExecutor(th) as ex: list(ex.map(work, range(th)))
            R["search_concurrency"][f"{th} clients"] = per * th / (time.perf_counter() - t0)
            print("concurrency", th, round(R["search_concurrency"][f"{th} clients"]), flush=True)
    finally:
        srv.terminate(); srv.wait()
    with open(a.out, "w") as fh: json.dump(R, fh, indent=1)


if __name__ == "__main__":
    main()
