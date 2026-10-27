"""
Read-only latency benchmark against a DEPLOYED Feather Cloud API.

Safe for shared/production servers: it never writes. Only GET endpoints,
/keyword_search, and /search with track=false (skipped entirely if the server's
OpenAPI schema has no `track` field, since then every search would bump recall
counters on real data). No context_chain (it touches salience). Load is light:
sequential requests plus at most --max-clients concurrent readers.

    python benchmarks/remote_readonly_bench.py --base http://localhost:8010 \
        --namespace work_hawky [--api-key $FEATHER_API_KEY] --out remote.json
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import httpx


def pct(xs):
    a = np.asarray(xs)
    return {"n": len(xs), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "req_per_s": 1e3 / float(a.mean())}


def timed(c, n, fn):
    lat, codes = [], {}
    for i in range(n):
        s = time.perf_counter(); r = fn(c, i); lat.append((time.perf_counter() - s) * 1e3)
        codes[r.status_code] = codes.get(r.status_code, 0) + 1
    out = pct(lat); out["status_codes"] = codes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8010")
    ap.add_argument("--namespace", default="work_hawky")
    ap.add_argument("--api-key", default=os.getenv("FEATHER_API_KEY", ""))
    ap.add_argument("--n", type=int, default=100, help="requests per endpoint")
    ap.add_argument("--max-clients", type=int, default=4)
    ap.add_argument("--out", default="remote_readonly.json")
    a = ap.parse_args()
    H = {"X-API-Key": a.api_key} if a.api_key else {}
    c = httpx.Client(base_url=a.base, headers=H, timeout=30)
    NS = a.namespace
    R = {"base": a.base, "namespace": NS}

    spec = c.get("/openapi.json").json()
    R["server_version"] = spec.get("info", {}).get("version")
    schemas = spec.get("components", {}).get("schemas", {})
    has_track = "track" in schemas.get("SearchRequest", {}).get("properties", {})
    R["search_supports_track"] = has_track
    print("server", R["server_version"], "| /search track=false supported:", has_track, flush=True)

    stats = c.get(f"/v1/namespaces/{NS}/stats").json()
    R["namespace_stats"] = stats
    dim = int(stats.get("dim") or 0)
    print("namespace stats:", {k: stats.get(k) for k in ("record_count", "dim")}, flush=True)

    rec = c.get(f"/v1/{NS}/records", params={"limit": 50}).json().get("results", [])
    ids = [r["id"] for r in rec] or [1]
    words = " ".join((r.get("metadata", {}) or {}).get("content", "") for r in rec).split()
    kq = [" ".join(np.random.default_rng(i).choice(words, 3)) for i in range(50)] if len(words) >= 3 else ["test"]

    ep = {
        "GET /health": lambda c, i: c.get("/health"),
        "GET /v1/namespaces": lambda c, i: c.get("/v1/namespaces"),
        f"GET /v1/namespaces/{NS}/stats": lambda c, i: c.get(f"/v1/namespaces/{NS}/stats"),
        "GET /records/{id}": lambda c, i: c.get(f"/v1/{NS}/records/{ids[i % len(ids)]}"),
        "GET /records?limit=100": lambda c, i: c.get(f"/v1/{NS}/records", params={"limit": 100}),
        "GET /records/{id}/edges": lambda c, i: c.get(f"/v1/{NS}/records/{ids[i % len(ids)]}/edges"),
        "POST /keyword_search k=10": lambda c, i: c.post(f"/v1/{NS}/keyword_search", json={"query": kq[i % len(kq)], "k": 10}),
    }
    Q = []
    if has_track and dim:
        Q = np.random.default_rng(0).standard_normal((50, dim)).astype(np.float32).tolist()
        ep["POST /search k=10 track=false"] = lambda c, i: c.post(f"/v1/{NS}/search", json={"vector": Q[i % 50], "k": 10, "track": False})
    else:
        print("skipping /search: server has no track flag, searches would mutate recall counters", flush=True)

    R["endpoints"] = {}
    for name, fn in ep.items():
        n = min(a.n, 30) if "stats" in name or "namespaces" in name else a.n
        R["endpoints"][name] = timed(c, n, fn)
        e = R["endpoints"][name]
        print(f"{name:40s} p50={e['p50_ms']:.2f}ms p95={e['p95_ms']:.2f}ms p99={e['p99_ms']:.2f}ms codes={e['status_codes']}", flush=True)

    # light concurrency on the cheapest read path
    target = (lambda cc, i: cc.post(f"/v1/{NS}/search", json={"vector": Q[i % 50], "k": 10, "track": False})) if Q \
        else (lambda cc, i: cc.get(f"/v1/{NS}/records/{ids[i % len(ids)]}"))
    R["concurrency"] = {}
    for th in sorted({1, 2, a.max_clients}):
        per = 200 // th
        def work(k):
            with httpx.Client(base_url=a.base, headers=H, timeout=30) as cc:
                for i in range(per): target(cc, k * per + i)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(th) as ex: list(ex.map(work, range(th)))
        R["concurrency"][f"{th} clients"] = per * th / (time.perf_counter() - t0)
        print(f"concurrency {th} clients: {R['concurrency'][f'{th} clients']:.0f} req/s", flush=True)

    with open(a.out, "w") as fh: json.dump(R, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
