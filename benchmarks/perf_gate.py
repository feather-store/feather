"""Performance gate: the measurements behind recent speedups, as checks.

    python benchmarks/perf_gate.py [--json out.json]

Runs in about a minute and exits non-zero if a win has regressed. CI machines
differ in speed, so each check is a count or a ratio measured on the same
machine in the same run, never an absolute number from a laptop. The few
absolute budgets are an order of magnitude loose: they catch an accidental
O(n^2), not noise.

  embed cache   a repeated text never reaches the provider again
  lazy BM25     opening a file with text costs about what one without does
  keep-alive    RemoteDB makes one connection for many calls
  search        vector / filtered search and a cold Pocket.hot() stay in budget
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import feather_db  # noqa: E402
from feather_db.integrations import embedders  # noqa: E402
from feather_db.pocket import pocket  # noqa: E402
from feather_db.remote import RemoteDB  # noqa: E402

RESULTS: list[dict] = []


def check(name: str, value: float, unit: str, ok: bool, rule: str) -> None:
    RESULTS.append({"name": name, "value": round(value, 3), "unit": unit,
                    "ok": bool(ok), "rule": rule})
    mark = "\033[32mok  \033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  {mark} {name:38} {value:10.3f} {unit:8} ({rule})")


def median_ms(fn, n: int) -> float:
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


# ── embed cache ─────────────────────────────────────────────────────────────
def bench_embed_cache() -> None:
    calls = []

    def slow_provider(text, model, key, dim, base_url):
        calls.append(text)
        time.sleep(0.02)                     # an API call, scaled down
        return [1.0] * dim

    embedders._DRIVERS["_gate"] = slow_provider
    embedders._DEFAULT_MODELS["_gate"] = "m"
    try:
        embed = embedders.make_embedder("_gate", api_key="k", dim=8)
        texts = [f"text {i}" for i in range(10)]
        t = time.perf_counter()
        for t_ in texts:
            embed(t_)
        miss = (time.perf_counter() - t) / len(texts) * 1e3
        t = time.perf_counter()
        for _ in range(100):
            for t_ in texts:
                embed(t_)
        hit = (time.perf_counter() - t) / 1000 * 1e3
    finally:
        del embedders._DRIVERS["_gate"], embedders._DEFAULT_MODELS["_gate"]
    check("embed: provider calls for 1010 embeds", len(calls), "calls",
          len(calls) == 10, "== 10 distinct texts")
    check("embed: cached / uncached speedup", miss / hit, "x", miss / hit > 100, "> 100x")


# ── lazy BM25 ───────────────────────────────────────────────────────────────
def _make(path: str, n: int, dim: int, text: bool) -> None:
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(20000)]
    p = 1.0 / np.arange(1, len(words) + 1)
    toks = rng.choice(len(words), size=(n, 50), p=p / p.sum())
    metas = []
    for i in range(n):
        m = feather_db.Metadata()
        m.namespace_id = f"ns{i % 4}"
        if text:
            m.content = " ".join(words[t] for t in toks[i])
        metas.append(m)
    db = feather_db.DB.open(path, dim=dim)
    db.add_batch(list(range(n)), rng.random((n, dim), dtype=np.float32), metas)
    db.close()


def _open_ms(path: str, dim: int) -> float:
    def once():
        feather_db.DB.open(path, dim=dim, read_only=True).close()
    return median_ms(once, 3)


def bench_open(d: str) -> None:
    n, dim = 20000, 128
    plain, text = os.path.join(d, "plain.feather"), os.path.join(d, "text.feather")
    _make(plain, n, dim, text=False)
    _make(text, n, dim, text=True)
    o_plain, o_text = _open_ms(plain, dim), _open_ms(text, dim)
    check("open: 20k plain", o_plain, "ms", True, "reported")
    check("open: 20k with text / plain", o_text / o_plain, "x", o_text / o_plain < 2.5,
          "< 2.5x (12x when BM25 was built at open)")

    db = feather_db.DB.open(text, dim=dim, read_only=True)
    t = time.perf_counter()
    db.keyword_search("w5 w17", k=10)
    first = (time.perf_counter() - t) * 1e3
    later = median_ms(lambda: db.keyword_search("w5 w17", k=10), 20)
    check("bm25: first keyword search (builds)", first, "ms", True, "reported")
    check("bm25: later keyword search", later, "ms", later < 50, "< 50 ms")

    q = np.random.default_rng(1).random(dim, dtype=np.float32)
    f = feather_db.FilterBuilder().namespace("ns1").build()
    s = median_ms(lambda: db.search(q, k=10, record_salience=False), 50)
    fs = median_ms(lambda: db.search(q, k=10, filter=f, record_salience=False), 50)
    check("search: 20k x 128 unfiltered", s, "ms", s < 5, "< 5 ms")
    check("search: 20k x 128 namespace filter", fs, "ms", fs < 25, "< 25 ms")
    db.close()


# ── Pocket.hot() ────────────────────────────────────────────────────────────
def bench_hot(d: str) -> None:
    db = feather_db.DB.open(os.path.join(d, "pocket.feather"), dim=64)
    pk = pocket(db, ("org", "brand", "agent"), budget_tokens=4000)
    for i in range(10000):
        pk.remember(f"k{i}", "captions stay lowercase for the brand voice")
    pk._hot_cache = {}
    t = time.perf_counter()
    pk.hot()
    cold = (time.perf_counter() - t) * 1e3
    warm = median_ms(pk.hot, 50)
    check("hot(): 10k cold", cold, "ms", cold < 1000, "< 1000 ms")
    check("hot(): warm (memoised)", warm, "ms", warm < 1, "< 1 ms")
    db.close()


# ── RemoteDB keep-alive ─────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        body = b'{"size": 1}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def bench_keepalive() -> None:
    accepted = []

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def get_request(self):
            conn = super().get_request()
            accepted.append(conn)
            return conn

    for v in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(v, None)
    srv = Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        db = RemoteDB(f"http://127.0.0.1:{srv.server_address[1]}", "ns")
        call = lambda: db._call("GET", "/stats")  # noqa: E731
        kept = median_ms(call, 200)
        conns = len(accepted)
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1/ns/stats"
        fresh = median_ms(lambda: db._session._urlopen("GET", url, None, {}), 200)
        db.close(save=False)
    finally:
        srv.shutdown()
        srv.server_close()
    check("remote: connections for 201 calls", conns, "conns", conns == 1, "== 1")
    # Reported only: on localhost a handshake is ~0.1 ms, inside the noise.
    # The connection count above is the check.
    check("remote: new-conn / kept-alive latency", fresh / kept, "x", True, "reported")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        print("embed cache");  bench_embed_cache()
        print("open / BM25 / search");  bench_open(d)
        print("pocket");  bench_hot(d)
        print("remote");  bench_keepalive()
    failed = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)} ok, {len(failed)} failed "
          f"in {time.perf_counter() - t0:.0f} s")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"results": RESULTS, "time": int(time.time())}, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
