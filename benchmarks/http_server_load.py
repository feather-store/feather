"""Fixed-arrival-rate load test: feather-api (Python) vs feather-server (Rust).

For each server and each offered rate, feather-loadgen schedules requests at
exact times (open loop, latency from the scheduled time) and this script
samples the server process's CPU time around the run. The answer to "is CPU
per request what limits it" is in two columns: CPU ms per request, and the
rate at which the server stops keeping up.

    .venv/bin/python benchmarks/http_server_load.py \
        [--n 20000] [--dim 768] [--duration 15] [--rounds 2] \
        [--rates 250,500,1000,1500,2000,3000,4000,6000,8000] \
        [--servers python,rust] [--route search] [--out bench/results/x.json]

`--route agent_turn` replays what an agent turn sends instead of one route:
an agent namespace is seeded through the SDK on the Python server
(`--episodes`, over 10 users), one turn is recorded as RemoteDB sends it
(AgentMemory.recall, Pocket.hot, Pocket.recall, PacketBuilder.build and an
episode write: 18-20 requests over 9 routes), and the turn is replayed with a
new episode key each time. Rates are requests/s; turns/s is that / the
recorded turn's length.

Needs: cd feather-server && cargo build --release. Same machine for client and
server, so the loadgen's own CPU (2 threads) competes with the server: say so
when quoting. Laptop numbers swing; rounds alternate the servers.
"""
import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import feather_db  # noqa: E402

RUST = os.path.join(ROOT, "feather-server", "target", "release", "feather-server")
LOADGEN = os.path.join(ROOT, "feather-server", "target", "release", "feather-loadgen")
NS = "bench"
WORDS = ("agent memory brand creative caption tone weekly report shipped late premium "
         "user prefers written async dark mode instagram budget launch campaign").split()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def build_dataset(path, n, dim):
    """n unit vectors with ~300-char agent text, a namespace, an entity and
    two attributes each: the shape of an agent-memory record."""
    rng = np.random.default_rng(42)
    db = feather_db.DB.open(path, dim=dim)
    now = int(time.time())
    step = 2000
    for start in range(0, n, step):
        ids = list(range(start + 1, min(n, start + step) + 1))
        vecs = rng.standard_normal((len(ids), dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        metas = []
        for i in ids:
            m = feather_db.Metadata()
            m.timestamp = now - (i % 365) * 86400
            m.importance = 1.0
            m.source = "agent"
            m.content = " ".join(rng.choice(WORDS, 45))[:300]
            m.namespace_id = f"org.user_{i % 100}"
            m.entity_id = f"user_{i % 100}"
            m.set_attribute("kind", "note")
            m.set_attribute("channel", ("ig", "fb", "web")[i % 3])
            metas.append(m)
        db.add_batch(ids, vecs, metas)
    db.save()
    db.close()


def write_bodies(path, route, dim, n=2000):
    rng = np.random.default_rng(7)
    with open(path, "w") as f:
        for _ in range(n):
            v = rng.standard_normal(dim).astype("<f4")
            v /= np.linalg.norm(v)
            b64 = base64.b64encode(v.tobytes()).decode()
            if route == "search":
                body = {"vector_b64": b64, "k": 10}          # what RemoteDB.search sends
            elif route == "hybrid":
                body = {"vector_b64": b64, "query": " ".join(rng.choice(WORDS, 3)), "k": 10}
            elif route == "keyword":
                body = {"query": " ".join(rng.choice(WORDS, 3)), "k": 10}
            else:
                raise SystemExit(f"unknown route {route}")
            f.write(json.dumps(body) + "\n")


ROUTE_PATH = {"search": "search", "hybrid": "hybrid_search", "keyword": "keyword_search", "agent_turn": ""}
AGENT_NS = "agents"


def build_agent_dataset(work, episodes, dim):
    """Seed an agent namespace through RemoteDB on the Python server, the way
    agents write it, and record one agent turn's requests exactly as sent.
    Returns (the namespace's file, the turn as [{method, path, body}])."""
    import urllib.parse
    sys.path.insert(0, os.path.join(ROOT, "feather-server", "tests"))
    import agent_session as a
    from feather_db import PacketBuilder
    from feather_db.integrations.agent_memory import AgentMemory
    from feather_db.integrations.langgraph_store import FeatherStore
    from feather_db.pocket import Pocket
    from feather_db.remote import RemoteDB, _json_dumps
    a.DIM = dim                      # the embedder's size (agent_session.embed)

    class Recorder(a.Tap):
        rec = None

        def request(self, method, url, json=None, params=None, headers=None):
            if self.rec is not None:
                q = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
                self.rec.append({"method": method, "path": urllib.parse.urlsplit(url).path + (f"?{q}" if q else ""),
                                 "body": _json_dumps(json) if json is not None else None})
            return super().request(method, url, json=json, params=params, headers=headers)

    data = os.path.join(work, "agent-seed")
    shutil.rmtree(data, ignore_errors=True)
    os.makedirs(data)
    port = free_port()
    p = start_server("python", data, port)
    try:
        tap = Recorder([])
        db = RemoteDB(f"http://127.0.0.1:{port}", AGENT_NS, session=tap)
        db.ensure_namespace(a.DIM)
        store = FeatherStore(db=db, embed=a.embed)
        pkt = Pocket(db, ("acme", "nike", "creative"), embed=a.embed)
        builder = PacketBuilder(pkt, budget_tokens=2000)
        users = [AgentMemory(store, org="acme", user=f"u{u}", agent="creative", brand="nike") for u in range(10)]
        for i in range(episodes):
            m = users[i % 10]
            m.episodes.record(f"shipped weekly report {i} for client {i % 37}", outcome="ok" if i % 3 else "late")
            if i % 4 == 0:
                pkt.remember(f"note{i}", f"captions stay lowercase, variant {i}")
            if i % 25 == 0:
                m.preferences.set("comms", f"prefers written async, v{i}")
                m.facts.state("positioning", f"premium, take {i}")
        mem = users[3]
        tap.rec = []
        mem.recall("how to contact", kinds=["preferences"])
        pkt.hot()
        pkt.recall("lowercase captions", k=5)
        builder.build(query="which caption style")
        mem.episodes.record("turn done: caption shipped", outcome="ok")
        turn, tap.rec = tap.rec, None
        store.close()
        db.close()
    finally:
        p.terminate()
        p.wait(timeout=60)
    return os.path.join(data, f"{AGENT_NS}.feather"), turn


def write_turns(path, turn, n=4000):
    """The recorded turn n times, each writing a new episode: its key (in the
    lookup paths and the batch) gets the turn's number."""
    key = json.loads(turn[-1]["body"])["ops"][0]["metadata"]["key"]
    with open(path, "w") as f:
        for i in range(n):
            for r in turn:
                f.write(json.dumps({k: (v.replace(key, f"{key}_{i}") if isinstance(v, str) else v)
                                    for k, v in r.items()}) + "\n")


def start_server(kind, data_dir, port):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FEATHER_")}
    env.update(FEATHER_DATA_DIR=data_dir, FEATHER_DEV_MODE="1", FEATHER_CHECKPOINT_POLL_S="3600")
    if kind == "python":
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port),
               "--workers", "1", "--log-level", "warning", "--no-access-log"]
        env["PYTHONPATH"] = ROOT
        cwd = os.path.join(ROOT, "feather-api")
    else:
        cmd = [RUST, "--host", "127.0.0.1", "--port", str(port)]
        cwd = ROOT
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit(f"{kind} exited: {p.stderr.read().decode()[-2000:]}")
            time.sleep(0.1)
    p.kill()
    raise SystemExit(f"{kind} did not start")


def cpu_seconds(proc):
    """User + system CPU of the server and any children it spawned."""
    total = 0.0
    for p in [proc] + proc.children(recursive=True):
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.NoSuchProcess:
            pass
    return total


def loadgen(url, bodies, rate, duration, conns, route="search"):
    # An agent turn's lookups of the episode it is about to write answer 404:
    # that is the normal path, not an error.
    feed = (["--requests", bodies, "--ok-statuses", "200,201,404"] if route == "agent_turn"
            else ["--bodies", bodies])
    out = subprocess.run([LOADGEN, "--url", url, *feed, "--rate", str(rate),
                          "--duration", str(duration), "--conns", str(conns), "--threads", "2",
                          "--timeout", "10"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def run_server(kind, base, work, bodies, route, rates, duration, conns, per_turn=0):
    data = os.path.join(work, f"data-{kind}")
    shutil.rmtree(data, ignore_errors=True)
    os.makedirs(data)
    ns = AGENT_NS if route == "agent_turn" else NS
    shutil.copy(base, os.path.join(data, f"{ns}.feather"))
    port = free_port()
    p = start_server(kind, data, port)
    proc = psutil.Process(p.pid)
    url = f"http://127.0.0.1:{port}/v1/{NS}/{ROUTE_PATH[route]}"
    if route == "agent_turn":
        url = f"http://127.0.0.1:{port}/"   # host only: each request carries its path
    rows = []
    try:
        # Opens the namespace, builds BM25 for keyword routes, and gets the
        # cores clocked up before anything is measured (an idle laptop core
        # turned a 1 ms p99 into 4-21 ms in an earlier report).
        loadgen(url, bodies, min(rates[0] * 2, 1000), 5, conns, route)
        for rate in rates:
            c0, t0 = cpu_seconds(proc), time.monotonic()
            r = loadgen(url, bodies, rate, duration, conns, route)
            c1, t1 = cpu_seconds(proc), time.monotonic()
            cpu = c1 - c0
            r.update(server=kind, route=route, cpu_s=round(cpu, 3),
                     cpu_ms_per_req=round(1000 * cpu / max(1, r["ok"]), 4),
                     cpu_cores=round(cpu / (t1 - t0), 3))
            rows.append(r)
            lat = r["latency_us"]
            turns = f"  = {r['achieved_rate'] / per_turn:6.0f} turns/s" if per_turn else ""
            print(f"{kind:12s} {rate:6d}/s  achieved {r['achieved_rate']:8.0f}/s{turns}  "
                  f"p50 {lat['p50'] / 1000:7.2f} ms  p99 {lat['p99'] / 1000:8.2f} ms  "
                  f"cpu {r['cpu_ms_per_req']:.3f} ms/req ({r['cpu_cores']:.2f} cores)  "
                  f"errors {r['errors']}", flush=True)
            # Past saturation the queue only grows; one more rate shows it.
            if r["achieved_rate"] < 0.9 * rate or lat["p99"] > 1_000_000:
                break
    finally:
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--conns", type=int, default=64)
    ap.add_argument("--rates", default="250,500,1000,1500,2000,3000,4000,6000,8000")
    ap.add_argument("--servers", default="python,rust")
    ap.add_argument("--route", default="search", choices=sorted(ROUTE_PATH))
    ap.add_argument("--episodes", type=int, default=2000, help="agent_turn: episodes seeded")
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    for b in (RUST, LOADGEN):
        if not os.path.exists(b):
            raise SystemExit(f"build first: (cd feather-server && cargo build --release); {b} missing")
    work = a.work or tempfile.mkdtemp(prefix="feather-load-")
    os.makedirs(work, exist_ok=True)
    per_turn = 0
    if a.route == "agent_turn":
        t = time.time()
        base, turn = build_agent_dataset(work, a.episodes, a.dim)
        per_turn = len(turn)
        bodies = os.path.join(work, "turns.jsonl")
        write_turns(bodies, turn)
        routes = sorted({f"{r['method']} {r['path'].split('?')[0].split('/', 3)[-1]}" for r in turn})
        print(f"agent namespace: {a.episodes} episodes x {a.dim}-d seeded in {time.time() - t:.1f}s; one turn = "
              f"{per_turn} requests over {len(routes)} routes: {', '.join(routes)}")
    else:
        base = os.path.join(work, f"base-{a.n}x{a.dim}.feather")
        if not os.path.exists(base):
            t = time.time()
            build_dataset(base, a.n, a.dim)
            print(f"dataset {a.n} x {a.dim} built in {time.time() - t:.1f}s: {base}")
        bodies = os.path.join(work, f"bodies-{a.route}-{a.dim}.jsonl")
        write_bodies(bodies, a.route, a.dim)
    rates = [int(r) for r in a.rates.split(",")]
    servers = a.servers.split(",")
    rows = []
    for rnd in range(a.rounds):
        order = servers if rnd % 2 == 0 else list(reversed(servers))
        for kind in order:
            for r in run_server(kind, base, work, bodies, a.route, rates, a.duration, a.conns, per_turn):
                r["round"] = rnd
                rows.append(r)
    meta = {"n": a.n, "dim": a.dim, "duration_s": a.duration, "conns": a.conns, "route": a.route,
            "episodes": a.episodes if a.route == "agent_turn" else None, "requests_per_turn": per_turn,
            "machine": f"{os.uname().sysname} {os.uname().machine}, {psutil.cpu_count()} cpus",
            "python": sys.version.split()[0], "feather_db": feather_db.__version__,
            "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = a.out or os.path.join(ROOT, "bench", "results",
                                f"http_load__{a.route}__{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
