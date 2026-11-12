"""Run one agent-memory session against feather-api (Python) and
feather-server (Rust) over real HTTP, through the SDK the agents use
(RemoteDB under FeatherStore, AgentMemory, Pocket and PacketBuilder), and
check that the Rust server answers every request the session sends and gives
the agent the same memory back.

    .venv/bin/python feather-server/tests/agent_session.py

parity.py compares routes one at a time; this checks that the routes an agent
actually sends are all there: setup, turns, and outside a turn the
corrections and erasure (pin, forget, a session's scratch dropped on close,
upkeep, erasing one item, one note and a whole subject). A namespace is served by one process (its
write lock), so a namespace moved to the Rust server must answer all of them:
a 501 there is an agent that crashes. Exits non-zero on any failed request
or any difference in what the agent reads.
"""
import collections
import hashlib
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parity import RUST_BIN, ROOT, free_port, start  # noqa: E402

sys.path.insert(0, ROOT)
from feather_db import PacketBuilder  # noqa: E402
from feather_db.integrations.agent_memory import AgentMemory  # noqa: E402
from feather_db.integrations.langgraph_store import FeatherStore  # noqa: E402
from feather_db.pocket import Pocket  # noqa: E402
from feather_db.remote import RemoteDB, _StdlibSession  # noqa: E402

DIM = 32
NS = "agents"


def embed(text):
    """A deterministic unit vector of DIM floats per text (set DIM first for
    another size: benchmarks/http_server_load.py --route agent_turn does)."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# Times a server stamps itself (the client's clock is fixed, see Clock).
CLOCK = {"recorded_at", "updated_at", "created_at"}


def canon(v):
    """A stored value as comparable text, without wall-clock fields."""
    if isinstance(v, dict):
        return repr(sorted((k, canon(x)) for k, x in v.items() if k not in CLOCK))
    if isinstance(v, (list, tuple)):
        return repr([canon(x) for x in v])
    return repr(v)


class Tap:
    """The default session, recording (method, route, status) per request."""

    def __init__(self, log):
        self.s, self.log = _StdlibSession(30.0), log

    def request(self, method, url, **kw):
        r = self.s.request(method, url, **kw)
        path = url.split("/v1/", 1)[1].split("?")[0]
        route = "/".join("{id}" if p.isdigit() else p for p in path.split("/")[1:]) or "namespaces"
        self.log.append((method, route, r.status_code))
        return r

    def close(self):
        self.s.close()


class Clock:
    """The client's time.time(), the same sequence in both sessions: an
    episode's happened_at is in the text that gets embedded, so with the wall
    clock the two sessions would store different vectors. Both start from
    one base: each reading the wall clock itself, the two sessions differed
    whenever they began in different seconds."""

    def __init__(self, base):
        self.t = float(base)

    def __call__(self):
        self.t += 0.001
        return self.t


_real_time = time.time
BASE = int(_real_time())


def session(url):
    """One agent's working session. Returns what the agent read, without ids
    (each server's version clock starts when its file is created, and ids
    of keyed records come from it)."""
    time.time = Clock(BASE)
    try:
        return _session(url)
    finally:
        time.time = _real_time


def receipt(r):
    """An erase receipt without ids (each server allocates its own), file
    sizes or run-specific fields: what was erased, and that it verified."""
    return {"matched": len(r["matched"]), "not_found": len(r["not_found"]),
            "derived": sorted(d["rel"] for d in r["derived"]), "pruned_edges": len(r["pruned_edges"]),
            "erased": r["erased"], "verified": r["verified"], "problems": r["problems"],
            "dry_run": r["dry_run"]}


def _session(url):
    log = []
    db = RemoteDB(url, NS, session=Tap(log))
    db.ensure_namespace(DIM)
    store = FeatherStore(db=db, embed=embed)
    mem = AgentMemory(store, org="acme", user="u1", agent="creative", brand="nike")
    pkt = Pocket(db, ("acme", "nike", "creative"), embed=embed)
    builder = PacketBuilder(pkt, budget_tokens=2000)

    for i in range(20):
        mem.episodes.record(f"shipped weekly report {i}", outcome="ok" if i % 3 else "late")
        pkt.remember(f"note{i}", f"captions stay lowercase, variant {i}")
    pkt.remember("tone", "never use exclamation marks", pinned=True)
    mem.preferences.set("comms", "prefers written async")
    mem.preferences.set("comms", "prefers a call for anything urgent")
    mem.facts.state("positioning", "premium")
    mem.facts.state("positioning", "premium, sport-first")
    mem.procedures.define("weekly", ["pull numbers", "draft", "send"])
    mem.entities.enrich("nike", {"industry": "apparel"})
    mem.entities.enrich("nike", {"hq": "beaverton"})
    mem.observations.note("style", "decisive", evidence=["comms"])

    out = {}
    for turn in range(3):
        out[f"recall{turn}"] = sorted(canon(v) for v in mem.recall("how to contact", kinds=["preferences"]))
        out[f"hot{turn}"] = sorted(i.content for i in pkt.hot())
        out[f"pocket_recall{turn}"] = sorted(i.content for i in pkt.recall("lowercase captions", k=5))
        p = builder.build(query="which caption style")
        out[f"packet{turn}"] = sorted(p.text.splitlines())
        mem.episodes.record(f"turn {turn} done", outcome="ok")
    out["about"] = {kind: sorted(canon(v) for v in vals) for kind, vals in mem.about().items()}
    out["count_episodes"] = store.count(("acme", "u1", "episodes"))
    out["newest_episodes"] = [canon(i.value) for i in store.newest(("acme", "u1", "episodes"), 5)]
    out["get_pref"] = canon(store.get(("acme", "u1", "preferences"), "comms").value)
    out["search"] = sorted(canon(i.value) for i in store.search(("acme", "u1", "episodes"), query="report", limit=5))
    out["dim"] = db.dim()

    # Outside a turn: corrections, a working session's scratch, upkeep and
    # erasure (PUT and DELETE /records/{id}, POST /erase).
    pkt.pin("note3")
    pkt.forget("note4")
    with pkt.session("s1") as s:                  # close() forgets the scratch
        s.note("draft caption: running at dawn")
        s.note("draft caption: the long way home")
        out["scratch_recall"] = sorted(i.content for i in s.recall("draft caption", k=3))
    out["maintain"] = pkt.maintain()
    out["hot_after_edits"] = sorted(i.content for i in pkt.hot())
    out["erase_dry_run"] = receipt(mem.erase_subject(dry_run=True))
    out["erase_pref"] = receipt(mem.erase("preferences", "comms"))   # and the observation citing it
    out["erase_note"] = receipt(pkt.erase("note5"))
    out["about_after_erase"] = {kind: sorted(canon(v) for v in vals) for kind, vals in mem.about().items()}
    out["pocket_recall_after_erase"] = sorted(i.content for i in pkt.recall("lowercase captions", k=5))
    out["erase_subject"] = receipt(mem.erase_subject())
    out["about_after_subject"] = {kind: sorted(canon(v) for v in vals) for kind, vals in mem.about().items()}
    out["count_after_subject"] = store.count(("acme", "u1", "episodes"))
    out["hot_after_subject"] = sorted(i.content for i in pkt.hot())
    store.close()
    db.close()
    return out, log


def main():
    if not os.path.exists(RUST_BIN):
        raise SystemExit(f"build it first: (cd feather-server && cargo build --release); {RUST_BIN} missing")
    work = tempfile.mkdtemp(prefix="feather-agent-session-")
    py_port, rs_port = free_port(), free_port()
    common = {k: v for k, v in os.environ.items() if not k.startswith("FEATHER_")}
    common.update(FEATHER_DEV_MODE="1", FEATHER_CHECKPOINT_POLL_S="3600")
    os.makedirs(os.path.join(work, "py"))
    os.makedirs(os.path.join(work, "rs"))
    py = start([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(py_port), "--log-level", "warning"],
               {**common, "FEATHER_DATA_DIR": os.path.join(work, "py"), "PYTHONPATH": ROOT}, py_port,
               os.path.join(ROOT, "feather-api"))
    rs = start([RUST_BIN, "--host", "127.0.0.1", "--port", str(rs_port)],
               {**common, "FEATHER_DATA_DIR": os.path.join(work, "rs")}, rs_port, ROOT)
    failures = 0
    try:
        t = time.perf_counter()
        py_out, py_log = session(f"http://127.0.0.1:{py_port}")
        t_py = time.perf_counter() - t
        t = time.perf_counter()
        try:
            rs_out, rs_log = session(f"http://127.0.0.1:{rs_port}")
        except Exception as e:                     # an agent that crashed
            print(f"FAIL  the session raised on the Rust server: {type(e).__name__}: {e}")
            raise SystemExit(1)
        t_rs = time.perf_counter() - t
    finally:
        for p in (py, rs):
            p.terminate()
            p.wait(timeout=30)

    routes = collections.Counter((m, r) for m, r, _ in rs_log)
    bad = [(m, r, s) for m, r, s in rs_log if s >= 500 or s == 405]
    print(f"{len(rs_log)} requests over {len(routes)} routes:")
    for (m, r), n in sorted(routes.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {m} {r}")
    if bad:
        failures += 1
        print(f"FAIL  {len(bad)} requests the Rust server could not serve: {sorted(set(bad))}")
    if [(m, r) for m, r, _ in py_log] != [(m, r) for m, r, _ in rs_log]:
        failures += 1
        print("FAIL  the session sent different requests to the two servers")
    for k in py_out:
        if py_out[k] != rs_out[k]:
            failures += 1
            print(f"DIFF  {k}:\n      python {py_out[k]!r}\n      rust   {rs_out[k]!r}")
    print(f"\nsession wall time: python {t_py:.2f} s, rust {t_rs:.2f} s (one client: latency, not capacity)")
    print("agent session identical on both servers" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
