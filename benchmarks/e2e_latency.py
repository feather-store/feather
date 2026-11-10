"""
End-to-end latency: what the engine micro-suite (full_suite.py) does not cover.

    python benchmarks/e2e_latency.py --out-dir bench/results/e2e_YYYYMMDD [--only bm25,...]

Sections (each writes <section>.json; p50/p95/p99 in ms throughout):

  build        100k records of real text: the first 100k unique LongMemEval
               turns (median 444 chars), 768-d random unit vectors, 50
               namespaces, timestamps over a year. Built once, reused.
  bm25         a fresh read-only process per run: open, the FIRST keyword
               search (builds the index), RSS before/after; then warm keyword
               and hybrid search over the 500 LongMemEval questions.
  time_travel  200 commits on the 100k file (15 writes each, a vacuum after
               commit 100, retention on): save, history(), as_of opens on the
               live file and on the kept file vs a plain read-only open,
               first search after an as_of open, diff() at 100k.
  agent        Pocket / PacketBuilder / FeatherStore / AgentMemory on an
               8k-memory agent (3 scope levels), local hash embedder, so the
               numbers are Feather's alone.
  live         PAID. LoCoMo dev conversations: per question, embed the
               question (Gemini, live), hybrid_search k=30, render, answer
               (gpt-4o-mini via OpenRouter); per-stage and total latency,
               sequential. --api repeats retrieval through the Cloud API
               (uvicorn + RemoteDB) for the HTTP hop.

Needs ~/.cache/feather/longmemeval/longmemeval_s_cleaned.json (bench.datasets
downloads it); `live` needs GOOGLE_API_KEY and OPENROUTER_API_KEY, and the
LoCoMo turn embeddings cached by `python -m bench.locomo`.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import feather_db  # noqa: E402

LME = os.path.expanduser("~/.cache/feather/longmemeval/longmemeval_s_cleaned.json")
DIM = 768


def pct(xs):
    """Percentiles, and the p99's 95% interval from order statistics
    (bench.stats.quantile_ci): a p99 of n samples rests on ~n/100 of them,
    and an upper end of None says there were too few to bound it."""
    from bench.stats import quantile_ci
    a = np.asarray(xs, dtype=float)
    lo, hi = quantile_ci(a.tolist(), q=0.99)
    r = lambda v: None if v is None else round(float(v), 3)          # noqa: E731
    return {"n": int(a.size), "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "p99_ci95_ms": [r(lo), r(hi)],
            "mean_ms": round(float(a.mean()), 3), "max_ms": round(float(a.max()), 3)}


def timed(fn, reps):
    lat = []
    for i in range(reps):
        t = time.perf_counter()
        fn(i)
        lat.append((time.perf_counter() - t) * 1e3)
    return pct(lat)


def rss_mb(pid=None):
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid or os.getpid())])
    return int(out) / 1024


def log(*a):
    print("[e2e]", *a, file=sys.stderr, flush=True)


# ── corpus ────────────────────────────────────────────────────────────────

def lme():
    with open(LME) as f:
        return json.load(f)


def corpus(n):
    seen, out = set(), []
    for e in lme():
        for s in e["haystack_sessions"]:
            for t in s:
                c = t["content"].strip()
                if len(c) >= 20 and c not in seen:
                    seen.add(c)
                    out.append(c)
                    if len(out) == n:
                        return out
    return out


def questions():
    return [e["question"] for e in lme()]


def unit(rng, n, dim=DIM):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build(work, n):
    path = os.path.join(work, "base.feather")
    if os.path.exists(path):
        return path
    texts = corpus(n)
    rng = np.random.default_rng(0)
    now = time.time()
    db = feather_db.DB.open(path, dim=DIM)
    t = time.perf_counter()
    for s in range(0, len(texts), 10_000):
        chunk = texts[s:s + 10_000]
        metas = []
        for i, c in enumerate(chunk, start=s):
            m = feather_db.Metadata()
            m.content = c
            m.namespace_id = f"org.user{i % 50}"
            m.timestamp = int(now - rng.uniform(0, 365 * 86400))
            metas.append(m)
        db.add_batch(list(range(s, s + len(chunk))), unit(rng, len(chunk)), metas)
    db.save()
    db.close()
    log(f"built {len(texts)} records in {time.perf_counter() - t:.0f} s")
    return path


# ── bm25: a cold process per run ─────────────────────────────────────────

_BM25_CHILD = r"""
import json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, sys.argv[3])
import feather_db
def rss(): return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())]))/1024
qs = json.load(open(sys.argv[2]))
r0 = rss(); t = time.perf_counter()
db = feather_db.DB.open(sys.argv[1], dim=768, read_only=True)
open_ms = (time.perf_counter() - t) * 1e3; r1 = rss()
t = time.perf_counter(); db.keyword_search(qs[0], k=10)
first_ms = (time.perf_counter() - t) * 1e3; r2 = rss()
out = {"open_ms": open_ms, "first_keyword_ms": first_ms, "rss_open_mb": r1 - r0,
       "rss_bm25_mb": r2 - r1}
if sys.argv[4] == "1":
    rng = np.random.default_rng(1)
    kw, hy = [], []
    for q in qs:
        t = time.perf_counter(); db.keyword_search(q, k=10, record_salience=False); kw.append((time.perf_counter()-t)*1e3)
    for q in qs:
        v = rng.standard_normal(768).astype(np.float32); v /= np.linalg.norm(v)
        t = time.perf_counter(); db.hybrid_search(v, q, k=10); hy.append((time.perf_counter()-t)*1e3)
    out["keyword"] = kw; out["hybrid"] = hy
print(json.dumps(out))
"""


def bm25(work, base, runs=5):
    qpath = os.path.join(work, "questions.json")
    with open(qpath, "w") as f:
        json.dump(questions(), f)
    cold = []
    warm = None
    for r in range(runs):
        out = subprocess.check_output([sys.executable, "-c", _BM25_CHILD, base, qpath, HERE,
                                       "1" if r == 0 else "0"])
        d = json.loads(out)
        if r == 0:
            warm = {"keyword_search": pct(d.pop("keyword")), "hybrid_search": pct(d.pop("hybrid"))}
        cold.append(d)
    res = {k: pct([c[k] for c in cold]) for k in ("open_ms", "first_keyword_ms")}
    res["rss_open_mb"] = round(float(np.median([c["rss_open_mb"] for c in cold])), 1)
    res["rss_bm25_mb"] = round(float(np.median([c["rss_bm25_mb"] for c in cold])), 1)
    res.update(warm)
    return res


# ── time travel ───────────────────────────────────────────────────────────

def time_travel(work, base, commits=200):
    path = os.path.join(work, "tt.feather")
    for p in [path, path + ".wal", path + ".lock"]:
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(path + ".history", ignore_errors=True)
    shutil.copyfile(base, path)
    rng = np.random.default_rng(2)
    texts = corpus(100_000)
    db = feather_db.DB.open(path, dim=DIM)
    n = len(db.record_versions())
    db.set_history_retention(float("inf"))
    saves, commit_bytes, marks = [], [], []
    next_id = n
    for c in range(1, commits + 1):
        for i in rng.integers(0, n, 10):
            m = db.get_metadata(int(i))
            m.content = texts[int(rng.integers(0, len(texts)))]
            db.update_metadata(int(i), m)
        for v in unit(rng, 5):
            m = feather_db.Metadata()
            m.content = texts[int(rng.integers(0, len(texts)))]
            m.namespace_id = "org.user0"
            db.add(next_id, v, m)
            next_id += 1
        t = time.perf_counter()
        db.save()
        saves.append((time.perf_counter() - t) * 1e3)
        st = db.storage_stats()
        commit_bytes.append(st["last_commit_bytes"])
        marks.append((st["generation"], time.time()))
        if c % 50 == 0:
            db.tag(f"run-{c}")
        if c == commits // 2:
            t = time.perf_counter()
            db.vacuum()
            vacuum_ms = (time.perf_counter() - t) * 1e3
        time.sleep(0.001)
    hist = timed(lambda _: db.history(), 20)
    hist_commits = len(db.history())
    st = db.storage_stats()
    db.close()

    early = [t for g, t in marks[: commits // 2 - 5]]      # in the kept file
    late = [t for g, t in marks[commits // 2 + 5:]]        # in the live file

    def opens(times, reps=15):
        lat, first = [], []
        for i in range(reps):
            t0 = time.perf_counter()
            h = feather_db.DB.open(path, as_of=times[int(rng.integers(0, len(times)))])
            lat.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            h.search(unit(rng, 1)[0], k=10)
            first.append((time.perf_counter() - t0) * 1e3)
            h.close()
        return pct(lat), pct(first)

    plain = timed(lambda _: feather_db.DB.open(path, read_only=True).close(), 10)
    live_open, live_first = opens(late)
    kept_open, kept_first = opens(early)
    tag_open = timed(lambda _: feather_db.DB.open(path, as_of="run-50").close(), 10)
    a = feather_db.DB.open(path, as_of="run-50")
    b = feather_db.DB.open(path, read_only=True)
    t = time.perf_counter()
    d = feather_db.diff(a, b)
    diff_ms = (time.perf_counter() - t) * 1e3
    a.close(); b.close()
    return {
        "records": n, "commits": commits,
        "save": pct(saves), "commit_bytes_median": int(np.median(commit_bytes)),
        "marker_bytes_per_commit": 51, "vacuum_ms": round(vacuum_ms, 1),
        "history_call": hist, "history_commits": hist_commits,
        "open_read_only_plain": plain,
        "open_as_of_time_live_file": live_open, "open_as_of_time_kept_file": kept_open,
        "open_as_of_tag": tag_open,
        "first_search_after_as_of_live": live_first, "first_search_after_as_of_kept": kept_first,
        "diff_100k_ms": round(diff_ms, 1), "diff_size": len(d),
        "history_files": st["history_files"], "history_mb": round(st["history_bytes"] / 2**20, 1),
        "live_file_mb": round(st["file_bytes"] / 2**20, 1),
    }


# ── agent layer ───────────────────────────────────────────────────────────

def agent(work):
    from bench.embedders import DeterministicEmbedder
    from feather_db.pocket import Pocket
    from feather_db.packet import PacketBuilder
    from feather_db.integrations.langgraph_store import FeatherStore
    from feather_db.integrations.agent_memory import AgentMemory
    emb = DeterministicEmbedder(DIM)
    texts = corpus(20_000)
    qs = questions()
    rng = np.random.default_rng(3)
    res = {}

    path = os.path.join(work, "agent.feather")
    for p in [path, path + ".wal", path + ".lock"]:
        if os.path.exists(p):
            os.remove(p)
    db = feather_db.DB.open(path, dim=DIM)
    layers = [(("org",), 1000), (("org", "brand"), 2000), (("org", "brand", "creative"), 5000)]
    pockets = {}
    lat = []
    k = 0
    for scope, count in layers:
        pk = Pocket(db, scope, budget_tokens=4000, embed=emb.embed)
        pockets[scope] = pk
        for i in range(count):
            t = time.perf_counter()
            pk.remember(f"note-{k}", texts[k], pinned=(i < 20), importance=float(rng.uniform(0.3, 1)))
            lat.append((time.perf_counter() - t) * 1e3)
            k += 1
    db.save()
    res["pocket_remember"] = pct(lat)
    pk = pockets[("org", "brand", "creative")]
    pk.remember("tone", "captions stay lowercase", pinned=True)
    cold = []
    for i in range(20):
        pk.remember(f"bump-{i}", texts[k + i])          # invalidates the hot() memo
        t = time.perf_counter(); pk.hot(); cold.append((time.perf_counter() - t) * 1e3)
    res["pocket_hot_cold"] = pct(cold)
    res["pocket_hot_warm"] = timed(lambda _: pk.hot(), 200)
    res["pocket_hot_text_warm"] = timed(lambda _: pk.hot_text(), 100)
    res["pocket_recall"] = timed(lambda i: pk.recall(qs[i % len(qs)], k=5), 200)
    b = PacketBuilder(pk, budget_tokens=4000)
    res["packet_build_query"] = timed(lambda i: b.build(required=["tone"], query=qs[i % len(qs)]), 200)
    res["packet_build_hot"] = timed(lambda i: b.build(required=["tone"]), 50)
    res["agent_records"] = len(db.record_versions())
    db.close()

    spath = os.path.join(work, "store.feather")
    for p in [spath, spath + ".wal", spath + ".lock"]:
        if os.path.exists(p):
            os.remove(p)
    store = FeatherStore(spath, dim=DIM, embed=emb.embed)
    ns = ("org", "user_1", "notes")
    res["store_put_autosave"] = timed(lambda i: store.put(ns, f"k{i}", {"text": texts[i]}), 2000)
    res["store_get"] = timed(lambda i: store.get(ns, f"k{int(rng.integers(0, 2000))}"), 500)
    res["store_search_query"] = timed(lambda i: store.search(ns, query=qs[i % len(qs)], limit=5), 200)
    mem = AgentMemory(store, org="hawky", user="user_842", agent="creative", brand="nike")
    res["memory_preferences_set"] = timed(lambda i: mem.preferences.set(f"p{i % 50}", texts[i]), 200)
    res["memory_episodes_record"] = timed(lambda i: mem.episodes.record(texts[1000 + i][:200], outcome="ok"), 200)
    res["memory_facts_state"] = timed(lambda i: mem.facts.state(f"f{i % 50}", texts[2000 + i][:200]), 200)
    res["memory_about"] = timed(lambda _: mem.about(), 50)
    res["memory_recall"] = timed(lambda i: mem.recall(qs[i % len(qs)]), 200)
    store.close()
    return res


# ── recall on real embeddings ────────────────────────────────────────────

def recall_real(work, nq=1000, k=10):
    """HNSW recall@10 vs brute force on every cached gemini-embedding-001
    vector (LongMemEval + LoCoMo, deduplicated): synthetic Gaussian clusters
    in 768-d have no meaningful nearest neighbours, real embeddings do."""
    import glob
    files = glob.glob(os.path.expanduser("~/.cache/feather/longmemeval/emb/gemini-embedding-001/**/*.npy"),
                      recursive=True)
    files += glob.glob(os.path.expanduser("~/.cache/feather/locomo/emb/gemini-embedding-001/*.npy"))
    vs = np.concatenate([np.load(f).astype(np.float32) for f in sorted(files)])
    vs /= np.linalg.norm(vs, axis=1, keepdims=True)
    _, keep = np.unique(np.round(vs[:, :16], 5), axis=0, return_index=True)
    vs = vs[np.sort(keep)]
    rng = np.random.default_rng(4)
    perm = rng.permutation(len(vs))
    qs, base = vs[perm[:nq]], vs[perm[nq:]]
    gt = np.argsort(-(qs @ base.T), axis=1)[:, :k]
    path = os.path.join(work, "recall_real.feather")
    for p in [path, path + ".wal", path + ".lock"]:
        if os.path.exists(p):
            os.remove(p)
    db = feather_db.DB.open(path, dim=base.shape[1])
    t = time.perf_counter()
    db.add_batch(list(range(len(base))), base, [feather_db.Metadata() for _ in range(len(base))])
    build_s = time.perf_counter() - t
    res = {"vectors": int(len(base)), "dim": int(base.shape[1]), "queries": nq,
           "build_s": round(build_s, 1)}
    for ef in (10, 50, 100, 200):
        db.set_ef(ef)
        lat, hit = [], 0
        for i, q in enumerate(qs):
            t = time.perf_counter()
            r = db.search(q, k=k)
            lat.append((time.perf_counter() - t) * 1e3)
            hit += len({x.id for x in r} & set(gt[i].tolist()))
        res[f"ef={ef}"] = {**pct(lat), "recall@10": round(hit / (nq * k), 4)}
    db.close(save=False)
    return res


# ── live: real models ────────────────────────────────────────────────────

def live(work, n_questions=100, via_api=False):
    from collections import Counter
    from bench import locomo
    from bench.providers_extra import GeminiEmbedder
    data = {s["sample_id"]: s for s in locomo.load()}
    split = json.load(open(os.path.join(HERE, "bench", "datasets", "locomo_split.json")))
    emb = GeminiEmbedder(locomo.EMB_MODEL)
    chat = locomo._Chat(locomo.READER, "reader")
    per_conv = max(1, n_questions // len(split["dev"]))
    rows, spent = [], Counter()
    remote = server = None
    if via_api:
        server, remote_for = _api_server(work)
    for sid in split["dev"]:
        s = data[sid]
        turns, qs = locomo.turns_of(s), locomo.questions_of(s)[:per_conv]
        vecs = locomo.vectors(s, turns, locomo.questions_of(s), spent)[:len(turns)]
        a, b = s["conversation"]["speaker_a"], s["conversation"]["speaker_b"]
        path = os.path.join(work, f"locomo_{sid}.feather")
        for p in [path, path + ".wal", path + ".lock"]:
            if os.path.exists(p):
                os.remove(p)
        db = feather_db.DB.open(path, dim=vecs.shape[1])
        for i, (t, v) in enumerate(zip(turns, vecs)):
            m = feather_db.Metadata()
            m.content = t["content"]
            m.timestamp = int(t["dt"].replace(tzinfo=__import__("datetime").timezone.utc).timestamp())
            m.namespace_id = sid
            m.source = t["dia_id"]
            db.add(i + 1, v, m)
        db.save()
        if via_api:
            remote = remote_for(sid, path, vecs.shape[1])
        for q in qs:
            r = {"sample_id": sid, "question": q["question"]}
            t0 = time.perf_counter()
            qv = emb.embed_batch([q["question"]])[0].astype(np.float32)
            t1 = time.perf_counter()
            hits = db.hybrid_search(qv, q["question"], k=30)
            t2 = time.perf_counter()
            idx = [h.id - 1 for h in hits]
            ctx = locomo.render(turns, idx)
            t3 = time.perf_counter()
            r["answer"] = chat([{"role": "system", "content": locomo.ANSWER_SYSTEM.format(a=a, b=b)},
                                {"role": "user", "content": locomo.ANSWER_USER.format(
                                    context=ctx, question=q["question"])}],
                               max_tokens=100, temperature=0.0)
            t4 = time.perf_counter()
            r.update(embed_ms=(t1 - t0) * 1e3, retrieve_ms=(t2 - t1) * 1e3,
                     render_ms=(t3 - t2) * 1e3, llm_ms=(t4 - t3) * 1e3, total_ms=(t4 - t0) * 1e3,
                     context_chars=len(ctx))
            if via_api:
                t5 = time.perf_counter()
                remote.hybrid_search(qv, q["question"], k=30)
                r["retrieve_api_ms"] = (time.perf_counter() - t5) * 1e3
            rows.append(r)
        db.close()
    if server is not None:
        server.terminate()
    out = {k: pct([r[k] for r in rows]) for k in
           ("embed_ms", "retrieve_ms", "render_ms", "llm_ms", "total_ms")}
    if via_api:
        out["retrieve_api_ms"] = pct([r["retrieve_api_ms"] for r in rows])
    out["questions"] = len(rows)
    out["context_chars_median"] = int(np.median([r["context_chars"] for r in rows]))
    out["feather_share_of_total_pct"] = round(
        100 * float(np.sum([r["retrieve_ms"] + r["render_ms"] for r in rows]))
        / float(np.sum([r["total_ms"] for r in rows])), 2)
    out["reader"] = locomo.READER
    out["reader_usd"] = chat.usd()
    out["served_by"] = dict(chat.served_by)
    out["embed_model"] = locomo.EMB_MODEL
    with open(os.path.join(work, "live_rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out


def _api_server(work):
    """uvicorn on a throwaway data dir; each LoCoMo DB copied in as a namespace."""
    data = os.path.join(work, "api_data")
    shutil.rmtree(data, ignore_errors=True)
    os.makedirs(data)
    env = {**os.environ, "FEATHER_DATA_DIR": data, "FEATHER_DEV_MODE": "1", "FEATHER_DB_DIM": "3072"}
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8799",
                            "--log-level", "warning"], cwd=os.path.join(HERE, "feather-api"),
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import httpx
    for _ in range(100):
        try:
            if httpx.get("http://127.0.0.1:8799/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    from feather_db.remote import RemoteDB

    def remote_for(sid, path, dim):
        shutil.copyfile(path, os.path.join(data, f"{sid.replace('-', '_')}.feather"))
        r = RemoteDB("http://127.0.0.1:8799", sid.replace("-", "_"))
        r.hybrid_search(np.full(dim, dim ** -0.5, np.float32), "warm", k=1)   # open the namespace
        return r
    return srv, remote_for


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work", default=os.path.expanduser("~/.cache/feather/e2e_latency"))
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--only", default="build,bm25,time_travel,agent")
    ap.add_argument("--questions", type=int, default=100)
    ap.add_argument("--api", action="store_true", help="live: also retrieve through the Cloud API")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    only = a.only.split(",")
    base = build(a.work, a.n) if set(only) & {"build", "bm25", "time_travel"} else None
    env = {"python": sys.version.split()[0], "feather": feather_db.__version__,
           "cpus": os.cpu_count(), "simd": getattr(feather_db.core, "simd_info", lambda: None)(),
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name in only:
        if name == "build":
            continue
        log(f"section {name} ...")
        t = time.perf_counter()
        if name == "bm25":
            r = bm25(a.work, base)
        elif name == "time_travel":
            r = time_travel(a.work, base)
        elif name == "agent":
            r = agent(a.work)
        elif name == "recall_real":
            r = recall_real(a.work)
        elif name == "live":
            r = live(a.work, a.questions, a.api)
        else:
            raise SystemExit(f"unknown section {name}")
        r["_env"] = env
        r["_section_s"] = round(time.perf_counter() - t, 1)
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as f:
            json.dump(r, f, indent=1)
        log(f"section {name} done in {r['_section_s']} s")
        print(json.dumps({name: {k: v for k, v in r.items() if not k.startswith('_')}}, indent=1))


if __name__ == "__main__":
    main()
