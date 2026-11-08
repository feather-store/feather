"""LongMemEval retrieval only: where do the gold sessions rank, and is scored
search exact? No reader, no judge, so a ranking change costs nothing to check.

    python -m bench.lme_retrieval embed knowledge-update    # paid, once
    python -m bench.lme_retrieval rank  knowledge-update    # scoring off / on
    python -m bench.lme_retrieval exact knowledge-update    # scored top-k vs exact
    python -m bench.lme_retrieval coverage abstention       # is "nothing relevant" visible?

`embed` caches gemini-embedding-001 vectors (3072-d, as the bench's runs use)
under ~/.cache/feather/longmemeval/emb/; it needs GOOGLE_API_KEY and costs
~$1.4 for knowledge-update, ~$2.4 for temporal-reasoning. The other commands
read only the cache.

`rank` reproduces bench/scenarios/longmemeval.py's retrieval: timestamps
shifted so the question date is "now", hybrid_search over a pool of k*5 turns,
sessions ordered by their best turn, top 20.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import feather_db

from . import lme_retrieve
from .datasets.longmemeval import iter_history_turns
from .scenarios.lme_context import parse_lme_date

DATA = os.path.expanduser("~/.cache/feather/longmemeval/longmemeval_s_cleaned.json")
CACHE = os.path.expanduser("~/.cache/feather/longmemeval/emb/gemini-embedding-001")
TOP_K, POOL = 20, 100
MODES = {"off": None, "w0.3": (30.0, 0.3), "w0.5": (30.0, 0.5)}


def questions(qtype: str) -> list:
    """One question type, or "abstention": the 30 unanswerable questions
    ('_abs' in the id) followed by the answerable ones they were made from
    (29 present), each with its own haystack."""
    data = json.load(open(DATA))
    if qtype != "abstention":
        return [q for q in data if q["question_type"] == qtype]
    by_id = {q["question_id"]: q for q in data}
    unanswerable = [q for q in data if "_abs" in q["question_id"]]
    parents = [by_id[p] for q in unanswerable
               if (p := q["question_id"].replace("_abs", "")) in by_id]
    return unanswerable + parents


def turns_of(q: dict) -> list:
    return [(sid, date, role, content)
            for sid, date, role, content, _ in iter_history_turns(q) if content.strip()]


def cache_path(qtype: str, qid: str) -> str:
    d = os.path.join(CACHE, qtype)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{qid}.npy")


def embed(qtype: str) -> None:
    from .providers_extra import GeminiEmbedder
    emb = GeminiEmbedder()

    def one(q):
        p = cache_path(qtype, q["question_id"])
        if os.path.exists(p):
            return
        texts = [t[3] for t in turns_of(q)] + [
            lme_retrieve.dense_query(q["question"], q.get("question_date") or "")]
        np.save(p, emb.embed_batch(texts).astype(np.float32))
        print("embedded", q["question_id"], len(texts), flush=True)

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, questions(qtype)))


def _haystack(qtype: str, q: dict):
    """(db, path, turns, stamps, turn vectors, query vector, now) as the bench builds it."""
    vecs = np.load(cache_path(qtype, q["question_id"]))
    ts = turns_of(q)
    now = int(time.time())
    q_dt = parse_lme_date(q.get("question_date") or "")
    shift = (now - int(q_dt.timestamp())) if q_dt else 0
    stamps = np.array([int(d.timestamp()) + shift if (d := parse_lme_date(date)) else 0
                       for _, date, _, _ in ts])
    path = tempfile.mktemp(suffix=".feather")
    db = feather_db.DB.open(path, dim=vecs.shape[1])
    for i, ((sid, _date, role, content), v) in enumerate(zip(ts, vecs[:-1])):
        m = feather_db.Metadata()
        m.content, m.timestamp, m.namespace_id = content, int(stamps[i]), q["question_id"]
        m.entity_id, m.source = sid, role
        db.add(id=i + 1, vec=v, meta=m)
    return db, path, ts, stamps, vecs[:-1], vecs[-1], now


def _drop(db, path) -> None:
    db.close(save=False)
    for f in (path, path + ".wal", path + ".lock"):
        if os.path.exists(f):
            os.remove(f)


def rank(qtype: str) -> list:
    out = []
    for q in questions(qtype):
        db, path, ts, _stamps, _tv, qv, _now = _haystack(qtype, q)
        date_of = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
        gold = sorted(q["answer_session_ids"], key=lambda s: parse_lme_date(date_of[s]))
        rec = {"qid": q["question_id"], "gold_old_to_new": gold}
        for mode, cfg in MODES.items():
            kw = {} if cfg is None else {"scoring": feather_db.ScoringConfig(
                half_life=cfg[0], weight=cfg[1], min=0.0)}
            for how in ("hybrid", "vector"):
                hits = (db.hybrid_search(qv, q["question"], k=POOL, **kw) if how == "hybrid"
                        else db.search(qv, k=POOL, record_salience=False, **kw))
                order = list(dict.fromkeys(ts[h.id - 1][0] for h in hits))[:TOP_K]
                rec[f"{how}/{mode}"] = [order.index(g) if g in order else None for g in gold]
        _drop(db, path)
        out.append(rec)
    print(f"== {qtype}: {len(out)} questions (newest gold = the latest evidence session) ==")
    for key in [k for k in out[0] if "/" in k]:
        new = [r[key][-1] for r in out]
        both = [None if None in r[key] else max(r[key]) for r in out]
        at = lambda xs, k: sum(x is not None and x < k for x in xs)       # noqa: E731
        mrr = sum(1 / (x + 1) for x in new if x is not None) / len(out)
        print(f"{key:12s} newest gold @1 {at(new, 1):3d} @5 {at(new, 5):3d} MRR {mrr:.3f} | "
              f"all gold @5 {at(both, 5):3d} @20 {at(both, 20):3d}")
    return out


def exact(qtype: str, w: float = 0.3, half_life: float = 30.0) -> None:
    """Scored search's top k against the exact scored top k over every turn."""
    cfg = feather_db.ScoringConfig(half_life=half_life, weight=w, min=0.0)
    res = {k: [] for k in (5, 10, 20)}
    for q in questions(qtype):
        db, path, _ts, stamps, tv, qv, now = _haystack(qtype, q)
        cos = tv @ qv / (np.linalg.norm(tv, axis=1) * np.linalg.norm(qv))
        age = np.maximum(0, now - stamps) / 86400
        truth = cos * ((1 - w) + w * 0.5 ** (age / half_life))
        for k in res:
            want = set(np.argsort(-truth)[:k] + 1)
            got = {r.id for r in db.search(qv, k=k, scoring=cfg, record_salience=False)}
            res[k].append(len(want & got) / k)
        _drop(db, path)
    for k, r in res.items():
        r = np.array(r)
        print(f"k={k:2d}  scored top-k vs exact: mean {r.mean():.4f}  min {r.min():.2f}  "
              f"short in {int((r < 1).sum())}/{len(r)}")


def coverage(qtype: str = "abstention") -> None:
    """Can the background floor tell "memory has it" from "it does not"?

    For each answerable question (the parents of the '_abs' ones) the best
    turn's relevance, (cos - floor) / (1 - floor) with floor =
    DB.background_cosine(), is compared with four negatives: the '_abs'
    question on its own haystack, each with its answer sessions removed,
    and the question asked against another question's haystack."""
    if qtype != "abstention":
        raise SystemExit("coverage runs on the abstention set: embed abstention first")
    qs = questions(qtype)
    data = {}
    for q in qs:
        db, path, ts, _s, tv, qv, _n = _haystack(qtype, q)
        data[q["question_id"]] = (q, ts, tv / np.linalg.norm(tv, axis=1, keepdims=True),
                                  qv / np.linalg.norm(qv), db.background_cosine())
        _drop(db, path)

    def rel(qid, hay_qid, drop_gold=False):
        q, ts, tv, _qv, floor = data[hay_qid]
        cos = tv @ data[qid][3]
        if drop_gold:
            cos = cos[[sid not in set(q["answer_session_ids"]) for sid, *_ in ts]]
        return (cos.max() - floor) / (1 - floor)

    parents = [q["question_id"] for q in qs if "_abs" not in q["question_id"]]
    unans = [q["question_id"] for q in qs if "_abs" in q["question_id"]]
    by = {
        "answerable":   [rel(p, p) for p in parents],
        "abstention":   [rel(u, u) for u in unans],
        "ans-gold":     [rel(p, p, True) for p in parents],
        "abs-gold":     [rel(u, u, True) for u in unans],
        "other-hay":    [rel(p, parents[(i + 7) % len(parents)]) for i, p in enumerate(parents)],
    }
    floors = [d[4] for d in data.values()]
    print(f"floor (background cosine): {min(floors):.3f}..{max(floors):.3f}")
    pos = np.array(by["answerable"])
    for c, xs in by.items():
        x = np.array(xs)
        auc = "" if c == "answerable" else f"  AUROC vs answerable {((pos[:, None] > x[None]).mean()):.3f}"
        print(f"{c:11s} n={len(x):2d} best-turn relevance min {x.min():.3f} "
              f"med {np.median(x):.3f} max {x.max():.3f}{auc}")
    print("flagged 'none' at relevance < t:")
    for t in (0.0, 0.15, 0.20, 0.25):
        print(f"  t={t:.2f}  " + "  ".join(f"{c} {int((np.array(x) < t).sum())}/{len(x)}"
                                       for c, x in by.items()))


if __name__ == "__main__":
    cmd, qtype = sys.argv[1], sys.argv[2]
    {"embed": embed, "rank": rank, "exact": exact, "coverage": coverage}[cmd](qtype)
