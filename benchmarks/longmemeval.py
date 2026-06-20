"""
LongMemEval retrieval benchmark for Feather DB
==============================================
Measures how well Feather *recalls the evidence session* for each LongMemEval
question — the retrieval half of the long-term-memory task that competitors
(e.g. HydraDB's "92% on LongMemEval") headline.

Dataset (ICLR 2025): https://github.com/xiaowu0162/LongMemEval
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Each question carries a haystack of ~40-50 user/assistant sessions and gold
`answer_session_ids`. We ingest every session as one record, retrieve top-k for
the question, and score recall@k = "did top-k include a gold evidence session".

Retrieval modes (Feather already ships all three):
    keyword  — BM25 over session text        (NO embedder / API key needed)
    dense    — HNSW vector search            (needs an embedder)
    hybrid   — BM25 + dense via RRF          (needs an embedder)

Usage
-----
    # real number, no API key — BM25 baseline over the full set:
    python benchmarks/longmemeval.py --data /tmp/lme_s.json --mode keyword

    # dense / hybrid with a real embedder:
    GOOGLE_API_KEY=... python benchmarks/longmemeval.py --data /tmp/lme_s.json \
        --mode hybrid --embed-provider gemini --limit 100

    # smoke test with no deps (deterministic hash embedder):
    python benchmarks/longmemeval.py --data /tmp/lme_s.json --mode dense \
        --embed-provider hash --limit 20
"""
import os, sys, glob, json, time, argparse, importlib.util
from collections import defaultdict
import numpy as np

# ── load the compiled core directly (no install needed) ─────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob(os.path.join(_HERE, "feather_db/core*.so")) if _tag in p]
_so = (_cands or sorted(glob.glob(os.path.join(_HERE, "feather_db/core*.so"))))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)


def get_embedder(provider, model, key, dim):
    """Reuse the MCP client-side embedders; 'hash' is deterministic & key-free."""
    if provider in (None, "", "none", "keyword"):
        return None, dim
    sys.path.insert(0, _HERE)
    from feather_db.integrations.embedders import make_embedder
    embed = make_embedder(provider, model=model, api_key=key, dim=dim)
    return embed, dim


def session_text(session):
    """Flatten one session (list of {role, content}) into a single document."""
    return "\n".join(f"{t.get('role','')}: {t.get('content','')}" for t in session)


def run(args):
    data = json.load(open(args.data))
    if args.limit:
        data = data[: args.limit]

    embed, dim = get_embedder(args.embed_provider, args.embed_model,
                              args.embed_key or os.getenv("GOOGLE_API_KEY")
                              or os.getenv("OPENAI_API_KEY"), args.dim)
    keyword_only = embed is None and args.mode == "keyword"
    if embed is None and args.mode != "keyword":
        sys.exit(f"mode={args.mode} needs an embedder (set --embed-provider)")
    vdim = dim if embed else 8   # keyword mode: tiny dummy vectors just to register records

    ks = sorted(args.ks)
    maxk = max(ks)
    hit = {k: 0 for k in ks}                       # overall recall@k numerator
    by_type = defaultdict(lambda: {k: 0 for k in ks})
    by_type_n = defaultdict(int)
    n = len(data)
    n_embed_calls = 0
    t0 = time.time()

    for qi, item in enumerate(data):
        sessions = item["haystack_sessions"]
        sids = item["haystack_session_ids"]
        gold = set(item["answer_session_ids"])
        qtype = item.get("question_type", "?")

        path = f"/tmp/lme_q{qi}.feather"
        for ext in ("", ".wal", ".tmp"):
            if os.path.exists(path + ext): os.remove(path + ext)
        db = fc.DB.open(path, dim=vdim)

        # ingest each session as one record (id == session index)
        for si, sess in enumerate(sessions):
            text = session_text(sess)
            m = fc.Metadata(); m.content = text; m.entity_id = sids[si]
            if embed:
                vec = np.asarray(embed(text), dtype=np.float32); n_embed_calls += 1
            else:
                vec = np.zeros(vdim, dtype=np.float32)
            db.add(id=si, vec=vec, meta=m)

        # retrieve
        q = item["question"]
        if args.mode == "keyword":
            res = db.keyword_search(q, k=maxk)
        elif args.mode == "dense":
            qv = np.asarray(embed(q), dtype=np.float32); n_embed_calls += 1
            res = db.search(qv, k=maxk)
        else:  # hybrid
            qv = np.asarray(embed(q), dtype=np.float32); n_embed_calls += 1
            res = db.hybrid_search(qv, q, k=maxk)

        ranked_sids = [sids[r.id] for r in res]
        by_type_n[qtype] += 1
        for k in ks:
            if gold & set(ranked_sids[:k]):
                hit[k] += 1; by_type[qtype][k] += 1

        for ext in ("", ".wal", ".tmp"):
            if os.path.exists(path + ext): os.remove(path + ext)
        if (qi + 1) % 25 == 0 or qi + 1 == n:
            r = hit[ks[len(ks)//2]] / (qi + 1)
            print(f"  [{qi+1:3d}/{n}] running recall@{ks[len(ks)//2]}={r:.3f}  "
                  f"({(time.time()-t0):.0f}s, {n_embed_calls} embeds)")

    dt = time.time() - t0
    print(f"\n=== Feather LongMemEval — mode={args.mode}"
          f"{'' if embed is None else ' / '+args.embed_provider}  "
          f"n={n}  {dt:.0f}s ===")
    print("Overall recall@k:")
    for k in ks:
        print(f"  recall@{k:<3d} = {hit[k]/n:.3f}")
    print("\nBy question type (recall@%d):" % maxk)
    for t in sorted(by_type_n):
        print(f"  {t:28s} n={by_type_n[t]:3d}  recall@{maxk}={by_type[t][maxk]/by_type_n[t]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/lme_s.json", help="longmemeval_s json")
    ap.add_argument("--mode", choices=["keyword", "dense", "hybrid"], default="keyword")
    ap.add_argument("--embed-provider", default=None,
                    help="gemini|openai|voyage|cohere|ollama|hash (omit for keyword)")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--embed-key", default=None)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0, help="first N questions (0=all)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    run(ap.parse_args())
