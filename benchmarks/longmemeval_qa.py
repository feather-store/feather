"""
LongMemEval QA-accuracy benchmark for Feather DB
================================================
The end-to-end metric: retrieve memory from Feather, have an LLM answer the
question from it, and judge the answer against the gold answer. This is the
metric vendor headlines ("92% on LongMemEval") refer to — distinct from the
pure retrieval recall in `longmemeval.py`.

Pipeline per question:
  1. ingest every haystack session as one Feather record
  2. retrieve top-k sessions for the question (keyword / dense / hybrid)
  3. answerer LLM writes an answer from the retrieved context (date-stamped)
  4. judge LLM marks it CORRECT / INCORRECT vs the gold answer
Reports accuracy overall and per question type.

Usage
-----
    # validate the whole pipeline with NO API calls (prints assembled prompts):
    python benchmarks/longmemeval_qa.py --data /tmp/lme_s.json --mode keyword \
        --limit 3 --dry-run

    # real run (BM25 retrieval, Gemini answer+judge — no embedder needed):
    GOOGLE_API_KEY=... python benchmarks/longmemeval_qa.py --data /tmp/lme_s.json \
        --mode keyword --answer-provider gemini --judge-provider gemini --limit 100

    # dense/hybrid retrieval needs an embedder too:
    GOOGLE_API_KEY=... python benchmarks/longmemeval_qa.py --data /tmp/lme_s.json \
        --mode hybrid --embed-provider gemini --answer-provider gemini

Note: the judge here is a single-rubric semantic-match prompt, simpler than
LongMemEval's official per-type GPT-4o rubric — treat the number as an internal
signal, not an official leaderboard submission.
"""
import os, sys, glob, json, time, argparse, importlib.util
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob(os.path.join(_HERE, "feather_db/core*.so")) if _tag in p]
_so = (_cands or sorted(glob.glob(os.path.join(_HERE, "feather_db/core*.so"))))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)
sys.path.insert(0, _HERE)

ANSWER_SYS = ("You answer questions about a user using excerpts from their past "
              "conversations with an assistant. Use ONLY the provided context. "
              "If the answer is not in the context, reply exactly: I don't know. "
              "Be concise — answer in one short phrase or sentence.")
JUDGE_SYS = ("You are grading an answer. Given the question, the gold answer, and "
             "the model's answer, decide if the model's answer is correct "
             "(it conveys the same key information as the gold answer). "
             "Reply with exactly one word: CORRECT or INCORRECT.")


def session_text(sess):
    return "\n".join(f"{t.get('role','')}: {t.get('content','')}" for t in sess)


def build_context(item, ranked_idx, k):
    dates = item.get("haystack_dates") or [""] * len(item["haystack_sessions"])
    blocks = []
    for si in ranked_idx[:k]:
        d = dates[si] if si < len(dates) else ""
        blocks.append(f"[Conversation on {d}]\n{session_text(item['haystack_sessions'][si])}")
    return "\n\n".join(blocks)


def get_embedder(provider, model, key, dim):
    if provider in (None, "", "none", "keyword"):
        return None
    from feather_db.integrations.embedders import make_embedder
    return make_embedder(provider, model=model, api_key=key, dim=dim)


def run(args):
    data = json.load(open(args.data))
    if args.limit:
        data = data[: args.limit]

    embed = get_embedder(args.embed_provider, args.embed_model, args.embed_key, args.dim)
    if args.mode != "keyword" and embed is None:
        sys.exit(f"mode={args.mode} needs --embed-provider")
    vdim = args.dim if embed else 8

    answer = judge = None
    if not args.dry_run:
        from feather_db.integrations.llm import make_chat
        answer = make_chat(args.answer_provider, args.answer_model, args.answer_key)
        judge = make_chat(args.judge_provider, args.judge_model, args.judge_key)

    correct = 0
    by_type = defaultdict(int); by_type_n = defaultdict(int)
    n = len(data); t0 = time.time()

    for qi, item in enumerate(data):
        sessions, sids = item["haystack_sessions"], item["haystack_session_ids"]
        gold = item["answer"]; q = item["question"]; qtype = item.get("question_type", "?")

        path = f"/tmp/lmeqa_{qi}.feather"
        for ext in ("", ".wal", ".tmp"):
            if os.path.exists(path + ext): os.remove(path + ext)
        db = fc.DB.open(path, dim=vdim)
        for si, sess in enumerate(sessions):
            m = fc.Metadata(); m.content = session_text(sess); m.entity_id = sids[si]
            vec = np.asarray(embed(m.content), dtype=np.float32) if embed else np.zeros(vdim, np.float32)
            db.add(id=si, vec=vec, meta=m)

        if args.mode == "keyword":
            res = db.keyword_search(q, k=args.k)
        elif args.mode == "dense":
            res = db.search(np.asarray(embed(q), np.float32), k=args.k)
        else:
            res = db.hybrid_search(np.asarray(embed(q), np.float32), q, k=args.k)
        ranked_idx = [r.id for r in res]
        ctx = build_context(item, ranked_idx, args.k)

        for ext in ("", ".wal", ".tmp"):
            if os.path.exists(path + ext): os.remove(path + ext)

        user = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        if args.dry_run:
            print(f"\n──── Q{qi} [{qtype}] ────")
            print(f"question: {q}")
            print(f"gold:     {gold}")
            print(f"retrieved sessions (idx): {ranked_idx}  "
                  f"(gold sids={item['answer_session_ids']})")
            print(f"answer-prompt chars: {len(user)} | first ctx block:\n"
                  f"{ctx[:280]}...")
            by_type_n[qtype] += 1
            continue

        a = answer(ANSWER_SYS, user).strip()
        verdict = judge(JUDGE_SYS,
                        f"Question: {q}\nGold answer: {gold}\nModel answer: {a}").strip().upper()
        ok = "CORRECT" in verdict and "INCORRECT" not in verdict
        correct += ok; by_type[qtype] += ok; by_type_n[qtype] += 1
        if (qi + 1) % 20 == 0 or qi + 1 == n:
            print(f"  [{qi+1:3d}/{n}] acc={correct/(qi+1):.3f}  ({time.time()-t0:.0f}s)")

    if args.dry_run:
        print(f"\nDRY RUN ok — pipeline assembled prompts for {sum(by_type_n.values())} "
              f"questions across {len(by_type_n)} types. Add --answer-provider to run for real.")
        return

    print(f"\n=== Feather LongMemEval QA accuracy — retrieval={args.mode} (k={args.k}), "
          f"answer={args.answer_provider}, judge={args.judge_provider}, n={n} ===")
    print(f"Overall accuracy: {correct/n:.3f}")
    print("By question type:")
    for t in sorted(by_type_n):
        print(f"  {t:28s} n={by_type_n[t]:3d}  acc={by_type[t]/by_type_n[t]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/lme_s.json")
    ap.add_argument("--mode", choices=["keyword", "dense", "hybrid"], default="keyword")
    ap.add_argument("--k", type=int, default=5, help="sessions of context to retrieve")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="assemble prompts, make NO API calls")
    # retrieval embedder (dense/hybrid only)
    ap.add_argument("--embed-provider", default=None)
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--embed-key", default=None)
    ap.add_argument("--dim", type=int, default=768)
    # answerer + judge LLMs
    ap.add_argument("--answer-provider", default="gemini")
    ap.add_argument("--answer-model", default=None)
    ap.add_argument("--answer-key", default=None)
    ap.add_argument("--judge-provider", default="gemini")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-key", default=None)
    run(ap.parse_args())
