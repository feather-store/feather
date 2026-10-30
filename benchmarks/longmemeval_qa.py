"""
LongMemEval QA-accuracy benchmark for Feather DB
================================================
The end-to-end metric: retrieve memory from Feather, have an LLM answer the
question from it, and judge the answer against the gold answer. This is the
metric vendor headlines ("92% on LongMemEval") refer to — distinct from the
pure retrieval recall in `longmemeval.py`.

Pipeline per question:
  1. ingest every haystack session as one Feather record
  2. retrieve top-k sessions for the question (hybrid when an embedder is
     given, else keyword; the embedded query carries the question date);
     no decay — temporal questions need old evidence. Count and comparison
     questions get a follow-up round (bench/lme_retrieve.py, --multi-round)
  3. answerer LLM writes an answer from the retrieved sessions, rendered
     by bench.scenarios.lme_context (dated, role-marked, oldest first,
     plus the current date) with bench.lme_answer's per-category prompt
     and DAYS_BETWEEN tool
  4. two judges mark it: LongMemEval's official per-type judge (on
     gpt-4o-2024-08-06 by default, as in the paper) and a lenient
     single-rubric one — see bench/lme_eval.py
  5. one JSON line per question (--out): retrieved sessions, gold coverage,
     answer, both verdicts, failure bucket
Reports both scores overall and per question type, abstention, judge
agreement, retrieval coverage and the failure buckets.

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

Compare the OFFICIAL score with other systems' published numbers; the lenient
one is a second opinion, and the questions the two disagree on are where to
look for judge noise. --official-judge-provider none skips the official judge.
"""
import os, sys, json, time, argparse, tempfile
from collections import defaultdict
import numpy as np

# The repo's feather_db, built in place (no install needed). Imported as the
# package, not by loading core*.so under another name: the embedders and LLM
# clients import feather_db too, and a second load of the extension fails
# ("type ... is already registered"), so every dense, hybrid or LLM run died.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import feather_db as fc  # noqa: E402
from bench import lme_answer, lme_eval, lme_retrieve  # noqa: E402
from bench.scenarios.lme_context import Turn, render_context  # noqa: E402


def session_text(sess):
    return "\n".join(f"{t.get('role','')}: {t.get('content','')}" for t in sess)


def session_turns(item):
    """(every turn as a Turn, {session index: its turns})."""
    sids = item["haystack_session_ids"]
    dates = item.get("haystack_dates") or [""] * len(item["haystack_sessions"])
    turns, by_session = [], {}
    for si, sess in enumerate(item["haystack_sessions"]):
        for t in sess:
            if not (t.get("content") or "").strip():
                continue
            turn = Turn(sids[si], dates[si] if si < len(dates) else "",
                        t.get("role", ""), t["content"], len(turns))
            turns.append(turn)
            by_session.setdefault(si, []).append(turn)
    return turns, by_session


def build_context(item, ranked_idx, max_chars):
    """(context, retrieved session ids, session ids in the context): the
    ranked sessions, dated, role-marked, oldest first. Retrieval ranked whole
    sessions, so every turn of one counts as a hit."""
    turns, by_session = session_turns(item)
    hits = [t for si in ranked_idx for t in by_session.get(si, [])]
    return render_context(hits, turns, unit="session", max_chars=max_chars,
                          max_sessions=len(ranked_idx),
                          question_date=item.get("question_date", ""))


def get_embedder(provider, model, key, dim):
    if provider in (None, "", "none", "keyword"):
        return None
    from feather_db.integrations.embedders import make_embedder
    return make_embedder(provider, model=model, api_key=key, dim=dim)


def run(args):
    data = json.load(open(args.data))
    if args.limit:
        data = data[: args.limit]
    if not data:
        sys.exit(f"{args.data}: no questions")

    embed = get_embedder(args.embed_provider, args.embed_model, args.embed_key, args.dim)
    if args.mode is None:
        args.mode = "hybrid" if embed else "keyword"
    if args.mode != "keyword" and embed is None:
        sys.exit(f"mode={args.mode} needs --embed-provider")
    vdim = args.dim if embed else 8

    answer = judge = official = None
    if not args.dry_run:
        from feather_db.integrations.llm import make_chat
        answer = make_chat(args.answer_provider, args.answer_model, args.answer_key)
        judge = make_chat(args.judge_provider, args.judge_model, args.judge_key)
        if args.official_judge_provider != "none":
            official = make_chat(args.official_judge_provider,
                                 args.official_judge_model, args.official_judge_key)

    records = []
    by_type_n = defaultdict(int)
    n = len(data); t0 = time.time()
    # Per-question files in this run's own directory (see longmemeval.py).
    tmp = tempfile.TemporaryDirectory(prefix="lmeqa_")

    for qi, item in enumerate(data):
        sessions, sids = item["haystack_sessions"], item["haystack_session_ids"]
        gold = item["answer"]; q = item["question"]; qtype = item.get("question_type", "?")

        db = fc.DB.open(os.path.join(tmp.name, f"q{qi}.feather"), dim=vdim)
        for si, sess in enumerate(sessions):
            m = fc.Metadata(); m.content = session_text(sess); m.entity_id = sids[si]
            vec = np.asarray(embed(m.content), dtype=np.float32) if embed else np.zeros(vdim, np.float32)
            db.add(id=si, vec=vec, meta=m)

        qdate = item.get("question_date", "")
        dates = item.get("haystack_dates") or []

        def search(text, after=None, before=None):
            """Ranked session indices; a date window filters the hits, so
            fetch more first."""
            k = args.k * (3 if (after or before) else 1)
            if args.mode == "keyword":
                res = db.keyword_search(text, k=k)
            else:
                v = np.asarray(embed(lme_retrieve.dense_query(text, qdate)),
                               np.float32)
                res = (db.search(v, k=k, record_salience=False)
                       if args.mode == "dense" else db.hybrid_search(v, text, k=k))
            ranked = [r.id for r in res]
            if after or before:
                ranked = [si for si in ranked if lme_retrieve.in_window(
                    dates[si] if si < len(dates) else "", after, before)]
            return ranked[: args.k]

        ranked_idx = search(q)
        round1 = list(ranked_idx)
        followups, missing = [], ""
        if answer is not None and (args.multi_round == "always" or (
                args.multi_round == "auto" and lme_retrieve.needs_multi_round(q))):
            _, by_session = session_turns(item)
            # The planner sees the user's side: what they said is what a
            # follow-up query has to match.
            seen = [t for si in round1 for t in by_session.get(si, [])
                    if t.role == "user"]
            followups, missing = lme_retrieve.plan_followups(
                answer, q, qdate, lme_retrieve.digest(seen, qdate, args.k))
            if followups:
                ranked_idx = [int(x) for x in lme_retrieve.merge_sessions(
                    [str(si) for si in round1],
                    [[str(si) for si in search(f["query"], f["after"], f["before"])]
                     for f in followups], args.k)]
        ctx, got, in_ctx = build_context(item, ranked_idx, args.context_chars)
        db.close(save=False)        # closed, not dropped: see longmemeval.py

        if args.dry_run:
            print(f"\n──── Q{qi} [{qtype}] ────")
            print(f"question: {q}")
            print(f"gold:     {gold}")
            cov = lme_eval.coverage(item["answer_session_ids"], in_ctx)
            print(f"retrieved sessions: {got}  gold: {item['answer_session_ids']}  "
                  f"(coverage in context: {cov})")
            print(f"context chars: {len(ctx)} | current date: "
                  f"{item.get('question_date', '?')} | first block:\n"
                  f"{ctx[:280]}...")
            by_type_n[qtype] += 1
            continue

        t_q = time.time()
        try:
            reply = lme_answer.answer(answer, q, ctx, qdate)
        except Exception as e:     # a failed call is an answer of its own
            reply = f"[answerer error: {e}]"
        a = lme_answer.final_answer(reply)

        def verdict(fn, *args):
            try:
                return fn(*args)
            except Exception as e:
                return {"correct": False, "raw": "", "error": str(e)}
        off = (verdict(lme_eval.official_verdict, official, qtype, q, gold, a,
                       lme_eval.is_abstention(item.get("question_id", "")))
               if official else None)
        len_ = verdict(lme_eval.lenient_verdict, judge, q, gold, a)
        records.append(lme_eval.make_record(
            item, retrieved_sessions=got, context_sessions=in_ctx,
            context_chars=len(ctx), answer=a, transcript=reply,
            official=off, lenient=len_, seconds=time.time() - t_q,
            followups=followups, missing=missing,
            round1_sessions=[sids[si] for si in round1]))
        if (qi + 1) % 20 == 0 or qi + 1 == n:
            head = [(r["official"] or r["lenient"])["correct"] for r in records]
            print(f"  [{qi+1:3d}/{n}] acc={sum(head)/len(head):.3f}  ({time.time()-t0:.0f}s)")

    tmp.cleanup()
    if args.dry_run:
        print(f"\nDRY RUN ok — pipeline assembled prompts for {sum(by_type_n.values())} "
              f"questions across {len(by_type_n)} types. Add --answer-provider to run for real.")
        return

    out = args.out or f"lmeqa_{args.mode}_k{args.k}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    lme_eval.write_jsonl(out, records)
    print(f"\n=== Feather LongMemEval QA accuracy — retrieval={args.mode} (k={args.k}), "
          f"answer={args.answer_provider}, judges: official="
          f"{args.official_judge_provider}/{args.official_judge_model if official else '-'}"
          f", lenient={args.judge_provider}, n={n} ===")
    print(lme_eval.format_summary(lme_eval.summarize(records)))
    print(f"per-question records: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/lme_s.json")
    ap.add_argument("--mode", choices=["keyword", "dense", "hybrid"], default=None,
                    help="default: hybrid with --embed-provider, else keyword")
    ap.add_argument("--multi-round", choices=["auto", "always", "off"], default="auto",
                    help="follow-up retrieval planned by the answer LLM; "
                         "auto = count and comparison questions")
    ap.add_argument("--k", type=int, default=20, help="sessions of context to retrieve")
    ap.add_argument("--context-chars", type=int, default=160_000,
                    help="context budget; sessions past it are shortened, then dropped")
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
    ap.add_argument("--judge-provider", default="gemini", help="lenient judge")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-key", default=None)
    # LongMemEval's official judge (the score to compare with other systems)
    ap.add_argument("--official-judge-provider", default="openai",
                    help="'none' skips it")
    ap.add_argument("--official-judge-model", default=lme_eval.OFFICIAL_JUDGE_MODEL)
    ap.add_argument("--official-judge-key", default=None)
    ap.add_argument("--out", default=None, help="per-question JSONL path")
    run(ap.parse_args())
