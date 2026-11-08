"""Compare two configurations of a benchmark, run by run, question by question.

    python -m bench.compare lme --a A1.jsonl A2.jsonl A3.jsonl --b B1.jsonl B2.jsonl B3.jsonl --split dev
    python -m bench.compare locomo --a A_r*.jsonl --b B_r*.jsonl --key sound --split test
    python -m bench.compare contradiction --a dev.json --variant-a notes \\
        --b dev.json --variant-b "notes + keys" --metric current_first

Each side is 3-5 runs of ONE configuration on the same questions (one file
per run; for contradiction, the repeats inside a report). Prints, overall and
per question type / category / attribute: both accuracies, the difference
with a 95% bootstrap CI, the questions that flipped and McNemar's exact p,
the rerun noise of each side, and a verdict. See bench/stats.py for what each
number means, and bench/protocol.py for the dev/test rule: compare on dev to
choose; run test once a choice is frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from . import stats


# ── adapters: a file -> runs of {question id: 0/1} and a group per id ─────

def lme_runs(paths, judge="official", split="all"):
    from .lme_eval import only_split, read_jsonl
    runs, groups = [], {}
    for p in paths:
        recs = only_split(read_jsonl(p), split)
        # A question that failed to run or to be judged counts wrong, as in
        # lme_eval: dropping it would shrink the denominator.
        runs.append({r["question_id"]: float(bool((r.get(judge) or {}).get("correct")))
                     for r in recs})
        groups.update({r["question_id"]: r.get("question_type", "?") for r in recs})
    return runs, groups


def locomo_runs(paths, key="original", split="all", judge="mem0"):
    from . import locomo
    corr = None if key == "original" else locomo.audit_corrections()
    runs, groups = [], {}
    for p in paths:
        recs = locomo.in_split(locomo.scored(locomo.read_records(p), key, corr, judge), split)
        runs.append({locomo.qid(r): float(r["llm_score"]) for r in recs})
        groups.update({locomo.qid(r): f"{r['category']} {r['category_name']}" for r in recs})
    return runs, groups


def clusters_of(bench: str, ids) -> dict | None:
    """What a question's correlated with: its LoCoMo conversation, or its
    contradiction world (trial). LongMemEval questions each have their own
    haystack: no clusters."""
    if bench == "locomo":
        return {q: q.split("#")[0] for q in ids}
    if bench == "contradiction":
        return {q: q.split("|")[0] for q in ids}
    return None


def contradiction_runs(paths, variant, metric="current_first", order=None):
    """Structural metrics are one deterministic run per report; `reader_correct`
    gives one run per reader repeat."""
    runs, groups = [], {}
    for p in paths:
        rep = json.loads(open(p).read())
        if "records" not in rep:
            raise SystemExit(f"{p}: no per-query records (written before 2026-11-08); rerun it")
        if metric == "reader_correct":
            by_rep = defaultdict(dict)
            for o in (rep.get("reader") or {}).get("outcomes", []):
                if o["variant"] == variant and (order is None or o["order"] == order):
                    by_rep[o["rep"]][o["qid"]] = float(o["correct"])
            if not by_rep:
                raise SystemExit(f"{p}: no reader outcomes for {variant!r} (run with --model)")
            runs += [by_rep[r] for r in sorted(by_rep)]
        else:
            run = {}
            for r in rep["records"]:
                if r["variant"] == variant and (order is None or r["order"] == order):
                    run[r["qid"]] = float(r[metric])
                    groups[r["qid"]] = r["attr"]
            if not run:
                raise SystemExit(f"{p}: no records for variant {variant!r}")
            runs.append(run)
    return runs, groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", choices=("lme", "locomo", "contradiction"))
    ap.add_argument("--a", nargs="+", required=True, help="runs of configuration A")
    ap.add_argument("--b", nargs="+", required=True, help="runs of configuration B")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="all")
    ap.add_argument("--judge", choices=("official", "lenient", "mem0", "strict"), default=None,
                    help="lme: official (default) or lenient; locomo: mem0 (default, "
                         "the published systems' judge) or strict (after rejudge)")
    ap.add_argument("--key", choices=("original", "sound", "audited"), default="original",
                    help="locomo: answer key (bench.locomo KEYS)")
    ap.add_argument("--metric", default="current_first",
                    choices=("current_first", "current_in", "stale_only", "reader_correct"),
                    help="contradiction: what counts as right (stale_only: lower is better)")
    ap.add_argument("--variant-a", default=None, help="contradiction: A's variant")
    ap.add_argument("--variant-b", default=None, help="contradiction: B's variant "
                    "(default: A's)")
    ap.add_argument("--order", default=None, choices=("in order", "shuffled"),
                    help="contradiction: one ingestion order only")
    ap.add_argument("--json", default=None, help="also write the comparison here")
    a = ap.parse_args(argv)

    deterministic = False
    allowed = {"lme": ("official", "lenient"), "locomo": ("mem0", "strict"),
               "contradiction": ()}[a.bench]
    if a.judge is not None and a.judge not in allowed:
        ap.error(f"--judge {a.judge} is not a {a.bench} judge ({', '.join(allowed) or 'none'})")
    if a.bench == "lme":
        judge = a.judge or "official"
        ra, groups = lme_runs(a.a, judge, a.split)
        rb, gb = lme_runs(a.b, judge, a.split)
    elif a.bench == "locomo":
        judge = a.judge or "mem0"
        ra, groups = locomo_runs(a.a, a.key, a.split, judge)
        rb, gb = locomo_runs(a.b, a.key, a.split, judge)
    else:
        if not a.variant_a:
            ap.error("contradiction needs --variant-a")
        ra, groups = contradiction_runs(a.a, a.variant_a, a.metric, a.order)
        rb, gb = contradiction_runs(a.b, a.variant_b or a.variant_a, a.metric, a.order)
        deterministic = a.metric != "reader_correct"
    groups.update(gb)
    c = stats.compare(ra, rb, groups=groups, deterministic=deterministic,
                      clusters=clusters_of(a.bench, stats.common_ids(ra + rb)))
    print(f"{a.bench}: B ({len(rb)} run(s)) vs A ({len(ra)} run(s)), split {a.split}"
          + (f", key {a.key}" if a.bench == "locomo" else "")
          + (f", {a.metric}" if a.bench == "contradiction" else ""))
    print(stats.format_compare(c))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(c, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
