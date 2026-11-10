"""LongMemEval accuracy against context size: one row per configuration.

    python -m bench.lme_curve bench/results/longmemeval__s__*.json
    python -m bench.lme_curve --ref "k=20 40k tok" RESULT.json ...

Takes result JSONs (each names its per-question file and its parameters);
runs of one configuration are pooled as reruns. A row reports what reached
the reader, not what was asked for: follow-up retrieval adds sessions past
--k, and a budget that never binds leaves the context smaller than it.

    acc         official judge, mean over the configuration's runs, 95% CI
    vs ref      paired difference from the reference row on the same
                questions, its CI, McNemar p (bench.stats)
    tokens      context tokens per question (answer model's tokenizer):
                mean and p90. Char-budgeted runs have no count; their
                size is shown as chars / 4.54 (measured on LongMemEval_S
                haystacks with the Gemini tokenizer), marked ≈
    sessions    sessions in the context, mean
    gold in     share of questions with every evidence session in context
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from . import stats
from .lme_eval import read_jsonl

CHARS_PER_TOKEN = 4.54
TYPES = (("knowledge-update", "KU"), ("multi-session", "MS"),
         ("single-session-assistant", "SSA"),
         ("single-session-preference", "PREF"),
         ("single-session-user", "SSU"), ("temporal-reasoning", "TR"))


def label(params: dict) -> str:
    if params.get("full_context"):
        return "full context"
    budget = (f"{params['context_tokens'] // 1000}k tok"
              if params.get("context_tokens")
              else f"{params.get('context_chars', 160_000) // 1000}k chars")
    return f"k={params.get('k')} {budget}"


def load(paths: list[str]) -> dict[str, list[list[dict]]]:
    """label -> runs, each a list of per-question records."""
    by: dict[str, list[list[dict]]] = defaultdict(list)
    for p in paths:
        res = json.load(open(p))
        qf = res.get("metrics", {}).get("questions_file")
        if not qf:
            print(f"skip {p}: no questions_file", file=sys.stderr)
            continue
        if not os.path.isabs(qf) and not os.path.exists(qf):
            qf = os.path.join(os.path.dirname(p), os.path.basename(qf))
        by[label(res.get("params", {}))].append(read_jsonl(qf))
    return by


def _run(recs: list[dict]) -> stats.Run:
    return {r["question_id"]: float(bool((r.get("official") or {}).get("correct")))
            for r in recs}


def row(name: str, runs: list[list[dict]], ref: list[list[dict]] | None) -> str:
    rs = [_run(r) for r in runs]
    d = stats.describe(rs)
    recs = [r for run in runs for r in run]
    toks = [r["context_tokens"] for r in recs if r.get("context_tokens")]
    approx = not toks
    if approx:
        toks = [r["context_chars"] / CHARS_PER_TOKEN for r in recs
                if r.get("context_chars")]
    tok = (f"{'≈' if approx else ''}{np.mean(toks) / 1000:.1f}k/"
           f"{np.percentile(toks, 90) / 1000:.1f}k") if toks else "-"
    sess = np.mean([len(r.get("context_sessions") or ()) for r in recs])
    gold = np.mean([(r.get("coverage") or 0) >= 1.0 for r in recs])
    acc = (f"{100 * d['accuracy']:.1f} [{100 * d['ci95'][0]:.1f},"
           f"{100 * d['ci95'][1]:.1f}]")
    vs = ""
    if ref is not None and runs is not ref:
        c = stats.compare([_run(r) for r in ref], rs)
        vs = (f"{100 * c['diff']:+.1f} [{100 * c['ci95'][0]:+.1f},"
              f"{100 * c['ci95'][1]:+.1f}] p={c['mcnemar_p']:.3f}")
    per = []
    for t, short in TYPES:
        sub = [[r for r in run if r.get("question_type") == t] for run in runs]
        n = sum(len(s) for s in sub)
        per.append(f"{short} {100 * sum(bool((r.get('official') or {}).get('correct')) for s in sub for r in s) / n:.0f}"
                   if n else f"{short} -")
    return (f"{name:16s} {len(runs):>2d} {d['n']:>4d}  {acc:20s} {vs:28s} "
            f"{tok:>12s} {sess:5.1f} {100 * gold:5.1f}%  " + " ".join(per))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result JSONs (not __questions.jsonl)")
    ap.add_argument("--ref", default=None,
                    help="row the others are compared with (default: the "
                         "largest non-full context)")
    a = ap.parse_args(argv)
    by = load(a.results)
    if not by:
        return 1

    def size(name):
        recs = [r for run in by[name] for r in run]
        t = [r.get("context_tokens") or r.get("context_chars", 0) / CHARS_PER_TOKEN
             for r in recs]
        return float(np.mean(t)) if t else 0.0
    order = sorted(by, key=size)
    ref = a.ref or max((n for n in order if n != "full context"), key=size,
                       default=order[-1])
    if ref not in by:
        print(f"--ref {ref!r} is not one of {order}", file=sys.stderr)
        return 2
    print(f"{'config':16s} {'runs':>2s} {'n':>4s}  {'acc % [95% CI]':20s} "
          f"{'vs ' + ref:28s} {'tokens mean/p90':>12s} {'sess':>5s} {'gold in':>6s}  by type")
    for name in order:
        print(row(name, by[name], by[ref]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
