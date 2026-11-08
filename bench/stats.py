"""Paired statistics for benchmark runs: is a difference bigger than noise?

Two sources of noise sit under every LLM-judged number here:

  rerun noise      the same configuration, rerun, flips questions: the reader
                   and judge are not deterministic at temperature 0. Measured
                   on LoCoMo, same-context reruns flipped ~3% of questions,
                   which is larger than most effects being judged.
  question sample  a benchmark is a sample of questions. A +0.6 on 1,540
                   questions is ~9 questions.

So a comparison takes 3-5 runs of each configuration on the same questions,
and reports, per question q:

    a_q, b_q   the share of A's (B's) runs that got q right
    d_q        b_q - a_q

    difference     mean(d_q), with a 95% bootstrap CI over questions
                   (resampling questions, every run of a question together)
    McNemar        exact two-sided test on the questions whose majority
                   verdict flipped (A-only right vs B-only right); with one
                   run per side it is the classic test on that run's flips
    rerun noise    within each side: the share of questions two runs of the
                   SAME configuration disagree on, and the spread of their
                   accuracies. A difference smaller than the rerun spread is
                   not a finding, whatever its p.

A difference is reported as real only when the CI excludes 0 and McNemar's
p < 0.05. Questions that share a conversation (LoCoMo) or a generated world
(contradiction) are correlated, and resampling them one by one understates
the uncertainty. Given `clusters`, a second interval resamples whole
clusters (conversations, worlds, every question of each), and a difference
is real only if that one excludes 0 as well. Below MIN_CLUSTERS clusters
there is too little to resample: no cluster interval, and a warning that
the question-level one assumes independence.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Dict, Iterable, List, Optional

import numpy as np

Run = Dict[str, float]          # question id -> 1.0 right, 0.0 wrong
MIN_RUNS = 3
MIN_CLUSTERS = 5


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p: b questions only A got right, c only B."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def common_ids(runs: Iterable[Run]) -> List[str]:
    runs = list(runs)
    if not runs:
        return []
    ids = set(runs[0])
    for r in runs[1:]:
        ids &= set(r)
    return sorted(ids)


def per_question(runs: List[Run], ids: List[str]) -> np.ndarray:
    """The share of `runs` that got each question in `ids` right."""
    return np.array([[float(r[q]) for q in ids] for r in runs]).mean(axis=0)


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> tuple:
    """Percentile CI of mean(values), resampling its entries."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    step = max(1, 2_000_000 // n)          # bound the index matrix
    for s in range(0, n_boot, step):
        m = min(step, n_boot - s)
        means[s:s + m] = values[rng.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def cluster_bootstrap_ci(values: np.ndarray, labels: List[str], n_boot: int = 10_000,
                         seed: int = 0, level: float = 0.95) -> tuple:
    """Percentile CI of mean(values) resampling CLUSTERS: each draw takes
    k clusters with replacement and every value in them."""
    names = sorted(set(labels))
    idx = {c: i for i, c in enumerate(names)}
    sums, sizes = np.zeros(len(names)), np.zeros(len(names))
    for v, c in zip(values, labels):
        sums[idx[c]] += v
        sizes[idx[c]] += 1
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(names), size=(n_boot, len(names)))
    means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def rerun_noise(runs: List[Run], ids: Optional[List[str]] = None) -> dict:
    """How much reruns of ONE configuration disagree: the share of questions
    two runs differ on (mean over pairs), and the spread of accuracies."""
    ids = ids if ids is not None else common_ids(runs)
    accs = [float(np.mean([r[q] for q in ids])) for r in runs] if ids else []
    flips = [float(np.mean([r1[q] != r2[q] for q in ids]))
             for r1, r2 in combinations(runs, 2)] if ids else []
    return {"runs": len(runs),
            "accuracy_per_run": [round(a, 4) for a in accs],
            "accuracy_sd": round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else None,
            "flip_rate": round(float(np.mean(flips)), 4) if flips else None}


def describe(runs: List[Run], ids: Optional[List[str]] = None, **boot) -> dict:
    """One configuration: mean accuracy over its runs with a 95% CI."""
    ids = ids if ids is not None else common_ids(runs)
    q = per_question(runs, ids)
    lo, hi = bootstrap_ci(q, **boot)
    return {"n": len(ids), "accuracy": round(float(q.mean()), 4) if len(ids) else None,
            "ci95": [round(lo, 4), round(hi, 4)], **rerun_noise(runs, ids)}


def compare(runs_a: List[Run], runs_b: List[Run], *, groups: Optional[Dict[str, str]] = None,
            clusters: Optional[Dict[str, str]] = None,
            n_boot: int = 10_000, seed: int = 0, deterministic: bool = False) -> dict:
    """B against A on the questions every run of both answered. `groups`
    (question id -> label, e.g. a category) adds a breakdown. `deterministic`:
    the metric has no rerun noise (no model in it), so one run is enough and
    the CI is question sampling alone."""
    ids = common_ids(runs_a + runs_b)
    out = _compare_on(runs_a, runs_b, ids, n_boot, seed, clusters)
    out["unpaired"] = {"only_some_runs_of_a_or_b":
                       len(set().union(*map(set, runs_a + runs_b)) - set(ids))}
    if groups:
        labels = sorted({groups[q] for q in ids if q in groups})
        out["by_group"] = {g: _compare_on(runs_a, runs_b,
                                          [q for q in ids if groups.get(q) == g], n_boot, seed,
                                          clusters)
                           for g in labels}
    warn = []
    if deterministic:
        out["note"] = "deterministic metric: no rerun noise; the CI is question sampling"
    elif min(len(runs_a), len(runs_b)) < MIN_RUNS:
        warn.append(f"{len(runs_a)} run(s) of A and {len(runs_b)} of B: with fewer than "
                    f"{MIN_RUNS} per side the rerun noise is not measured, and a difference "
                    "cannot be told from it. Run 3-5 of each.")
    if clusters and out.get("n_clusters") is not None and out["n_clusters"] < MIN_CLUSTERS:
        warn.append(f"{out['n_clusters']} clusters (conversations / worlds): too few to "
                    "resample; the interval treats their questions as independent, which "
                    "understates the uncertainty")
    out["warnings"] = warn
    return out


def _compare_on(runs_a, runs_b, ids, n_boot, seed, clusters=None) -> dict:
    if not ids:
        return {"n": 0}
    qa, qb = per_question(runs_a, ids), per_question(runs_b, ids)
    d = qb - qa
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    # Majority verdict per question; a tie (an even number of runs split
    # evenly) is neither side's, and drops out of the flip counts.
    ma, mb = np.sign(qa - 0.5), np.sign(qb - 0.5)
    a_only = [q for q, x, y in zip(ids, ma, mb) if x > 0 and y < 0]
    b_only = [q for q, x, y in zip(ids, ma, mb) if x < 0 and y > 0]
    p = mcnemar_p(len(a_only), len(b_only))
    diff = float(d.mean())
    real = (lo > 0 or hi < 0) and p < 0.05
    cl = None
    n_clusters = len({clusters[q] for q in ids}) if clusters else None
    if clusters and n_clusters >= MIN_CLUSTERS:
        cl = cluster_bootstrap_ci(d, [clusters[q] for q in ids], n_boot=n_boot, seed=seed)
        real = real and (cl[0] > 0 or cl[1] < 0)
    noise_a, noise_b = rerun_noise(runs_a, ids), rerun_noise(runs_b, ids)
    return {
        "n": len(ids),
        "a": round(float(qa.mean()), 4), "b": round(float(qb.mean()), 4),
        "diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "n_clusters": n_clusters,
        "ci95_cluster": [round(cl[0], 4), round(cl[1], 4)] if cl else None,
        "a_only": len(a_only), "b_only": len(b_only), "mcnemar_p": p,
        "fixed": b_only, "broken": a_only,
        "noise_a": noise_a, "noise_b": noise_b,
        "verdict": ("B better" if real and diff > 0 else "B worse" if real
                    else "not distinguishable from noise"),
    }


def _p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def format_compare(c: dict, label_a: str = "A", label_b: str = "B", pct: bool = True) -> str:
    f = (lambda x: f"{100 * x:.2f}") if pct else (lambda x: f"{x:.4f}")
    df = (lambda x: f"{100 * x:+.2f}") if pct else (lambda x: f"{x:+.4f}")

    def row(name, s):
        if not s.get("n"):
            return f"{name:24s} n=0"
        cl = s.get("ci95_cluster")
        return (f"{name:24s} n={s['n']:<5d} {label_a} {f(s['a'])}  {label_b} {f(s['b'])}  "
                f"diff {df(s['diff'])} [{df(s['ci95'][0])}, {df(s['ci95'][1])}]"
                + (f" by {s['n_clusters']} clusters [{df(cl[0])}, {df(cl[1])}]" if cl else "")
                + "  "
                f"flips {s['a_only']}/{s['b_only']}  McNemar p={_p(s['mcnemar_p'])}  "
                f"-> {s['verdict']}")

    lines = [row("overall", c)]
    for g, s in (c.get("by_group") or {}).items():
        lines.append(row(f"  {g}", s))
    for side, key in ((label_a, "noise_a"), (label_b, "noise_b")):
        nz = c.get(key) or {}
        if nz.get("runs", 0) > 1:
            lines.append(f"rerun noise {side}: {nz['runs']} runs, accuracy "
                         f"{', '.join(f(a) for a in nz['accuracy_per_run'])} "
                         f"(sd {f(nz['accuracy_sd'])}), two runs disagree on "
                         f"{f(nz['flip_rate'])}% of questions")
    if c.get("unpaired", {}).get("only_some_runs_of_a_or_b"):
        lines.append(f"unpaired: {c['unpaired']['only_some_runs_of_a_or_b']} questions "
                     "missing from some run (excluded)")
    if c.get("note"):
        lines.append(c["note"])
    lines += [f"WARNING: {w}" for w in c.get("warnings", [])]
    return "\n".join(lines)
