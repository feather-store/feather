"""CLI: python -m bench run <scenario> --dataset <name> ..."""
from __future__ import annotations
import argparse
import sys

from .runner import BenchRunner
from .report import write_report
from .datasets.synthetic import load_synthetic
from .datasets.sift import load_sift
from .datasets.longmemeval import load_longmemeval
from .scenarios import vector_ann, vector_ann_real
from .scenarios import longmemeval as lme_scenario
from .embedders import get_embedder
from .judges import get_judge


SCENARIOS = {
    "vector_ann":      "synthetic only — brute-force GT computed locally",
    "vector_ann_real": "datasets that ship pre-computed ground truth (sift1m, siftsmall)",
    "longmemeval":     "long-term memory QA benchmark (Xu et al., 2024)",
}

REAL_DATASETS = {"sift1m", "siftsmall"}
LME_VARIANTS = {"oracle", "s", "m"}


def cmd_run(args):
    runner = BenchRunner(scenario=args.scenario, dataset=args.dataset)

    if args.scenario == "vector_ann":
        if args.dataset != "synthetic":
            print("vector_ann scenario only supports --dataset synthetic", file=sys.stderr)
            return 2
        base, queries = load_synthetic(n=args.n, dim=args.dim,
                                       n_queries=args.queries)
        n, dim = base.shape
        result = runner.run(
            lambda _r: vector_ann.run(base, queries, k=args.k, ef=args.ef),
            n=n, dim=dim, k=args.k, ef=args.ef, queries=len(queries),
        )

    elif args.scenario == "vector_ann_real":
        if args.dataset not in REAL_DATASETS:
            print(f"vector_ann_real needs --dataset one of {sorted(REAL_DATASETS)}",
                  file=sys.stderr)
            return 2
        base, queries, gt = load_sift(
            variant=args.dataset,
            n_base=args.n if args.n > 0 else None,
            n_queries=args.queries if args.queries > 0 else None,
        )
        n, dim = base.shape
        ef_arg = args.ef_sweep if args.ef_sweep else args.ef
        result = runner.run(
            lambda _r: vector_ann_real.run(base, queries, gt, k=args.k, ef=ef_arg),
            n=n, dim=dim, k=args.k, ef=str(ef_arg), queries=len(queries),
        )

    elif args.scenario == "longmemeval":
        if args.dataset not in LME_VARIANTS:
            print(f"longmemeval needs --dataset one of {sorted(LME_VARIANTS)}",
                  file=sys.stderr)
            return 2
        questions = load_longmemeval(
            variant=args.dataset,
            limit=(args.limit if args.limit > 0 else None),
        )

        # Build embedder
        if args.embedder == "openai":
            from .embedders_openai import OpenAIEmbedder
            embedder = OpenAIEmbedder(model=args.embedder_model,
                                      dim=(args.dim or None))
        else:
            embedder = get_embedder(args.embedder, dim=args.dim)

        # Build judge
        if args.judge == "llm":
            from .judges_llm import LLMJudge
            judge = LLMJudge(provider=args.judge_provider,
                             model=args.judge_model,
                             answerer_provider=args.answerer_provider,
                             answerer_model=args.answerer_model)
        else:
            judge = get_judge(args.judge)

        result = runner.run(
            lambda _r: lme_scenario.run(
                questions, embedder=embedder, judge=judge,
                top_k=args.k, ef=args.ef,
            ),
            n=len(questions), dim=embedder.dim, k=args.k,
            ef=args.ef, embedder=embedder.name, judge=judge.name,
            variant=args.dataset,
        )

    else:
        print(f"unknown scenario: {args.scenario}", file=sys.stderr)
        return 2

    # Pretty-print headline metrics (skip nested sweep dict)
    print()
    print(f"=== {args.scenario} / {args.dataset} / n={result.n} dim={result.dim} ===")
    for k, v in result.metrics.items():
        if isinstance(v, dict):
            print(f"  {k:>20} :")
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    summary = (f"recall@{args.k}={sv.get(f'recall@{args.k}', 0):.3f} "
                               f"p50={sv.get('p50_ms', 0):.2f}ms "
                               f"p95={sv.get('p95_ms', 0):.2f}ms")
                    print(f"    {sk:>14} : {summary}")
                else:
                    print(f"    {sk:>14} : {sv}")
        elif isinstance(v, float):
            print(f"  {k:>20} : {v:.3f}")
        else:
            print(f"  {k:>20} : {v}")
    return 0


def cmd_report(args):
    path = write_report()
    print(f"wrote {path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a scenario against a dataset")
    p_run.add_argument("scenario", choices=SCENARIOS.keys())
    p_run.add_argument("--dataset", default="synthetic")
    p_run.add_argument("--n", type=int, default=10_000,
                       help="Subset base set to first N. Use 0 for full dataset.")
    p_run.add_argument("--dim", type=int, default=128,
                       help="Synthetic only.")
    p_run.add_argument("--k", type=int, default=10)
    p_run.add_argument("--ef", type=int, default=None,
                       help="HNSW search beam width. Default = DB default (50).")
    p_run.add_argument("--ef-sweep", type=lambda s: [int(x) for x in s.split(",")],
                       default=None,
                       help="Comma-separated ef values to sweep, e.g. 10,50,100,200")
    p_run.add_argument("--queries", type=int, default=200,
                       help="Subset queries. Use 0 for all.")

    # LongMemEval-specific knobs
    p_run.add_argument("--limit", type=int, default=5,
                       help="LongMemEval: max # questions to evaluate (0 = all).")
    p_run.add_argument("--embedder", default="deterministic",
                       choices=["deterministic", "openai"],
                       help="LongMemEval embedder. Phase 1 = deterministic.")
    p_run.add_argument("--judge", default="substring",
                       choices=["substring", "llm"],
                       help="LongMemEval scorer. Phase 1 = substring (free).")
    p_run.add_argument("--embedder-model", default="text-embedding-3-small",
                       help="OpenAI embedder model name.")
    p_run.add_argument("--judge-provider", default="gemini",
                       choices=["gemini", "claude", "openai", "ollama"])
    p_run.add_argument("--judge-model", default=None,
                       help="Override judge model (default = provider's default).")
    p_run.add_argument("--answerer-provider", default=None,
                       choices=[None, "gemini", "claude", "openai", "ollama"],
                       help="Provider for the answer-generation step. "
                            "Defaults to same as --judge-provider.")
    p_run.add_argument("--answerer-model", default=None,
                       help="Override answerer model.")

    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="Regenerate Markdown report")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
