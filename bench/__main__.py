"""CLI: python -m bench run <scenario> --dataset <name> ..."""
from __future__ import annotations
import argparse
import sys

from .runner import BenchRunner
from .report import write_report
from .datasets.synthetic import load_synthetic
from .scenarios import vector_ann


SCENARIOS = {
    "vector_ann": vector_ann.run,
}

DATASETS = {
    "synthetic": load_synthetic,
}


def cmd_run(args):
    scenario_fn = SCENARIOS.get(args.scenario)
    if not scenario_fn:
        print(f"unknown scenario: {args.scenario}", file=sys.stderr)
        return 2

    dataset_fn = DATASETS.get(args.dataset)
    if not dataset_fn:
        print(f"unknown dataset: {args.dataset}", file=sys.stderr)
        return 2

    base, queries = dataset_fn(n=args.n, dim=args.dim, n_queries=args.queries)
    runner = BenchRunner(scenario=args.scenario, dataset=args.dataset)

    result = runner.run(
        lambda _r: scenario_fn(base, queries, k=args.k, ef=args.ef),
        n=args.n, dim=args.dim, k=args.k, ef=args.ef, queries=args.queries,
    )

    # Pretty-print headline metrics
    print()
    print(f"=== {args.scenario} / {args.dataset} / n={args.n} dim={args.dim} ===")
    for k, v in result.metrics.items():
        if isinstance(v, float):
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
    p_run.add_argument("--dataset", default="synthetic", choices=DATASETS.keys())
    p_run.add_argument("--n", type=int, default=10_000)
    p_run.add_argument("--dim", type=int, default=128)
    p_run.add_argument("--k", type=int, default=10)
    p_run.add_argument("--ef", type=int, default=50)
    p_run.add_argument("--queries", type=int, default=200)
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="Regenerate Markdown report")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
