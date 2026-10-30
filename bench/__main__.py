"""CLI: python -m bench run <scenario> --dataset <name> ..."""
from __future__ import annotations
import argparse
import sys
import time

from .runner import BenchRunner, RESULTS_DIR
from .report import write_report
from .datasets.synthetic import load_synthetic
from .datasets.sift import load_sift
from .datasets.longmemeval import load_longmemeval
from .scenarios import vector_ann, vector_ann_real
from .scenarios import longmemeval as lme_scenario
from .scenarios import longmemeval_phase9 as lme9_scenario
from .embedders import get_embedder
from .judges import get_judge


SCENARIOS = {
    "vector_ann":         "synthetic only — brute-force GT computed locally",
    "vector_ann_real":    "datasets that ship pre-computed ground truth (sift1m, siftsmall)",
    "longmemeval":        "long-term memory QA benchmark (Xu et al., 2024)",
    "longmemeval_phase9": "longmemeval with Phase 9 fact extraction at ingest",
}

REAL_DATASETS = {"sift1m", "siftsmall"}
LME_VARIANTS = {"oracle", "s", "m"}


def cmd_run(args):
    lme = args.scenario.startswith("longmemeval")
    if args.k is None:
        args.k = 20 if lme else 10      # longmemeval: sessions, not turns
    if args.dim is None and not (lme and args.embedder in ("openai", "gemini")):
        args.dim = 128                  # openai embedder: model's native dim
    if lme and (args.decay_time_weight or args.decay_half_life):
        # Temporal and multi-session questions need old evidence, and decay
        # ranks it lower. Recency reaches the answerer as session dates and
        # the "latest wins" rule in the prompt instead.
        print("longmemeval runs without decay; drop --decay-half-life / "
              "--decay-time-weight", file=sys.stderr)
        return 2

    runner = BenchRunner(scenario=args.scenario, dataset=args.dataset)

    official_judge = questions_path = None
    if lme:
        questions_path = RESULTS_DIR / (
            f"{args.scenario}__{args.dataset}__"
            f"{time.strftime('%Y%m%d_%H%M%S')}__questions.jsonl")
        if args.judge == "llm" and args.official_judge_provider != "none":
            from .judges_llm import OfficialJudge
            official_judge = OfficialJudge(args.official_judge_provider,
                                           args.official_judge_model)

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

    elif args.scenario == "longmemeval_phase9":
        if args.dataset not in LME_VARIANTS:
            print(f"longmemeval_phase9 needs --dataset one of {sorted(LME_VARIANTS)}",
                  file=sys.stderr)
            return 2
        questions = load_longmemeval(variant=args.dataset, limit=args.limit)
        # Embedder
        if args.embedder == "openai":
            from .embedders_openai import OpenAIEmbedder
            embedder = OpenAIEmbedder(model=args.embedder_model,
                                      dim=(args.dim or None))
        elif args.embedder == "gemini":
            from .providers_extra import GeminiEmbedder
            embedder = GeminiEmbedder(dim=(args.dim or None))
        else:
            embedder = get_embedder(args.embedder, dim=args.dim)
        # Judge
        if args.judge == "llm":
            from .judges_llm import LLMJudge
            judge = LLMJudge(provider=args.judge_provider,
                             model=args.judge_model,
                             answerer_provider=args.answerer_provider,
                             answerer_model=args.answerer_model,
                             max_context_chars=args.context_chars,
                             answer_max_tokens=args.answer_max_tokens,
                             count_votes=args.count_votes)
        else:
            judge = get_judge(args.judge)
        # System LLM (always ours; Phase 9's "internal" role)
        from .judges_llm import _provider_from_name
        system_provider = _provider_from_name(
            args.system_provider, args.system_model)
        result = runner.run(
            lambda _r: lme9_scenario.run(
                questions, embedder=embedder, judge=judge,
                system_provider=system_provider,
                top_k=args.k, ef=args.ef,
                scoring_half_life=args.decay_half_life,
                scoring_time_weight=args.decay_time_weight,
                max_facts_per_turn=args.max_facts_per_turn,
                unit=args.unit, context_chars=args.context_chars,
                official_judge=official_judge, questions_path=questions_path,
            ),
            n=len(questions), dim=embedder.dim, k=args.k,
            ef=args.ef, embedder=embedder.name, judge=judge.name,
            variant=args.dataset, scenario_variant="phase9",
            official_judge=getattr(official_judge, "name", None),
            unit=args.unit, context_chars=args.context_chars,
            system_provider=args.system_provider,
            system_model=args.system_model,
            decay_half_life=args.decay_half_life,
            decay_time_weight=args.decay_time_weight,
            max_facts_per_turn=args.max_facts_per_turn,
        )

    elif args.scenario == "longmemeval":
        if args.dataset not in LME_VARIANTS:
            print(f"longmemeval needs --dataset one of {sorted(LME_VARIANTS)}",
                  file=sys.stderr)
            return 2
        if args.split != "all" and (args.dataset != "s"
                                    or args.limit is not None):
            print("--split needs --dataset s and no --limit", file=sys.stderr)
            return 2
        questions = load_longmemeval(variant=args.dataset, limit=args.limit)
        if args.split != "all":
            from .datasets.lme_split import select
            questions = select(questions, args.split)
        count_tokens = None
        if args.context_tokens:
            from .tokens import gemini_counter
            count_tokens = gemini_counter(args.tokenizer_model)

        # Build embedder
        if args.embedder == "openai":
            from .embedders_openai import OpenAIEmbedder
            embedder = OpenAIEmbedder(model=args.embedder_model,
                                      dim=(args.dim or None))
        elif args.embedder == "gemini":
            from .providers_extra import GeminiEmbedder
            embedder = GeminiEmbedder(dim=(args.dim or None))
        else:
            embedder = get_embedder(args.embedder, dim=args.dim)

        # Build judge
        if args.judge == "llm":
            from .judges_llm import LLMJudge
            judge = LLMJudge(provider=args.judge_provider,
                             model=args.judge_model,
                             answerer_provider=args.answerer_provider,
                             answerer_model=args.answerer_model,
                             # A token budget is the budget; a char
                             # backstop would cut what it admitted.
                             max_context_chars=(None if args.context_tokens
                                                else args.context_chars),
                             answer_max_tokens=args.answer_max_tokens,
                             count_votes=args.count_votes,
                             map_provider=args.map_provider,
                             map_model=args.map_model)
        else:
            judge = get_judge(args.judge)

        fact_cache = None
        if args.facts == "session":
            from .judges_llm import _provider_from_name
            from .lme_facts import SessionFactCache
            if args.system_provider == "gemini-text":
                from .providers_extra import GeminiTextProvider
                extractor_llm = GeminiTextProvider(args.system_model, think=False)
            else:
                extractor_llm = _provider_from_name(args.system_provider,
                                                    args.system_model)
            fact_cache = SessionFactCache(
                extractor_llm,
                f"{args.system_provider}-{args.system_model or 'default'}")

        result = runner.run(
            lambda _r: lme_scenario.run(
                questions, embedder=embedder, judge=judge,
                top_k=args.k, ef=args.ef,
                scoring_half_life=args.decay_half_life,
                scoring_time_weight=args.decay_time_weight,
                unit=args.unit, context_chars=args.context_chars,
                official_judge=official_judge, questions_path=questions_path,
                multi_round=args.multi_round, fact_cache=fact_cache,
                count_map=args.count_map, workers=args.workers,
                context_tokens=args.context_tokens, count_tokens=count_tokens,
                reader=args.reader,
            ),
            n=len(questions), dim=embedder.dim, k=args.k,
            ef=args.ef, embedder=embedder.name, judge=judge.name,
            variant=args.dataset, unit=args.unit,
            multi_round=args.multi_round, facts=args.facts,
            count_map=args.count_map, split=args.split, reader=args.reader,
            context_tokens=args.context_tokens,
            tokenizer_model=(args.tokenizer_model if args.context_tokens
                             else None),
            count_votes=args.count_votes, context_chars_budget=args.context_chars,
            system_provider=(args.system_provider if fact_cache else None),
            system_model=(args.system_model if fact_cache else None),
            official_judge=getattr(official_judge, "name", None),
            context_chars=args.context_chars,
            decay_half_life=args.decay_half_life,
            decay_time_weight=args.decay_time_weight,
            verdict=lambda metrics, dirty: lme_reportability(
                args, metrics, dirty),
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
                if isinstance(sv, dict) and f"recall@{args.k}" in sv:
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
    if "reportable" in result.metrics:
        why = result.metrics["not_reportable_because"]
        print("\n" + ("REPORTABLE: the full benchmark, official judge, "
                      "committed code, no errors" if not why else
                      "NOT REPORTABLE:\n" + "\n".join(f"  - {w}" for w in why)))
    return 0


def lme_reportability(args, metrics: dict, dirty: bool) -> list[str]:
    """Why a LongMemEval result is not a number to put next to another
    system's: anything that changes WHAT was measured, or measured it
    with something other than the official judge. Each of these once
    produced a confident score that meant something else (`--limit`
    defaulted to 5 and a "final" run scored 5 questions)."""
    from .lme_eval import OFFICIAL_JUDGE_MODEL
    why = []
    if args.limit is not None:
        why.append(f"--limit {args.limit}: the first {args.limit} questions")
    if args.split != "all":
        why.append(f"--split {args.split}: not all 500 questions")
    if args.dataset == "oracle":
        why.append("--dataset oracle: the evidence sessions are given")
    if args.judge != "llm":
        why.append(f"--judge {args.judge}: no LLM reader or judge")
    if args.embedder == "deterministic":
        why.append("--embedder deterministic: hash vectors, no semantics")
    if not metrics.get("official_judge"):
        why.append("no official judge: the score is the lenient one")
    elif OFFICIAL_JUDGE_MODEL not in str(metrics["official_judge"]):
        why.append(f"official judge is {metrics['official_judge']}, "
                   f"not {OFFICIAL_JUDGE_MODEL}")
    n_q, n_s = metrics.get("n_questions"), metrics.get("n_scored")
    if n_q is not None and n_s != n_q:
        why.append(f"{n_s} of {n_q} questions scored")
    for key, what in (("run_errors", "failed to run (counted wrong)"),
                      ("judge_errors", "were not judged (counted wrong)")):
        if metrics.get(key):
            why.append(f"{metrics[key]} questions {what}: rerun them")
    if dirty:
        why.append("uncommitted changes: no commit reproduces this run")
    return why


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
    p_run.add_argument("--dim", type=int, default=None,
                       help="Vector dim (default 128). longmemeval with "
                            "--embedder openai defaults to the model's "
                            "native dim: 128 truncates text-embedding-3.")
    p_run.add_argument("--k", type=int, default=None,
                       help="top-k (default 10); longmemeval: sessions of "
                            "context (default 20)")
    p_run.add_argument("--ef", type=int, default=None,
                       help="HNSW search beam width. Default = DB default (50).")
    p_run.add_argument("--ef-sweep", type=lambda s: [int(x) for x in s.split(",")],
                       default=None,
                       help="Comma-separated ef values to sweep, e.g. 10,50,100,200")
    p_run.add_argument("--queries", type=int, default=200,
                       help="Subset queries. Use 0 for all.")

    # LongMemEval-specific knobs
    p_run.add_argument("--limit", type=lambda s: int(s) or None, default=None,
                       help="LongMemEval: only the first N questions, for a "
                            "smoke run (default: all; 0 also means all). A "
                            "limited run is marked not reportable.")
    p_run.add_argument("--embedder", default="deterministic",
                       choices=["deterministic", "openai", "gemini"],
                       help="LongMemEval embedder. Phase 1 = deterministic.")
    p_run.add_argument("--judge", default="substring",
                       choices=["substring", "llm"],
                       help="LongMemEval scorer. Phase 1 = substring (free).")
    p_run.add_argument("--embedder-model", default="text-embedding-3-large",
                       help="OpenAI embedder model (Azure: AZURE_OPENAI_"
                            "DEPLOYMENT overrides it).")
    p_run.add_argument("--judge-provider", default="gemini",
                       choices=["gemini", "claude", "openai", "ollama",
                                "azure", "azure-openai", "gemini-text", "openrouter"])
    p_run.add_argument("--judge-model", default=None,
                       help="Override judge model (default = provider's default).")
    p_run.add_argument("--answerer-provider", default=None,
                       choices=[None, "gemini", "claude", "openai", "ollama",
                                "azure", "azure-openai", "gemini-text", "openrouter"],
                       help="Provider for the answer-generation step. "
                            "Defaults to same as --judge-provider.")
    p_run.add_argument("--answerer-model", default=None,
                       help="Override answerer model.")
    p_run.add_argument("--decay-half-life", type=float, default=None,
                       help="Engage adaptive decay: half-life in days. "
                            "Refused for longmemeval.")
    p_run.add_argument("--decay-time-weight", type=float, default=None,
                       help="Adaptive decay time-weight in [0,1]. Engages "
                            "ScoringConfig in retrieval. 0.3-0.5 typical.")
    p_run.add_argument("--unit", default="session",
                       choices=["session", "turn"],
                       help="longmemeval: expand retrieved turns to their "
                            "sessions (k = sessions), or keep turns only.")
    p_run.add_argument("--official-judge-provider", default="openai",
                       choices=["openai", "azure", "azure-openai", "claude",
                                "gemini", "ollama", "gemini-text",
                                "openrouter", "none"],
                       help="longmemeval + --judge llm: LongMemEval's own "
                            "per-type judge, scored alongside the lenient "
                            "one. 'none' skips it.")
    p_run.add_argument("--official-judge-model",
                       default="gpt-4o-2024-08-06",
                       help="Official judge model (the paper's). For "
                            "openrouter: openai/gpt-4o-2024-08-06. For azure, "
                            "the deployment name.")
    p_run.add_argument("--multi-round", default="auto",
                       choices=["auto", "always", "off"],
                       help="longmemeval: follow-up retrieval round planned "
                            "by the answer LLM. auto = count and comparison "
                            "questions only.")
    p_run.add_argument("--workers", type=int, default=8,
                       help="longmemeval: questions answered in parallel.")
    p_run.add_argument("--answer-max-tokens", type=int, default=2048,
                       help="Answerer output cap. Reasoning models need more.")
    p_run.add_argument("--count-votes", type=int, default=1,
                       help="longmemeval: answer count questions this many "
                            "times (1 greedy + sampled), majority wins.")
    p_run.add_argument("--map-provider", default=None,
                       choices=[None, "gemini", "claude", "openai", "ollama",
                                "azure", "azure-openai", "gemini-text",
                                "openrouter"],
                       help="longmemeval --count-map: provider for the "
                            "per-session map calls (default: the answerer).")
    p_run.add_argument("--map-model", default=None,
                       help="longmemeval --count-map: model for the map "
                            "calls, e.g. a flash model under a pro answerer.")
    p_run.add_argument("--split", default="all",
                       choices=["all", "dev", "test"],
                       help="longmemeval -d s: the frozen split "
                            "(bench/datasets/longmemeval_s_dev.json). Tune "
                            "on dev; run test once per frozen config.")
    p_run.add_argument("--context-tokens", type=int, default=None,
                       help="longmemeval: budget the context in tokens of "
                            "--tokenizer-model (overrides --context-chars).")
    p_run.add_argument("--tokenizer-model", default="gemini-2.5-flash",
                       help="Tokenizer for --context-tokens (Gemini local "
                            "tokenizer; needs sentencepiece + protobuf).")
    p_run.add_argument("--count-map", default="off",
                       choices=["off", "retrieved", "haystack"],
                       help="longmemeval: answer count/total questions by "
                            "map-reduce: one call per session lists the "
                            "instances, one call merges and counts. "
                            "haystack = every session, not just retrieved.")
    p_run.add_argument("--reader", default="single",
                       choices=["single", "verified", "routed"],
                       help="longmemeval: single = one prompt with every "
                            "procedure; verified = the same, and an "
                            "abstention is re-searched and must be "
                            "overturned by a real quote; routed = measured "
                            "WORSE (0.794 vs 0.904), kept to reproduce it.")
    p_run.add_argument("--facts", default="off", choices=["off", "session"],
                       help="longmemeval: extract dated facts per session "
                            "at ingest (one --system-provider call per "
                            "session, cached under FEATHER_CACHE) and "
                            "search them beside the turns.")
    p_run.add_argument("--context-chars", type=int, default=160_000,
                       help="longmemeval: answerer context budget, chars.")
    # Phase 9 scenario knobs
    p_run.add_argument("--system-provider", default="gemini",
                       choices=["gemini", "claude", "openai", "ollama",
                                "azure", "azure-openai", "gemini-text", "openrouter"],
                       help="LLM used by extractors (phase9; longmemeval "
                            "--facts session). "
                            "Default = gemini (works with GOOGLE_API_KEY). "
                            "Cloud will lock this to Claude Haiku.")
    p_run.add_argument("--system-model", default=None,
                       help="Phase 9: model for system_provider.")
    p_run.add_argument("--max-facts-per-turn", type=int, default=8,
                       help="Phase 9: cap facts extracted per haystack turn.")

    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="Regenerate Markdown report")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
