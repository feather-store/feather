# Feather DB Benchmark Harness

Reproducible benchmarks. One command per run; results and reports are version-controlled.

## Layout

```
bench/
├── runner.py          # BenchRunner: orchestrates, persists JSON
├── metrics.py         # latency percentiles, recall@k, NDCG@k, QPS
├── report.py          # renders Markdown report from results/
├── datasets/          # pluggable dataset loaders
├── scenarios/         # one file per scenario (vector_ann, hybrid, ...)
├── results/           # per-run JSON (checked in)
├── reports/latest.md  # rolled-up Markdown
└── __main__.py        # python -m bench CLI
```

## Quick start

```bash
# Install deps (numpy is required; feather_db must be built)
python setup.py build_ext --inplace

# Run the vector ANN scenario on synthetic data
python -m bench run vector_ann --dataset synthetic --n 10000 --dim 128 --k 10

# Regenerate the rolled-up report
python -m bench report
cat bench/reports/latest.md
```

## Adding a scenario

1. Drop a file under `bench/scenarios/my_scenario.py` exposing `run(...)` that returns a metrics dict.
2. Register it in `bench/__main__.py` `SCENARIOS` dict.
3. If it needs a new dataset, add a loader under `bench/datasets/`.

## Planned scenarios

| Scenario | Status | Purpose |
|----------|--------|---------|
| `vector_ann` | ✅ | HNSW latency + recall@k vs brute force |
| `keyword_bm25` | ⏳ | BM25 NDCG@10 on BEIR |
| `hybrid_rrf` | ⏳ | Hybrid RRF NDCG@10 on BEIR |
| `wal_fuzz` | ⏳ | Crash-recovery correctness |
| `longmemeval` | ⏳ | Long-term memory QA (5 axes) |
| `locomo` | ⏳ | 10 conversations × ~300 turns |
| `thread_stress` | ⏳ | Concurrent add/search/link race test |
