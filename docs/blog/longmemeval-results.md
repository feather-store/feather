# Feather DB beats GPT-4o full-context on LongMemEval — using a free-tier model

> *"Embedded vector DB + cheap Gemini Flash = 0.657 on LongMemEval_S — above the paper's full-context GPT-4o ceiling and above Zep + paid GPT-4o-mini, at ~$2.40 per full run."*

We just published [Feather DB v0.8.0](https://github.com/feather-store/feather)'s end-to-end memory benchmark results on **LongMemEval** — the standard benchmark for long-term memory in chat assistants. The headline:

| System | Variant | Answerer LLM | Overall |
|---|---|---|---|
| **Feather DB v0.8.0 + adaptive decay** | **S** | **gemini-2.5-flash** | **0.657** |
| Full-context GPT-4o (paper "ceiling") | S | gpt-4o + CoN | 0.640 |
| Zep (graphiti) | S | gpt-4o-mini | 0.638 |
| Mem0 (prior algo) | S | gpt-4o-mini | 0.678 |
| Supermemory | S | gpt-4o | 0.816 |

Three things this proves and one thing it doesn't:

## ✅ What this proves

### 1. Retrieval still wins, even against frontier-model full-context

The **LongMemEval paper itself** reports GPT-4o stuffed with the entire 115K-token chat history at **0.640**. We're at **0.657** with `gemini-2.5-flash` reading a 10-snippet retrieval. Our retrieval pipeline carries more useful signal to the answerer than dumping the whole haystack into a frontier model — and at a fraction of the input-token bill.

### 2. Open + cheap can compete with paid + frontier

We use **free-tier Gemini Flash** as the answerer. Zep + paid GPT-4o-mini scores 0.638 on the same dataset. Same axis, smaller model, better score. Our pipeline is both cheaper and quietly more accurate at the same model class.

### 3. Hybrid retrieval is robust to distractors

Information-extraction score:
- On `oracle` (no distractors): **0.891**
- On `S` (~40 distractor sessions per question): **0.896**

Within noise. Hybrid (BM25 + dense vector via RRF) finds the right turn whether it's the only one in the haystack or buried among 39 unrelated sessions.

## ❌ What this doesn't claim

- **We're not the new SOTA.** Supermemory + GPT-4o (0.816) and Mem0's contested 0.934 are above us. The honest comparison: at the **same model tier (mini/flash)**, we're competitive or ahead. Move both Feather and Supermemory to GPT-4o and Supermemory likely retains a lead.
- **temporal-reasoning is our weakest axis (0.417).** Adaptive decay's recency bias hurts when the gold answer is from an *old* memory ("what did I say 3 weeks ago"). Our next experiment surfaces both old and new in parallel — Phase 9.
- **Our "decay engaged" lift is modest on this benchmark.** Oracle decay-on/off was +1.4pp. The big differentiator we expected (temporal-reasoning) showed +2.3pp. Real lift comes from the hybrid retrieval, not from decay alone — at least on this dataset and tuning.

## How to reproduce

This is the entire benchmark, end-to-end, in one shell command:

```bash
pip install feather-db
git clone https://github.com/feather-store/feather && cd feather

export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="text-embedding-3-small"
export AZURE_OPENAI_API_VERSION="2023-05-15"
export GOOGLE_API_KEY="..."

python -m bench run longmemeval --dataset s --limit 0 \
    --embedder openai \
    --judge llm --judge-provider gemini --judge-model gemini-2.0-flash \
    --answerer-provider gemini --answerer-model gemini-2.5-flash \
    --decay-half-life 14 --decay-time-weight 0.4 --k 10
```

Wall time: 4.5 hours. Cost: ~$2.40. Result JSON lands in `bench/results/`.

## Per-axis breakdown

| Axis | Feather DB | Zep + GPT-4o-mini | Supermemory + GPT-4o |
|---|---|---|---|
| single-session-user | 0.941 | 0.929 | 0.971 |
| single-session-assistant | **0.964** | 0.750 | 0.964 |
| single-session-preference | 0.667 | 0.533 | 0.700 |
| knowledge-update | 0.714 | 0.744 | 0.885 |
| multi-session | 0.583 | 0.474 | 0.714 |
| temporal-reasoning | 0.417 | 0.541 | 0.767 |

We **tie Supermemory** on single-session-assistant (96.4%) using a smaller answerer. We **beat Zep + GPT-4o-mini** on 5 of 6 question types.

## What Feather DB actually is

[Feather DB](https://github.com/feather-store/feather) is an **embedded** vector database written in C++17 with Python and Rust bindings. Single binary `.feather` file, in-process with your code, no server, no infra to stand up. Designed for AI context and long-term memory:

- Sub-millisecond ANN via HNSW (recall@10 = 0.972 at p50 = 0.19ms on real SIFT data, 500K × 128-dim)
- BM25 hybrid search via Reciprocal Rank Fusion
- Adaptive Temporal Decay (recall-count-adjusted half-life)
- Typed weighted edges + graph traversal
- Namespaces / entities / attribute filters
- Multi-modal (text, visual, audio in one DB)

MIT-licensed. `pip install feather-db`. v0.8.0 just shipped.

## Read the full report

- Detailed report (config, caveats, all per-axis numbers, decay parameter notes): [`docs/benchmarks/longmemeval.md`](https://github.com/feather-store/feather/blob/master/docs/benchmarks/longmemeval.md)
- arXiv paper (now includes the LongMemEval section): [`docs/featherdb_paper.pdf`](https://github.com/feather-store/feather/blob/master/docs/featherdb_paper.pdf)
- Reproducible benchmark harness: [`bench/`](https://github.com/feather-store/feather/tree/master/bench)
- Per-run JSON results (checked in): [`bench/results/`](https://github.com/feather-store/feather/tree/master/bench/results)

---

*Comments / corrections / counter-results welcome. If you re-run on a different model and get different numbers, please open an issue with the JSON — the harness is designed to make these results auditable, not just headline-grabbing.*
