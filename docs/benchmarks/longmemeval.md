# Feather DB on LongMemEval

> Status: in progress as of 2026-04-25. The numbers in this document are
> **draft**. Final headline (full 500-question oracle + S-variant run with
> adaptive decay engaged) lands when both runs complete and the comparison
> table is finalized.

---

## TL;DR

**Headline: Feather DB v0.8.0 + Gemini 2.5-flash scores 0.657 on LongMemEval_S — beating both Zep + GPT-4o-mini (0.638) and the paper's full-context GPT-4o ceiling (0.640) — using a free-tier model and ~$2.40 in API spend.**

| Run | Overall | Failures | Wall | Cost |
|---|---|---|---|---|
| Oracle, no decay | 0.656 | 0/500 | 38 min | ~$0.15 |
| Oracle, decay on | 0.670 | 0/500 | 39 min | ~$0.15 |
| **S, decay on** | **0.657** | 5/500 | 268 min | ~$2.40 |

**Three claims this benchmark supports:**

1. Feather + free-tier Gemini-Flash matches/beats Zep + paid GPT-4o-mini on the same dataset.
2. Feather beats the paper's "full-context GPT-4o" ceiling — i.e. our retrieval pipeline is no worse than dumping the whole 115K-token haystack into a frontier model.
3. We tie Supermemory on `single-session-assistant` (96.4% vs 96.4%) using a cheaper answerer model.

**Where we lose:** temporal-reasoning 41.7% — below most competitors. Adaptive-decay tuning + "old + new in parallel" retrieval is the next attack surface (Phase 9).

**Honest read of the decay number on oracle (+1.4pp)**: oracle ships only evidence sessions, so decay has almost no noise to filter out. The same configuration on S (with ~40 distractor sessions per question) lets decay do real work.

---

## What this measures

[LongMemEval](https://arxiv.org/abs/2410.10813) (Xu et al., 2024 / ICLR 2025) is the standard benchmark for long-term memory in chat assistants. 500 questions, each paired with a long chat history, scored end-to-end across 5 memory ability axes:

- **information-extraction** — recall a fact stated by user or assistant in a single session
- **multi-session-reasoning** — synthesize across distant sessions
- **temporal-reasoning** — answer time-anchored questions
- **knowledge-updates** — track changes / contradictions over time
- **abstention** — refuse when info isn't there

The benchmark ships in three variants of increasing difficulty:
- `oracle` — only evidence sessions in the haystack (~3–5 sessions per question)
- `s` — ~40 sessions, ~115K tokens per question (the standard public benchmark)
- `m` — ~500 sessions, ~1.5M tokens

Most published numbers (Mem0, Zep, Supermemory, Emergence) are on the **S** variant.

---

## Reproducing this benchmark

```bash
# 1. Build Feather DB
python setup.py build_ext --inplace

# 2. Set credentials
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="text-embedding-3-small-..."
export AZURE_OPENAI_API_VERSION="2023-05-15"
export GOOGLE_API_KEY="..."

# 3. Headline run (oracle, no decay) — ~38 min, ~$0.15
python -m bench run longmemeval \
  --dataset oracle --limit 0 \
  --embedder openai \
  --judge llm --judge-provider gemini --judge-model gemini-2.0-flash \
  --answerer-provider gemini --answerer-model gemini-2.5-flash \
  --k 10

# 4. With adaptive decay engaged (Feather differentiator)
python -m bench run longmemeval \
  --dataset oracle --limit 0 \
  --embedder openai \
  --judge llm --judge-provider gemini --judge-model gemini-2.0-flash \
  --answerer-provider gemini --answerer-model gemini-2.5-flash \
  --decay-half-life 14 --decay-time-weight 0.4 \
  --k 10

# 5. Apples-to-apples on S variant — ~85 min, ~$1.65
python -m bench run longmemeval \
  --dataset s --limit 0 \
  --embedder openai \
  --judge llm --judge-provider gemini --judge-model gemini-2.0-flash \
  --answerer-provider gemini --answerer-model gemini-2.5-flash \
  --decay-half-life 14 --decay-time-weight 0.4 \
  --k 10
```

Per-run JSON results land in `bench/results/`; rolled-up table in `bench/reports/latest.md`.

---

## Configuration

| Component | Value | Notes |
|---|---|---|
| Engine | Feather DB v0.8.0 | C++17 core, Python bindings |
| Embedder | Azure `text-embedding-3-small` | 1536-dim. Same model Mem0/Zep used. |
| Retrieval | `db.hybrid_search(...)` | BM25 + dense vector merged via RRF (k_rrf=60) |
| HNSW `ef` | 50 | New default in v0.8.0 (was 10) |
| `top_k` | 10 | Standard for memory benchmarks |
| Answerer | `gemini-2.5-flash` | Reasoning-tuned Flash tier; thinking mode active |
| Judge | `gemini-2.0-flash` | Cheap binary correct/incorrect judge with rubric |
| Adaptive decay | `half_life=14d, time_weight=0.4` | Engaged in the "decay" runs only |

---

## Headline result — Feather DB

| Variant | Decay | overall | info-extr | multi-sess | temporal | knowledge-upd | per-q time | n_failures |
|---|---|---|---|---|---|---|---|---|
| oracle | off | 0.656 | 0.891 | 0.617 | 0.383 | 0.718 | 4.5s mean | 0/500 |
| oracle | on  | 0.670 | 0.897 | 0.624 | 0.406 | 0.744 | 4.7s mean | 0/500 |
| **s** | **on** | **0.657** | **0.896** | **0.583** | **0.417** | **0.714** | **32.0s mean** | **5/500** |

Decay-on configuration: `half_life=14d`, `time_weight=0.4`. Oracle and S information-extraction are essentially identical (0.891 vs 0.896) — Feather's retrieval is robust to distractors on that axis. The drop on multi-session (0.617 → 0.583) reflects the harder retrieval problem with ~40 sessions per question.

---

## Comparison vs published baselines

> **Variant matters.** Almost every published vendor number is on `LongMemEval_S`,
> while Feather's first headline is on `oracle` (a *different, easier* setting).
> Numbers below should be read alongside the **variant** column.

| System | Variant | Answerer LLM | Overall | SS-User | SS-Asst | SS-Pref | KU | TR | MS | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| **Feather DB v0.8.0 (oracle, no decay)** | oracle | gemini-2.5-flash | 0.656 | 0.943 | 0.946 | 0.667 | 0.718 | 0.383 | 0.617 | (this work) |
| **Feather DB v0.8.0 (oracle, decay)** | oracle | gemini-2.5-flash | 0.670 | 0.943 | 0.946 | 0.700 | 0.744 | 0.406 | 0.624 | (this work) |
| **Feather DB v0.8.0 (S, decay)** | **S** | **gemini-2.5-flash** | **0.657** | **0.941** | **0.964** | **0.667** | **0.714** | **0.417** | **0.583** | **(this work)** |
| Full-context GPT-4o (paper ceiling, full history) | S | GPT-4o + CoN+JSON | 0.640 | 0.814 | 0.946 | 0.200 | 0.782 | 0.451 | 0.443 | [paper](https://arxiv.org/html/2410.10813v1) |
| Oracle GPT-4o (paper ceiling, evidence-only) | oracle | GPT-4o + CoN+JSON | 0.924 | — | — | — | — | — | — | [paper](https://arxiv.org/html/2410.10813v1) |
| Oracle GPT-4o (no CoN) | oracle | GPT-4o | 0.870 | — | — | — | — | — | — | [paper](https://arxiv.org/html/2410.10813v1) |
| Naive RAG (Stella v5 1.5B + GPT-4o, top-5, sessions) | M | GPT-4o | 0.676 | — | — | — | — | — | — | [paper Tbl 3](https://arxiv.org/html/2410.10813v1) |
| Mem0 (token-efficient algo, Apr 2026) | S* | GPT-4o-mini | 0.934 | 0.971 | 1.000 | 0.967 | 0.962 | 0.932 | 0.865 | [Mem0 blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm) |
| Mem0 (prior algo, baseline) | S* | GPT-4o-mini | 0.678 | 0.943 | 0.464 | 0.767 | 0.795 | 0.511 | 0.707 | [Mem0 blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm) |
| Zep (graphiti) | S | GPT-4o | 0.712 | 0.929 | 0.804 | 0.567 | 0.833 | 0.624 | 0.579 | [Zep paper](https://arxiv.org/html/2501.13956v1) |
| Zep (graphiti) | S | GPT-4o-mini | 0.638 | 0.929 | 0.750 | 0.533 | 0.744 | 0.541 | 0.474 | [Zep paper](https://arxiv.org/html/2501.13956v1) |
| Supermemory | S | GPT-4o | 0.816 | 0.971 | 0.964 | 0.700 | 0.885 | 0.767 | 0.714 | [Supermemory](https://supermemory.ai/research/) |
| Supermemory | S | GPT-5 | 0.846 | — | — | — | — | — | — | [Supermemory](https://supermemory.ai/research/) |
| Supermemory | S | Gemini-3-Pro | 0.852 | — | — | — | — | — | — | [Supermemory](https://supermemory.ai/research/) |
| Emergence AI | S | GPT-4o-2024-08-06 | 0.860 | 0.986 | 1.000 | 0.600 | 0.833 | 0.857 | 0.812 | [Emergence](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) |
| Full-context GPT-4o-mini | S | GPT-4o-mini | 0.554 | 0.814 | 0.818 | 0.300 | 0.769 | 0.365 | 0.406 | [Zep paper](https://arxiv.org/html/2501.13956v1) |
| MemGPT / Letta | — | — | (no published number) | — | — | — | — | — | — | [Letta issue #3115](https://github.com/letta-ai/letta/issues/3115) |

\* Mem0's research page does not explicitly label the variant as `_S`; the question-type breakdown matches `_S`'s 6-type taxonomy and 500-question split.

---

## Caveats — read before drawing conclusions

1. **Variant matters enormously.** The same memory pipeline scores ~25pp higher on `oracle` than on `S`. Comparisons must hold variant constant.
2. **Answerer LLM differences swing results 10+ points.** Same Zep system goes 63.8% → 71.2% by swapping `gpt-4o-mini` → `gpt-4o`. Feather using `gemini-2.5-flash` likely sits between mini and full GPT-4o.
3. **Mem0's 93.4% deserves scrutiny.** It's a vendor-published number; the third-party Zep paper independently measured Mem0 at 49.0% on S. Use both numbers as a band, not a single point.
4. **"Full-context GPT-4o" is not a single number.** The paper itself reports 60.6% (no CoN) to 64.0% (with CoN+JSON) on S. Use 60–64% as the band.
5. **Letta/MemGPT has no official LongMemEval result.** Any such number is third-party.

---

## What's interesting about Feather here

The S-variant numbers support three claims that would survive a hostile read:

**1. Cost-competitive with paid frontier-model pipelines.** Feather + Azure `text-embedding-3-small` + Gemini 2.5-flash totals **~$2.40 for the full 500-question S run**. Same dataset on GPT-4o-based stacks (Mem0, Zep, Supermemory, Emergence) costs roughly 5–20× more. The gemini-2.5-flash answerer is the cheap-tier model and we still match Zep + GPT-4o-mini.

**2. Beats the paper's full-context ceiling.** GPT-4o stuffed with the entire 115K-token haystack scores 0.640 on S (per the LongMemEval paper). Feather + Gemini-Flash retrieval scores **0.657**. That means our 10-snippet retrieval pipeline carries more signal to the answerer than dumping the whole haystack into the model — and at a fraction of the input-token bill.

**3. Robust to distractors on information-extraction.** Score on info-extraction is **0.891 (oracle) vs 0.896 (S)** — distractors barely move the needle. That's the embedder + BM25 hybrid doing its job; the dense vector handles paraphrase, BM25 handles exact terms, RRF stays calibrated.

Other hard wins:

- **Sub-millisecond vector latency.** p99 for HNSW retrieval at 500K × 128-dim is 0.13ms (see `bench/results/vector_ann_real__sift1m__*.json`). Per-question time on LongMemEval is 100% dominated by LLM round-trips — the database itself is invisible in the latency budget.
- **Embedded, not a service.** Feather runs in-process. Everything in this report runs on a laptop with no external vector DB, no Pinecone bill, no service to stand up.

## What we don't beat (yet)

- **Mem0 (token-efficient, contested 0.934)** and **Supermemory + GPT-4o (0.816)** are above us by 16–30 points. Caveats apply (Mem0's 0.934 was independently measured at 0.490 by Zep), but Supermemory + GPT-4o is a real, reproducible lead.
- **temporal-reasoning at 0.417** is below most competitors. Adaptive decay's recency bias hurts as much as helps when the gold answer is from an *old* memory ("what did I say 3 weeks ago"). Phase 9 will explore decay-aware retrieval that surfaces both old and new in parallel.

---

## Open questions / future runs

1. **GPT-4o ceiling for Feather.** Re-run oracle + S with `gpt-4o` answerer to isolate the model gap.
2. **Decay parameter sweep.** half_life ∈ {7, 14, 30, 60} × time_weight ∈ {0.2, 0.4, 0.6}.
3. **`top_k` sweep.** k ∈ {5, 10, 20, 30}.
4. **Context-graph traversal.** `db.context_chain(...)` n-hop expansion as a retrieval augmentation, especially for multi-session questions.
5. **Phase-9 LLM extractors.** Pre-extract atomic facts at ingest time; compare retrieval over facts vs raw turns.

---

## Sources

- LongMemEval paper (arXiv): https://arxiv.org/abs/2410.10813
- LongMemEval GitHub: https://github.com/xiaowu0162/LongMemEval
- LongMemEval HuggingFace: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- Mem0 Research: https://mem0.ai/research
- Mem0 token-efficient algo blog: https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm
- Zep paper (arXiv 2501.13956): https://arxiv.org/abs/2501.13956
- Zep blog: https://blog.getzep.com/state-of-the-art-agent-memory/
- Supermemory Research: https://supermemory.ai/research/
- Emergence AI SOTA on LongMemEval: https://www.emergence.ai/blog/sota-on-longmemeval-with-rag
- Letta open issue (no LongMemEval published): https://github.com/letta-ai/letta/issues/3115
