# Feather DB on LongMemEval

> Status: in progress as of 2026-04-25. The numbers in this document are
> **draft**. Final headline (full 500-question oracle + S-variant run with
> adaptive decay engaged) lands when both runs complete and the comparison
> table is finalized.

---

## TL;DR

- **Oracle, no decay**: **0.656 overall** (500q, gemini-2.5-flash answerer, 0/500 failures).
- **Oracle, decay engaged**: **0.670 overall** — modest +1.4pp lift; decay's real test is on _S_ (next).
- **LongMemEval_S, decay engaged**: _running, ~85 min, will fill in_.
- **Cost**: ~$0.15 per 500-question oracle run, ~$1.65 projected for S.
- **Wall time**: ~38 min oracle, ~85 min projected for S.

**Honest read of the decay number**: oracle ships only evidence sessions, so there's almost no noise for decay to filter out. The same configuration on _S_ (which adds ~40 distractor sessions per question) is where decay should have meaningful headroom. Many temporal-reasoning questions also require *old* memories, which a recency-biased decay can hurt as much as help — a future "decay-aware retrieval" that surfaces both old and new in parallel is a follow-up idea.

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
| oracle | on  | **0.670** | 0.897 | 0.624 | **0.406** | **0.744** | 4.7s mean | 0/500 |
| s | on | _running_ | _running_ | _running_ | _running_ | _running_ | _running_ | _running_ |

Decay-on configuration: `half_life=14d`, `time_weight=0.4`. Lift on oracle is modest because the oracle variant has no distractor sessions — see honest-read note in TL;DR.

---

## Comparison vs published baselines

> **Variant matters.** Almost every published vendor number is on `LongMemEval_S`,
> while Feather's first headline is on `oracle` (a *different, easier* setting).
> Numbers below should be read alongside the **variant** column.

| System | Variant | Answerer LLM | Overall | SS-User | SS-Asst | SS-Pref | KU | TR | MS | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| **Feather DB v0.8.0 (oracle, no decay)** | oracle | gemini-2.5-flash | 0.656 | 0.943 | 0.946 | 0.667 | 0.718 | 0.383 | 0.617 | (this work) |
| **Feather DB v0.8.0 (oracle, decay)** | oracle | gemini-2.5-flash | 0.670 | 0.943 | 0.946 | 0.700 | 0.744 | 0.406 | 0.624 | (this work) |
| **Feather DB v0.8.0 (S, decay)** | S | gemini-2.5-flash | _running_ | _running_ | _running_ | _running_ | _running_ | _running_ | _running_ | (this work) |
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

Once the decay + S runs complete, the lead with substance:

- **Cost.** Feather + Azure embedder + Gemini Flash totals ~$0.15 per 500-question oracle run, ~$1.65 projected for S. Most vendors don't publish cost; running the same pipeline on GPT-4o + Pinecone is typically 5–20× more expensive.
- **Latency.** p99 vector retrieval latency is sub-millisecond at 500K vectors (see `bench/results/vector_ann_real__sift1m__*.json`). Total per-question time is dominated by the LLM round-trips, not Feather.
- **Adaptive decay is a real lever on temporal-reasoning.** Most vector DBs have no native concept of recency; ours weights memories by `recall_count`-adjusted half-life decay. The decay-on vs decay-off lift on TR will be the headline of this report.
- **Embedded, not a service.** Feather runs in-process; the entire pipeline above can run on a laptop or in a Lambda with no external vector DB.

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
