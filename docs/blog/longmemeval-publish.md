# You don't need GPT-4o full-context for AI memory — Feather DB beats it for $2.40

*Published: April 2026 · Hawky.ai · [github.com/feather-store/feather](https://github.com/feather-store/feather)*

---

## TL;DR

We just ran the [LongMemEval](https://arxiv.org/abs/2410.10813) benchmark — the standard 500-question evaluation for long-term memory in chat assistants — on **Feather DB v0.8.0**, our embedded vector database. Two runs, same retrieval pipeline, two different answerer models:

| Configuration | LongMemEval_S Score | Cost / full run | Wall time |
|---|---|---|---|
| **Feather DB + GPT-4o**           | **0.693** | ~$8 | 4.5 hours |
| **Feather DB + Gemini 2.5-flash** | **0.657** | ~$2.40 | 4.5 hours |
| Full-context GPT-4o (paper "ceiling") | 0.640 | (paper-reported) | — |
| Full-context GPT-4o-mini | 0.554 | (paper-reported) | — |
| Naive vector RAG (Stella + GPT-4o) | ~0.31 | (paper-reported) | — |

Feather DB **beats the LongMemEval paper's full-context GPT-4o ceiling** — meaning a 10-snippet retrieval from a single `.feather` file delivers more useful signal to the answerer than dumping the entire 115K-token chat history into a frontier model.

The whole pipeline runs in-process. There's no service to host. The reproduce command is one shell line. The total benchmark cost — embeddings + answer generation + LLM judge for all 500 questions — is the price of an espresso.

Below: what this means, the per-axis breakdown, and how to run it yourself.

---

## What LongMemEval actually measures

LongMemEval (Xu et al., 2024 / ICLR 2025) is the standard end-to-end benchmark for memory in chat assistants. 500 questions, each paired with a long conversation history (~115K tokens, ~40 sessions of which most are distractors). Each question tests one of five memory abilities:

- **information-extraction** — recall a fact stated by user or assistant
- **multi-session reasoning** — synthesize facts across distant sessions
- **temporal reasoning** — answer time-anchored questions ("what did I say three weeks ago?")
- **knowledge-updates** — track changes / contradictions over time
- **abstention** — refuse to answer when info isn't there

The eval is run end-to-end: ingest the haystack, retrieve relevant memories at question time, hand them to an LLM to answer, and have a separate LLM judge correctness against gold. **It only measures whether the assistant gave the right answer** — perfect retrieval that doesn't translate to a correct answer scores zero.

This is the right test of "does memory work" because every other metric (recall@k, NDCG, etc.) can look great while the actual user experience is broken.

---

## The headline result, per-axis

Same retrieval pipeline (Feather + Azure `text-embedding-3-small` + adaptive temporal decay), two answerer models:

| Axis | Feather + Gemini-Flash | Feather + GPT-4o |
|---|---|---|
| information-extraction | 0.896 | **0.942** |
| knowledge-updates | 0.714 | 0.714 |
| multi-session-reasoning | 0.583 | 0.606 |
| temporal-reasoning | 0.417 | 0.477 |
| **overall** | **0.657** | **0.693** |

By question_type:

| Type | Gemini-Flash | GPT-4o |
|---|---|---|
| single-session-user | 0.941 | **1.000** *(perfect)* |
| single-session-assistant | 0.964 | 0.964 |
| single-session-preference | 0.667 | 0.767 |
| knowledge-update | 0.714 | 0.714 |
| multi-session | 0.583 | 0.606 |
| temporal-reasoning | 0.417 | 0.477 |

Three things worth noticing:

1. **Switching to GPT-4o lifts the overall by +3.6pp**, but the lift is *concentrated*: temporal-reasoning +6, single-session-preference +10, single-session-user +5.9. Other axes are flat or marginal.
2. **Knowledge-update is identical across model classes** (0.714). That's a fingerprint of a *retrieval-side* gap that no answerer can fix — a richer answer model can't make up for memories that weren't surfaced.
3. **Single-session-user hits 100%** with GPT-4o. Combined with single-session-assistant at 96.4%, simple recall is essentially solved.

---

## The headline-headline: we beat full-context GPT-4o

The LongMemEval paper reports **full-context GPT-4o + Chain-of-Note + JSON output = 0.640** on the same dataset. Their setup: stuff the entire 115K-token chat history into GPT-4o's context window, ask the question, score the answer.

Feather + GPT-4o = **0.693** with a 10-snippet retrieval. **+5.3pp over the paper's full-context ceiling.**

What this means: our retrieval pipeline isn't just *acceptable* given a long-context-capable model — it's actually *better than handing the model everything*. The reasons are mechanical:

- **Less noise**: 10 carefully-selected memories vs 115K tokens of distractors.
- **Lower input cost**: ~3K tokens to GPT-4o per question vs ~115K. **40× cheaper per query.**
- **Lower latency**: smaller prompts = faster responses, regardless of model.
- **Frontier-model ceiling stops mattering**: Feather + Flash beats Full-context + GPT-4o-mini (0.554 paper-reported). You can pick the cheapest model that the *retrieval* hands enough context to.

Retrieval has been an evergreen idea since RAG. What's new is that **the retrieval is now in-process, file-based, sub-millisecond, and free.**

---

## Reproducing this

The whole pipeline, one shell command:

```bash
pip install feather-db
git clone https://github.com/feather-store/feather && cd feather

# Set your credentials
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="text-embedding-3-small"
export AZURE_OPENAI_API_VERSION="2023-05-15"
export GOOGLE_API_KEY="..."        # for the judge

# Run on LongMemEval_S — 500 questions, ~$2.40, ~4.5 hours
python -m bench run longmemeval --dataset s --limit 0 \
    --embedder openai \
    --judge llm --judge-provider gemini --judge-model gemini-2.0-flash \
    --answerer-provider gemini --answerer-model gemini-2.5-flash \
    --decay-half-life 14 --decay-time-weight 0.4 --k 10
```

That's the entire benchmark. The dataset auto-downloads on first run, the result lands as a JSON in `bench/results/`, and a Markdown rolled-up table sits in `bench/reports/latest.md`.

To run with GPT-4o instead, swap the answerer flags:

```bash
    --answerer-provider azure --answerer-model gpt-4o-feather
```

(Plus the appropriate `AZURE_OPENAI_CHAT_*` env vars for the chat deployment.)

Per-question JSON results are checked into the repo at `bench/results/`. Every number in this post is auditable. If you re-run on different models and get different numbers, please open an issue with the JSON.

---

## What Feather DB actually is

[Feather DB](https://github.com/feather-store/feather) is an **embedded** vector database written in C++17 with Python and Rust bindings. Single binary `.feather` file. In-process with your code. No server. No infrastructure to stand up.

Designed specifically for AI long-term memory:

- **HNSW** for sub-ms ANN — p50 = 0.19 ms, recall@10 = 0.972 on 500K × 128-dim SIFT data
- **BM25 hybrid search** via Reciprocal Rank Fusion — handles paraphrase + exact terms
- **Adaptive Temporal Decay** — recall-count-adjusted half-life for "stickiness" of memories
- **Typed weighted edges + graph traversal** — context chains across sessions
- **Namespaces / entities / attribute filters** — multi-tenant friendly
- **Multimodal pockets** — text, visual, audio in one DB

MIT-licensed. `pip install feather-db`. v0.8.0 just shipped.

---

## What's coming next

Two threads pull on the result above:

**Knowledge-update is unchanged across model classes.** That tells us where the next investment should go. Feather already stores raw chat turns; competitor memory layers extract atomic facts at ingest time and resolve contradictions automatically. We're shipping **`feather_db.extractors`** in v0.9.0 to do the same — pluggable LLM-based fact extraction, ontology-aware edges, contradiction resolution. Expected to specifically lift knowledge-updates and multi-session-reasoning.

**Same engine, different surface.** Feather DB is embedded today. We're building a managed cloud version for teams that want the same engine, same file format, but a hosted API surface they can call over HTTPS. Coming Q3 2026 — your `.feather` file stays portable, your data stays yours, only the deployment topology changes.

**Cloud waitlist is open.** Drop your email below and we'll ping when there's a beta to try.

→ **[Join the Cloud waitlist](https://www.getfeather.store/cloud)** ←

---

## Resources

- **GitHub**: [github.com/feather-store/feather](https://github.com/feather-store/feather)
- **PyPI**: `pip install feather-db`
- **Crates.io**: `feather-db-cli`
- **Detailed report** (config, all per-axis numbers, caveats): [`docs/benchmarks/longmemeval.md`](https://github.com/feather-store/feather/blob/master/docs/benchmarks/longmemeval.md)
- **arXiv paper** (now includes the §4.7 LongMemEval section): [`docs/featherdb_paper.pdf`](https://github.com/feather-store/feather/blob/master/docs/featherdb_paper.pdf)
- **Reproducible benchmark harness**: [`bench/`](https://github.com/feather-store/feather/tree/master/bench)
- **Per-run JSON results** (audit trail): [`bench/results/`](https://github.com/feather-store/feather/tree/master/bench/results)

---

*Found this useful? We'd love your star on [GitHub](https://github.com/feather-store/feather) and your feedback on whatever you build with it. Found something wrong? Open an issue with the JSON — we mean it about the audit trail.*

*Feather DB is part of [Hawky.ai](https://hawky.ai) — AI-native development tools.*
