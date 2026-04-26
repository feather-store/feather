# Hacker News Submission — Feather DB / LongMemEval

## Submission

**Title** *(70 chars max — HN truncates aggressively)*:

> Feather DB beats GPT-4o full-context on LongMemEval — embedded vector DB

**URL**: `https://hawky.ai/blog/longmemeval-results` (or wherever the article ends up)

**Type**: Link post (NOT "Show HN" — Show HN is for hosted demos. Our demo is "run this command.")

**Best time to post**: Tue / Wed, 07:30 PT. The first 60 minutes determine HN front-page reach. Post at 07:30 to catch the morning EU-leaving-work-and-US-just-waking-up window.

---

## First-comment template (post within 2 minutes of submission)

We don't put marketing copy in the article, but HN deserves the inline context. The author should comment first with:

> Author here. Quick context for HN —
>
> What I find interesting: the LongMemEval paper itself reports
> full-context GPT-4o stuffed with the 115K-token haystack at 0.640.
> Our retrieval pipeline (10 snippets, in-process Feather DB)
> hands those snippets to the same GPT-4o and gets 0.693. So
> retrieval isn't just acceptable when long-context is available —
> it's actually better, and ~40x cheaper per query.
>
> Cheap-tier run (Gemini-Flash answerer) is 0.657 for $2.40 a
> benchmark run. Pre-existing competitor numbers in the same range
> are using paid GPT-4o-mini.
>
> Open to questions on the methodology. Per-question JSON results
> are checked into bench/results/ — happy to walk through any
> question we got right or wrong.

This earns trust before the comment thread spirals.

---

## Reply templates for the predictable HN questions

### Q1: "Why not Pinecone / Weaviate / Chroma / pgvector?"

> Same family but different tradeoff. Those are *services* you stand up
> (Chroma can run embedded, but most folks deploy it as a service).
> Feather is in-process, single binary file, sub-ms latency. Closest
> analogy: SQLite vs MySQL. If you need a hosted vector DB, the others
> are great. If you want one to ship inside a CLI, a Lambda, or a
> mobile app, that's where we live.

### Q2: "How does it compare to Mem0 / Letta / MemGPT?"

> Different layer. Mem0 / Letta / MemGPT are memory *frameworks* — they
> sit above an LLM and decide what to extract, when to summarize, etc.
> Feather is the *database layer* underneath that. They could (and
> should) compose. We provide the vectors-+-graph + decay primitives;
> a memory framework decides what to store in them.

### Q3: "Cost claim seems handwavy"

> Fair, here's the breakdown for the $2.40 number:
> – Azure text-embedding-3-small @ $0.02/1M tokens × ~58M tokens (500
>   questions × ~116K tokens of haystack each) = ~$1.16
> – Gemini 2.5-flash answerer @ $0.30 in / $2.50 out per 1M × ~1.5M
>   in + 250K out = ~$1.07
> – Gemini 2.0-flash judge @ $0.075 in / $0.30 out × tiny volumes = ~$0.02
> Total ~$2.25. Rounded to $2.40 to give a margin. Per-call telemetry
> is in bench/results/longmemeval__s__20260425_175723.json.

### Q4: "Sub-ms latency claim — on what hardware?"

> M1 / M2 Mac, single thread, 500K vectors × 128-dim, real SIFT
> distribution (no synthetic). p50 = 0.19 ms, p99 = 0.23 ms at the
> default ef=50, recall@10 = 0.972. Full sweep table in
> bench/results/vector_ann_real__sift1m__*.json. We don't claim
> faster than FAISS or hnswlib direct — we use hnswlib internally, of
> course — just that we don't add overhead on top.

### Q5: "Why C++17, why not Rust?"

> The core is C++17 because we vendor hnswlib (which is C++) and
> embedding it directly avoids an FFI layer. The CLI and one of the
> bindings is Rust. We're not religious about the choice; if a Rust
> HNSW becomes the consensus performance bar we'd port. Today, C++
> hnswlib is the gold standard.

### Q6: "Is this actually open source or 'open core'?"

> The engine, all the bindings, the benchmark harness, the entire
> codebase used to produce the LongMemEval numbers — all MIT,
> all on GitHub. Cloud (Q3 2026, managed API) will be a separate
> proprietary service. Same engine binary; different deployment
> surface. The OSS is not a teaser version — it's the whole engine.

### Q7: "Why a new benchmark? What about [something else]?"

> Not a new benchmark — LongMemEval is the standard for memory
> evaluation in agents (ICLR 2025, Xu et al., already cited by Mem0
> and Zep in their published numbers). What we did is run it on
> Feather, end-to-end, with checked-in JSON results.

---

## Backup angles if the post tanks

If the submission gets <10 points in the first 30 minutes:

- **Don't resubmit same URL same week.** HN penalizes reposts. Wait 7 days.
- **Try a different title angle**: "An embedded vector DB that runs the LongMemEval benchmark for $2.40" or "Show HN: Feather DB v0.8.0 — sub-ms vector DB with adaptive memory" (if we have a Web demo by then).
- **Pivot to /r/MachineLearning** — has slower decay than HN and rewards detailed posts.
