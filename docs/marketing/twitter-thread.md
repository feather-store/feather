# Twitter Thread — Feather DB v0.8.0 / LongMemEval Launch

*7 tweets. Post sequentially with ~1-min spacing.*

**Account:** founder personal account preferred, @hawky_ai retweets/quotes after the thread is up.

**Best post time:** Tuesday or Wednesday, 09:00 PT.

---

## 1/7 (the hook — needs an image)

> Built an embedded vector DB. Just ran the LongMemEval benchmark on it.
>
> 0.693 on LongMemEval_S — beats the paper's full-context GPT-4o ceiling (0.640) and Zep + GPT-4o-mini (0.638).
>
> A single .feather file. No server. $2.40 to run a full 500-question benchmark with cheap-tier models.
>
> Numbers + reproduce command in 🧵

*[Image: bar chart — Feather (0.693) and Feather-Flash (0.657) above the paper-ceiling line at 0.640. White bg, blue/orange. No competitor names except "Full-context GPT-4o (paper ceiling)".]*

## 2/7

> What it actually measures:
>
> LongMemEval (Xu et al., ICLR 2025) tests long-term memory in chat assistants end-to-end.
>
> 500 questions, 115K-token chat history each, scored across 5 axes:
> ↳ information extraction
> ↳ multi-session reasoning
> ↳ temporal reasoning
> ↳ knowledge updates
> ↳ abstention

## 3/7

> The "wait what" moment:
>
> Stuffing all 115K tokens into GPT-4o's context window scores 0.640 (paper).
>
> Feather DB retrieves 10 snippets and hands them to GPT-4o → scores 0.693.
>
> Retrieval > brute-force long-context. ~40× cheaper per query. You can use a smaller answerer.

## 4/7

> Per-axis with GPT-4o:
>
> single-session-user      1.000  (perfect)
> single-session-assistant 0.964
> single-session-preference 0.767
> knowledge-update         0.714
> multi-session            0.606
> temporal-reasoning       0.477  (still our weakest — Phase 9 attacks this)

## 5/7

> Same retrieval, free-tier Gemini-Flash answerer = 0.657.
>
> Total cost for the full 500-question benchmark: $2.40.
>
> Still beats Zep + GPT-4o-mini (0.638) and Naive RAG + GPT-4o (~0.31, paper).
>
> "Memory" doesn't have to be expensive.

## 6/7

> Reproduce the whole thing in one shell command:
>
> ```
> pip install feather-db
> git clone github.com/feather-store/feather
> python -m bench run longmemeval --dataset s --limit 0 \
>     --embedder openai \
>     --answerer-provider gemini --answerer-model gemini-2.5-flash \
>     --decay-half-life 14 --decay-time-weight 0.4 --k 10
> ```
>
> Per-q JSON results auto-checked-in to bench/results/. Audit trail.

## 7/7

> Feather DB v0.8.0:
> ↳ Embedded — single .feather file, in-process
> ↳ Sub-ms ANN (0.19ms p50 @ 500K vectors)
> ↳ HNSW + BM25 hybrid + adaptive decay
> ↳ MIT, free forever
>
> Cloud (managed API) coming Q3 2026 — waitlist open.
>
> → github.com/feather-store/feather
> → pip install feather-db

---

## Notes for Claude Cowork

- Don't @-mention Mem0, Zep, Supermemory in any tweet. We can name "Zep + GPT-4o-mini" as a *number on a chart* (it's published, fair game) but no @-handles, no quote-tweets.
- The image in 1/7 is the most important asset. White background, Inter font, 4 bars with labels:
  - "Naive vector RAG (paper)" — ~0.31
  - "Full-context GPT-4o-mini (paper)" — 0.554
  - "Full-context GPT-4o ceiling (paper)" — 0.640 (label this "the bar to beat")
  - "Feather DB + GPT-4o" — 0.693 (highlight in brand color)
  - "Feather DB + Gemini-Flash" — 0.657
- Replies prepared for predictable Qs:
  - "But Mem0 says X% / Supermemory says Y%" → "Different config; full table is in our internal report. Run their command, run ours, compare. We're optimizing for cost-efficiency on commodity models."
  - "Why not just use Pinecone/Weaviate/Chroma?" → "If you need a hosted service, those are great. Feather is for when you want it embedded — like SQLite vs MySQL."
  - "What about LMM/Letta/MemGPT?" → "Different architecture. They publish a memory abstraction over an LLM; we publish a database for vectors and graphs. Composable."
- After thread is up: founder retweets/quote-tweets a key insight throughout the day to feed the algorithm.
