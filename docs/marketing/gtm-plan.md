# Feather DB — GTM Plan (v0.8.0 Launch)

> **For Claude Cowork**: this brief is the source-of-truth for the Feather DB
> v0.8.0 / LongMemEval launch. Treat the *Positioning* and *Core Message*
> sections as locked; everything else (channels, copy, schedule) is
> editable. **Do not include direct comparison vs Supermemory in any
> public asset.** We can name them as "competitors" if needed but no
> head-to-head numbers in launch creative.

---

## 1. Positioning

> **Feather DB is the embedded vector database that gives AI agents long-term memory — without a service to run, a frontier model in the loop, or a $5K/month token bill.**

One sentence. Memorable. Concrete.

## 2. Ideal Customer Profile (ICP)

| Tier | Who | What they care about | Why Feather wins |
|---|---|---|---|
| Primary | AI engineers building chatbots / agents / copilots at startups | Token cost, latency, infra simplicity | Embedded = no infra; sub-ms search; competitive memory quality at cheap-tier model cost |
| Primary | Indie hackers / solo devs shipping AI SaaS | "I can't afford $5K/mo Pinecone" | Free, MIT, single binary, runs in Lambda/Cloud Run/laptop |
| Secondary | Founding engineers at Series A AI startups | Reproducibility, vendor lock-in risk | Open file format, can leave anytime, code on GitHub |
| Secondary | Privacy / on-prem-mandated enterprises | Data leaves nowhere, audit-friendly | In-process, single file, no third-party service |

**Non-ICP** (don't waste energy on these in launch):
- Hyperscalers building their own foundation-model platforms.
- Pure research labs (they'll cite the paper, not adopt the tool).
- People shopping for "the next Pinecone" expecting a hosted service today.

## 3. Core Message + Three Proof Points

**Message:** *"You don't need a frontier model and a hosted vector DB to build agents with real long-term memory. You need a single .feather file."*

**Proof points (in priority order):**

1. **Beats the LongMemEval paper's full-context GPT-4o ceiling — 0.693 vs 0.640.** Our 10-snippet retrieval carries more usable signal to the answerer than dumping 115K tokens at GPT-4o does. Reproducible, $7–9 per full run.

2. **Same pipeline at cheap-tier costs: 0.657 with free-tier Gemini-Flash, $2.40 per full benchmark run.** Beats Zep + paid GPT-4o-mini (0.638) on the same dataset. Indie-affordable.

3. **Sub-millisecond ANN at 500K vectors.** p50 = 0.19 ms with recall@10 = 0.972 on the canonical SIFT1M benchmark. Embedded, in-process, no service to host.

## 4. The Article (Day-0 Asset)

**Title (working):** *"You don't need GPT-4o full-context for AI memory — Feather DB beats it for $2.40 a benchmark run"*

**Length:** 800–1200 words.
**Tone:** technical, direct, numbers-first. No marketing fluff.
**Has:** runnable code block, results table, link to GitHub, cloud waitlist CTA.
**Does not have:** Supermemory, Mem0, Zep head-to-head numbers in the comparison table. (Internal report at `docs/benchmarks/longmemeval.md` keeps those for honesty; the public article does not.)

Draft is committed at `docs/blog/longmemeval-publish.md`.

## 5. Channel Strategy + Launch Sequence

### Day 0 (Tuesday best, Wed acceptable)

- **08:00 PT — Blog post live** at `hawky.ai/blog` (preferred) or `getfeather.store/blog`.
- **08:30 PT — Hacker News submission** (canonical title — see `hn-submission.md`).
  - Don't say "Show HN" unless we have a hosted demo; "Feather DB beats GPT-4o full-context on LongMemEval (github.com)" or similar.
  - Pre-write three reply templates for predictable questions (cost, comparison vs ${competitor}, why embedded > hosted).
- **09:00 PT — Twitter thread** from @hawky_ai or founder account (see `twitter-thread.md`).
  - Top tweet has the result number + chart.
  - Tweet 2 = the cost comparison.
  - Tweet 3 = link to article + reproduce command.
- **All day — Reddit cross-posts**:
  - r/LocalLLaMA (best fit — they care about cost, on-prem, OSS)
  - r/MachineLearning (paper-style framing — link to arXiv PDF)
  - r/LangChain (positioning as a memory backend they can wire)

### Day 1–3

- **Direct outreach** to 8–12 high-signal accounts (engineers, not influencers): swyx, simonw, andrew___ng, jxnlco, mathemagic1an, indie AI builders. Personal note + link.
- **Discord posts**: LangChain, LlamaIndex, Latent.Space, Indie Hackers AI.
- **Newsletter mentions**: Latent Space (if they bite), TLDR AI, AI Engineer.
- **Linear / Mention / Polymarket** type comparison sites — submit listing.

### Week 1

- **Follow-up technical post** on Phase 9 design (LLM extractors). Keeps the audience engaged after the launch peak.
- **Quick demo video** (3–5 min) screen-record running the bench locally.
- **One conference CFP** (NeurIPS workshop, ML in Production, etc.) — paper is camera-ready.

### Week 2–4

- **Cloud waitlist landing page** (if not already up). The article and HN both end with "Cloud Q3 — waitlist" CTA. Capture leads.
- **Podcast pitch**: Latent Space, ML Engineered, Practical AI, This Week in ML.
- **Two-stage email sequence** for waitlist signups: (1) thanks + give them deeper material, (2) two weeks later: "what would you build?" survey.

## 6. Asset Checklist for Claude Cowork

Build / produce in this order:

| Asset | Status | Owner | Notes |
|---|---|---|---|
| `docs/blog/longmemeval-publish.md` | ✅ drafted | done | The article, no Supermemory |
| `docs/marketing/twitter-thread.md` | ✅ drafted | done | 7-tweet thread |
| `docs/marketing/hn-submission.md` | ✅ drafted | done | Title + 3 reply templates |
| Cover image / chart | ⏳ | Cowork | Bar chart: Feather (both tiers) vs ceiling, naive RAG, full-context. Inter font, white bg, blue/orange. Ref `docs/featherdb_business_diagram.html` style. |
| Demo screencast 3–5 min | ⏳ | Cowork | Run `python -m bench run longmemeval ...` and tail output. No talking-head needed. |
| Cloud waitlist landing page | ⏳ | Cowork | One-fielder. Email + "what use case?" optional. UTM-tag the article CTA. |
| Reddit-flavored variants | ⏳ | Cowork | r/LocalLLaMA reframe ("$2.40 to benchmark agent memory locally") · r/MachineLearning ("LongMemEval result for an embedded VDB") |
| Founder Twitter bio update | ⏳ | Cowork | Add: "Feather DB · Embedded memory for AI agents · 0.693 LongMemEval_S" |

## 7. KPIs (90-day window)

Stop reporting vanity. Track these in priority order:

| Metric | 30d target | 90d target | Source |
|---|---|---|---|
| **Cloud waitlist emails** | 250 | 1,000 | Form backend |
| GitHub stars | +400 | +1,500 | github.com/feather-store/feather |
| PyPI weekly downloads | 2× baseline | 10× baseline | pypistats |
| HN top-page hours | ≥4 | — | one-shot |
| Inbound DMs / "we want to use it" | 10 | 40 | personal inbox |
| arXiv paper views | 500 | 2,000 | arxiv |

## 8. Conversion Goal

Single conversion: **email capture for Cloud waitlist.**

Why: starring a GitHub repo is free engagement; an email is qualified intent. Cloud is the monetization path; the 0.693 number proves the OSS works. Use the OSS to win trust → Cloud to win revenue.

CTA copy (use exactly): **"Cloud is open-waitlist for Q3 2026. Drop your email — we'll ping when there's a beta to try."**

## 9. What we explicitly do NOT do

- **No competitor head-to-head charts** in launch creative. The internal benchmark report has them; the launch doesn't lead with them.
- **No claims we haven't reproduced.** Every number cited links to a JSON in `bench/results/`.
- **No "the best" / "the fastest" superlatives.** Stick to "X with Y for $Z" structure.
- **No paid acquisition (yet).** OSS launches die from paid promo. Pure organic until we see what sticks.

## 10. Founder talking points (if asked)

- *"Why open source?"* — Memory is infrastructure. We don't think anyone wants to be locked into a closed memory layer.
- *"Why embedded, not a service?"* — Same reason SQLite beat MySQL for 90% of apps that "need a database." Most agent memory doesn't need a service.
- *"What's the moat?"* — File format, ecosystem, the cloud version we're building. The OSS is the gateway, not the moat.
- *"How do you compete with $X-funded competitor?"* — We don't, today. We're cheaper, embeddable, open. The market is huge; we're playing for the long-tail of indie / on-prem / privacy-sensitive builders.
- *"Roadmap?"* — Phase 9 = LLM extractors at ingest (matches what closed competitors do). Cloud Q3 2026. Same engine, different surface.

## 11. Roadmap Communications

This is what we say is coming, in launch and follow-up content:

```
Now (v0.8.0):       BM25 hybrid + WAL + atomic saves + LongMemEval 0.693
Q2 2026 (v0.9.0):   LLM extractors module — fact extraction, contradiction
                    resolution, ontology-aware edges
Q3 2026:            Feather Cloud (managed API) — waitlist open
Q4 2026:            VPC / on-prem deployment, SOC2 in motion
```

Tease v0.9.0 in launch ("we know exactly which axes to attack, and we'll show you in 6 weeks"). Don't promise dates; do anchor seasons.

---

**Last updated:** 2026-04-26 (post LongMemEval_S GPT-4o run).
**Owner:** Hawky.ai team.
**Hand off to:** Claude Cowork for execution.
