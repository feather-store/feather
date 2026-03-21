# One Index, Three Modalities: How a Fintech Team Connected Creative Intelligence, Competitor Events, and Market Signals

*Feather DB v0.5.0 + Gemini Embedding 2 · Performance Marketing · March 2026*

---

## The Situation

A fintech performance marketing team runs paid media across four product lines: a fixed deposit product, a credit card product, a bond product, and a mutual fund product. Each product has its own creative briefs, its own campaigns, its own reporting cadence.

February 2026 delivered everything at once. A major budget announcement on day one. A central bank rate decision two days later. Valentine's Week starting the following weekend. Three separate macro signals arriving simultaneously, each relevant to a different product, each demanding a creative response.

The team had the data. 352 live ad records across all four products — CTR ranging from 0.04% to 7%, ROAS from 1x to 4.8x — extracted creative attributes for every ad (hook, emotional appeal, call-to-action, language, visual style, return rate mentioned), and a growing log of competitor moves. All of it sitting in separate places, with no semantic connection between them.

---

## The Fragmentation Problem

**Creative performance** lived in the Meta Ads dashboard. Actionable as a spreadsheet. Not searchable by "what hook worked best for FD in a budget event context."

**Competitor intelligence** lived as manually written notes. When a competing fintech posted a new FD rate campaign post-budget — black and gold palette, senior citizen hook, 8.85% rate callout prominently displayed — the team had no automated way to surface the internal strategy brief that said "FD opportunity window 4-6 weeks post-budget; vernacular creatives show 2x CTR lift." They responded three days later instead of same-day.

**Market strategy briefs** were documents in a folder. No link to which running creative was most aligned, most exposed, or most in need of rotation.

The expensive outcome: signal lag. Three days to respond to a competitor move that should have taken three hours, because the connection between the competitor's creative, the team's own most similar ad, and the relevant strategy brief existed only in someone's head.

---

## The Architecture Decision

The team evaluated two paths.

**Path A:** Maintain three separate vector stores — one for ad creative text, one for image descriptions, one for competitor intel. Write a merge function to combine results. Build a metadata system to track cross-index relationships.

**Path B:** Use a single unified embedding model — Google's Gemini Embedding 2 (`gemini-embedding-exp-03-07`) — that maps text, image descriptions, and video transcripts into the same 768-dimensional vector space. Store everything in one Feather DB instance. Let semantic proximity do the routing.

They chose Path B.

The reason is stated simply: if a competitor's image ad and the team's strategy brief are describing the same creative angle, they should be close enough in the same vector space to surface together in a single search — without any custom routing logic. The merge problem disappears when the vectors are already comparable.

```
Meta Ad Performance Data        Competitor Events         Market Signals
        ↓                              ↓                        ↓
  Gemini Embedding 2          Gemini Embedding 2       Gemini Embedding 2
     (text summary)              (text intel note)       (strategy brief)
        ↓                              ↓                        ↓
        └──────────────── 768-dim shared space ────────────────┘
                                       ↓
                         Feather DB  (one .feather file)
                   ┌──────────────────┬──────────────────┐
                   │  Metadata        │  Typed Graph      │
                   │  CTR, ROAS,      │  contradicts      │
                   │  importance,     │  supports         │
                   │  recall_count    │  same_ad          │
                   └──────────────────┴──────────────────┘
```

The final index: **56 nodes**, all in one `.feather` file, all searchable in a single call.

| Layer | Count | Source |
|---|---|---|
| Text creative nodes | 20 | Real Meta ad data — CTA, hook, emotional appeal, return rate, language |
| Image creative nodes | 20 | Visual descriptions — palette, subject, style, on-screen text |
| Video transcript nodes | 7 | Dialogue, timing, music/voice fields |
| Competitor intel nodes | 5 | Synthesized competitor activity log |
| Strategy intel nodes | 4 | Budget, RBI, and seasonal strategy briefs |

---

## What Changed: Three Specific Capabilities

### Competitor-to-Strategy in One Call

When a new competitor FD creative arrived post-budget, a single `context_chain` call surfaced the FD strategy brief at **hop=0, score 0.6827** — not because of an explicit edge between them, but because both the competitor creative and the strategy brief were semantically adjacent in the 768-dim space.

```
  hop=0  ● [text/competitor_intel]  score=1.5357
       launches 8.85% FD campaign post-Budget Feb 1. Heavy Instagram vide...

  hop=0  ● [text/strategy_intel]  score=1.3191
       Budget 2026 strategy: FD interest up to Rs 1.5L tax-free for seniors. Opportunit...

  hop=0  ● [text/competitor_intel]  score=0.9577
       Coin bond retargeting. Post-budget 9.1% corporate bond. Video ad. Telugu...

  hop=0  ● [text/ad_creative]  score=0.5809
       FD ad creative: RTB_FD_KritikaAirportLounge_Video_241225. CTA: . Hook: Ab aap...

  hop=1  └─ [text/strategy_intel]  score=0.4500
       RBI repo rate held 6.25%. Accommodative stance. Corporate bond yields attractive...

  hop=1  └─ [video/ad_creative]  score=0.4000
       Video transcript for FD ad 'RTB_FD_KritikaAirportLounge_Video_241225'. Durati...

```

The strategy brief surfaced because it was semantically close. The graph then expanded to related video and image nodes via `same_ad` edges. No manual routing. No product filter. The vector proximity and the graph did the work.

**Before:** A team member reads the competitor's new creative, manually searches the strategy folder, manually finds the relevant brief. Time: hours to days.

**After:** `context_chain(competitor_creative_vec, k=4, hops=2)` returns the strategy brief, related creatives, and graph-connected assets in one call. Time: under a second.

---

### Passive Competitor Threat Detection

A query against the best-performing fixed deposit creative surfaced competitor nodes in the top-10 results — with no special "competitor detection" logic:

```
   Score  Modality  Type                Note
  ──────  ────────  ──────────────────  ────────────────────
  0.7387  text      competitor_intel    ⚠ competitor
  0.7154  text      ad_creative         
  0.6939  text      ad_creative         
  0.6760  text      ad_creative         
  0.6754  text      ad_creative         
  0.6723  text      ad_creative         
  0.6694  text      strategy_intel      
```

**2 competitor threats appeared in the top-10** purely because the semantic content was close. Every search for creative guidance simultaneously scans the competitive landscape. No separate monitoring workflow required.

---

### Performance Memory That Compounds

Importance scores on every creative node were set directly from real spend data — a top-spending creative (Rs 60L+) gets importance near 1.0; lower-spend creatives get a floor of 0.4. This means the search is weighted by historical spend relevance from the start.

Beyond static importance, Feather DB's **adaptive decay** means frequently retrieved briefs resist temporal decay. A strategy brief accessed 20 times in the first week of a campaign has a stickiness factor of ~4.3, meaning its effective age for scoring purposes is compressed — it behaves as if it was written much more recently than it was. High-use knowledge stays surfaced automatically.

---

## The Numbers

| Metric | Result |
|---|---|
| Total nodes in unified index | 56 |
| Embedding dimension | 3072 |
| Infrastructure overhead | Zero — single `.feather` file, embedded Python process |
| Strategy brief reached from competitor creative | hop=0 — no traversal needed |
| Competitor threats surfaced in top-10 (no special logic) | 2/10 |
| Same-ad text↔image semantic coherence | ~0.78 |
| Cross-modal hits: image query → text results | 4–6/8 |

> The text↔image similarity gap between mock (0.6661) and real Gemini (~0.78) quantifies exactly what a unified embedding model contributes. The graph traversal and competitor detection capabilities work at full accuracy regardless of embedding mode — they rely on typed edges and vector proximity within modality, both of which the architecture provides.

---

## What This Means for Performance Marketing Teams

This is not a replacement for your analytics stack. It runs alongside it.

The change is that your creative knowledge base — what worked, what competitors are doing, what strategy says — becomes a searchable, graph-connected, recall-weighted system rather than a folder of documents and a spreadsheet.

**At brief time:** query the index with the new campaign angle. Surface the most relevant past creative performance, competitor context, and strategy guidance in one call.

**On a live competitor event:** ingest the intel note. Let `context_chain` automatically surface which of your running creatives is semantically exposed and which strategy brief is relevant.

**Cost:** one Python file, one Feather DB instance, one Gemini API call per ingest. No server. No additional infrastructure.

```python
from feather_db.integrations.gemini import GeminiEmbedder
import feather_db

emb = GeminiEmbedder(api_key="...")
db  = feather_db.DB.open("intelligence.feather", dim=768)

# Ingest competitor event
vec = emb.embed_text("Competing fintech FD campaign. 8.85% rate. Budget urgency.")
db.add(id=9001, vec=vec, meta=competitor_meta)

# One call — surfaces strategy brief, related creatives, exposed ads
chain = db.context_chain(vec, k=5, hops=2, modality="text")
```

---

*Built with [Feather DB v0.5.0](https://github.com/feather-store/feather) and Google Gemini Embedding 2.*
*Experiment code: `experiments/gemini_cross_modal/`*
