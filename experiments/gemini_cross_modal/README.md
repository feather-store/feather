# Feather DB × Gemini Embedding 2 — Cross-Modal Experiment

**Status:** Mock mode verified ✓ · Real mode ready (needs `GOOGLE_API_KEY`)

## Files

```
experiments/gemini_cross_modal/
├── README.md          this file
├── embedder.py        GeminiEmbedder — real + mock mode
├── experiment.py      4 experiments using real Stable Money ad data
├── blog.md            full technical blog post
└── results/
    ├── gemini_cross_modal.feather   Feather DB (generated on run)
    └── experiment_results.json      raw experiment output
```

## Run

```bash
# From repo root
source repro_venv/bin/activate

# Mock mode — no API key, runs fully offline
python3 experiments/gemini_cross_modal/experiment.py

# Real Gemini Embedding 2 — requires Google AI Studio API key
GOOGLE_API_KEY=AIza... python3 experiments/gemini_cross_modal/experiment.py --real
```

## What it tests

| Experiment | What it proves |
|---|---|
| EXP 1: Cross-modal search | Image query → text results (unified space) |
| EXP 2: context_chain | Competitor creative → strategy traversal in 2 hops |
| EXP 3: Same-ad coherence | text↔image similarity for same creative |
| EXP 4: Threat detection | Competitor surfaces in top-10 with no special logic |

## Mock vs Real

| Metric | Mock | Expected (Real Gemini) |
|---|---|---|
| text↔image same-ad similarity | ~0.10 | ~0.78 |
| Cross-modal hits in EXP 1 | 0/8 | 4–6/8 |
| Mock purpose | Architecture + graph validation | True cross-modal semantic search |

Mock mode validates that the Feather architecture (context_chain, typed edges, BFS traversal) works correctly. Real mode shows the cross-modal semantic power that Gemini Embedding 2 adds.

## Blog post

See `blog.md` for the full write-up.
