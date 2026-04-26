# arXiv Submission — Feather DB Paper (April 2026 update)

This package is what you upload to arxiv.org to update / replace the existing paper. **I (Claude) cannot upload to arXiv programmatically — it's a manual web form**. This guide is the runbook.

## What's in this directory

```
docs/arxiv-submission/
├── featherdb_paper.tex      ← the LaTeX source (single file, no \include)
├── featherdb_paper.pdf      ← compiled PDF for sanity-checking before upload
└── SUBMISSION_GUIDE.md      ← this file
```

## What changed vs. the previous version on arXiv

The diff is **§4.7 End-to-End Memory Benchmark: LongMemEval**:

- New empirical section reporting Feather DB's LongMemEval_S results.
- Two answerer tiers: Gemini-2.5-flash (0.657) and GPT-4o (0.693).
- New comparison table (Table 6 / `tab:longmemeval`) vs the LongMemEval paper's full-context ceilings, Zep, Mem0, Supermemory.
- Discussion of where the remaining gap to closed competitors lives (knowledge-update, multi-session, temporal-reasoning) and why it's structural rather than answerer-bounded.
- Four new bibliography entries: Xu et al. 2024 (LongMemEval), Rasmussen et al. 2025 (Zep), Mem0 token-efficient blog, Supermemory research.

No other section is materially changed (abstract was already general enough to cover this).

## Upload steps (manual)

### 1. Verify the PDF compiles cleanly

```bash
cd docs/arxiv-submission
tectonic featherdb_paper.tex
# expect "Writing `featherdb_paper.pdf` (...)" with no errors
# warnings about underfull/overfull boxes are typesetting noise, ignore
```

If you don't have `tectonic`: `brew install tectonic` (no sudo, single binary).

### 2. Sanity check

Open the PDF and confirm:
- Section numbering is intact.
- Table 6 (LongMemEval comparison) appears with our scores in bold.
- Bibliography includes the four new entries (search for "Xu et~al.", "Rasmussen", "Mem0", "Supermemory").

### 3. Pack for arXiv

arXiv accepts a single `.tex` upload. They'll run their LaTeX build server (it auto-detects packages). Pack:

```bash
cd docs/arxiv-submission
zip featherdb_paper.zip featherdb_paper.tex
```

(Don't include the PDF — arXiv recompiles from source. Upload only the .tex.)

### 4. Upload via arxiv.org

1. Log in at https://arxiv.org/user
2. Find the existing paper in "Articles" listing
3. Click "Replace" → "Replace this article"
4. Upload `featherdb_paper.tex` (or the zip if you have multiple files)
5. Step through metadata pages — keep **same primary category** (cs.IR or cs.DB depending on what you used) and **same authors**
6. Update the abstract if needed (current one is general; new section doesn't strictly require an abstract change, but you may want to mention "now includes empirical validation on LongMemEval"). Suggested abstract addendum (one sentence):

> *"We additionally report end-to-end results on the LongMemEval benchmark (500 questions, S variant), where Feather DB scores 0.693 with GPT-4o and 0.657 with Gemini-2.5-flash, exceeding the LongMemEval paper's full-context GPT-4o ceiling of 0.640 with a 10-snippet retrieval pipeline."*

7. Submit. arXiv runs its LaTeX build, you get a confirmation email when it's announced (typically within 24h).

### 5. Update the link in the README and HF Space

Once arXiv assigns the new version (e.g. `arXiv:XXXX.XXXXXv2`):

- Update `README.md` — "Architecture" or "Resources" section.
- Update HF Space `Hawky-ai/feather-db` `main` README.
- Tweet the link (optional, since the launch already covered the result).

## Common pitfalls (from prior submissions)

- **Bibliography format mismatch**: arXiv prefers `natbib` with one of its known styles. We use `\begin{thebibliography}` directly (manual entries) — this works fine.
- **Missing `\maketitle`**: ensure it's present (line ~120 in the .tex).
- **PDF-only submission**: don't do this; arXiv strongly prefers source. If you must submit PDF-only (e.g. you don't want them to recompile), use the "PDF only" option but it's discouraged.
- **License mismatch**: keep the same license you submitted with the v1 paper.

## If arXiv rejects the build

The build log will tell you the exact line. Most common cause is a missing package — arXiv's TeXLive distribution should have everything we use (`amsmath`, `booktabs`, `natbib`, etc.). If something's missing, add the package import to `featherdb_paper.tex` line ~4-30 and resubmit.

For local debugging (matching arXiv's environment as closely as possible):

```bash
docker run --rm -v "$(pwd)":/work -w /work texlive/texlive:latest \
  pdflatex featherdb_paper.tex
```

## Timing

arXiv announces submissions in batches (typically 4 PM ET, weekdays only). Submit by **2 PM ET on a weekday** for same-day announcement. Friday afternoon submissions wait until Monday.

## After it's live

- Verify the arXiv abstract page renders correctly: https://arxiv.org/abs/<your-id>
- The PDF link should work: https://arxiv.org/pdf/<your-id>
- Update Citations / Mentions tracking — Semantic Scholar, Google Scholar will pick it up within a few days.
