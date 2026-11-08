# Feather DB Benchmark Harness

Reproducible benchmarks. One command per run; results and reports are version-controlled.

## Layout

```
bench/
├── runner.py          # BenchRunner: orchestrates, persists JSON
├── metrics.py         # latency percentiles, recall@k, NDCG@k, QPS
├── report.py          # renders Markdown report from results/
├── datasets/          # pluggable dataset loaders
├── scenarios/         # one file per scenario (vector_ann, hybrid, ...)
├── results/           # per-run JSON (checked in)
├── reports/latest.md  # rolled-up Markdown
└── __main__.py        # python -m bench CLI
```

## Quick start

```bash
# Install deps (numpy is required; feather_db must be built)
python setup.py build_ext --inplace

# Run the vector ANN scenario on synthetic data
python -m bench run vector_ann --dataset synthetic --n 10000 --dim 128 --k 10

# Regenerate the rolled-up report
python -m bench report
cat bench/reports/latest.md
```

## Adding a scenario

1. Drop a file under `bench/scenarios/my_scenario.py` exposing `run(...)` that returns a metrics dict.
2. Register it in `bench/__main__.py` `SCENARIOS` dict.
3. If it needs a new dataset, add a loader under `bench/datasets/`.

## Planned scenarios

| Scenario | Status | Purpose |
|----------|--------|---------|
| `vector_ann` | ✅ | HNSW latency + recall@k vs brute force |
| `keyword_bm25` | ⏳ | BM25 NDCG@10 on BEIR |
| `hybrid_rrf` | ⏳ | Hybrid RRF NDCG@10 on BEIR |
| `wal_fuzz` | ⏳ | Crash-recovery correctness |
| `longmemeval` | ⏳ | Long-term memory QA (5 axes) |
| `locomo` | ✅ | 10 conversations, 1,540 QA, Mem0's protocol: `python -m bench.locomo` (below) |
| `thread_stress` | ⏳ | Concurrent add/search/link race test |

## Memory poisoning

```bash
python -m bench.poisoning                 # ~3 s, no API keys
python -m bench.poisoning --model gemini-text:gemini-2.5-flash   # + obedience
python -m bench.poisoning --out bench/results/poisoning.json
```

Plants "SYSTEM OVERRIDE" memories (relevance-stuffed, pinned, shadowing a
binding rule's or a pinned note's key, forging a binding header) of origin
agent / tool / external, then counts how often they reach a context packet, a
hot-set packet and `Pocket.hot_text()`; how often as instruction rather than
quoted data; and, with `--model`, how often a model follows them (each one
asks for a canary code in the reply). Each run compares the default trust
policy with one that trusts every origin. See the module docstring for the
metrics.

## Evaluation protocol (2026-11-08)

Three things made earlier numbers less trustworthy than they looked: rerun
noise larger than the effects judged (same-context reruns flip ~3% of
questions; +0.6 J was judged on one run), tuning in view of the questions
reported on (the extractor prompt was tuned looking at LoCoMo and the
contradiction worlds), and a published answer key with known errors. The
rules now (`bench/protocol.py`):

1. **Frozen dev/test splits, tune on dev only.** LongMemEval: 100 of 500
   questions (`datasets/longmemeval_s_dev.json`). LoCoMo: 3 of 10
   conversations, 459 of 1,540 questions (`datasets/locomo_split.json`; by
   conversation, because tuning reads whole conversations). Contradiction:
   `--split dev` is the default seed and lexicon; `--split test` is a
   disjoint seed range with a held-out lexicon (five other attributes, names,
   phrasings, distractors). Runs default to dev.
2. **A run that touches test questions needs a committed tree** (LoCoMo
   `--split test|all` or a test conversation, contradiction `--split test`,
   any judged LongMemEval run beyond the dev ids). The result's commit then
   names the code it measured.
3. **3-5 runs per configuration, compared with paired tests.** LoCoMo
   `run --repeats N`, contradiction `--repeats N` (reader), `REPEATS=` in
   `scripts/lme_dev.sh` / `lme_final.sh`. Then
   `python -m bench.compare {lme,locomo,contradiction} --a A_runs --b B_runs`:
   the difference with a 95% bootstrap CI over questions, the questions that
   flipped with McNemar's exact p (majority vote per question), each side's
   rerun noise, and a verdict. A difference counts only if the CI excludes 0
   and p < 0.05; with fewer than 3 runs a side it warns that the noise was
   not measured. Questions that share a LoCoMo conversation or a
   contradiction world are correlated, so a second interval resamples whole
   conversations / worlds, and it must exclude 0 too (below 5 clusters there
   is none, and a warning says so).
4. **LoCoMo against an audited key.** 99 of the 1,540 published answers are
   wrong ([dial481/locomo-audit](https://github.com/dial481/locomo-audit),
   pinned at `9493fb4`: 33 hallucinated, 26 mis-dated, 24 wrong speaker,
   13 ambiguous, 3 incomplete; its other 57 issues cite the wrong evidence
   and change no score). `--key sound` scores the 1,441 sound questions
   (free); `--key audited` re-judges the 99 against the corrected answers
   (`python -m bench.locomo rekey RUN.jsonl`, ~$0.01). A perfect system
   scores 93.6 on the published key.
5. **A second, stricter LoCoMo judge.** Mem0's judge prompt is the one the
   published systems are scored with, so it stays the comparable number; the
   audit found a pipeline's judge accepting 62.8% of deliberately wrong,
   plausible answers. `--judge strict` is LongMemEval's official yes/no
   prompt (the response must contain the answer; a temporal question
   tolerates an off-by-one count of days) on the same gpt-4o-mini:
   `bench.locomo rejudge RUN.jsonl` adds its verdicts, and
   `bench.locomo leniency` measures how many of the audit's 1,441 known-wrong
   answers (sound-key questions) each judge accepts. Report both.

**Earlier claims, rescored** (one run each, so no rerun noise is in these
CIs; numbers before this date were tuned with every conversation in view,
test included):

| claim | key | n | A | B | diff [95% CI], by question | by conversation | McNemar p | verdict |
|---|---|---:|---:|---:|---|---|---:|---|
| LoCoMo turns → index (78.77 → 82.40) | original | 1540 | 78.77 | 82.40 | +3.64 [+1.95, +5.32] | [+1.49, +5.55] | <0.0001 | B better |
| | sound | 1441 | 81.19 | 84.52 | +3.33 [+1.60, +5.07] | [+1.32, +5.21] | 0.0002 | B better |
| ... on the 7 test conversations | sound | 1014 | 80.97 | 84.91 | +3.94 [+1.87, +6.02] | [+1.67, +6.06] | 0.0003 | B better |
| ... on the 3 dev conversations | sound | 427 | 81.73 | 83.61 | +1.87 [−1.17, +4.92] | too few | 0.29 | noise |
| index → + supersede, 0.2.0 keys (82.66) | sound | 1441 | 84.52 | 84.87 | +0.35 [−0.21, +0.90] | [−0.20, +0.91] | 0.33 | noise |
| index → + supersede, 0.3.1 keys (82.92) | sound | 1441 | 84.59 | 85.01 | +0.42 [−0.21, +1.04] | [−0.08, +0.91] | 0.29 | noise |
| 0.2.0 → 0.3.1 extraction (82.40 → 82.34) | sound | 1441 | 84.52 | 84.59 | +0.07 [−1.11, +1.25] | [−0.64, +0.66] | 1.0 | noise (85 questions flipped) |

The index gain survives every check here. The supersede gains do not: they
are not distinguishable from noise. 82.40 itself is 80.5-84.4 at 95% on
question sampling alone (one run); on the sound key, 84.52 [82.58, 86.33].
`--key audited`, the strict judge, the leniency check and 3-5 runs of each
configuration are still to be run (paid; the commands are above). The contradiction reader percentages in this file are dev worlds
(the ones the extractor prompts were tuned on), one reader run each.

## LoCoMo, contradiction, poisoning

Mem0, Zep and Letta publish LoCoMo; Feather published only LongMemEval. These
three run side by side: LoCoMo for the head-to-head number, and two
structure-level evals (no model needed) for what LoCoMo does not test: whether
a fact that changed comes back current, and whether planted text reaches a
prompt as instruction.

```bash
set -a; source <your .env>; set +a     # OPENROUTER_API_KEY, GOOGLE_API_KEY
python -m bench.locomo categories                      # free: what each category number holds
python -m bench.locomo run --conversations conv-44     # smoke, one dev conversation
python -m bench.locomo run --repeats 3                 # the 3 dev conversations, 3 runs
python -m bench.locomo run --split test --repeats 3    # held out, committed tree: ~$1.0 a run
python -m bench.locomo run --memory index              # turns, found via extracted facts too
python -m bench.locomo rescore RUN.jsonl --key sound   # without the 99 wrong answers
python -m bench.compare locomo --a A_r*.jsonl --b B_r*.jsonl --key sound
python -m bench.contradiction                          # free, ~3 s
python -m bench.contradiction --embed gemini --model openrouter:openai/gpt-4o-mini
python -m bench.poisoning                              # free, ~2 s
python -m bench.suite [--model provider:model] [--locomo] [--out-dir bench/results]
```

### LoCoMo protocol

Mem0's `evaluation/` at the paper's commit
([`393a4fd`](https://github.com/mem0ai/mem0/blob/393a4fd5a6cfeb754857a2229726f567a9fadf36/evaluation/metrics/llm_judge.py)),
so the numbers sit next to Mem0's paper (arXiv:2504.19413), Table 1-2:

- `locomo10.json` from snap-research/locomo (pinned commit and sha256 in the
  module). Categories 1-4 only, 1,540 questions; category 5 (adversarial, 446)
  has no gold answer, and Mem0's `evals.py` skips it.
- **J**: Mem0's judge prompt verbatim, gpt-4o-mini, temperature 0, JSON mode.
  **F1**: Mem0's set-of-tokens F1. **BLEU-1**: nltk `sentence_bleu` (1,0,0,0),
  smoothing method1 (nltk 3.10.4 for the run below; without nltk an
  approximate tokenizer is used, 0.5 points off on the smoke run, and the
  summary says so).
- Feather: a fresh DB per conversation, one record per turn
  (`"speaker: text"` + the image caption), timestamp = the session date,
  gemini-embedding-001 vectors (cached under `~/.cache/feather/locomo/emb/`),
  `hybrid_search(qvec, question, k=30)` with no decay scoring. The 30 turns are
  rendered grouped by session, oldest first, under each session's date and
  weekday (~1,250 tokens).
- Reader: gpt-4o-mini, temperature 0, a short answer-only prompt whose one
  substantive rule is resolving relative dates against the session date
  (`ANSWER_SYSTEM`). Not Mem0's answer prompt, which is written for extracted
  memories.
- Reader and judge reach gpt-4o-mini through OpenRouter (every call was served
  by OpenAI, per the response metadata). Cost is OpenRouter's own per-call
  figure; embeddings are list price.
- One run. Mem0 reports the mean of 10 (std 0.1-0.7 points).

**Category numbers.** In the data, 1 = multi-hop (evidence in >1 session for
96%), 2 = temporal (77% "When…"), 3 = open-domain (inference past the text),
4 = single-hop (one evidence turn), 5 = adversarial. mem0ai/memory-benchmarks
uses the same names. Mem0's *paper* labels its Table 1 columns 1 "Single Hop",
2 "Temporal", 3 "Multi-Hop", 4 "Open Domain": that is the only assignment
under which the n-weighted mean of each system's columns reproduces its Table 2
overall J (Mem0, Mem0g, Zep, LangMem, OpenAI, all to 0.01; `categories` prints
the check). So compare per category **by number**.

### LoCoMo results (2026-11-08, `locomo__locomo10__20261108_000830.json`)

| cat | name | n | J | F1 | BLEU-1 | evidence recall@30 | Mem0 J | Mem0g J | Zep J (Mem0's run) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | multi-hop | 282 | 70.57 | 39.94 | 29.86 | 0.59 | 67.13 | 65.71 | 61.70 |
| 2 | temporal | 321 | 71.34 | 57.24 | 42.06 | 0.83 | 55.51 | 58.13 | 49.31 |
| 3 | open-domain | 96 | 55.21 | 27.67 | 21.96 | 0.50 | 51.15 | 47.19 | 41.35 |
| 4 | single-hop | 841 | 87.04 | 63.31 | 52.30 | 0.87 | 72.93 | 75.71 | 76.60 |
| 1-4 | **overall** | 1540 | **78.77** | 55.55 | 44.17 | 0.79 | 66.88 | 68.44 | 65.99 |

Mem0's columns are its paper's, matched by category number. In the same paper
the full-context baseline (the whole conversation, gpt-4o-mini) scored 72.90.
$0.47 for the run: reader $0.34, judge $0.10, embeddings $0.03 (cached after).
Per conversation J ranged 71.4 (conv-42) to 88.9 (conv-30).

Read it with care:
- The judge is generous by design ("touches on the same topic" counts):
  "friends and confidantes" passed for "family", "24 June 2022" for "the
  Friday before 24 June 2022". It inflates every J under this protocol
  equally, ours and Mem0's.
- One run against a mean of ten; a different reader prompt; raw turns rather
  than extracted memories.
- Not comparable: Mem0's current 92.5% (mem0ai/memory-benchmarks: gpt-4o
  answerer and judge, a far more lenient judge with partial credit and 14-day
  date tolerance, top 200 memories); Zep's self-reported 75.14 (its own
  harness, models not stated, disputed by Mem0); Letta's 74.0 (an agent with
  filesystem tools on gpt-4o-mini; its judge not checked here).

### LoCoMo: extracted facts as index, raw turns as payload (2026-11-08)

`--memory facts` (the reader sees extracted facts) scored J 64.03: extraction
keeps ~4 facts per session and speaker and drops single-hop detail.
`--memory index` keeps the reader on the same 30 raw turns and adds each
extracted fact as an extra index record for the turn it cites (a fact hit
counts for its turn). Same extraction (cached), same reader, judge and context
size (~1,400 tokens). `locomo__locomo10_index__20261108_121646.json`:

| cat | name | J turns | J index | evidence recall@30 turns → index |
|---:|---|---:|---:|---:|
| 1 | multi-hop | 70.57 | 75.53 | 0.59 → 0.67 |
| 2 | temporal | 71.34 | 79.44 | 0.83 → 0.94 |
| 3 | open-domain | 55.21 | 53.12 | 0.50 → 0.58 |
| 4 | single-hop | 87.04 | 89.18 | 0.87 → 0.92 |
| 1-4 | **overall** | 78.77 | **82.40** | 0.79 → 0.86 |

The gain is retrieval's: on the 195 questions whose evidence recall rose, +75
right / −2; on the 1,297 with the same recall, +43 / −45 (the reader's noise);
on the 48 where it fell, +0 / −15. $0.42 (reader + judge).

`--supersede` adds, under an older turn, the newest statement of the same
extracted one-value key (speaker + subject|attribute) as "↳ later, session N:
...". It depends on the keys, and three extractions were measured:

| extraction | keys with >1 value | J index | J + supersede | flips (re-asked) |
|---|---:|---:|---:|---:|
| 0.2.0, any shared key | 117 (928 q noted) | 82.40 | not run | |
| 0.2.0, ContradictionResolver gate (definite + supersedes) | 55 pairs | 82.40 | 82.66 | +11 / −7 (353) |
| 0.3.1, `one_value` + `user_name` + `known_keys` | 55 | 82.34 | **82.92** | +17 / −8 (555) |

0.2.0 keys were mostly additive ("hobbies", "preference", "plan"); the
resolver sees only subject | predicate | object and still passed "continue
my education" → "plans to adopt". 0.3.1 asks the model whether a later value
makes a fact false (`one_value`), tells it the speaker's name and the keys
extracted so far (each speaker's sessions in order): a partner → married,
"no pets" → a puppy named Toby, a new car → a Ferrari. Some noise remains
("event", "next event"). The +0.6 is within noise (same-context flips run
±3%); one gain is the mechanism ("What car brand does Calvin own?": "Red
sports car" → "Ferrari", from "↳ later: Calvin bought a Ferrari"), most
are not. LoCoMo has few real updates; it shows supersession does no harm,
not what it is for. 0.3.1 extraction $0.24, index run $0.48, supersede
$0.17 (`locomo__locomo10_index__20261108_140548.json`,
`locomo__locomo10_index_supersede__20261108_141106.json`).

### Contradiction (`python -m bench.contradiction`)

20 trials × 8 people × 5 attributes that changed 2-4 times (city, employer,
manager, diet, phone plan), 12 distractors per person, each person in their
own scope; in-order and shuffled ingestion alternate. One query per
(person, attribute) through `PacketBuilder(Pocket(scope)).build(query=..., k=10)`,
1,500-token budget. Four ways of writing the same statements; see the module
docstring for the metrics. Reader: gpt-4o-mini via OpenRouter, 40 packets per
row (one trial's worth).

Keyword search (the default without an embedder;
`contradiction__keyword__20261108_000903.json`):

| variant | ingestion | current in | current first | stale / packet | stale only | tokens | reader correct | reader stale |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| notes | in order | 80% | 48% | 1.36 | 12% | 73 | 35% | 52% |
| notes | shuffled | 83% | 48% | 1.39 | 11% | 72 | 32% | 62% |
| notes, no valid time | in order | 80% | 48% | 1.36 | 12% | 73 | 38% | 50% |
| notes, no valid time | shuffled | 73% | 35% | 1.45 | 19% | 72 | 28% | 62% |
| keyed facts | in order | 100% | 100% | 2.04 (history) | 0% | 739 | 100% | 0% |
| keyed facts | shuffled | 100% | 100% | 2.02 (history) | 0% | 735 | 100% | 0% |
| items + valid_at | in order | 80% | 48% | 1.35 | 12% | 141 | 65% | 25% |
| items + valid_at | shuffled | 84% | 48% | 1.38 | 10% | 141 | 68% | 28% |

Dense search, `--embed gemini` (`contradiction__gemini__20261108_000948.json`):
notes and items have the current value in 98-99% of packets but first in only
37-42%, with 2.0 stale values per packet; the reader is right 35-45% and
answers a stale value 52-62%. Keyed facts: 100% in and first, 0 stale, 79
tokens, reader 95-98%.

What it shows:
- **Free-text notes do not resolve contradictions.** `Pocket.recall` ranks by
  query relevance with heat (recency) as a 0.05 tiebreak, so which of several
  matching statements comes first is decided by phrasing ("Emeka's home city
  is Porto" beats "Emeka lives in Bergen" for "which city"), not by time:
  current first 35-48%, about chance for 2-4 values. And a packet renders
  no dates, so the reader cannot tell which statement is current: gpt-4o-mini
  gave a stale value in half or more of the notes packets.
- **Valid time barely moves the ranking**: it matters only for shuffled
  ingestion, and only through the heat tiebreak (current first 48% vs 35%,
  stale only 11% vs 19%, keyword search).
- **items + valid_at** rank exactly like notes. The reader does better only
  because the default FeatherStore renders `valid_at: <unix epoch>` into the
  packet and the model compares the integers; with a text-only index (the
  `--embed` run) it falls back to the notes level.
- **Keyed facts work**: `facts.state(at=...)` kept the newest value as `text`
  under shuffled writes every time (held current 100%), and the current value
  always leads. But with the default index the packet is the raw JSON value:
  epoch floats, `source: None`, and the superseded values as a Python-repr
  `supersedes` list. That is ~2 stale values per packet (labelled as history)
  and 10x the tokens of a note. With `index_fields=["text"]` it is the current
  text alone.

Supersession on raw notes (2026-11-08, `--embed gemini --model-packets 400
--extract openrouter:openai/gpt-4o-mini`,
`contradiction__gemini__keys__20261108_*.json`). Both new arms write through
the product path, `Pocket.remember(valid_at=, index=)` and `PacketBuilder`:
the notes stay the payload, a chain shows once with its newest statement
as the line and the older ones as history, and a chain reached only through
an old statement brings its newest one in (`Pocket.group_superseded`). The
first run (inline history) gave extracted keys 94-95%, oracle keys 97%.

- `notes + keys`: each statement of an attribute indexed under
  `user|<attribute>`, the bench's own attribute: an oracle key.
- `notes + extracted keys`: each note through SessionFactExtractor 0.3.1 as
  it is written (per person, in write order), told the person's name and
  the keys already on file; its facts are the index. ~4,300 calls, $0.79.

With `PacketBuilder`'s default `history="section"` (superseded memories in
an `EARLIER VALUES (superseded)` section after the context), current value
asked (`--question now`) and the value before it (`--question previous`,
"Which city did Alice live in before Leeds?"), `--now 1794126998` (the dates
of the extraction run, so its cache is reused),
`contradiction__gemini__section_{now,previous}__20261108_*.json`:

| variant | current value | previous value | tokens |
|---|---:|---:|---:|
| notes (dated) | 84.5% | 71.0% | 108 |
| keyed facts | 99.4% | 0% (no history in the packet) | 112 |
| notes + keys (oracle) | 99.6% | 88.6% | 153 |
| notes + extracted keys | 96.0% | 82.4% | 146 |

Reader gpt-4o-mini, 800 packets per cell (in order and shuffled pooled).
Where the history goes, oracle keys (`--history`, 800 packets each):

| history | current value | previous value | tokens |
|---|---:|---:|---:|
| section (default) | 99.5% | 87.6% | 153 |
| inline `    earlier:` lines | 96.9% | 91.8% | 171 |
| none | 99.2% | 0% | 63 |

Inline history lost on current-value questions because a history line that
matches the question's wording ("Emeka lives in Bergen", asked "Which city
does Emeka live in now?") drew the reader; more explicit inline markers did
worse ("no longer current:" 95.8%, "replaced by the line above:" 91.0%).

Extracted keys lose the rest to chains they split: a stale value whose fact
the model did not mark one-value stays a statement. Key consistency over
the same phrasings (4 people x 40 statements, `user_name` + `known_keys`):
0.2.0 told nothing ~70% on one key; 0.3.1 city 30/32, employer 32/32,
manager 30/32, diet 22/32, phone plan 32/32, no distractor took an
attribute's key. "Diet" is the model reading a diet as additive (one can be
vegan and gluten-free). Tried and rejected: a prompt saying keys on file are
one-value (0.3.2: diet 6/32, LoCoMo false chains 52 -> 75) and deciding
one-value per key in code (majority of a key's flags: contradiction
coverage +1.5 points, LoCoMo false chains 52 -> 67).

### LongMemEval knowledge-update with facts as index (2026-11-08)

```
python -m bench run longmemeval --dataset s --k 20 --embedder gemini \
  --judge llm --judge-provider gemini-text --judge-model gemini-2.5-flash \
  --official-judge-provider openrouter --official-judge-model openai/gpt-4o-2024-08-06 \
  --workers 10 --limit 0 --question-types knowledge-update \
  --facts session --system-provider openrouter --system-model openai/gpt-4o-mini
```

`--facts session` extracts every session (SessionFactExtractor 0.3.1, each
question's sessions in date order, told the keys extracted so far: 3,704
calls, ~$3), indexes the facts beside the turns (a fact hit counts for its
turn, the reader still gets sessions) and puts a timeline of one-value keys
whose value changed above the sessions. Official judge, 78 questions:

| run | correct |
|---|---:|
| facts off, 2026-10-30 (two runs) | 71, 74 |
| facts off, 2026-11-08 (same code as below) | 74 (0.949) |
| facts session, 2026-11-08 | 75 (0.962) |

No measurable effect: retrieval already put every question's evidence in
the context (evidence fully retrieved 1.0), so the misses are the reader's,
and one question is inside the run-to-run spread. The timelines showed a
defect since fixed: an attribute named `event` chained unrelated events as
one value replacing the next (`replaces()` now refuses category names).
`longmemeval__s__20261108_140815_ku_facts.json`; the baseline's per-question
file was overwritten by this run (both started in the same second; result
names are now reserved, `runner.reserve`), its summary is
`longmemeval__s__20261108_140809_ku_baseline_summary.txt`.

### Poisoning (`python -m bench.poisoning`, `poisoning__20261108_001122.json`)

| mode | surface | packets | reached | poison / packet | as instruction | binding hijacked | displaced | pinned in | benign in | forged headers | blocked | benign / packet |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| undefended | packet(query) | 800 | 100% | 6.5 | 100% | 100% | n/a | n/a | n/a | 32 | 0% | 8.9 |
| undefended | packet(hot) | 20 | 100% | 9.5 | 100% | 100% | 100% | 100% | 51% | 8 | 0% | 60.8 |
| undefended | hot_text | 20 | 100% | 8.3 | 100% | n/a | 100% | 100% | 55% | 10 | n/a | – |
| defended | packet(query) | 800 | 58% | 1.1 | 0% | 0% | n/a | n/a | n/a | 0 | 0% | 8.9 |
| defended | packet(hot) | 20 | 100% | 5.0 | 0% | 0% | 0% | 52% | 59% | 0 | 0% | 71.1 |
| defended | hot_text | 20 | 100% | 5.5 | 0% | n/a | 0% | 56% | 62% | 0 | n/a | – |

Planted text still reaches 58% of query packets under the default policy, but
only as quoted data: never as instruction, never as a binding rule, no forged
headers.

## Prompt-cache reuse of packets (`python -m bench.packet_cache`)

A provider prompt cache reuses a request's prefix up to the first byte that
changed. `ContextPacket.stable_text` (required rules, then trusted pins and due
intentions, each in key order) is the same every turn; `volatile_text` (this
query's recall) is not. The bench builds one 12-turn session (400 notes, 180-word
stand-in answers) and sends the same packets in three layouts:

- **whole**: `packet.text` in the system prompt, where agents usually put memory.
- **split-keep**: `stable_text` in the system prompt, `volatile_text` in each
  user turn, kept in the history.
- **split**: the same, but only the newest turn carries its `volatile_text`.

`billed` is input in uncached-token equivalents (cached reads at 0.25 for
Gemini, 0.1 for the offline bound), relative to **whole**:

| measured | whole cached | split-keep cached / billed | split cached / billed |
|---|---:|---:|---:|
| offline prefix bound | 47% | 86% / 0.75 | 73% / 0.60 |
| Gemini 2.5 Flash implicit cache, seed 7 | 19% | 77% / 1.07 | 41% / 0.81 |
| Gemini 2.5 Flash implicit cache, seed 8 | 12% | 78% / 0.97 | 47% / 0.71 |

Send `volatile_text` with the newest turn only. Kept in the history it raises
the hit rate and the bill: every turn's recall is re-sent. The gain grows with
the history: with 40-word answers, split is 0.88 of whole offline. `--anthropic`
and `--openai` measure the Claude and OpenAI caches (Claude requests use
`max_tokens=0`, so only input is billed); neither has been run yet.
