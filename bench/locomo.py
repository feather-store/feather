"""LoCoMo QA on Feather, scored the way Mem0 scored it, so the numbers sit next
to the ones Mem0, Zep and Letta publish.

    python -m bench.locomo categories                    # mapping check, free
    python -m bench.locomo run --conversations conv-44   # smoke: one dev conversation
    python -m bench.locomo run --repeats 3               # the 3 dev conversations, 3 runs
    python -m bench.locomo run --split test --repeats 3  # held out: committed tree only
    python -m bench.locomo run --categories 1,3 --k 50   # rerun some categories only
    python -m bench.locomo run --memory facts            # extracted facts, not raw turns
    python -m bench.locomo run --memory index            # raw turns, found via facts too
    python -m bench.locomo run --memory index --supersede --reuse <index run>.jsonl
    python -m bench.locomo rescore bench/results/locomo__locomo10__<stamp>__questions.jsonl
    python -m bench.locomo rescore RUN.jsonl --key sound   # without the 99 bad keys, free
    python -m bench.locomo rekey RUN.jsonl               # re-judge those 99 against the audit
    python -m bench.compare locomo --a A*.jsonl --b B*.jsonl --key sound

Protocol (bench/protocol.py): tune on the 3 dev conversations
(bench/datasets/locomo_split.json); run test from a committed tree, 3-5
repeats per configuration, compared with bench.compare. Numbers measured
before the split (2026-11-08) were tuned with all 10 conversations in view.

Answer key: 99 of the 1,540 category 1-4 answers are wrong (hallucinated,
mis-dated, attributed to the wrong speaker, ambiguous or incomplete) per the
dial481/locomo-audit audit (AUDIT_URL, pinned; 156 issues, the other 57 cite
the wrong evidence only and do not change a score). `--key original` scores
against the published key; `sound` drops those 99 (free); `audited` scores
them against the audit's corrected answers, re-judged by `rekey`.

`--memory facts` retrieves what Feather's write path extracts instead of raw
turns: SessionFactExtractor (gpt-4o-mini, one call per session and speaker,
cached under ~/.cache/feather/locomo/facts/) -> AgentMemory.ingest_facts over
a FeatherStore, then the same hybrid_search k, reader and judge (FACTS_SYSTEM
in place of ANSWER_SYSTEM). See the comment above FACTS_SYSTEM.

LoCoMo: Maharana et al., ACL 2024 (arXiv:2402.17753), `data/locomo10.json` from
snap-research/locomo at DATA_URL's commit: 10 conversations, 5,882 turns, 1,986
questions. Downloaded once to ~/.cache/feather/locomo/.

Protocol: Mem0's `evaluation/` (arXiv:2504.19413), read at the commit in
JUDGE_SOURCE.
  - Categories 1-4 only: 1,540 questions. Category 5 (adversarial, 446) has no
    `answer`; Mem0's evals.py skips it, and so does this.
  - J: Mem0's LLM judge. Prompt verbatim (JUDGE_PROMPT), gpt-4o-mini,
    temperature 0, JSON mode, one user message, `label == "CORRECT"` scores 1.
    Reader and judge reach gpt-4o-mini through OpenRouter by default.
  - F1: Mem0's token F1 (metrics/utils.py: lower-case, drop . , ! ?, split on
    whitespace, SETS of tokens).
  - BLEU-1: nltk `sentence_bleu`, weights (1,0,0,0), smoothing method1, over
    `nltk.word_tokenize` of the lower-cased texts. Needs nltk (and its
    punkt_tab data); without it an approximate tokenizer is used, and the
    summary says which (`bleu1_impl`). `rescore` recomputes F1/BLEU offline.
  - One run. Mem0 reports the mean of 10 (std 0.1-0.7 points).
  - Mem0's paper calls its judge "a separate, more capable LLM"; its code uses
    gpt-4o-mini. This follows the code.

Category numbers -> names. Verified from the data (`categories` prints it):
    1  multi-hop     evidence in >1 session for 96% (mean 3.1 evidence turns)
    2  temporal      77% start with "When"; answers are dates
    3  open-domain   "Would X likely...", answers need inference past the text
    4  single-hop    one evidence turn (mean 1.06), always one session
    5  adversarial   no gold answer
This agrees with mem0ai/memory-benchmarks (benchmarks/locomo/prompts.py,
CATEGORY_NAMES). Mem0's PAPER labels the same numbers differently: its Table 1
columns are 1 "Single Hop", 2 "Temporal", 3 "Multi-Hop", 4 "Open Domain". That
is the only assignment under which the n-weighted mean of each system's Table 1
columns reproduces its Table 2 overall J (Mem0 66.88, Mem0g 68.44, Zep 65.99,
LangMem 58.10, OpenAI 52.90, all to 0.01; MEM0_TABLE1). So to compare per
category with Mem0's paper, match by NUMBER, not by name: their "Single Hop"
column is our multi-hop row. Overall J does not depend on the mapping.

Feather side:
  - A fresh DB per conversation (namespace = sample_id), one record per turn:
    content "speaker: text" (+ " [shares a photo: <blip caption>]"), timestamp
    = the session's date and time (read as UTC).
  - Vectors: gemini-embedding-001 (3072-d), turns and questions, cached under
    ~/.cache/feather/locomo/emb/ so a rerun pays only for the reader and judge.
  - Retrieval: `db.hybrid_search(qvec, question, k=K)` (BM25 + dense, RRF), no
    decay scoring. K = 30 turns.
  - Context: those turns grouped by session, sessions oldest first, turns in
    dialogue order, each session headed by its date and weekday.
  - Reader: gpt-4o-mini, temperature 0, ANSWER_SYSTEM (short, answer only).
    Not Mem0's answer prompt: theirs is written for extracted memories. Ours
    answers from raw turns, so its one rule of substance is resolving relative
    dates against the session date.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

DATA_URL = ("https://raw.githubusercontent.com/snap-research/locomo/"
            "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json")
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
# The answer-key audit (CC BY-NC 4.0, as LoCoMo): downloaded, never vendored.
AUDIT_URL = ("https://raw.githubusercontent.com/dial481/locomo-audit/"
             "9493fb4b4af4256ed17a18e8fd0b3cfdeec29539/errors.json")
AUDIT_SHA256 = "f298ef46263fada20688a1d672be37d4f488c6da3b19343eea55ae5c1b5fe55e"
# Error types that change a score; WRONG_CITATION leaves the answer right.
SCORE_CORRUPTING = ("HALLUCINATION", "TEMPORAL_ERROR", "ATTRIBUTION_ERROR",
                    "AMBIGUOUS", "INCOMPLETE")
KEYS = ("original", "sound", "audited")
# The audit's judge-leniency test: a deliberately wrong, plausible answer to
# every question (Claude Opus 4.6, given the key), the same pin.
LENIENCY_URL = ("https://raw.githubusercontent.com/dial481/locomo-audit/"
                "9493fb4b4af4256ed17a18e8fd0b3cfdeec29539/ap-baseline/v2/"
                "ap_eval_results_scored.json")
LENIENCY_SHA256 = "3c32acbdcf667967a4e1f9f541b06566ae5e7361dc92f79f9043af7d068adc6b"
# Two judges. mem0: Mem0's prompt (JUDGE_PROMPT), the one published systems
# are scored with, so the comparable number. strict: LongMemEval's official
# yes/no prompts, verbatim (bench.lme_eval), on the same model: the response
# must contain the answer, a vaguer or partial one is wrong; a temporal
# question tolerates an off-by-one count of days. `leniency` measures how
# many known-wrong answers each accepts.
JUDGES = ("mem0", "strict")
SCORE_FIELDS = {"mem0": ("llm_score", "llm_score_audited"),
                "strict": ("strict_score", "strict_score_audited")}
CACHE = Path(os.environ.get("FEATHER_CACHE", Path.home() / ".cache" / "feather")) / "locomo"
EMB_MODEL = "gemini-embedding-001"
K = 30
# gpt-4o-mini through OpenRouter (the OpenAI key on hand is invalid). Same
# model Mem0 used for both roles; OpenRouter serves it from OpenAI or Azure,
# and the summary records which (`served_by`).
READER = "openrouter:openai/gpt-4o-mini"
JUDGE = "openrouter:openai/gpt-4o-mini"

# number -> (name verified from the data, the column Mem0's paper reports it under)
CATEGORIES = {
    1: ("multi-hop", "Single Hop"),
    2: ("temporal", "Temporal"),
    3: ("open-domain", "Multi-Hop"),
    4: ("single-hop", "Open Domain"),
}
ADVERSARIAL = 5

# Mem0 paper (arXiv:2504.19413) Table 1 J per column, and Table 2 overall J.
MEM0_TABLE1 = {
    # system: ({column: J}, overall)
    "Mem0":    ({"Single Hop": 67.13, "Multi-Hop": 51.15, "Open Domain": 72.93,
                 "Temporal": 55.51}, 66.88),
    "Mem0g":   ({"Single Hop": 65.71, "Multi-Hop": 47.19, "Open Domain": 75.71,
                 "Temporal": 58.13}, 68.44),
    "Zep":     ({"Single Hop": 61.70, "Multi-Hop": 41.35, "Open Domain": 76.60,
                 "Temporal": 49.31}, 65.99),
    "LangMem": ({"Single Hop": 62.23, "Multi-Hop": 47.92, "Open Domain": 71.12,
                 "Temporal": 23.43}, 58.10),
    "OpenAI":  ({"Single Hop": 63.79, "Multi-Hop": 42.92, "Open Domain": 62.29,
                 "Temporal": 21.71}, 52.90),
}

# USD per 1M tokens (input, output): list prices; check before trusting.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00),
          "gemini-embedding-001": (0.15, 0.0)}

# Source: Mem0's evaluation/metrics/llm_judge.py at the paper's commit,
# https://github.com/mem0ai/mem0/blob/393a4fd5a6cfeb754857a2229726f567a9fadf36/evaluation/metrics/llm_judge.py
# (2025-04-29; arXiv v1 was 2025-04-28). Unchanged except one apostrophe
# (’ -> ') through aae5989e78a6188b3b047c104d960c9ad0927e75, the last version
# before evaluation/ moved to mem0ai/memory-benchmarks (mem0 9315e30,
# 2026-06-13), which uses a different, more lenient judge. Verbatim, trailing
# spaces included.
JUDGE_SOURCE = ("https://github.com/mem0ai/mem0/blob/393a4fd5a6cfeb754857a2229726f567a9fadf36"
                "/evaluation/metrics/llm_judge.py")
JUDGE_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

ANSWER_SYSTEM = """\
You answer questions about a long conversation between {a} and {b}, from \
excerpts of it retrieved from memory.

The excerpts are grouped by session, oldest first; each session is headed by \
the date it took place.

- Resolve every relative date against the date of the session it was said in: \
"yesterday" in a session of Monday 8 May 2023 is 7 May 2023, "last year" is \
2022, "last Saturday" is 6 May 2023. Answer with the absolute date, month or year.
- Give the answer only: a short phrase, no explanation. For a list, give every \
item the excerpts support.
- If the excerpts do not state the answer outright, give the most likely answer \
they support."""

ANSWER_USER = "Excerpts:\n\n{context}\n\nQuestion: {question}\nAnswer:"

_SESSION = re.compile(r"session_(\d+)$")
_DIA = re.compile(r"D:?(\d+):(\d+)")


# ── data ─────────────────────────────────────────────────────────────────

def load() -> list[dict]:
    path = CACHE / "locomo10.json"
    if not path.exists() or path.stat().st_size == 0:
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[locomo] downloading {DATA_URL} -> {path}", file=sys.stderr)
        subprocess.check_call(["curl", "-fL", "--retry", "3", "-o", str(path), DATA_URL])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        print(f"[locomo] warning: {path} is not the pinned file (sha256 differs)",
              file=sys.stderr)
    return json.loads(raw)


def load_audit() -> list[dict]:
    """The audit's error records, downloaded once and checked against the
    pinned sha256 (refused if different: a moved key is a different key)."""
    path = CACHE / "audit_errors.json"
    if not path.exists() or path.stat().st_size == 0:
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[locomo] downloading {AUDIT_URL} -> {path}", file=sys.stderr)
        subprocess.check_call(["curl", "-fL", "--retry", "3", "-o", str(path), AUDIT_URL])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUDIT_SHA256:
        raise SystemExit(f"[locomo] {path} is not the pinned audit (sha256 differs)")
    return json.loads(raw)


def audit_corrections(data: Optional[list] = None) -> dict:
    """(sample_id, qa_index) -> {"error_type", "correct_answer"} for the 99
    score-corrupting answers. The audit's `locomo_<i>_qa<n>` is conversation
    i in file order, qa_index n (0-based); every record is checked against
    the question and the published answer, and a mismatch stops the run."""
    data = data if data is not None else load()
    norm = lambda x: " ".join(str(x).split())          # noqa: E731
    out = {}
    for e in load_audit():
        m = re.fullmatch(r"locomo_(\d+)_qa(\d+)", e["question_id"])
        ci, qi = int(m.group(1)), int(m.group(2))
        q = data[ci]["qa"][qi]
        if (norm(q["question"]), norm(q.get("answer")), q["category"]) != \
                (norm(e["question"]), norm(e["golden_answer"]), e["category"]):
            raise SystemExit(f"[locomo] audit record {e['question_id']} does not match "
                             "the dataset: the mapping is wrong")
        if e["error_type"] in SCORE_CORRUPTING:
            out[(data[ci]["sample_id"], qi)] = {"error_type": e["error_type"],
                                               "correct_answer": e["correct_answer"]}
    return out


def parse_date(s: str) -> _dt.datetime:
    """'1:56 pm on 8 May, 2023' -> datetime (naive, read as UTC)."""
    return _dt.datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")


def dia_key(dia_id: str) -> Optional[tuple[int, int]]:
    """(session, turn) for 'D3:12'. Evidence lists hold a few malformed ids
    ('D:11:26', 'D8:6; D9:17'); evidence_ids() splits those."""
    m = _DIA.fullmatch(dia_id.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def evidence_ids(q: dict) -> set[tuple[int, int]]:
    out = set()
    for e in q.get("evidence") or []:
        for part in re.split(r"[;\s]+", e):
            k = dia_key(part) if part else None
            if k:
                out.add(k)
    return out


def turns_of(sample: dict) -> list[dict]:
    """Every turn, sessions oldest first, in dialogue order."""
    conv = sample["conversation"]
    sessions = sorted((int(_SESSION.match(k).group(1)), k) for k in conv if _SESSION.match(k))
    out = []
    for n, key in sessions:
        when = conv[f"{key}_date_time"]
        dt = parse_date(when)
        for t in conv[key]:
            cap = t.get("blip_caption")
            out.append({
                "session": n, "date": when, "dt": dt, "dia_id": t["dia_id"],
                "speaker": t["speaker"],
                "content": f"{t['speaker']}: {t.get('text', '')}"
                           + (f" [shares a photo: {cap}]" if cap else ""),
            })
    out.sort(key=lambda t: (t["dt"], t["session"]))      # stable: dialogue order kept
    return out


def questions_of(sample: dict) -> list[dict]:
    """Categories 1-4, in file order, each with its index in `qa`."""
    return [{**q, "qa_index": i} for i, q in enumerate(sample["qa"])
            if q["category"] != ADVERSARIAL]


# ── embeddings (cached) ──────────────────────────────────────────────────

def vectors(sample: dict, turns: list[dict], qs: list[dict], spent: Counter) -> np.ndarray:
    """Turn vectors then question vectors, from cache or gemini-embedding-001.
    The cache file name carries a hash of the texts, so a change in what is
    embedded is a miss, never a stale hit."""
    texts = [t["content"] for t in turns] + [q["question"] for q in qs]
    h = hashlib.sha1("\n\x00".join(texts).encode()).hexdigest()[:12]
    d = CACHE / "emb" / EMB_MODEL
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sample['sample_id']}_{h}.npy"
    if path.exists():
        return np.load(path)
    from .providers_extra import GeminiEmbedder
    v = GeminiEmbedder(EMB_MODEL).embed_batch(texts).astype(np.float32)
    np.save(path, v)
    spent["embed_tokens"] += sum(len(t) for t in texts) // 4
    print(f"[locomo] embedded {sample['sample_id']}: {len(texts)} texts", file=sys.stderr)
    return v


# ── retrieval ────────────────────────────────────────────────────────────

_FACT_ID = 1_000_000      # fact index entries: ids from here; turns are 1..N


def retrieve(sample: dict, turns: list[dict], qs: list[dict], vecs: np.ndarray,
             k: int, index: Optional[list] = None, select: str = "rank") -> list[list[int]]:
    """Per question, the indices into `turns` of its top-k hybrid hits.

    `index` ([(turn index, statement, vector)], --memory index): extracted
    facts added as extra records beside the turns. A hit on one counts for the
    turn it cites, so the result is still k distinct raw turns; the search
    over-fetches 4k records to fill them.

    `select` "coverage": k turns chosen from the top 2k by information
    covered (feather_db.coverage; units: the facts each turn was found by,
    else its sentences), so a fact said in several turns takes one slot."""
    import feather_db
    tv, qv = vecs[:len(turns)], vecs[len(turns):]
    owner = {i + 1: i for i in range(len(turns))}
    with tempfile.TemporaryDirectory() as d:
        db = feather_db.DB.open(os.path.join(d, "locomo.feather"), dim=tv.shape[1])
        try:
            for i, (t, v) in enumerate(zip(turns, tv)):
                m = feather_db.Metadata()
                m.content = t["content"]
                m.timestamp = int(t["dt"].replace(tzinfo=_dt.timezone.utc).timestamp())
                m.namespace_id = sample["sample_id"]
                m.entity_id = t["speaker"]
                m.source = t["dia_id"]
                db.add(id=i + 1, vec=v, meta=m)
            for j, (ti, text, v) in enumerate(index or []):
                m = feather_db.Metadata()
                m.content = text
                m.namespace_id = sample["sample_id"]
                m.source = turns[ti]["dia_id"]
                db.add(id=_FACT_ID + j, vec=v, meta=m)
                owner[_FACT_ID + j] = ti
            out = []
            cover = select == "coverage"
            want = 2 * k if cover else k
            for j, q in enumerate(qs):
                seen: list[int] = []
                score: dict[int, float] = {}
                found_by: dict[int, list[str]] = {}
                for h in db.hybrid_search(qv[j], q["question"],
                                          k=4 * want if index else want):
                    ti = owner[h.id]
                    if h.id >= _FACT_ID:
                        found_by.setdefault(ti, []).append(h.metadata.content)
                    score[ti] = max(score.get(ti, 0.0), float(h.score))
                    if ti not in seen and len(seen) < want:
                        seen.append(ti)
                if cover and seen:
                    from feather_db import coverage
                    units = [coverage.units_of(found_by.get(ti, []))
                             or coverage.units_of(coverage.sentences(turns[ti]["content"]))
                             for ti in seen]
                    keep = coverage.select([score[ti] for ti in seen], [1.0] * len(seen), units,
                                           budget=float(k), max_items=k)
                    seen = [seen[i] for i in keep]
                out.append(seen)
            return out
        finally:
            db.close(save=False)


def render(turns: list[dict], idx: list[int], notes: Optional[dict] = None) -> str:
    """`notes` (--supersede): turn index -> [(later turn index, statement)].
    Each is written under its turn unless the later turn is in the context
    already, where the reader sees it in its own session."""
    lines, cur = [], None
    shown = set(idx)
    for i in sorted(shown):
        t = turns[i]
        if t["session"] != cur:
            cur = t["session"]
            lines.append(f"\n[Session {cur} · {t['date']} ({t['dt'].strftime('%A')})]")
        lines.append(t["content"])
        for later, text in (notes or {}).get(i, ()):
            if later not in shown:
                lines.append(f"  ↳ later, session {turns[later]['session']}: {text}")
    return "\n".join(lines).strip()


# ── extracted facts (--memory facts) ─────────────────────────────────────
#
# Mem0 retrieves memories an LLM extracted at write time; the default here
# retrieves raw turns. This arm runs Feather's own write path instead:
# SessionFactExtractor (one call per session) -> AgentMemory.ingest_facts
# (events -> episodes, preferences, states -> facts with supersession) over a
# FeatherStore, then the same hybrid_search, reader and judge. The extractor
# is written for a user and an assistant, and keeps facts about the user
# only; LoCoMo is two people, so each session is extracted twice, once with
# each speaker as the user (Mem0 also adds each conversation once per
# speaker), and "the user" / "the assistant" in a statement become names.

FACTS_SYSTEM = """\
You answer questions about a long conversation between {a} and {b}, from \
memories extracted from it.

Each memory is one statement that starts with the date it refers to; they are \
listed oldest first.

- Answer with the absolute date, month or year when asked when.
- Give the answer only: a short phrase, no explanation. For a list, give every \
item the memories support.
- If the memories do not state the answer outright, give the most likely answer \
they support."""

FACTS_USER = "Memories:\n\n{context}\n\nQuestion: {question}\nAnswer:"


def _named(text: str, me: str, other: str) -> str:
    for role, name in (("user", me), ("assistant", other)):
        text = re.sub(rf"\b[Tt]he {role}'s\b", f"{name}'s", text)
        text = re.sub(rf"\b[Tt]he {role}\b", name, text)
    return text


def extract_facts(sample: dict, chat: "_Chat", workers: int) -> list[tuple[str, list]]:
    """[(speaker, [Fact])] for every (session, speaker), sessions oldest
    first. Each extraction is cached on disk (not a failed one), so a rerun
    pays nothing for it.

    Each speaker's sessions are extracted in order, each call told the
    speaker's name and the one-value keys already extracted for them
    (extractor 0.3.0: `user_name`, `known_keys`), so one attribute keeps one
    name across sessions. The two speakers run side by side."""
    from feather_db.extractors.base import Fact
    from feather_db.extractors.session_facts import (SessionFactExtractor, fact_key,
                                                     replaces)

    class _Provider:
        def complete(self, messages, max_tokens=3000, temperature=0.0):
            return chat(messages, max_tokens=max_tokens, temperature=temperature)

    ex = SessionFactExtractor(_Provider())
    conv = sample["conversation"]
    a, b = conv["speaker_a"], conv["speaker_b"]
    d = CACHE / "facts" / f"{ex.name}-{ex.version}-{chat.model.replace('/', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    sessions = sorted((int(_SESSION.match(k).group(1)), k) for k in conv if _SESSION.match(k))
    def one(n, key, me, other, known):
        slug = re.sub(r"\W+", "_", me)
        path = d / f"{sample['sample_id']}_s{n}_{slug}.json"
        if path.exists():
            return [Fact(**f) for f in json.loads(path.read_text())]
        turns = conv[key]
        msgs = [{"role": "user" if t["speaker"] == me else "assistant",
                 "content": t.get("text", "") + (f" [shares a photo: {t['blip_caption']}]"
                                                 if t.get("blip_caption") else "")}
                for t in turns]
        errors = []
        facts = ex.extract_session(msgs, parse_date(conv[f"{key}_date_time"]),
                                   context={"on_error": errors.append, "user_name": me,
                                            "known_keys": known})
        for f in facts:
            f.metadata["statement"] = _named(f.metadata["statement"], me, other)
            turn = f.metadata.get("turn")
            # The cited turn's dia_id, which ingest_facts stores as `source`:
            # what evidence recall is computed from.
            f.metadata["session_id"] = turns[turn]["dia_id"] if turn is not None else f"D{n}"
        if errors:
            print(f"[locomo] extraction {path.name}: {errors[0][:200]}", file=sys.stderr)
        else:
            path.write_text(json.dumps([f.__dict__ for f in facts]))
        return facts

    def speaker(me):
        other, known, out = (b if me == a else a), set(), {}
        for n, key in sessions:
            out[n] = one(n, key, me, other, sorted(known))
            known |= {fact_key(f) for f in out[n] if replaces(f)}
        return out

    with ThreadPoolExecutor(2) as pool:
        per = dict(zip((a, b), pool.map(speaker, (a, b))))
    return [(me, per[me][n]) for n, _ in sessions for me in (a, b)]


def _fact_vectors(sample: dict, texts: list[str], spent: Counter) -> dict:
    """sha1(text) -> vector for every fact statement, from cache or gemini."""
    h = lambda t: hashlib.sha1(t.encode()).hexdigest()       # noqa: E731
    path = CACHE / "emb" / EMB_MODEL / f"{sample['sample_id']}_facts.npz"
    cache = dict(np.load(path)) if path.exists() else {}
    missing = sorted({t for t in texts if h(t) not in cache})
    if missing:
        from .providers_extra import GeminiEmbedder
        vecs = GeminiEmbedder(EMB_MODEL).embed_batch(missing).astype(np.float32)
        cache.update({h(t): v for t, v in zip(missing, vecs)})
        np.savez(path, **cache)
        spent["embed_tokens"] += sum(len(t) for t in missing) // 4
        print(f"[locomo] embedded {sample['sample_id']}: {len(missing)} facts", file=sys.stderr)
    return cache


def retrieve_facts(sample: dict, qs: list[dict], qvecs: np.ndarray, k: int,
                   chat: "_Chat", spent: Counter, workers: int) -> tuple[list, dict]:
    """Per question, its top-k facts as (dia_id, valid_at, statement); and
    counts of what ingest_facts kept."""
    from feather_db.integrations.agent_memory import AgentMemory
    from feather_db.integrations.langgraph_store import FeatherStore
    from feather_db.pocket import display_text

    per = extract_facts(sample, chat, workers)
    texts = [f.metadata["statement"] for _, facts in per for f in facts]
    vec = _fact_vectors(sample, texts, spent)
    embed = lambda t: vec[hashlib.sha1(t.encode()).hexdigest()]   # noqa: E731
    counts = Counter(extracted=len(texts))
    with tempfile.TemporaryDirectory() as d:
        store = FeatherStore(os.path.join(d, "facts.feather"), dim=qvecs.shape[1],
                             embed=embed, index_fields=["text"], auto_save=False)
        try:
            mems = {}
            for me, facts in per:
                mem = mems.get(me) or mems.setdefault(
                    me, AgentMemory(store, sample["sample_id"], user=re.sub(r"\W+", "_", me)))
                counts.update(mem.ingest_facts(facts))
            counts["records"] = sum(1 for _ in store.scan((sample["sample_id"],), prefix=True))
            out = []
            for j, q in enumerate(qs):
                hits = []
                for h in store.db.hybrid_search(qvecs[j], q["question"], k=k):
                    m = h.metadata
                    if m.source != "langgraph":
                        continue
                    value = json.loads(m.get_attribute("_lg_value") or "{}")
                    hits.append((value.get("source") or "", m.valid_time, display_text(m)))
                out.append(hits)
            return out, dict(counts)
        finally:
            store.db.close(save=False)


def render_facts(hits: list) -> str:
    return "\n".join(f"- {text}" for _, _, text in sorted(hits, key=lambda h: h[1]))


# ── facts as index, turns as payload (--memory index) ────────────────────
#
# The same extraction as --memory facts, but the reader never sees a fact:
# each one is an extra index entry for the raw turn it cites (`retrieve`'s
# `index`), so a question can reach a turn through either its own words or
# what was extracted from it. With --supersede, facts the extractor marks
# one-value (`replaces`, extractor 0.3.0) that share a key (speaker +
# fact_key: subject|attribute) and disagree form a supersession chain, and
# an older turn in the context carries the chain's newest statement under it
# ("↳ later, session N: ..."): history, not a verdict.
#
# Measured with extractor 0.2.0 (no `replaces`, keys named per call): 117
# keys disagreed, mostly additive ("hobbies", "preference", "plan"); ungated
# notes reached 928 of 1540 questions ("When did Melanie paint a sunrise?"
# got "later: Melanie loves camping trips"), and gating them with
# ContradictionResolver (definite + supersedes) still left "continue my
# education" superseded by "plans to adopt" (J 82.66 vs 82.40, noise).

def fact_index(sample: dict, turns: list[dict], chat: "_Chat", spent: Counter,
               workers: int, supersede: bool = False) -> tuple[list, dict, Counter]:
    """([(turn index, statement, vector)], notes, counts). `notes` (with
    `supersede`): turn index -> [(turn index of the newest statement of a key,
    that statement)] for every turn holding an older, different value of a
    one-value key."""
    from feather_db.extractors.session_facts import fact_key, replaces
    dia = {t["dia_id"]: i for i, t in enumerate(turns)}
    per = extract_facts(sample, chat, workers)
    entries, chains = [], defaultdict(list)
    counts = Counter()
    for me, facts in per:
        for f in facts:
            ti = dia.get(f.metadata.get("session_id"))
            if ti is None:                 # no cited turn: nothing to point at
                counts["uncited"] += 1
                continue
            entries.append((ti, f.metadata["statement"]))
            if replaces(f):
                chains[(me, fact_key(f))].append((f.valid_at or 0, ti, f))
    vec = _fact_vectors(sample, [s for _, s in entries], spent)
    index = [(ti, s, vec[hashlib.sha1(s.encode()).hexdigest()]) for ti, s in entries]
    notes = defaultdict(list)
    for chain in (chains.values() if supersede else ()):
        chain.sort(key=lambda c: (c[0], c[1]))
        _, newest_ti, newest = chain[-1]
        cur = newest.object.strip().lower()
        for _, ti, f in chain[:-1]:
            note = (newest_ti, newest.metadata["statement"])
            if ti != newest_ti and f.object.strip().lower() != cur and note not in notes[ti]:
                notes[ti].append(note)
    counts.update(index_entries=len(index), keys=len(chains),
                  superseded_keys=sum(1 for c in chains.values()
                                      if len({f.object.strip().lower() for *_, f in c}) > 1),
                  turns_with_notes=len(notes))
    return index, dict(notes), counts


# ── models ───────────────────────────────────────────────────────────────

class _Chat:
    """complete(messages, max_tokens, temperature, json_mode) -> text, with
    token usage counted. OpenAI and OpenRouter go through the OpenAI SDK
    directly: feather_db's OpenAIProvider forces JSON mode, and neither it nor
    bench's OpenRouterProvider returns usage or takes response_format.
    OpenRouter also reports what each call cost (`usage.cost`), summed here.
    Anything else goes through bench.judges_llm's providers, tokens estimated
    (chars / 4)."""

    def __init__(self, spec: str, role: str):
        self.name, _, self.model = spec.partition(":")
        self.role = role
        self.lock = threading.Lock()
        self.tokens = Counter()
        self.cost = 0.0                  # OpenRouter's own figure, USD
        self.served_by = Counter()
        if self.name in ("openai", "openrouter"):
            from openai import OpenAI
            if self.name == "openai":
                self.model = self.model or "gpt-4o-mini"
                self._client = OpenAI(max_retries=0, timeout=60)
            else:
                from .providers_extra import OPENROUTER_BASE
                self.model = self.model or "openai/gpt-4o-mini"
                self._client = OpenAI(base_url=OPENROUTER_BASE, max_retries=0, timeout=60,
                                      api_key=os.environ.get("OPENROUTER_API_KEY"))
        else:
            from .judges_llm import _provider_from_name
            self._provider = _provider_from_name(self.name, self.model or None)

    def __call__(self, messages: list[dict], *, max_tokens: Optional[int] = None,
                 temperature: float = 0.0, json_mode: bool = False) -> str:
        from .judges_llm import _retry
        if self.name in ("openai", "openrouter"):
            kw = {"model": self.model, "messages": messages, "temperature": temperature}
            if max_tokens:
                kw["max_tokens"] = max_tokens
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            if self.name == "openrouter":
                kw["extra_body"] = {"usage": {"include": True}}
            r = _retry(lambda: self._client.chat.completions.create(**kw), label=self.role)
            if not r.choices:
                raise RuntimeError(f"{self.role}: empty reply")
            with self.lock:
                if r.usage:
                    self.tokens["in"] += r.usage.prompt_tokens
                    self.tokens["out"] += r.usage.completion_tokens
                    self.cost += float(getattr(r.usage, "cost", 0) or 0)
                self.served_by[getattr(r, "provider", None) or self.name] += 1
            return (r.choices[0].message.content or "").strip()
        text = _retry(lambda: self._provider.complete(
            messages, max_tokens=max_tokens or 512, temperature=temperature),
            label=self.role)
        with self.lock:
            self.tokens["in"] += sum(len(m["content"]) for m in messages) // 4
            self.tokens["out"] += len(text or "") // 4
            self.tokens["estimated"] = 1
        return (text or "").strip()

    def usd(self) -> Optional[float]:
        """What the calls cost: OpenRouter's figure when it gave one, else
        tokens x list price."""
        if self.cost:
            return self.cost
        p = PRICES.get(self.model.split("/")[-1])
        if p is None:
            return None
        return (self.tokens["in"] * p[0] + self.tokens["out"] * p[1]) / 1e6


def judge(chat: _Chat, question: str, gold: str, response: str) -> tuple[int, str, str]:
    """(score, label, raw). Mem0's evaluate_llm_judge: one user message, JSON
    mode, temperature 0; 1 iff the label is exactly "CORRECT". A reply that
    does not parse scores 0 (Mem0's script would raise) and is counted."""
    raw = chat([{"role": "user", "content": JUDGE_PROMPT.format(
        question=question, gold_answer=gold, generated_answer=response)}],
        temperature=0.0, json_mode=True)
    try:
        label = str(json.loads(raw)["label"])
    except (ValueError, KeyError, TypeError):
        m = re.search(r'"label"\s*:\s*"(\w+)"', raw)
        label = m.group(1) if m else "UNPARSED"
    return int(label == "CORRECT"), label, raw


def judge_strict(chat: _Chat, category: int, question: str, gold: str,
                 response: str) -> tuple[int, str]:
    """(score, raw). LongMemEval's official prompt for the category's kind of
    question: one user message, temperature 0, 1 iff 'yes' is in the reply."""
    from .lme_eval import official_prompt
    qtype = "temporal-reasoning" if int(category) == 2 else "single-session-user"
    raw = chat([{"role": "user", "content": official_prompt(qtype, question, gold, response)}],
               max_tokens=10, temperature=0.0)
    return int("yes" in (raw or "").lower()), raw


class _JudgeCache:
    """Judgments by (judge, model, question, gold, response), appended to a
    file: a rerun of rejudge, rekey or leniency pays only for what is new."""

    def __init__(self, name: str):
        self.path = CACHE / name
        self.lock = threading.Lock()
        self.data = {}
        if self.path.exists():
            for line in open(self.path):
                if line.strip():
                    c = json.loads(line)
                    self.data[c["k"]] = c

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps([str(p) for p in parts]).encode()).hexdigest()

    def get_or(self, k: str, compute):
        if k not in self.data:
            score, raw = compute()
            with self.lock:
                self.data[k] = {"k": k, "score": score, "raw": raw}
                CACHE.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as fh:
                    fh.write(json.dumps(self.data[k]) + "\n")
        return self.data[k]


def _verdict(judge_name: str, chat: _Chat, cache: _JudgeCache, spec: str, category,
             question: str, gold: str, response: str) -> dict:
    k = cache.key(judge_name, spec, question, gold, response)
    if judge_name == "strict":
        return cache.get_or(k, lambda: judge_strict(chat, category, question, gold, response))
    return cache.get_or(k, lambda: (lambda s, _l, raw: (s, raw))(
        *judge(chat, question, gold, response)))


# ── Mem0's lexical metrics ───────────────────────────────────────────────

def _simple_tokenize(text: str) -> list[str]:
    return str(text).lower().replace(".", " ").replace(",", " ").replace(
        "!", " ").replace("?", " ").split()


def f1(prediction: str, reference: str) -> float:
    """Mem0 metrics/utils.py calculate_metrics()['f1']: over token SETS."""
    prediction, reference = str(prediction).strip(), str(reference).strip()
    if not prediction or not reference:
        return 0.0
    p, r = set(_simple_tokenize(prediction)), set(_simple_tokenize(reference))
    common = p & r
    if not p or not r or not common:
        return 0.0
    prec, rec = len(common) / len(p), len(common) / len(r)
    return 2 * prec * rec / (prec + rec)


_NLTK = None


def _nltk():
    """nltk with punkt_tab, or False. Mem0's utils.py downloads its data on
    import; this does the same, once, only if a tokenize fails for want of it."""
    global _NLTK
    if _NLTK is None:
        try:
            import nltk
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
            try:
                nltk.word_tokenize("a b.")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                nltk.word_tokenize("a b.")
            _NLTK = (nltk.word_tokenize, sentence_bleu, SmoothingFunction().method1,
                     f"nltk {nltk.__version__}")
        except Exception:
            _NLTK = False
    return _NLTK


def bleu1_impl() -> str:
    n = _nltk()
    return n[3] if n else "approximate (nltk missing: regex tokenizer, same BLEU-1 formula)"


def bleu1(prediction: str, reference: str) -> float:
    """Mem0 calculate_bleu_scores()['bleu1']."""
    n = _nltk()
    if n:
        tok, sentence_bleu, smooth, _ = n
        try:
            return float(sentence_bleu([tok(str(reference).lower())],
                                       tok(str(prediction).lower()),
                                       weights=(1, 0, 0, 0), smoothing_function=smooth))
        except Exception:
            return 0.0
    # nltk's sentence_bleu for weights (1,0,0,0) and smoothing method1:
    # clipped unigram precision (epsilon 0.1 when nothing matches) x brevity.
    tok = lambda s: re.findall(r"\w+|[^\w\s]", str(s).lower())   # noqa: E731
    hyp, ref = tok(prediction), tok(reference)
    if not hyp:
        return 0.0
    rc = Counter(ref)
    match = sum(min(c, rc[w]) for w, c in Counter(hyp).items())
    p1 = (match or 0.1) / len(hyp)
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return bp * p1


# ── run ──────────────────────────────────────────────────────────────────

def _select(data: list[dict], spec: Optional[str]) -> list[dict]:
    if not spec or spec == "all":
        return data
    want = [s.strip() for s in spec.split(",") if s.strip()]
    by_id = {s["sample_id"]: s for s in data}
    return [by_id[w] if w in by_id else data[int(w)] for w in want]


def run(conversations: Optional[str] = None, *, k: int = K, reader: str = READER,
        judge_spec: str = JUDGE, workers: int = 12,
        out_dir: Optional[Path] = None, memory: str = "turns",
        extractor: str = READER, categories: Optional[set] = None,
        supersede: bool = False, reuse: Optional[str] = None,
        split: str = "dev", repeats: int = 1, select: str = "rank") -> dict:
    from .runner import release_unused_claims
    try:
        return _run(conversations, k=k, reader=reader, judge_spec=judge_spec,
                    workers=workers, out_dir=out_dir, memory=memory, extractor=extractor,
                    categories=categories, supersede=supersede, reuse=reuse,
                    split=split, repeats=repeats, select=select)
    finally:
        release_unused_claims()


def _run(conversations, *, k, reader, judge_spec, workers, out_dir, memory, extractor,
         categories, supersede, reuse, split, repeats, select="rank") -> dict:
    from .runner import RESULTS_DIR, _git_commit, _git_dirty
    from .datasets.locomo_split import dev_conversations
    from .protocol import require_committed
    import feather_db

    if supersede and memory != "index":
        raise SystemExit("--supersede needs --memory index")
    if select != "rank" and memory == "facts":
        raise SystemExit("--select chooses among raw turns: --memory turns or index")
    if repeats > 1 and reuse:
        raise SystemExit("--repeats measures rerun noise; --reuse copies answers "
                         "instead of asking again")
    t0 = time.time()
    data = load()
    dev = dev_conversations()
    if conversations:
        samples = _select(data, conversations)
    elif split in ("dev", "test"):
        samples = [s for s in data if (s["sample_id"] in dev) == (split == "dev")]
    elif split == "all":
        samples = data
    else:
        raise SystemExit(f"split must be dev, test or all, not {split!r}")
    held_out = [s["sample_id"] for s in samples if s["sample_id"] not in dev]
    if held_out:
        require_committed(f"LoCoMo on {', '.join(held_out)}")
    total_q = sum(1 for s in data for q in questions_of(s)
                  if not categories or q["category"] in categories)
    spent = Counter()
    reader_chat, judge_chat = _Chat(reader, "reader"), _Chat(judge_spec, "judge")
    extract_chat = _Chat(extractor, "extractor") if memory in ("facts", "index") else None
    kept = Counter()
    # --reuse: a prior run's record stands for a question whose context is
    # byte-identical here. Only the context is compared: the prior run must
    # have used the same reader, judge and system prompt.
    prior = {}
    if reuse:
        for line in open(reuse):
            if line.strip():
                r = json.loads(line)
                prior[(r["sample_id"], r["qa_index"])] = r

    tasks = []
    for s in samples:
        turns, qs = turns_of(s), questions_of(s)
        vecs = vectors(s, turns, qs, spent)     # all questions: the cache key
        if categories:
            keep = [j for j, q in enumerate(qs) if q["category"] in categories]
            qs = [qs[j] for j in keep]
            vecs = np.concatenate([vecs[:len(turns)], vecs[len(turns):][keep]])
        a, b = s["conversation"]["speaker_a"], s["conversation"]["speaker_b"]
        dia_index = {dia_key(t["dia_id"]): i for i, t in enumerate(turns)}
        if memory == "facts":
            fact_hits, counts = retrieve_facts(s, qs, vecs[len(turns):], k,
                                               extract_chat, spent, workers)
            kept.update(counts)
            for q, hits in zip(qs, fact_hits):
                ev = evidence_ids(q)
                got = {dia_key(src) for src, _, _ in hits} - {None}
                tasks.append({
                    "sample_id": s["sample_id"], "qa_index": q["qa_index"],
                    "category": q["category"], "category_name": CATEGORIES[q["category"]][0],
                    "question": q["question"], "gold": str(q["answer"]),
                    "evidence": sorted(f"D{x}:{y}" for x, y in ev),
                    "evidence_missing_from_data": sum(1 for e in ev if e not in dia_index),
                    "retrieved": [src for src, _, _ in hits],
                    "evidence_recall": (len(ev & got) / len(ev)) if ev else None,
                    "_system": FACTS_SYSTEM.format(a=a, b=b),
                    "_user": FACTS_USER,
                    "_context": render_facts(hits),
                })
            continue
        index, notes = None, None
        if memory == "index":
            index, notes, counts = fact_index(s, turns, extract_chat, spent, workers,
                                              supersede=supersede)
            kept.update(counts)
            if not supersede:
                notes = None
        hits = retrieve(s, turns, qs, vecs, k, index=index, select=select)
        for q, idx in zip(qs, hits):
            ev = evidence_ids(q)
            got = {dia_key(turns[i]["dia_id"]) for i in idx}
            tasks.append({
                "sample_id": s["sample_id"], "qa_index": q["qa_index"],
                "category": q["category"], "category_name": CATEGORIES[q["category"]][0],
                "question": q["question"], "gold": str(q["answer"]),
                "evidence": sorted(f"D{x}:{y}" for x, y in ev),
                "evidence_missing_from_data": sum(1 for e in ev if e not in dia_index),
                "retrieved": [turns[i]["dia_id"] for i in idx],
                "evidence_recall": (len(ev & got) / len(ev)) if ev else None,
                "_system": ANSWER_SYSTEM.format(a=a, b=b),
                "_user": ANSWER_USER,
                "_context": render(turns, idx, notes),
                **({"notes_shown": sum(1 for i in set(idx) for later, _ in notes.get(i, ())
                                       if later not in set(idx))} if notes is not None else {}),
            })

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if memory == "turns" else f"_{memory}"
    tag += "_supersede" if supersede else ""
    from .runner import reserve
    summaries = []
    for rep in range(1, repeats + 1):
        if rep > 1:     # fresh accounting, so each repeat's cost is its own
            reader_chat, judge_chat = _Chat(reader, "reader"), _Chat(judge_spec, "judge")
        rtag = f"_r{rep}" if repeats > 1 else ""
        qpath = reserve(f"locomo__locomo10{tag}__{stamp}{rtag}", "__questions.jsonl", out_dir)
        summaries.append(_answer_and_judge(
            tasks, qpath, prior, reuse, reader_chat, judge_chat, extract_chat, workers,
            spent, total_q, t0, samples, k, memory, categories, extractor, kept, supersede,
            reader, judge_spec, split, rep, repeats, select))
    if repeats > 1:
        from .stats import rerun_noise
        runs = []
        for sm in summaries:
            recs = [json.loads(line) for line in open(sm["questions_file"]) if line.strip()]
            runs.append({qid(r): float(r["llm_score"]) for r in recs})
        noise = rerun_noise(runs)
        js = [sm["metrics"]["overall"]["J"] for sm in summaries]
        print(f"\n{repeats} repeats: J {', '.join(f'{j:.2f}' for j in js)} "
              f"(mean {sum(js) / len(js):.2f}, sd {100 * (noise['accuracy_sd'] or 0):.2f}); "
              f"two runs disagree on {100 * noise['flip_rate']:.2f}% of questions")
        rpath = out_dir / f"locomo__locomo10{tag}__{stamp}_repeats.json"
        rpath.write_text(json.dumps({"repeats": repeats, "J": js, "rerun_noise": noise,
                                     "questions_files": [sm["questions_file"]
                                                         for sm in summaries]}, indent=2))
        print(f"wrote {rpath}", file=sys.stderr)
    return summaries[-1]


def qid(rec: dict) -> str:
    """A LoCoMo question's id across runs: conversation and qa index."""
    return f"{rec['sample_id']}#{rec['qa_index']}"


def _answer_and_judge(tasks, qpath, prior, reuse, reader_chat, judge_chat, extract_chat,
                      workers, spent, total_q, t0, samples, k, memory, categories,
                      extractor, kept, supersede, reader, judge_spec, split, rep,
                      repeats, select="rank") -> dict:
    from .runner import _git_commit, _git_dirty
    import feather_db
    lock = threading.Lock()
    reused = Counter()

    def work(task: dict) -> dict:
        rec = {k_: v for k_, v in task.items() if not k_.startswith("_")}
        rec["context_tokens_est"] = len(task["_context"]) // 4
        old = prior.get((task["sample_id"], task["qa_index"]))
        if old is not None and old.get("context") == task["_context"] \
                and "reader_error" not in old and old.get("judge_label") not in ("ERROR", "UNPARSED"):
            for k_ in ("response", "llm_score", "judge_label", "judge_raw", "f1", "bleu1"):
                rec[k_] = old.get(k_)
            rec["context"], rec["reused_from"] = task["_context"], reuse
            with lock:
                reused["n"] += 1
                with open(qpath, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            return rec
        try:
            rec["response"] = reader_chat(
                [{"role": "system", "content": task["_system"]},
                 {"role": "user", "content": task["_user"].format(
                     context=task["_context"], question=task["question"])}],
                max_tokens=100, temperature=0.0)
        except Exception as e:
            rec["response"], rec["reader_error"] = "", str(e)
        try:
            rec["llm_score"], rec["judge_label"], rec["judge_raw"] = judge(
                judge_chat, rec["question"], rec["gold"], rec["response"])
        except Exception as e:
            rec["llm_score"], rec["judge_label"], rec["judge_error"] = 0, "ERROR", str(e)
        rec["f1"] = f1(rec["response"], rec["gold"])
        rec["bleu1"] = bleu1(rec["response"], rec["gold"])
        rec["context"] = task["_context"]
        with lock, open(qpath, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    print(f"[locomo] {len(tasks)} questions from {len(samples)} conversation(s), "
          f"k={k}, reader={reader}, judge={judge_spec}", file=sys.stderr)
    records, done = [], 0
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(work, t) for t in tasks]):
            records.append(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"[locomo] {done}/{len(tasks)}", file=sys.stderr)
    records.sort(key=lambda r: (r["sample_id"], r["qa_index"]))

    cost = {
        "reader_tokens": dict(reader_chat.tokens), "judge_tokens": dict(judge_chat.tokens),
        "embed_tokens_new": spent["embed_tokens"],
        "reader_usd": reader_chat.usd(), "judge_usd": judge_chat.usd(),
        "embed_usd": spent["embed_tokens"] * PRICES[EMB_MODEL][0] / 1e6,
        "prices_per_1m": PRICES,
        "cost_source": {"reader": "openrouter usage.cost" if reader_chat.cost else "list price",
                        "judge": "openrouter usage.cost" if judge_chat.cost else "list price",
                        "embed": "list price x chars/4"},
        "served_by": {"reader": dict(reader_chat.served_by),
                      "judge": dict(judge_chat.served_by)},
    }
    if extract_chat is not None:
        cost["extractor_tokens"] = dict(extract_chat.tokens)
        cost["extractor_usd"] = extract_chat.usd()
        cost["extractor_note"] = "0 when every extraction came from the cache"
    paid = [c for c in (cost["reader_usd"], cost["judge_usd"], cost.get("extractor_usd"))
            if c is not None]
    cost["usd"] = round(sum(paid) + cost["embed_usd"], 4)
    llm_usd = sum(paid)
    cost["projected_all_10_llm_usd"] = (round((llm_usd - (cost.get("extractor_usd") or 0))
                                              / len(records) * total_q, 3)
                                        if records else None)

    summary = {
        "scenario": "locomo", "dataset": "locomo10", "n": len(records), "dim": 3072,
        "feather_version": getattr(feather_db, "__version__", "unknown"),
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "platform": f"{platform.system()} {platform.machine()}",
        "commit": _git_commit(), "dirty": _git_dirty(),
        "argv": sys.argv, "started_at": t0, "wall_seconds": round(time.time() - t0, 1),
        "params": {
            "conversations": [s["sample_id"] for s in samples], "k": k,
            "split": split, "repeat": rep, "repeats": repeats,
            "memory": memory,
            "select": select,
            "categories": sorted(categories) if categories else [1, 2, 3, 4],
            "retrieval": "hybrid_search (BM25 + dense, RRF), no scoring"
                         + (" over turns + extracted facts; a fact hit counts for "
                            "the turn it cites; k distinct turns" if memory == "index" else ""),
            **({"extractor": extractor, "extracted": dict(kept),
                "facts_system": FACTS_SYSTEM} if memory == "facts" else {}),
            **({"extractor": extractor, "index": dict(kept), "supersede": supersede,
                "supersede_gate": "extractor replaces (one_value)",
                "note_format": "  ↳ later, session N: <newest statement of the key>"}
               if memory == "index" else {}),
            **({"reused_from": reuse, "reused": reused["n"]} if reuse else {}),
            "embedder": EMB_MODEL, "reader": reader, "judge": judge_spec,
            "judge_source": JUDGE_SOURCE, "data_url": DATA_URL, "data_sha256": DATA_SHA256,
            "answer_system": ANSWER_SYSTEM, "answer_user": ANSWER_USER,
            "bleu1_impl": bleu1_impl(),
            "category_mapping": {str(c): {"name": n, "mem0_paper_column": col}
                                 for c, (n, col) in CATEGORIES.items()},
        },
        "metrics": metrics(records),
        "cost": cost,
        "questions_file": str(qpath),
    }
    spath = qpath.with_name(qpath.name.replace("__questions.jsonl", ".json"))
    spath.write_text(json.dumps(summary, indent=2))
    print(table(summary))
    print(f"\nwrote {qpath}\nwrote {spath}", file=sys.stderr)
    return summary


def metrics(records: list[dict]) -> dict:
    def agg(rs):
        ev = [r["evidence_recall"] for r in rs if r.get("evidence_recall") is not None]
        return {"n": len(rs),
                "J": round(100 * sum(r["llm_score"] for r in rs) / len(rs), 2) if rs else None,
                "F1": round(100 * sum(r["f1"] for r in rs) / len(rs), 2) if rs else None,
                "BLEU1": round(100 * sum(r["bleu1"] for r in rs) / len(rs), 2) if rs else None,
                "evidence_recall": round(sum(ev) / len(ev), 4) if ev else None,
                "all_evidence_retrieved": round(sum(e == 1.0 for e in ev) / len(ev), 4)
                if ev else None}
    by = defaultdict(list)
    for r in records:
        by[r["category"]].append(r)
    out = {"overall": agg(records),
           "by_category": {str(c): {"name": CATEGORIES[c][0],
                                    "mem0_paper_column": CATEGORIES[c][1], **agg(by[c])}
                           for c in sorted(by)},
           "judge_unparsed": sum(r.get("judge_label") in ("UNPARSED", "ERROR") for r in records),
           "reader_errors": sum("reader_error" in r for r in records),
           "context_tokens_est_mean": round(sum(r.get("context_tokens_est", 0)
                                                for r in records) / max(1, len(records)))}
    return out


def table(summary: dict) -> str:
    m = summary["metrics"]
    rows = ["| category | name (from the data) | Mem0 paper column | n | J | F1 | BLEU-1 "
            "| evidence recall@k |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for c, r in m["by_category"].items():
        rows.append(f"| {c} | {r['name']} | {r['mem0_paper_column']} | {r['n']} | {r['J']:.2f} "
                    f"| {r['F1']:.2f} | {r['BLEU1']:.2f} | {r['evidence_recall']:.3f} |")
    o = m["overall"]
    rows.append(f"| 1-4 | overall | | {o['n']} | **{o['J']:.2f}** | {o['F1']:.2f} "
                f"| {o['BLEU1']:.2f} | {o['evidence_recall']:.3f} |")
    c = summary.get("cost", {})
    p = summary.get("params", {})
    rows += ["", f"k={p.get('k')}  reader={p.get('reader')}  judge={p.get('judge')}  "
                 f"BLEU-1: {p.get('bleu1_impl')}  context ~{m['context_tokens_est_mean']} tokens",
             f"judge unparsed/errors: {m['judge_unparsed']}  reader errors: {m['reader_errors']}"]
    if c:
        rows.append(f"cost: ${c['usd']:.4f} (reader {c['reader_tokens']}, judge {c['judge_tokens']}, "
                    f"new embedding tokens {c['embed_tokens_new']}); all 10 conversations, "
                    f"reader+judge: ~${c['projected_all_10_llm_usd']}")
    return "\n".join(rows)


def read_records(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def in_split(records: list[dict], split: str) -> list[dict]:
    if split == "all":
        return records
    from .datasets.locomo_split import dev_conversations
    dev = dev_conversations()
    return [r for r in records if (r["sample_id"] in dev) == (split == "dev")]


def scored(records: list[dict], key: str = "original",
           corrections: Optional[dict] = None, judge: str = "mem0") -> list[dict]:
    """The records as answer key `key` and judge `judge` score them.
    original: as judged. sound: without the 99 questions whose published
    answer is wrong. audited: those 99 re-judged against the audit's
    corrected answer (`rekey` for mem0, `rejudge` for strict, first)."""
    if key not in KEYS:
        raise SystemExit(f"key must be one of {', '.join(KEYS)}, not {key!r}")
    if judge not in JUDGES:
        raise SystemExit(f"judge must be one of {', '.join(JUDGES)}, not {judge!r}")
    field, field_audited = SCORE_FIELDS[judge]
    if judge != "mem0":
        if any(field not in r for r in records):
            raise SystemExit(f"--judge {judge}: this file has no {field}; run "
                             "`python -m bench.locomo rejudge FILE` and use its output")
        records = [{**r, "llm_score": r[field],
                    **({"llm_score_audited": r[field_audited]} if field_audited in r else {})}
                   for r in records]
    if key == "original":
        return records
    corrections = corrections if corrections is not None else audit_corrections()
    wrong_key = lambda r: (r["sample_id"], int(r["qa_index"])) in corrections   # noqa: E731
    if key == "sound":
        return [r for r in records if not wrong_key(r)]
    out = []
    for r in records:
        if wrong_key(r):
            if "llm_score_audited" not in r:
                raise SystemExit("--key audited: this file has no re-judged answers; run "
                                 "`python -m bench.locomo rekey FILE` and use its output")
            r = {**r, "gold": r["gold_audited"], "llm_score": r["llm_score_audited"],
                 "f1": f1(r["response"], r["gold_audited"]),
                 "bleu1": bleu1(r["response"], r["gold_audited"])}
        out.append(r)
    return out


def rescore(path: str, key: str = "original", split: str = "all",
            judge_name: str = "mem0") -> dict:
    """Recompute the tables from a questions JSONL under answer key `key`,
    judge `judge_name`, on `split`. Free: it reads judgments already made."""
    records = read_records(path)
    for r in records:
        r["f1"], r["bleu1"] = f1(r["response"], r["gold"]), bleu1(r["response"], r["gold"])
    records = in_split(scored(records, key, judge=judge_name), split)
    summary = {"metrics": metrics(records), "params": {"bleu1_impl": bleu1_impl()}}
    print(f"answer key: {key}   judge: {judge_name}   split: {split}   "
          f"questions: {len(records)}")
    print(table(summary))
    return summary


def rekey(path: str, judge_spec: str = JUDGE, workers: int = 12) -> str:
    """Re-judge a run's answers to the 99 wrong-key questions against the
    audit's corrected answers, with the run's judge (Mem0's prompt). Writes
    <run>_audited__questions.jsonl, every record kept, those 99 carrying
    `gold_audited`, `audit_error`, `llm_score_audited`. Judgments are cached
    by (question, corrected answer, response, judge), so a rerun is free."""
    records = read_records(path)
    corr = audit_corrections()
    chat = _Chat(judge_spec, "judge")
    cache_path = CACHE / "audit_judgments.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            if line.strip():
                c = json.loads(line)
                cache[c["k"]] = c
    lock = threading.Lock()

    def ck(r, gold):
        return hashlib.sha256(json.dumps([r["question"], gold, r["response"], judge_spec])
                              .encode()).hexdigest()

    def one(r):
        c = corr[(r["sample_id"], int(r["qa_index"]))]
        k_ = ck(r, c["correct_answer"])
        if k_ not in cache:
            score, label, raw = judge(chat, r["question"], c["correct_answer"], r["response"])
            with lock:
                cache[k_] = {"k": k_, "score": score, "label": label, "raw": raw}
                with open(cache_path, "a") as fh:
                    fh.write(json.dumps(cache[k_]) + "\n")
        return r, c, cache[k_]

    todo = [r for r in records if (r["sample_id"], int(r["qa_index"])) in corr]
    with ThreadPoolExecutor(workers) as ex:
        for r, c, j in ex.map(one, todo):
            r.update({"gold_audited": c["correct_answer"], "audit_error": c["error_type"],
                      "llm_score_audited": j["score"], "judge_label_audited": j["label"]})
    out = path.replace("__questions.jsonl", "_audited__questions.jsonl")
    if out == path:
        out = path + ".audited.jsonl"
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    for key in KEYS:
        m = metrics(scored(records, key, corr))["overall"]
        print(f"{key:9s} n={m['n']:<5d} J {m['J']:.2f}")
    print(f"wrote {out}", file=sys.stderr)
    return out


def rejudge(path: str, judge_spec: str = JUDGE, workers: int = 12) -> str:
    """Judge a run's answers again with the strict judge: every answer
    against the published key (`strict_score`), and the 99 wrong-key
    questions against the corrected answers too (`strict_score_audited`).
    Writes <run>_strict__questions.jsonl, every record and field kept.
    Cached, so a rerun is free. Prints mem0 against strict."""
    records = read_records(path)
    corr = audit_corrections()
    chat = _Chat(judge_spec, "judge")
    cache = _JudgeCache("strict_judgments.jsonl")

    def one(r):
        out = {"strict_score": _verdict("strict", chat, cache, judge_spec, r["category"],
                                        r["question"], r["gold"], r["response"])["score"]}
        c = corr.get((r["sample_id"], int(r["qa_index"])))
        if c:
            out["strict_score_audited"] = _verdict(
                "strict", chat, cache, judge_spec, r["category"], r["question"],
                c["correct_answer"], r["response"])["score"]
        return r, out

    with ThreadPoolExecutor(workers) as ex:
        for r, out in ex.map(one, records):
            r.update(out)
    dst = path.replace("__questions.jsonl", "_strict__questions.jsonl")
    if dst == path:
        dst = path + ".strict.jsonl"
    with open(dst, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    agree = sum(int(r["llm_score"]) == int(r["strict_score"]) for r in records)
    lenient = sum(int(r["llm_score"]) == 1 and int(r["strict_score"]) == 0 for r in records)
    m0 = metrics(records)["overall"]["J"]
    st = metrics([{**r, "llm_score": r["strict_score"]} for r in records])["overall"]["J"]
    print(f"published key: mem0 judge J {m0:.2f}, strict judge {st:.2f}; they agree on "
          f"{agree}/{len(records)}; mem0 alone accepts {lenient}")
    print(f"wrote {dst}", file=sys.stderr)
    return dst


def load_leniency() -> list[dict]:
    """The audit's deliberately wrong answers: [{question_id, question,
    golden_answer, generated_answer, category}], pinned and checked."""
    path = CACHE / "audit_leniency.json"
    if not path.exists() or path.stat().st_size == 0:
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[locomo] downloading {LENIENCY_URL} -> {path}", file=sys.stderr)
        subprocess.check_call(["curl", "-fL", "--retry", "3", "-o", str(path), LENIENCY_URL])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LENIENCY_SHA256:
        raise SystemExit(f"[locomo] {path} is not the pinned file (sha256 differs)")
    return [x for recs in json.loads(raw)["detailed_results"].values() for x in recs]


def leniency(judge_spec: str = JUDGE, workers: int = 12, split: str = "all",
             judges: tuple = JUDGES) -> dict:
    """How often each judge accepts an answer known to be wrong: the audit's
    plausible wrong answers, on the questions whose key is sound (a wrong
    answer to a wrong key may be right). Lower is better; the audit measured
    62.8% for the judge prompt of the EverMemOS pipeline."""
    data = load()
    corr = audit_corrections(data)
    from .datasets.locomo_split import dev_conversations
    dev = dev_conversations()
    chat = _Chat(judge_spec, "judge")
    cache = _JudgeCache("leniency_judgments.jsonl")
    items = []
    for x in load_leniency():
        m = re.fullmatch(r"locomo_(\d+)_qa(\d+)", x["question_id"])
        ci, qi = int(m.group(1)), int(m.group(2))
        sid = data[ci]["sample_id"]
        if " ".join(str(data[ci]["qa"][qi]["question"]).split()) != " ".join(
                str(x["question"]).split()):
            raise SystemExit(f"[locomo] leniency record {x['question_id']} does not match")
        if (sid, qi) in corr:
            continue
        if split != "all" and (sid in dev) != (split == "dev"):
            continue
        items.append((f"{sid}#{qi}", int(x["category"]), x["question"],
                      str(x["golden_answer"]), str(x["generated_answer"])))
    from .stats import describe
    out = {"n": len(items), "split": split, "judge_model": judge_spec}
    for jn in judges:
        with ThreadPoolExecutor(workers) as ex:
            verdicts = list(ex.map(lambda it: _verdict(jn, chat, cache, judge_spec, it[1],
                                                       it[2], it[3], it[4])["score"], items))
        run = {it[0]: float(v) for it, v in zip(items, verdicts)}
        d = describe([run])
        by_cat = defaultdict(list)
        for it, v in zip(items, verdicts):
            by_cat[CATEGORIES[it[1]][0]].append(v)
        out[jn] = {"accepts_wrong": d["accuracy"], "ci95": d["ci95"],
                   "by_category": {c: round(sum(v) / len(v), 4) for c, v in by_cat.items()}}
        print(f"{jn:7s} accepts {100 * d['accuracy']:.1f}% of known-wrong answers "
              f"[{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}] (n={len(items)}) "
              + " ".join(f"{c} {100 * v:.0f}%" for c, v in out[jn]["by_category"].items()))
    return out


def categories() -> None:
    """Print what each category number holds, and which labels reproduce
    Mem0's overall J from its per-column numbers."""
    data = load()
    qs = [q for s in data for q in s["qa"]]
    print("| cat | n | has `answer` | evidence turns (mean) | evidence in >1 session "
          "| starts with 'When' | example |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for c in (1, 2, 3, 4, 5):
        cq = [q for q in qs if q["category"] == c]
        ev = [evidence_ids(q) for q in cq]
        multi = sum(len({s for s, _ in e}) > 1 for e in ev)
        when = sum(q["question"].lower().startswith("when") for q in cq)
        print(f"| {c} | {len(cq)} | {sum('answer' in q for q in cq)} "
              f"| {sum(map(len, ev)) / len(cq):.2f} | {100 * multi / len(cq):.0f}% "
              f"| {100 * when / len(cq):.0f}% | {cq[7]['question']} |")
    n = {c: sum(q["category"] == c for q in qs) for c in CATEGORIES}
    total = sum(n.values())
    print("\nMem0 paper: n-weighted mean of Table 1 columns vs its Table 2 overall J")
    print("| system | Table 2 | as labelled here (1 SH, 2 T, 3 MH, 4 OD) "
          "| as the LoCoMo paper orders them (1 SH, 2 MH, 3 T, 4 OD) |")
    print("|---|---:|---:|---:|")
    alt = {1: "Single Hop", 2: "Multi-Hop", 3: "Temporal", 4: "Open Domain"}
    for name, (cols, overall) in MEM0_TABLE1.items():
        here = sum(n[c] * cols[CATEGORIES[c][1]] for c in n) / total
        other = sum(n[c] * cols[alt[c]] for c in n) / total
        print(f"| {name} | {overall:.2f} | {here:.2f} | {other:.2f} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest, retrieve, answer, judge")
    r.add_argument("--conversations", default=None,
                   help="comma-separated sample_ids or indices (default: all 10)")
    r.add_argument("--k", type=int, default=K, help="turns retrieved per question")
    r.add_argument("--reader", default=READER, help="provider:model (bench.judges_llm names)")
    r.add_argument("--judge", default=JUDGE,
                   help="provider:model for Mem0's judge (openai or openrouter: JSON mode)")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--out-dir", default=None)
    r.add_argument("--memory", choices=("turns", "facts", "index"), default="turns",
                   help="retrieve raw turns (default), facts extracted at write "
                        "time (SessionFactExtractor -> AgentMemory.ingest_facts), or "
                        "raw turns found through those facts as well (index)")
    r.add_argument("--select", choices=("rank", "coverage"), default="rank",
                   help="turns/index: the top k (rank), or k of the top 2k by "
                        "information covered (feather_db.coverage)")
    r.add_argument("--supersede", action="store_true",
                   help="--memory index: under an older turn, the newest statement "
                        "of the same one-value extracted key (extractor `replaces`)")
    r.add_argument("--reuse", default=None,
                   help="questions JSONL of a prior run: copy its answer and judgment "
                        "for every question whose context is identical")
    r.add_argument("--categories", default=None,
                   help="comma-separated category numbers to run (default: 1-4)")
    r.add_argument("--extractor", default=READER,
                   help="provider:model for --memory facts extraction")
    r.add_argument("--split", choices=("dev", "test", "all"), default="dev",
                   help="the frozen conversation split (bench/datasets/locomo_split.json); "
                        "test and all need a committed tree")
    r.add_argument("--repeats", type=int, default=1,
                   help="answer and judge every question N times (3-5 to compare "
                        "configurations: same-context reruns flip ~3%% of questions)")
    s = sub.add_parser("rescore", help="recompute the tables from a questions JSONL")
    s.add_argument("path")
    s.add_argument("--key", choices=KEYS, default="original",
                   help="answer key: published (original), without the 99 wrong "
                        "answers (sound), or with them corrected (audited, after rekey)")
    s.add_argument("--split", choices=("dev", "test", "all"), default="all")
    s.add_argument("--judge", choices=JUDGES, default="mem0",
                   help="which judge's verdicts (strict: after rejudge)")
    rj = sub.add_parser("rejudge", help="judge a run again with the strict judge "
                                        "(paid, cached)")
    rj.add_argument("path")
    rj.add_argument("--judge", default=JUDGE)
    rj.add_argument("--workers", type=int, default=12)
    le = sub.add_parser("leniency", help="how often each judge accepts the audit's "
                                         "known-wrong answers (paid, cached)")
    le.add_argument("--judge", default=JUDGE)
    le.add_argument("--workers", type=int, default=12)
    le.add_argument("--split", choices=("dev", "test", "all"), default="all")
    rk = sub.add_parser("rekey", help="re-judge the 99 wrong-key questions against "
                                      "the audit's corrected answers (paid, cached)")
    rk.add_argument("path")
    rk.add_argument("--judge", default=JUDGE)
    rk.add_argument("--workers", type=int, default=12)
    sub.add_parser("categories", help="what each category number holds")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.conversations, k=a.k, reader=a.reader, judge_spec=a.judge,
            workers=a.workers, out_dir=a.out_dir, memory=a.memory,
            extractor=a.extractor, supersede=a.supersede, reuse=a.reuse,
            categories={int(c) for c in a.categories.split(",")} if a.categories else None,
            split=a.split, repeats=a.repeats, select=a.select)
    elif a.cmd == "rescore":
        rescore(a.path, a.key, a.split, a.judge)
    elif a.cmd == "rejudge":
        rejudge(a.path, a.judge, a.workers)
    elif a.cmd == "leniency":
        leniency(a.judge, a.workers, a.split)
    elif a.cmd == "rekey":
        rekey(a.path, a.judge, a.workers)
    else:
        categories()
    return 0


if __name__ == "__main__":
    sys.exit(main())
