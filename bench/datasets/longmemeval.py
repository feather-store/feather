"""LongMemEval loader (Xu et al. 2024 / ICLR 2025, arXiv:2410.10813).

Downloads from HuggingFace dataset xiaowu0162/longmemeval-cleaned and
caches to ~/.cache/feather/longmemeval/. Three variants:

    oracle       15 MB    only evidence sessions, no distractors
    s            277 MB   ~115K tokens / ~40 sessions per question
    m            2.7 GB   ~500 sessions per question, exceeds 128k context

Schema per question (from the official repo):
    question_id          : str
    question_type        : str  e.g. "single-session-user", "multi-session",
                                "temporal-reasoning", "knowledge-update",
                                "_abs" suffix on abstention questions
    question             : str
    answer               : str   gold reference answer
    question_date        : str   ISO timestamp the question was posed
    haystack_session_ids : list[str]
    haystack_dates       : list[str]
    haystack_sessions    : list[list[{"role": "user"|"assistant",
                                       "content": str,
                                       "has_answer"?: bool}]]
    answer_session_ids   : list[str]   evidence sessions
"""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from typing import Iterator


CACHE_ROOT = Path(os.environ.get("FEATHER_CACHE", Path.home() / ".cache" / "feather"))
LME_DIR = CACHE_ROOT / "longmemeval"

VARIANTS = {
    "oracle": "longmemeval_oracle.json",
    "s":      "longmemeval_s_cleaned.json",
    "m":      "longmemeval_m_cleaned.json",
}

HF_REPO_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"


def _ensure(variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; one of {list(VARIANTS)}")
    LME_DIR.mkdir(parents=True, exist_ok=True)
    fname = VARIANTS[variant]
    dest = LME_DIR / fname
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{HF_REPO_BASE}/{fname}"
    print(f"[longmemeval] downloading {url} ({variant}) -> {dest}")
    subprocess.check_call([
        "curl", "-fL", "--progress-bar", "--retry", "3",
        "-o", str(dest), url,
    ])
    return dest


def load_longmemeval(variant: str = "oracle", limit: int | None = None,
                     stratify: bool = True, seed: int = 42) -> list[dict]:
    """Returns the parsed list of question dicts.

    Args:
        variant:   oracle | s | m
        limit:     if set, return at most `limit` questions.
        stratify:  if True (default), when limit < len(data), draw a
                   roughly proportional sample across question_type so
                   smoke runs are not all one axis. The dataset ships
                   sorted by type, so stratifying matters for small
                   samples. Uses `seed` for reproducibility.
        seed:      RNG seed for stratification.
    """
    import random
    path = _ensure(variant)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{path} top-level is not a list (got {type(data).__name__})")

    if limit is None or limit >= len(data):
        return data

    if not stratify:
        return data[:limit]

    # Stratified by question_type, proportional, leftover via shuffle.
    by_type: dict[str, list[dict]] = {}
    for q in data:
        by_type.setdefault(q.get("question_type", "unknown"), []).append(q)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for qs in by_type.values():
        rng.shuffle(qs)
    # Round-robin draw until we hit the limit.
    type_keys = sorted(by_type.keys())
    rng.shuffle(type_keys)
    cursors = {t: 0 for t in type_keys}
    while len(sampled) < limit:
        progressed = False
        for t in type_keys:
            if cursors[t] < len(by_type[t]) and len(sampled) < limit:
                sampled.append(by_type[t][cursors[t]])
                cursors[t] += 1
                progressed = True
        if not progressed:
            break
    return sampled


def iter_history_turns(question: dict) -> Iterator[tuple[str, str, str, str, str]]:
    """Yields (session_id, session_date, role, content, has_answer_flag) for each turn.

    has_answer_flag is "1" if the turn was marked with has_answer=True (evidence
    turn), else "0".
    """
    sids = question.get("haystack_session_ids") or []
    dates = question.get("haystack_dates") or []
    sessions = question.get("haystack_sessions") or []
    for sid, date, session in zip(sids, dates, sessions):
        for turn in session:
            yield (
                sid,
                date,
                turn.get("role", ""),
                turn.get("content", ""),
                "1" if turn.get("has_answer") else "0",
            )


def question_type_axis(qtype: str) -> str:
    """Map question_type to one of the LongMemEval ability axes.

    The cleaned dataset ships these question types:
        single-session-user / -assistant / -preference  -> information-extraction
        multi-session                                   -> multi-session-reasoning
        temporal-reasoning                              -> temporal-reasoning
        knowledge-update                                -> knowledge-updates
        *_abs (if present)                              -> abstention
    """
    if qtype.endswith("_abs"):
        return "abstention"
    base = qtype
    if base in ("single-session-user",
                "single-session-assistant",
                "single-session-preference"):
        return "information-extraction"
    if base == "multi-session":
        return "multi-session-reasoning"
    if base == "temporal-reasoning":
        return "temporal-reasoning"
    if base == "knowledge-update":
        return "knowledge-updates"
    return base or "unknown"
