"""A frozen dev/test split of LongMemEval_S.

Prompts and knobs are tuned on DEV (100 questions) only; TEST (the other
400) is run once per frozen configuration. Tuning against all 500 and
reporting all 500 measures how well the prompt fits those 500.

The split is stratified by (question_type, abstention) and picked by a
hash of the question id, so it depends on nothing but the ids. It is
generated once and committed (`longmemeval_s_dev.json`); `dev_ids()`
reads the file, never regenerates it: a split that moved when the
dataset file was reordered would leak test questions into tuning.

    python -m bench.datasets.lme_split     # writes the file if missing
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path

DEV_FILE = Path(__file__).with_name("longmemeval_s_dev.json")
DEV_SIZE = 100


def _stratum(q: dict) -> tuple[str, bool]:
    return q.get("question_type", "unknown"), "_abs" in q["question_id"]


def make_dev_split(questions: list[dict], size: int = DEV_SIZE) -> list[str]:
    """Proportional allocation per stratum (largest remainder), then the
    ids with the smallest sha1 within each stratum."""
    strata: dict = defaultdict(list)
    for q in questions:
        strata[_stratum(q)].append(q["question_id"])
    n = len(questions)
    quota = {k: size * len(v) / n for k, v in strata.items()}
    take = {k: int(x) for k, x in quota.items()}
    for k in sorted(quota, key=lambda k: (take[k] - quota[k], k))[:size - sum(take.values())]:
        take[k] += 1
    h = lambda qid: hashlib.sha1(qid.encode()).hexdigest()   # noqa: E731
    return sorted(qid for k, ids in strata.items()
                  for qid in sorted(ids, key=h)[:take[k]])


def dev_ids() -> set[str]:
    if not DEV_FILE.exists():
        raise FileNotFoundError(
            f"{DEV_FILE} missing: run `python -m bench.datasets.lme_split` "
            "once and commit it")
    return set(json.loads(DEV_FILE.read_text())["dev"])


def select(questions: list[dict], split: str) -> list[dict]:
    """split: all | dev | test."""
    if split == "all":
        return questions
    dev = dev_ids()
    if split == "dev":
        return [q for q in questions if q["question_id"] in dev]
    if split == "test":
        return [q for q in questions if q["question_id"] not in dev]
    raise ValueError(f"split must be all, dev or test, not {split!r}")


if __name__ == "__main__":
    from .longmemeval import load_longmemeval
    if DEV_FILE.exists():
        raise SystemExit(f"{DEV_FILE} exists; the split is frozen")
    qs = load_longmemeval("s")
    dev = make_dev_split(qs)
    DEV_FILE.write_text(json.dumps({
        "dataset": "longmemeval_s_cleaned", "n_total": len(qs),
        "strata": "question_type x abstention, proportional; sha1(question_id)",
        "dev": dev}, indent=1) + "\n")
    print(f"wrote {len(dev)} dev ids to {DEV_FILE}")
