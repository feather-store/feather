"""A frozen dev/test split of LoCoMo (locomo10), by conversation.

Tuning reads whole conversations (the extractor prompt runs over every
session), so the split is by conversation, not by question: a question-level
split would tune on the very sessions the test questions ask about.

DEV is the 3 conversations with the smallest sha1(sample_id): 459 of the
1,540 category 1-4 questions (30%), category mix close to the whole. TEST is
the other 7 (1,081). Generated once and committed (`locomo_split.json`);
`dev_conversations()` reads the file and never regenerates it.

Frozen 2026-11-08, AFTER the 0.2.0-0.3.1 extractor prompts were tuned with
every conversation in view: LoCoMo numbers for those prompts are not held
out, on test or anywhere.

    python -m bench.datasets.locomo_split     # writes the file if missing
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLIT_FILE = Path(__file__).with_name("locomo_split.json")
DEV_SIZE = 3


def make_split(sample_ids: list[str], size: int = DEV_SIZE) -> list[str]:
    return sorted(sorted(sample_ids, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:size])


def dev_conversations() -> set[str]:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"{SPLIT_FILE} missing: run `python -m bench.datasets."
                                "locomo_split` once and commit it")
    return set(json.loads(SPLIT_FILE.read_text())["dev"])


def split_of(sample_id: str) -> str:
    return "dev" if sample_id in dev_conversations() else "test"


if __name__ == "__main__":
    from bench.locomo import load, questions_of
    if SPLIT_FILE.exists():
        raise SystemExit(f"{SPLIT_FILE} exists; the split is frozen")
    data = load()
    ids = [s["sample_id"] for s in data]
    dev = make_split(ids)
    n = {s["sample_id"]: len(questions_of(s)) for s in data}
    SPLIT_FILE.write_text(json.dumps({
        "dataset": "locomo10", "unit": "conversation",
        "rule": "the 3 sample_ids with the smallest sha1(sample_id)",
        "frozen": "2026-11-08",
        "dev": dev, "test": sorted(set(ids) - set(dev)),
        "questions_cat1_4": {"dev": sum(n[s] for s in dev),
                             "test": sum(n[s] for s in ids if s not in dev)},
        "not_held_out": "the 0.2.0-0.3.1 extractor prompts were tuned with all 10 "
                        "conversations in view",
    }, indent=1) + "\n")
    print(f"wrote {SPLIT_FILE}")
