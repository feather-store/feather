"""The rules that keep a benchmark number honest, shared by every bench.

  1. Each bench has a frozen dev/test split, committed with the code:
       LongMemEval  bench/datasets/longmemeval_s_dev.json   (100 of 500 questions)
       LoCoMo       bench/datasets/locomo_split.json        (3 of 10 conversations)
       contradiction  dev worlds from --seed, test worlds from TEST_SEED_BASE
                    with a held-out lexicon (bench/contradiction.py)
     Prompts, thresholds and knobs are tuned on dev, and only on dev.
  2. A run that touches TEST questions needs a committed tree. Its result
     then names, by commit, exactly the code it measured, and a test number
     cannot be fed back into tuning without leaving a commit behind.
  3. Two configurations are compared on 3-5 runs of each, with paired tests
     (bench.compare): same-context reruns flip ~3% of questions.

A test result produced before a split existed is not held out. Say so next
to the number.
"""
from __future__ import annotations


def require_committed(what: str) -> None:
    """Refuse a run on TEST questions from a tree with uncommitted code
    (bench/results and bench/reports do not count)."""
    from .runner import _git_dirty
    if _git_dirty():
        raise SystemExit(
            f"{what} touches TEST questions, and the working tree has uncommitted "
            "code. Tune on the dev split; commit before running test, so the result "
            "names the code it measured (bench/protocol.py).")
