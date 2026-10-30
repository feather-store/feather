"""The frozen LongMemEval dev/test split."""
import json
import random
from collections import Counter

from bench.datasets import lme_split as S


def _qs():
    qs = []
    for t, n in (("multi-session", 133), ("temporal-reasoning", 127),
                 ("single-session-user", 70), ("knowledge-update", 78)):
        qs += [{"question_id": f"{t}-{i}", "question_type": t}
               for i in range(n)]
    qs += [{"question_id": f"x{i}_abs", "question_type": "multi-session"}
           for i in range(12)]
    return qs


def test_split_is_proportional_and_order_free():
    qs = _qs()
    dev = S.make_dev_split(qs, 100)
    assert len(dev) == 100 == len(set(dev))
    per = Counter(S._stratum(q) for q in qs if q["question_id"] in set(dev))
    assert per[("multi-session", False)] in (31, 32)
    assert per[("multi-session", True)] in (2, 3)
    shuffled = list(qs)
    random.Random(1).shuffle(shuffled)
    assert S.make_dev_split(shuffled, 100) == dev


def test_frozen_file_holds_100_ids():
    data = json.loads(S.DEV_FILE.read_text())
    assert len(data["dev"]) == 100 == len(set(data["dev"]))
    assert S.dev_ids() == set(data["dev"])


def test_select_partitions():
    dev = S.dev_ids()
    qs = [{"question_id": q} for q in sorted(dev)[:3]] + \
         [{"question_id": "not-in-dev"}]
    assert [q["question_id"] for q in S.select(qs, "test")] == ["not-in-dev"]
    assert len(S.select(qs, "dev")) == 3 and len(S.select(qs, "all")) == 4
