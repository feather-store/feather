"""Map-reduce for LongMemEval count questions: one read per session, one
merge. Maps list candidates; only the reduce decides what counts."""
import json
import re

from bench import lme_answer as A
from bench.embedders import get_embedder
from bench.scenarios import longmemeval as lme


def _sessions():
    return ["[Session 2023/05/01 (Mon) 10:00]\n[user] I bought a road bike.",
            "[Session 2023/05/02 (Tue) 10:00]\n[user] Let's plan a recipe.",
            "[Session 2023/05/03 (Wed) 10:00]\n[user] Got a mountain bike."]


def _map(system, user):
    assert system == A.MAP_SYSTEM and "do not answer it" in user
    m = re.search(r"a (\w+ bike)", user)
    if not m:
        return "NONE"
    date = re.search(r"Session (\S+)", user).group(1)
    return f'- {date} | {m.group(1)} | user did/has | "{m.group(0)}"'


def test_gate_excludes_date_spans():
    assert A.is_map_reduce_question("How many bikes did I buy?")
    assert A.is_map_reduce_question("How much did I spend on bikes in total?")
    assert not A.is_map_reduce_question("How many days ago did I buy a bike?")
    assert not A.is_map_reduce_question("What colour is my bike?")


def test_reduce_reads_only_the_lists():
    seen = []

    def reduce(system, user):
        seen.append(user)
        return "ledger...\nFINAL ANSWER: 2 (road bike, mountain bike)"

    final, transcript, stats = A.count_map_reduce(
        _map, reduce, "How many bikes did I buy?", _sessions(),
        "2023/05/30 (Tue) 12:00")
    assert final.startswith("2")
    assert stats == {"sessions_read": 3, "map_failed": 0,
                     "sessions_listed": 2, "candidates": 2}
    # The reduce sees dated lists, not the sessions' text.
    assert "road bike | user did/has" in seen[0]
    assert "2023/05/03 (Wed) 10:00" in seen[0] and "recipe" not in seen[0]
    assert "2 of 3 sessions" in seen[0] and "[map lists]" in transcript


def test_failed_maps_are_reported_and_all_failing_raises():
    calls = iter([RuntimeError("down"), None, None])

    def flaky(system, user):
        e = next(calls)
        if e:
            raise e
        return _map(system, user)

    seen = []
    A.count_map_reduce(flaky, lambda s, u: seen.append(u) or "FINAL ANSWER: 1",
                       "How many bikes?", _sessions(), "", workers=1)
    assert "(1 could not be read)" in seen[0]

    def down(system, user):
        raise RuntimeError("down")
    try:
        A.count_map_reduce(down, None, "How many bikes?", _sessions(), "")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_reduce_gets_the_date_tool():
    seen = []

    def reduce(system, user):
        seen.append(user)
        if "Tool results" in user:
            return "FINAL ANSWER: 1"
        return "DAYS_BETWEEN(2023/05/01, 2023/05/30)"

    A.count_map_reduce(_map, reduce, "How many bikes in the last month?",
                       _sessions(), "2023/05/30 (Tue) 12:00")
    assert len(seen) == 2 and "= 29 days" in seen[1]


class _Answerer:
    name = "fake"

    def answer_full(self, question, context, question_date=""):
        n = sum(w in context for w in ("road bike", "mountain bike"))
        return str(n), f"FINAL ANSWER: {n}"

    def verdict(self, question, gold, predicted):
        return {"correct": predicted.split()[0] == gold, "raw": ""}

    def map_chat(self):
        return _map

    def answer_chat(self):
        def reduce(system, user):
            n = sum(1 for line in user.splitlines() if "user did/has" in line)
            return f"FINAL ANSWER: {n}"
        return reduce


def test_scenario_haystack_map_counts_what_one_read_missed(tmp_path):
    sessions = [("s1", "I bought a road bike today.")] + \
        [(f"d{i}", f"Let's talk about recipe number {i}.") for i in range(8)] + \
        [("s2", "Picked up a mountain bike at the shop.")]
    q = {"question_id": "q1", "question_type": "multi-session",
         "question": "How many bikes did I buy? road bike", "answer": "2",
         "question_date": "2023/05/30 (Tue) 12:00",
         "haystack_session_ids": [s for s, _ in sessions],
         "haystack_dates": ["2023/05/01 (Mon) 10:00"] * len(sessions),
         "haystack_sessions": [[{"role": "user", "content": c}]
                               for _, c in sessions],
         "answer_session_ids": ["s1", "s2"]}
    emb = get_embedder("deterministic", dim=32)
    run = lambda mode, name: lme.run(   # noqa: E731
        [q], embedder=emb, judge=_Answerer(), top_k=1, multi_round="off",
        count_map=mode, questions_path=tmp_path / name)

    assert run("off", "off.jsonl")["overall"] == 0.0
    assert run("haystack", "hay.jsonl")["overall"] == 1.0
    rec = json.loads((tmp_path / "hay.jsonl").read_text())
    assert rec["count_map"]["sessions_read"] == 10
    assert rec["count_map"]["sessions_listed"] == 2


def test_pro_models_never_ask_for_zero_thinking():
    # 2.5 Pro rejects thinking_budget=0; the planner's failure was
    # swallowed, so multi-round retrieval stopped without an error.
    from bench.providers_extra import _no_thinking
    assert _no_thinking("gemini-2.5-flash") == (0, 0)
    assert _no_thinking("gemini-2.5-pro")[0] >= 128


def test_answer_prompt_checks_premises_with_invented_examples():
    # Examples come from no LongMemEval question: tuning on test failures
    # would inflate the held-out score.
    s = A.ANSWER_SYSTEM
    assert "PREMISE CHECK" in s and "only in an assistant reply" in s
    prompts = s + A.MAP_SYSTEM + A.REDUCE_SYSTEM
    for leaked in ("Peter", "Software Engineer Manager", "bus", "smoker",
                   "necklace", "jewelry", "Trek", "plant", "wedding",
                   "sixth"):
        assert leaked not in prompts, leaked
