"""Multi-round retrieval for LongMemEval count/comparison questions."""
import json

from bench import lme_retrieve as R
from bench.embedders import get_embedder
from bench.scenarios import longmemeval as lme
from bench.scenarios.lme_context import Turn


def test_dense_query_carries_the_date():
    assert R.dense_query("q?", "2023/05/30 (Tue) 12:00") == \
        "[Current date: 2023/05/30 (Tue) 12:00] q?"
    assert R.dense_query("q?", "") == "q?"


def test_gate_picks_count_and_comparison_questions():
    assert R.needs_multi_round("How many bikes do I own?")
    assert R.needs_multi_round("Which did I attend first, the workshop or the webinar?")
    assert R.needs_multi_round("How much did I spend in total on concerts?")
    assert not R.needs_multi_round("What colour is my bike?")
    assert not R.needs_multi_round("What did I eat last week?")


def test_plan_followups_parses_caps_and_survives_garbage():
    reply = ('Sure:\n```json\n{"missing": "the second bike", "queries": ['
             '{"query": "got a mountain bike", "after": "2023/05/01", "before": "nope"},'
             '"bought a helmet", {"query": ""}, {"query": "a"}, {"query": "b"}]}\n```')
    qs, missing = R.plan_followups(lambda s, u: reply, "q", "d", "dig")
    assert missing == "the second bike"
    assert qs[0] == {"query": "got a mountain bike", "after": "2023/05/01",
                     "before": None}
    assert qs[1]["query"] == "bought a helmet" and len(qs) == R.MAX_FOLLOWUPS
    assert R.plan_followups(lambda s, u: "no json here", "q", "d", "")[0] == []

    def boom(s, u):
        raise RuntimeError("down")
    qs, missing = R.plan_followups(boom, "q", "d", "")
    assert qs == [] and "planner error" in missing


def test_merge_keeps_round_one_and_adds_new_sessions():
    out = R.merge_sessions(["a", "b", "c"], [["b", "x", "y", "z", "w"], ["a", "q"]],
                           k=2, per_followup=3)
    assert out == ["a", "b", "x", "y", "z", "q"]


def test_window_and_digest():
    assert R.in_window("2023/05/14 (Sun) 10:00", "2023/05/14", "2023/05/20")
    assert not R.in_window("2023/05/13 (Sat) 23:59", "2023/05/14", None)
    assert R.in_window("", "2023/05/14", None)
    hits = [Turn("s1", "2023/05/01 (Mon) 10:00", "user", "x" * 1000, 0),
            Turn("s2", "2023/05/02 (Tue) 10:00", "user", "short", 1)]
    d = R.digest(hits, "2023/05/03 (Wed) 10:00", max_sessions=1)
    assert "2 days before current date" in d and "[…]" in d and "short" not in d


class _Planner:
    name = "fake"
    asked = []

    def answer_full(self, question, context, question_date=""):
        n = sum(w in context for w in ("road bike", "mountain bike"))
        return str(n), f"FINAL ANSWER: {n}"

    def verdict(self, question, gold, predicted):
        return {"correct": predicted == gold, "raw": ""}

    def planner_chat(self):
        def chat(system, user):
            self.asked.append(user)
            return json.dumps({"missing": "other bikes",
                               "queries": [{"query": "mountain bike"}]})
        return chat


def test_followup_round_recovers_the_missed_session(tmp_path):
    sessions = [("s1", "I bought a road bike today.")] + \
        [(f"d{i}", f"Let's talk about recipe number {i}.") for i in range(8)] + \
        [("s2", "Picked up a mountain bike at the shop.")]
    q = {"question_id": "q1", "question_type": "multi-session",
         "question": "How many bikes did I buy? road bike", "answer": "2",
         "question_date": "2023/05/30 (Tue) 12:00",
         "haystack_session_ids": [s for s, _ in sessions],
         "haystack_dates": ["2023/05/01 (Mon) 10:00"] * len(sessions),
         "haystack_sessions": [[{"role": "user", "content": c}] for _, c in sessions],
         "answer_session_ids": ["s1", "s2"]}
    emb = get_embedder("deterministic", dim=32)

    off = lme.run([q], embedder=emb, judge=_Planner(), top_k=1,
                  multi_round="off", questions_path=tmp_path / "off.jsonl")
    on = lme.run([q], embedder=emb, judge=_Planner(), top_k=1,
                 multi_round="auto", questions_path=tmp_path / "on.jsonl")
    rec = json.loads((tmp_path / "on.jsonl").read_text())
    assert off["overall"] == 0.0 and on["overall"] == 1.0
    assert rec["followups"] == [{"query": "mountain bike", "after": None,
                                 "before": None}]
    assert rec["round1_coverage"] == 0.5 and rec["coverage"] == 1.0
    assert on["multi_round_evidence_round1"] == 0.0
    assert on["multi_round_evidence_final"] == 1.0
    assert "Current date" in _Planner.asked[0]


def test_count_vote_takes_the_majority_and_keeps_its_transcript():
    from bench import lme_answer as A
    assert A.is_count_question("How many bikes do I own?")
    assert not A.is_count_question("Which came first, the bike or the car?")
    assert A.vote_key("You have three bikes.") == A.vote_key("3 bikes") == "3"
    assert A.vote_key("$1,200 in total") == "1200"
    replies = iter(["FINAL ANSWER: 2", "listing…\nFINAL ANSWER: three",
                    "FINAL ANSWER: 3 bikes"])
    lock = __import__("threading").Lock()

    def chat(system, user):
        with lock:
            return next(replies)
    final, transcript = A.answer_vote([chat, chat, chat], "How many?", "ctx", "")
    assert A.vote_key(final) == "3" and "'3'×2" in transcript

    def boom(system, user):
        raise RuntimeError("down")
    final, transcript = A.answer_vote([boom, lambda s, u: "FINAL ANSWER: 4"],
                                      "How many?", "ctx", "")
    assert final == "4"
