"""LongMemEval measurement: official judge prompts, per-question records,
failure buckets, and the scenario writing them end to end (fake LLMs)."""
import json

from bench import lme_eval
from bench.embedders import get_embedder
from bench.scenarios import longmemeval as lme


def test_official_prompt_per_type_and_abstention():
    p = lme_eval.official_prompt
    assert "off-by-one" in p("temporal-reasoning", "q", "a", "r")
    assert "updated answer" in p("knowledge-update", "q", "a", "r")
    assert "Rubric: a" in p("single-session-preference", "q", "a", "r")
    assert p("multi-session", "q", "a", "r").endswith(
        "Is the model response correct? Answer yes or no only.")
    # Abstention overrides the type, as in evaluate_qa.py.
    assert "unanswerable" in p("multi-session", "q", "a", "r", abstention=True)
    assert lme_eval.is_abstention("gpt4_70e84552_abs")
    assert not lme_eval.is_abstention("gpt4_70e84552")


def test_official_verdict_is_yes_in_reply_with_no_system_prompt():
    seen = []

    def chat(system, user):
        seen.append(system)
        return "Yes."
    v = lme_eval.official_verdict(chat, "multi-session", "q", "a", "r", False)
    assert v["correct"] and seen == [""]


def _rec(qtype="multi-session", qid="q1", cov_sessions=("s1",),
         official=False, lenient=False, err=None,
         retrieved=("s1", "s2", "s3")):
    off = {"correct": official, "raw": ""}
    if err:
        off["error"] = err
    return lme_eval.make_record(
        {"question_id": qid, "question_type": qtype, "question": "q",
         "answer": "a", "answer_session_ids": ["s1", "s2"]},
        retrieved_sessions=list(retrieved),
        context_sessions=list(cov_sessions), context_chars=10, answer="x",
        official=off, lenient={"correct": lenient, "raw": ""})


def test_buckets():
    assert _rec(official=True)["bucket"] is None
    # Retrieved, then cut by the budget: packing, not retrieval.
    assert _rec(cov_sessions=("s1",))["bucket"] == "packing_miss"
    assert _rec(cov_sessions=("s1",), retrieved=("s1", "s3"))["bucket"] == \
        "retrieval_miss"
    assert _rec(cov_sessions=("s1", "s2"))["bucket"] == "reader_wrong"
    assert _rec(cov_sessions=("s1", "s2"), lenient=True)["bucket"] == \
        "judge_disagreement"
    assert _rec(err="timeout")["bucket"] == "judge_error"
    # Abstention evidence is a decoy: never a retrieval miss.
    assert _rec(qid="x_abs", cov_sessions=())["bucket"] == "reader_wrong"
    r = _rec(cov_sessions=("s1",))
    assert r["coverage"] == 0.5 and r["retrieval_coverage"] == 1.0


def test_summarize_reports_both_judges():
    recs = [_rec(official=True, lenient=True, cov_sessions=("s1", "s2")),
            _rec(official=False, lenient=True, cov_sessions=("s1", "s2")),
            _rec(qtype="temporal-reasoning", qid="t_abs")]
    s = lme_eval.summarize(recs)
    assert s["overall_official"] == round(1 / 3, 4)
    assert s["overall_lenient"] == round(2 / 3, 4)
    assert s["by_question_type_official"] == {"multi-session": 0.5,
                                              "temporal-reasoning": 0.0}
    assert s["task_averaged_official"] == 0.25
    assert s["abstention_official"] == 0.0
    assert s["judge_agreement"] == round(2 / 3, 4)
    assert s["failure_buckets"] == {"judge_disagreement": 1, "reader_wrong": 1}
    assert s["evidence_fully_in_context"] == 1.0
    assert "official judge" in lme_eval.format_summary(s)


class _FakeAnswerer:
    name = "fake"

    def answer_full(self, question, context, question_date=""):
        return ("blue" if "blue" in context else "I don't know",
                "reasoning…\nFINAL ANSWER: …")

    def verdict(self, question, gold, predicted):
        return {"correct": gold in predicted, "raw": "lenient"}

    def score(self, predicted, gold, question=""):
        raise AssertionError("verdict() should be used")


class _FakeOfficial:
    name = "official=fake"
    calls = []

    def verdict(self, qtype, question, gold, predicted, abstention):
        self.calls.append((qtype, abstention))
        return {"correct": gold in predicted, "raw": "yes" if gold in predicted
                else "no"}


def _question(qid, answer, sessions):
    return {
        "question_id": qid, "question_type": "single-session-user",
        "question": "What colour is my bike?", "answer": answer,
        "question_date": "2023/05/30 (Tue) 12:00",
        "haystack_session_ids": [s for s, _ in sessions],
        "haystack_dates": ["2023/05/01 (Mon) 10:00"] * len(sessions),
        "haystack_sessions": [[{"role": "user", "content": c}]
                              for _, c in sessions],
        "answer_session_ids": [sessions[0][0]],
    }


def test_scenario_writes_records_and_reports_both_scores(tmp_path):
    qs = [_question("q1", "blue", [("s1", "My bike is blue."),
                                   ("s2", "I like tea.")]),
          _question("q2_abs", "not mentioned", [("s3", "My car is red.")])]
    out = tmp_path / "q.jsonl"
    m = lme.run(qs, embedder=get_embedder("deterministic", dim=32),
                judge=_FakeAnswerer(), top_k=5, official_judge=_FakeOfficial(),
                questions_path=out)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["question_id"] for r in recs] == ["q1", "q2_abs"]
    assert recs[0]["coverage"] == 1.0 and recs[0]["answer"] == "blue"
    assert recs[0]["official"]["correct"] and recs[0]["bucket"] is None
    assert recs[1]["abstention"] and recs[1]["bucket"] == "reader_wrong"
    assert ("single-session-user", True) in _FakeOfficial.calls
    assert m["overall"] == m["overall_official"] == 0.5
    assert m["headline_judge"] == "official"
    assert m["by_axis"] == {"abstention": 0.0, "information-extraction": 1.0}
    assert m["questions_file"] == str(out)


def test_mcnemar_exact():
    assert lme_eval.mcnemar_p(0, 0) == 1.0
    assert lme_eval.mcnemar_p(5, 5) == 1.0
    # 10 discordant, all one way: 2 * 0.5**10
    assert abs(lme_eval.mcnemar_p(0, 10) - 2 / 1024) < 1e-12
    assert lme_eval.mcnemar_p(3, 9) == lme_eval.mcnemar_p(9, 3)


def test_compare_pairs_by_question():
    def rec(qid, ok, t="multi-session"):
        return {"question_id": qid, "question_type": t,
                "official": {"correct": ok}}
    a = [rec("q1", True), rec("q2", False), rec("q3", False), rec("q4", True)]
    b = [rec("q1", True), rec("q2", True), rec("q3", True), rec("q5", True)]
    c = lme_eval.compare(a, b)
    assert c["overall"]["n"] == 3 and c["overall"]["b_only"] == 2
    assert c["fixed"] == ["q2", "q3"] and c["broken"] == []
    assert c["only_in_a"] == 1 and c["only_in_b"] == 1
    assert "McNemar" in lme_eval.format_compare(c)


# ── a run's score is over every question it was given ─────────────────────

def test_failed_and_unjudged_questions_count_wrong_not_absent():
    ok = _rec(official=True, lenient=True, cov_sessions=("s1", "s2"))
    crashed = _rec(qid="q2", cov_sessions=())
    crashed["run_error"] = "TimeoutError: gemini"
    crashed["bucket"] = lme_eval.bucket(crashed)
    unjudged = _rec(qid="q3", official=True, lenient=True,
                    cov_sessions=("s1", "s2"))
    unjudged["official"] = None
    errored = _rec(qid="q4", cov_sessions=("s1", "s2"))
    errored["answer"] = "[answerer error: 429]"
    s = lme_eval.summarize([ok, crashed, unjudged, errored])
    assert crashed["bucket"] == "run_error"
    assert lme_eval.bucket(errored) == "run_error"
    assert s["overall_official"] == 0.25          # 1 of 4, not 1 of 1
    assert s["run_errors"] == 2
    assert "Rerun before reporting" in lme_eval.format_summary(s)


def test_check_names_what_makes_a_run_incomplete():
    full = [_rec(qid=f"q{i}", official=True, cov_sessions=("s1", "s2"))
            for i in range(3)]
    assert lme_eval.check(full, 3) == []
    assert lme_eval.check(full[:2], 3) == ["2 questions, expected 3"]
    bad = full + [dict(full[0])]
    bad[1] = dict(bad[1], run_error="boom")
    bad[2] = dict(bad[2], official=None)
    why = lme_eval.check(bad, 3)
    assert "1 questions appear twice" in why
    assert "1 questions failed to run" in why
    assert "1 questions have no official verdict" in why


def test_a_question_that_crashes_is_scored_wrong_not_dropped(tmp_path):
    from bench.scenarios import longmemeval as lme
    from tests.test_session_facts import _BagEmbedder

    class Reader:
        name = "fake"

        def answer_full(self, question, context, question_date=""):
            if "crash" in question:
                raise ConnectionError("provider down")
            return "blue", ""

        def verdict(self, question, gold, predicted):
            return {"correct": predicted == gold, "raw": ""}

    def q(qid, text):
        return {"question_id": qid, "question_type": "single-session-user",
                "question": text, "answer": "blue",
                "question_date": "2023/05/09 (Tue) 10:00",
                "haystack_session_ids": ["s1"],
                "haystack_dates": ["2023/05/01 (Mon) 10:00"],
                "haystack_sessions": [[{"role": "user",
                                        "content": "My bike is blue."}]],
                "answer_session_ids": ["s1"]}
    out = lme.run([q("a", "What colour is my bike?"), q("b", "crash please")],
                  embedder=_BagEmbedder(), judge=Reader(), top_k=2,
                  multi_round="off", questions_path=tmp_path / "q.jsonl")
    recs = lme_eval.read_jsonl(tmp_path / "q.jsonl")
    assert [r["question_id"] for r in recs] == ["a", "b"]
    assert recs[1]["bucket"] == "run_error"
    assert "provider down" in recs[1]["run_error"]
    assert out["overall"] == 0.5 and out["n_scored"] == 2
    assert out["run_errors"] == 1 and out["n_failures"] == 1
