"""The LongMemEval answerer sees only what build_context renders: dates,
roles and whole sessions must survive into it."""
from bench import lme_answer
from bench.scenarios.lme_context import (Turn, build_context, parse_lme_date,
                                        render, session_header)


def _haystack():
    # Two sessions, deliberately listed newest first.
    return [
        Turn("s2", "2023/05/20 (Sat) 09:00", "user", "I bought a second bike.", 0),
        Turn("s2", "2023/05/20 (Sat) 09:00", "assistant", "Nice! " + "x" * 2000, 1),
        Turn("s1", "2023/04/10 (Mon) 17:50", "user", "I bought a bike.", 2),
        Turn("s1", "2023/04/10 (Mon) 17:50", "assistant", "Enjoy it.", 3),
        Turn("s1", "2023/04/10 (Mon) 17:50", "user", "It is red.", 4),
    ]


def test_parses_the_dataset_date_format():
    # fromisoformat rejects this; every turn used to be stored at ts=0.
    d = parse_lme_date("2023/04/10 (Mon) 23:07")
    assert (d.year, d.month, d.day, d.hour, d.minute) == (2023, 4, 10, 23, 7)
    assert parse_lme_date("2023-04-10T23:07:00") is not None
    assert parse_lme_date("garbage") is None


def test_sessions_are_dated_role_marked_and_chronological():
    turns = _haystack()
    ctx = build_context([turns[0], turns[2]], turns)   # s2 ranked first
    assert "2023/04/10 (Mon) 17:50" in ctx and "2023/05/20 (Sat) 09:00" in ctx
    assert ctx.index("2023/04/10") < ctx.index("2023/05/20")
    assert "s1" not in ctx and "s2" not in ctx   # ids leak gold in oracle
    assert "[user] I bought a bike." in ctx and "[assistant] Enjoy it." in ctx


def test_session_unit_expands_a_hit_to_its_whole_session():
    turns = _haystack()
    ctx = build_context([turns[2]], turns, unit="session")
    assert "It is red." in ctx            # not retrieved, same session
    assert "second bike" not in ctx       # other session, not retrieved


def test_turn_unit_keeps_only_hits():
    turns = _haystack()
    ctx = build_context([turns[2]], turns, unit="turn")
    assert "I bought a bike." in ctx and "It is red." not in ctx
    assert "2023/04/10" in ctx


def test_budget_degrades_a_session_before_dropping_it():
    turns = _haystack()
    ctx = build_context([turns[0], turns[2]], turns, max_chars=600)
    assert "second bike" in ctx and "I bought a bike." in ctx
    assert "x" * 2000 not in ctx and len(ctx) <= 600


def test_header_carries_distance_to_question_date():
    h = session_header("2023/04/10 (Mon) 17:50", "2023/04/26 (Wed) 09:00")
    assert h == "[Session 2023/04/10 (Mon) 17:50 · 16 days before current date]"
    assert "same day" in session_header("2023/04/26 (Wed) 01:00",
                                        "2023/04/26 (Wed) 09:00")
    assert session_header("2023/04/10 (Mon) 17:50") == \
        "[Session 2023/04/10 (Mon) 17:50]"


def test_days_between():
    r = lme_answer.days_between("2023/01/31", "2023/03/15")
    assert "43 days" in r and "6 weeks 1 days" in r and "1 full calendar" in r
    assert "earlier" in lme_answer.days_between("2023-03-15", "2023-01-31")
    assert "error" in lme_answer.days_between("2023/02/30", "2023/03/01")


def test_tool_loop_serves_calls_then_returns_final():
    seen = []

    def chat(system, user):
        seen.append(user)
        if "Tool results" not in user:
            return "Need the span.\nDAYS_BETWEEN(2023/04/10, 2023/05/20)"
        assert "= 40 days" in user and "Current date: 2023/05/20" in user
        return "Reasoning.\nFINAL ANSWER: 40 days"

    out = lme_answer.answer(chat, "How long ago?", "ctx", "2023/05/20 (Sat) 10:00")
    assert lme_answer.final_answer(out) == "40 days" and len(seen) == 2


def test_tool_loop_stops_without_calls_and_bounds_rounds():
    calls = iter(f"DAYS_BETWEEN(2023/01/{d:02d}, 2023/02/01)" for d in range(1, 30))
    n = []

    def chat(system, user):
        n.append(1)
        return next(calls)

    lme_answer.answer(chat, "q", "ctx", "")
    assert len(n) == lme_answer.MAX_TOOL_ROUNDS + 1
    assert lme_answer.final_answer("no marker") == "no marker"


def _long_haystack(n_sessions=6):
    """Sessions of one short user turn and one long assistant reply."""
    turns = []
    for s in range(n_sessions):
        date = f"2023/04/{10 + s:02d} (Mon) 10:00"
        turns.append(Turn(f"s{s}", date, "user", f"Fact number {s}: " + "u" * 400,
                          len(turns)))
        turns.append(Turn(f"s{s}", date, "assistant", "a" * 5000, len(turns)))
        turns.append(Turn(f"s{s}", date, "user", f"Later remark {s}.", len(turns)))
    return turns


def test_budget_admits_every_session_before_upgrading_any():
    # Rank-order greedy used to give the top sessions verbatim and cut the
    # rest: 14 of ~22 retrieved sessions reached the reader.
    turns = _long_haystack()
    hits = [t for t in turns if t.content.startswith("Later")]
    r = render(hits, turns, max_chars=12_000)
    assert len(r.in_context) == 6 and len(r.text) <= 12_000
    assert r.levels["s0"] == 0                     # best-ranked upgraded
    assert max(r.levels.values()) > 0              # the rest shortened


def test_dense_hits_do_not_crowd_out_low_ranked_sessions():
    # Multi-round retrieval merges ~100 turns per round; a top session with
    # a dozen hits cost ~25k chars at HITS_ONLY and 160k admitted 12-15 of
    # ~25 sessions, cutting golds retrieved at ranks 10-24 (packing misses).
    turns = []
    for s in range(25):
        for j in range(12):
            turns.append(Turn(f"s{s}", f"2023/04/{1 + s:02d} (Mon) 10:00",
                              "user", f"s{s} turn {j} " + "w" * 2500,
                              len(turns)))
    hits = [t for t in turns if t.session_id in {f"s{s}" for s in range(5)}]
    hits += [t for t in turns if t.session_id not in {h.session_id
                                                       for h in hits}][::12]
    r = render(hits, turns, max_chars=160_000)
    assert len(r.retrieved) == 25 and r.in_context == r.retrieved
    assert len(r.text) <= 160_000


def test_shortening_cuts_assistant_prose_never_user_turns():
    # 94% of LongMemEval evidence is in user turns; they used to be
    # clipped to 300 chars, or dropped, when away from a retrieved turn.
    turns = _long_haystack()
    hits = [t for t in turns if t.content.startswith("Later")]
    r = render(hits, turns, max_chars=6_000)
    assert len(r.in_context) == 6
    for s in range(6):
        assert f"Fact number {s}: " + "u" * 400 in r.text
    assert "a" * 5000 not in r.text and "[…]" in r.text


def test_answer_prompt_carries_a_calendar():
    seen = []
    lme_answer.answer(lambda s, u: seen.append(u) or "FINAL ANSWER: x",
                      "Who gave me jewelry last Saturday?", "ctx",
                      "2023/03/09 (Thu) 15:47")
    assert "Sat 2023/03/04" in seen[0] and "Thu 2023/03/02" in seen[0]
    assert lme_answer.calendar("") == ""


def test_span_answer_without_the_tool_is_sent_back_once():
    seen = []

    def chat(system, user):
        seen.append(user)
        if "Tool results" in user:
            return "FINAL ANSWER: 21 days"
        if lme_answer.TOOL_NUDGE in user:
            return "DAYS_BETWEEN(2022/03/25, 2022/04/15)"
        return "Mental math says 26.\nFINAL ANSWER: 26 days"

    out = lme_answer.answer(chat, "How many days ago did I bake?", "ctx",
                            "2022/04/15 (Fri) 18:46")
    assert lme_answer.final_answer(out) == "21 days" and len(seen) == 3
    assert "= 21 days" in seen[-1]
    # Not a span question: no nudge.
    n = []
    lme_answer.answer(lambda s, u: n.append(1) or "FINAL ANSWER: red",
                      "What colour is my bike?", "ctx", "2023/01/01")
    assert len(n) == 1


def test_days_between_reports_rounded_months():
    assert "≈5.7 months" in lme_answer.days_between("2022/12/20", "2023/06/10")


def test_budget_in_tokens_uses_the_counter():
    turns = _long_haystack()
    hits = [t for t in turns if t.content.startswith("Later")]
    words = lambda s: len(s.split())     # noqa: E731  a stand-in tokenizer
    r = render(hits, turns, max_tokens=90, count_tokens=words)
    assert words(r.text) <= 90 and len(r.in_context) == 6
    # The same text by chars would blow through a 90-char budget.
    assert len(r.text) > 90
    try:
        render(hits, turns, max_tokens=60)
        assert False, "max_tokens without a counter must raise"
    except ValueError:
        pass
