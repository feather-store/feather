"""The routed reader (bench/lme_reader): one procedure per question kind,
abstentions re-searched and quoted, premises checked on their own."""
import json

from bench import lme_answer, lme_reader as r


def _chat(handlers, calls=None):
    """A chat that answers by which prompt it was given."""
    def chat(system, user):
        if calls is not None:
            calls.append((system, user))
        for prefix, reply in handlers.items():
            if system.startswith(prefix):
                return reply(user) if callable(reply) else reply
        raise AssertionError(f"unexpected prompt: {system[:60]!r}")
    return chat


ROUTE = r.CLASSIFY_SYSTEM[:30]
ANSWER = lme_answer.ANSWER_SYSTEM[:30]          # every routed prompt's head
VERIFY = r.VERIFY_SYSTEM[:30]
PREMISE = r.PREMISE_SYSTEM[:30]
RESEARCH = r.RESEARCH_SYSTEM[:30]

CONTEXT = ("[Session 2023/05/06 (Sat) 10:00 · 3 days before current date]\n"
           "[user] I tried a lavender gin fizz last weekend and loved it.\n"
           "[assistant] Lovely choice!")


# ── prompts ───────────────────────────────────────────────────────────────

def test_routed_prompts_are_slices_of_the_measured_prompt():
    for route in r.ROUTES:
        p = r.system_for(route)
        assert p.startswith(r._HEAD) and p.endswith(r._TAIL)
        assert "PREMISE CHECK" not in p              # its own call now
        others = [h for k, h in (("aggregate", "COUNTING ("),
                                 ("temporal", "TIME SPANS ("),
                                 ("update", "KNOWLEDGE UPDATES ("),
                                 ("preference", "PREFERENCES (")) if k != route]
        assert not any(h in p for h in others)
        assert ("ABSTENTION (" in p) == (route != r.PREFERENCE)
    for section in r._SECS.values():
        assert section in lme_answer.ANSWER_SYSTEM


def test_classify_reads_one_word_and_falls_back():
    q = "Which store did I spend the most at?"
    assert r.classify(lambda s, u: "Temporal", "When did I go?") == r.TEMPORAL
    assert r.classify(lambda s, u: "fact or update", q) == r.AGGREGATE

    def broken(s, u):
        raise TimeoutError
    assert r.classify(broken, "Any tips for Tokyo?") == r.PREFERENCE


def test_aggregation_covers_superlatives_and_orders_not_spans():
    assert r.is_aggregation_question("How many weddings did I attend?")
    assert r.is_aggregation_question("Which grocery store did I spend the "
                                     "most money at?")
    assert r.is_aggregation_question("In what order did I visit the parks?")
    assert not r.is_aggregation_question("How many days ago did I buy it?")


def test_quote_in_tolerates_copying_slips_not_inventions():
    assert r.quote_in("I tried a lavender gin fizz last weekend", CONTEXT)
    assert r.quote_in("i tried a Lavender gin-fizz last weekend, and loved",
                      CONTEXT)
    assert not r.quote_in("I tried a mojito last weekend and loved it", CONTEXT)
    assert not r.quote_in("loved it", CONTEXT)          # too short to prove


# ── the reader ────────────────────────────────────────────────────────────

def _research_and_route(route):
    return {ROUTE: route, RESEARCH: json.dumps(
        {"queries": ["cocktail I made", "drink recipe"], "nouns": "cocktail"})}


def test_an_abstention_with_a_real_quote_becomes_the_answer():
    aux = _chat(_research_and_route("fact"))
    main = _chat({
        ANSWER: "FINAL ANSWER: It was not mentioned.",
        VERIFY: 'QUOTE: "I tried a lavender gin fizz last weekend and loved '
                'it."\nFINAL ANSWER: a lavender gin fizz',
        PREMISE: 'cocktail recipe | SUPPORTED | "I tried a lavender gin fizz"'
                 "\nVERDICT: SUPPORTED"})
    searched = []
    final, transcript, trace = r.routed_answer(
        main, aux, "What cocktail did I try last weekend?", CONTEXT,
        "2023/05/09", search=lambda q, n: searched.append((q, n)) or "")
    assert final == "a lavender gin fizz"
    assert searched == [(["cocktail I made", "drink recipe"], "cocktail")]
    assert trace["verify"]["quote_found"] and trace["premises"]["verdict"] == "supported"


def test_an_invented_quote_keeps_the_abstention():
    main = _chat({
        ANSWER: "FINAL ANSWER: It was not mentioned.",
        VERIFY: 'QUOTE: "I made a spicy margarita for the party on Friday"\n'
                "FINAL ANSWER: a spicy margarita"})
    final, _, trace = r.routed_answer(
        main, _chat(_research_and_route("fact")),
        "What cocktail did I try last weekend?", CONTEXT, "2023/05/09",
        search=lambda q, n: "")
    assert final == "It was not mentioned."
    assert trace["verify"]["quote"] and not trace["verify"]["quote_found"]


def test_an_unsupported_premise_turns_an_answer_into_not_mentioned():
    main = _chat({
        ANSWER: "FINAL ANSWER: 4 engineers",
        PREMISE: "role of Software Engineer Manager | UNSUPPORTED | the user "
                 "started as Senior Software Engineer\nVERDICT: UNSUPPORTED"})
    final, transcript, trace = r.routed_answer(
        main, _chat({ROUTE: "fact"}),
        "How many engineers did I lead as Software Engineer Manager?",
        CONTEXT, "2023/05/09")
    assert final.startswith("Not mentioned: role of Software Engineer Manager")
    assert "Senior Software Engineer" in final and "[premises]" in transcript


def test_an_unclear_premise_reply_leaves_the_answer_alone():
    main = _chat({ANSWER: "FINAL ANSWER: 4",
                  PREMISE: "I could not tell. VERDICT: UNSUPPORTED"})
    final, _, trace = r.routed_answer(main, _chat({ROUTE: "fact"}), "q?",
                                      CONTEXT, "")
    assert final == "4" and trace["premises"]["verdict"] == "unclear"


def test_preferences_are_neither_verified_nor_premise_checked():
    calls = []
    main = _chat({ANSWER: "FINAL ANSWER: Use your Suica card.", }, calls)
    final, _, trace = r.routed_answer(main, _chat({ROUTE: "preference"}),
                                      "Any tips for Tokyo?", CONTEXT, "",
                                      search=lambda q, n: 1 / 0)
    assert final == "Use your Suica card." and len(calls) == 1
    assert "premises" not in trace and "verify" not in trace
    assert calls[0][0] == r.system_for(r.PREFERENCE)


def test_aggregates_go_to_map_reduce_and_fall_back_to_one_read():
    main = _chat({ANSWER: "FINAL ANSWER: 3",
                  PREMISE: "x | SUPPORTED | \"I tried a lavender gin fizz\"\n"
                           "VERDICT: SUPPORTED"})
    aux = _chat({ROUTE: "aggregate"})
    final, transcript, trace = r.routed_answer(
        main, aux, "How many?", CONTEXT, "",
        map_reduce=lambda q: ("2 weddings", "[map lists]"))
    assert final == "2 weddings" and trace["map_reduce"] is True

    def fails(q):
        raise RuntimeError("every map call failed")
    final, _, trace = r.routed_answer(main, aux, "How many?", CONTEXT, "",
                                      map_reduce=fails)
    assert final == "3" and "every map call failed" in trace["map_reduce"]


# ── map-reduce: aggregations and the ledger ───────────────────────────────

def test_reduce_reads_the_ledger_and_compares_with_its_own_prompt():
    maps = lambda s, u: "2023/05/20 | Walmart $120 | user did/has | \"$120\""
    seen = []

    def reduce(system, user):
        seen.append((system, user))
        return "FINAL ANSWER: Thrive Market"
    ledger = "- 2023/05/02 | thrive market order | $150 | own"
    final, _, _ = lme_answer.count_map_reduce(
        maps, reduce, "Which grocery store did I spend the most money at?",
        ["[Session 2023/05/20]\n[user] Walmart, $120."], "2023/06/01",
        ledger=ledger)
    assert final == "Thrive Market"
    system, user = seen[0]
    assert system == lme_answer.AGG_REDUCE_SYSTEM
    assert user.startswith("Event ledger") and ledger in user

    seen.clear()
    lme_answer.count_map_reduce(maps, reduce, "How many stores did I visit?",
                                ["[Session 2023/05/20]\n[user] Walmart."],
                                "2023/06/01")
    system, user = seen[0]
    assert system == lme_answer.REDUCE_SYSTEM
    assert user.startswith("Candidates from 1 of 1 sessions read")


# ── the scenario ──────────────────────────────────────────────────────────

def test_scenario_routes_and_researches_an_abstention(tmp_path):
    from bench.scenarios import longmemeval as lme
    from tests.test_session_facts import _BagEmbedder

    filler = [(f"d{i}", "2023/03/01 (Wed) 10:00",
               f"Recipe {i}: go easy on the salt.") for i in range(30)]
    sessions = filler + [("g", "2023/05/06 (Sat) 10:00",
                          "My aunt gave me a pearl necklace for my birthday.")]
    researched = []

    def answer_reply(user):
        return "FINAL ANSWER: not mentioned"

    def verify_reply(user):
        researched.append(user)
        return ('QUOTE: "My aunt gave me a pearl necklace for my birthday."\n'
                "FINAL ANSWER: your aunt")

    class Reader:
        name = "fake"

        def answer_full(self, *a, **k):
            raise AssertionError("routed reader must not use answer_full")

        def answer_chat(self):
            return _chat({ANSWER: answer_reply, VERIFY: verify_reply,
                          PREMISE: 'jewelry | SUPPORTED | "My aunt gave me a '
                                   'pearl necklace"\nVERDICT: SUPPORTED'})

        def planner_chat(self):
            return _chat({ROUTE: "fact", RESEARCH: json.dumps(
                {"queries": ["gift of jewellery", "necklace present"],
                 "nouns": "necklace pearl aunt"})})

        def verdict(self, question, gold, predicted):
            return {"correct": predicted == gold, "raw": ""}

    q = {"question_id": "t1", "question_type": "temporal-reasoning",
         "question": "Who gave me the jewelry I received?",
         "answer": "your aunt", "question_date": "2023/05/09 (Tue) 10:00",
         "haystack_session_ids": [s for s, _, _ in sessions],
         "haystack_dates": [d for _, d, _ in sessions],
         "haystack_sessions": [[{"role": "user", "content": c}]
                               for _, _, c in sessions],
         "answer_session_ids": ["g"]}
    out = lme.run([q], embedder=_BagEmbedder(), judge=Reader(), top_k=2,
                  multi_round="off", reader="routed",
                  questions_path=tmp_path / "q.jsonl")
    rec = json.loads((tmp_path / "q.jsonl").read_text())
    assert out["overall"] == 1.0 and out["reader"] == "routed"
    assert rec["reader"]["route"] == "fact"
    assert rec["reader"]["draft"] == "not mentioned"
    assert rec["reader"]["research"]["queries"] == ["gift of jewellery",
                                                    "necklace present"]
    assert "pearl necklace" in researched[0].split("Turns found by the "
                                                   "second search:")[1]


def test_a_failed_check_keeps_the_draft_and_says_so():
    def premise_down(system, user):
        if system.startswith(PREMISE):
            raise RuntimeError("official judge failed after 6 attempts")
        return "FINAL ANSWER: 4"
    final, _, trace = r.routed_answer(premise_down, _chat({ROUTE: "fact"}),
                                      "q?", CONTEXT, "")
    assert final == "4" and "failed after 6 attempts" in trace["premise_error"]

    def verify_down(system, user):
        if system.startswith(VERIFY):
            raise RuntimeError("down")
        return "FINAL ANSWER: not mentioned"
    final, _, trace = r.routed_answer(
        verify_down, _chat(_research_and_route("fact")), "q?", CONTEXT, "",
        search=lambda q, n: "")
    assert final == "not mentioned" and trace["verify_error"]


# ── the verified reader: the single prompt, plus a safe abstention check ──

def test_verified_reader_drafts_with_the_single_prompt():
    calls = []
    main = _chat({ANSWER: "FINAL ANSWER: blue"}, calls)
    final, _, trace = r.verified_answer(main, _chat({}), "What colour?",
                                        CONTEXT, "2023/05/09",
                                        search=lambda q, n: 1 / 0)
    assert final == "blue" and len(calls) == 1
    assert calls[0][0] == lme_answer.ANSWER_SYSTEM     # premise check included
    assert "verify" not in trace and "premises" not in trace


def test_verified_reader_overturns_an_abstention_only_with_a_real_quote():
    aux = _chat({RESEARCH: json.dumps({"queries": ["drink"], "nouns": "gin"})})
    real = _chat({ANSWER: "FINAL ANSWER: not mentioned",
                  VERIFY: 'QUOTE: "I tried a lavender gin fizz last weekend '
                          'and loved it."\nFINAL ANSWER: a lavender gin fizz'})
    final, transcript, trace = r.verified_answer(
        real, aux, "What cocktail?", CONTEXT, "", search=lambda q, n: "")
    assert final == "a lavender gin fizz" and "[verify]" in transcript
    invented = _chat({ANSWER: "FINAL ANSWER: not mentioned",
                      VERIFY: 'QUOTE: "I made a spicy margarita on Friday '
                              'night"\nFINAL ANSWER: a spicy margarita'})
    final, _, _ = r.verified_answer(invented, aux, "What cocktail?", CONTEXT,
                                    "", search=lambda q, n: "")
    assert final == "not mentioned"
