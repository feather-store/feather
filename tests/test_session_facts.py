"""SessionFactExtractor + link_validity (library) and their LongMemEval
wiring: facts find sessions, the reader gets sessions, changes keep both
values with their dates."""
import json

from feather_db.extractors import (Fact, SessionFactExtractor, fact_key,
                                   ledger, link_validity, profile, timeline)


class _Provider:
    def __init__(self, reply):
        self.reply, self.messages = reply, []

    def complete(self, messages, max_tokens=512, temperature=0.0):
        self.messages.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply(messages) if callable(self.reply) else self.reply


TURNS = [{"role": "user", "content": "I switched from yoga to pilates last week."},
         {"role": "assistant", "content": "Nice! " + "tip " * 1000}]


def test_extracts_dated_facts_with_turn_and_session_date_in_prompt():
    p = _Provider(json.dumps({"facts": [   # JSON mode wraps the array
        {"statement": "On 2023/05/07 the user switched from yoga to pilates.",
         "date": "2023/05/07", "subject": "User", "attribute": "Exercise class",
         "value": "pilates", "turn": 0},
        {"statement": "no date given", "subject": "user", "attribute": "x",
         "value": "y", "turn": 99},
        {"date": "2023/01/01"}]}))
    ex = SessionFactExtractor(p, max_turn_chars=50)
    facts = ex.extract_session(TURNS, "2023/05/14 (Sun) 10:00",
                               context={"session_id": "s1"})
    user = p.messages[0][1]["content"]
    assert "Session date: 2023/05/14 (Sun)" in user and "[0] user:" in user
    assert "[1] assistant: Nice!" in user and "tip " * 100 not in user
    assert len(facts) == 2
    f = facts[0]
    assert (f.subject, f.predicate, f.object) == ("User", "Exercise class", "pilates")
    assert f.metadata["date"] == "2023/05/07" and f.metadata["turn"] == 0
    assert f.metadata["session_id"] == "s1"
    # no date -> the session's; a turn out of range -> None
    assert facts[1].metadata["date"] == "2023/05/14" and facts[1].metadata["turn"] is None


def test_failures_return_empty_and_report():
    errs = []
    assert SessionFactExtractor(_Provider("not json"), max_retries=0).extract_session(
        TURNS, "2023/05/14", context={"on_error": errs.append}) == []
    assert SessionFactExtractor(_Provider(RuntimeError("down")), max_retries=0
                                ).extract_session(TURNS, "2023/05/14",
                                                  context={"on_error": errs.append}) == []
    assert len(errs) == 2
    assert SessionFactExtractor(_Provider("[]")).extract_session(
        [{"role": "user", "content": "  "}], "2023/05/14") == []


def _f(value, date, subject="user", attr="job"):
    import datetime
    d = datetime.datetime.strptime(date, "%Y/%m/%d")
    return Fact(subject=subject, predicate=attr, object=value,
                valid_at=int(d.timestamp()),
                metadata={"statement": f"On {date} {subject} {attr} {value}",
                          "date": date})


def test_link_validity_keeps_every_value_with_its_dates():
    a, a2, b = _f("junior dev", "2023/01/10"), _f("Junior dev", "2023/01/10"), \
        _f("senior dev", "2023/05/14")
    other = _f("blue", "2023/02/01", attr="car colour")
    chains = link_validity([b, a2, other, a])
    chain = chains[fact_key(a)]
    assert [f.object for f in chain][-1] == "senior dev"
    # both junior-dev mentions are closed by the change, not by each other
    assert a.metadata["valid_until"] == a2.metadata["valid_until"] == "2023/05/14"
    assert "valid_until" not in b.metadata and b.metadata["valid_from"] == "2023/05/14"
    assert "valid_until" not in other.metadata
    t = timeline(chains, [fact_key(b), fact_key(other)])
    assert t.lower() == ("- user / job: junior dev (2023/01/10 – 2023/05/14) → "
                         "senior dev (2023/05/14 – now)")
    assert fact_key(_f("x", "2023/01/01", "User's", "Job!")) == "user's|job"


def test_cache_skips_repeat_sessions_and_never_caches_failures(tmp_path):
    from bench.lme_facts import SessionFactCache
    reply = json.dumps([{"statement": "On 2023/05/14 the user likes tea.",
                         "date": "2023/05/14", "subject": "user",
                         "attribute": "drink", "value": "tea", "turn": 0}])
    p = _Provider(reply)
    c = SessionFactCache(p, "m", path=tmp_path / "c.jsonl")
    f1 = c.facts("s1", "2023/05/14", TURNS)
    f2 = c.facts("other_id", "2023/05/14", TURNS)       # same content
    assert len(p.messages) == 1 and c.hits == 1
    assert f2[0].metadata["session_id"] == "other_id" != f1[0].metadata["session_id"]
    again = SessionFactCache(_Provider(RuntimeError("unused")), "m",
                             path=tmp_path / "c.jsonl")
    assert again.facts("s1", "2023/05/14", TURNS)[0].object == "tea"
    bad = SessionFactCache(_Provider("garbage"), "m2", path=tmp_path / "d.jsonl")
    bad._extractor._max_retries = 0
    assert bad.facts("s1", "2023/05/14", TURNS) == [] and bad.failures == 1
    assert not (tmp_path / "d.jsonl").exists() or \
        (tmp_path / "d.jsonl").read_text() == ""


class _BagEmbedder:
    """Hashed bag of words: overlap decides the dense ranking too, so the
    test is not at the mercy of a random-vector embedder."""
    dim, name = 64, "bag"

    def embed(self, text):
        import re
        import zlib
        import numpy as np
        v = np.zeros(self.dim, np.float32)
        for w in re.findall(r"[a-z]+", text.lower()):
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v + 1.0 / self.dim ** 0.5


def test_facts_find_the_session_and_the_reader_gets_it(tmp_path):
    from bench.lme_facts import SessionFactCache
    from bench.scenarios import longmemeval as lme

    sessions = [("s1", "2023/01/10 (Tue) 10:00", "Yoga twice a week keeps me sane."),
                ("s2", "2023/05/14 (Sun) 10:00", "Dropped it. Pilates for me these days!")] + \
        [(f"d{i}", "2023/03/01 (Wed) 10:00", f"Recipe {i}: exercise caution with salt, "
          f"class.") for i in range(30)]

    def extract(messages):
        text = messages[1]["content"]
        if "Yoga" in text:
            v, d = "yoga", "2023/01/10"
        elif "Pilates" in text:
            v, d = "pilates", "2023/05/14"
        else:
            return "[]"
        return json.dumps([{"statement": f"On {d} the user's exercise class "
                            f"is {v}.", "date": d, "subject": "user",
                            "attribute": "exercise class", "value": v, "turn": 0}])

    class Reader:
        name = "fake"
        contexts = []

        def answer_full(self, question, context, question_date=""):
            self.contexts.append(context)
            return ("pilates" if "Pilates" in context else "unknown"), ""

        def verdict(self, question, gold, predicted):
            return {"correct": predicted == gold, "raw": ""}

    q = {"question_id": "ku1", "question_type": "knowledge-update",
         "question": "Which exercise class does the user attend now?",
         "answer": "pilates", "question_date": "2023/06/01 (Thu) 10:00",
         "haystack_session_ids": [s for s, _, _ in sessions],
         "haystack_dates": [d for _, d, _ in sessions],
         "haystack_sessions": [[{"role": "user", "content": c}] for _, _, c in sessions],
         "answer_session_ids": ["s2"]}
    kw = dict(embedder=_BagEmbedder(), judge=Reader(), top_k=2,
              multi_round="off")

    plain = lme.run([q], **kw, questions_path=tmp_path / "a.jsonl")
    cache = SessionFactCache(_Provider(extract), "fake", path=tmp_path / "f.jsonl")
    with_facts = lme.run([q], **kw, fact_cache=cache,
                         questions_path=tmp_path / "b.jsonl")
    rec = json.loads((tmp_path / "b.jsonl").read_text())
    assert plain["overall"] == 0.0 and with_facts["overall"] == 1.0
    ctx = Reader.contexts[-1]
    assert "Dropped it. Pilates for me these days!" in ctx          # the session itself
    assert "yoga (2023/01/10 – 2023/05/14) → pilates (2023/05/14 – now)" in ctx
    assert rec["fact_hits"] and rec["changes"]
    assert with_facts["fact_extraction_calls"] == 32
    # the per-question DB is closed and removed with its .wal/.lock
    import glob, tempfile, os
    assert not glob.glob(os.path.join(tempfile.gettempdir(), "*_ku1.feather*"))


def _event(date, entity, own=True, amount="", kind="event", stmt=None):
    return Fact(subject="user", predicate=entity, object=entity,
                metadata={"statement": stmt or f"On {date} {entity}.",
                          "date": date, "kind": kind, "entity": entity,
                          "amount": amount, "own": own})


def test_extractor_reads_kind_entity_amount_own():
    p = _Provider(json.dumps([
        {"statement": "On 2023/05/06 the user attended the wedding of Rachel "
         "and Mike.", "date": "2023/05/06", "subject": "user",
         "attribute": "wedding attended", "value": "Rachel and Mike",
         "turn": 0, "kind": "Event", "entity": "Wedding of Rachel & Mike",
         "amount": "", "own": True},
        {"statement": "x", "subject": "user", "attribute": "a", "value": "b",
         "turn": 0, "kind": "nonsense", "own": False}]))
    a, b = SessionFactExtractor(p).extract_session(TURNS, "2023/05/14")
    assert a.metadata["kind"] == "event" and a.metadata["own"] is True
    assert a.metadata["entity"] == "wedding of rachel mike"
    assert b.metadata["kind"] == "other" and b.metadata["own"] is False


def test_ledger_counts_an_event_once_and_marks_others():
    facts = [_event("2023/05/06", "wedding of rachel and mike"),
             _event("2023/05/06", "wedding of rachel and mike"),   # re-told
             _event("2023/06/10", "wedding of emily and sarah"),
             _event("2023/07/01", "wedding of my cousin", own=False),
             _event("2023/02/01", "bike chain replacement", amount="$25"),
             _event("2023/02/01", "tone", kind="preference")]
    rows = ledger(facts).splitlines()
    assert len(rows) == 4                       # 3 weddings + bike, no pref
    assert rows[0].startswith("- 2023/02/01 | bike chain replacement | $25")
    assert sum("wedding of rachel and mike" in r for r in rows) == 1
    assert "wedding of my cousin | NOT the user's own" in ledger(facts)
    assert "cousin" not in ledger(facts, own_only=True)


def test_profile_keeps_each_preference_at_its_latest_value():
    old = _event("2023/01/01", "drink", kind="preference",
                 stmt="On 2023/01/01 the user prefers tea.")
    new = _event("2023/04/01", "drink", kind="preference",
                 stmt="On 2023/04/01 the user prefers coffee.")
    chains = link_validity([old, new, _event("2023/02/01", "x")])
    out = profile(chains)
    assert out == "- On 2023/04/01 the user prefers coffee."


def test_count_question_gets_a_deduplicated_ledger(tmp_path):
    from bench.lme_facts import SessionFactCache
    from bench.scenarios import longmemeval as lme

    weddings = {"w1": ("2023/05/08 (Mon) 10:00", "rachel and mike", True),
                "w2": ("2023/06/12 (Mon) 10:00", "rachel and mike", True),
                "w3": ("2023/07/03 (Mon) 10:00", "emily and sarah", True),
                "w4": ("2023/07/20 (Thu) 10:00", "my coworker", False)}
    sessions = [(sid, d, f"Wedding talk: {who}.") for sid, (d, who, _)
                in weddings.items()] + [
        ("p1", "2023/03/01 (Wed) 10:00", "I always carry my power bank.")]

    def extract(messages):
        text = messages[1]["content"]
        for sid, (d, who, own) in weddings.items():
            if f"Wedding talk: {who}." in text and d[:10] in messages[1]["content"]:
                return json.dumps([{
                    "statement": f"On {d[:10]} the user attended the wedding "
                    f"of {who}.", "date": d[:10], "subject": "user",
                    "attribute": f"wedding of {who}", "value": "attended",
                    "turn": 0, "kind": "event", "entity": f"wedding of {who}",
                    "amount": "", "own": own}])
        if "power bank" in text:
            return json.dumps([{
                "statement": "On 2023/03/01 the user carries a power bank.",
                "date": "2023/03/01", "subject": "user",
                "attribute": "phone charging", "value": "power bank",
                "turn": 0, "kind": "preference", "entity": "", "own": True}])
        return "[]"

    class Reader:
        name = "fake"
        contexts = []

        def answer_full(self, question, context, question_date=""):
            self.contexts.append(context)
            return "2", ""

        def verdict(self, question, gold, predicted):
            return {"correct": predicted == gold, "raw": ""}

    q = {"question_id": "ms1", "question_type": "multi-session",
         "question": "How many weddings have I attended this year?",
         "answer": "2", "question_date": "2023/08/01 (Tue) 10:00",
         "haystack_session_ids": [s for s, _, _ in sessions],
         "haystack_dates": [d for _, d, _ in sessions],
         "haystack_sessions": [[{"role": "user", "content": c}]
                               for _, _, c in sessions],
         "answer_session_ids": ["w1", "w3"]}
    cache = SessionFactCache(_Provider(extract), "fake", path=tmp_path / "f.jsonl")
    lme.run([q], embedder=_BagEmbedder(), judge=Reader(), top_k=5,
            multi_round="off", fact_cache=cache,
            questions_path=tmp_path / "q.jsonl")
    rec = json.loads((tmp_path / "q.jsonl").read_text())
    ctx = Reader.contexts[-1]
    rows = rec["ledger"].splitlines()
    assert len(rows) == 3, rows
    assert ("2023/05/08 | wedding of rachel and mike | own | mentioned "
            "2023/05/08, 2023/06/12") in rec["ledger"]
    assert "wedding of my coworker | NOT the user's own" in rec["ledger"]
    assert "the user carries a power bank" in rec["profile"]
    assert ctx.index("[About the user") < ctx.index("[Event ledger") \
        < ctx.index("[Session ")
