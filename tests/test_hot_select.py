"""Pocket.hot() by the engine (DB.heat_select) against the Python scan it
mirrors, which stays the specification: same items, same heat to the bit,
same order, on random scopes with everything that changes the answer."""
import importlib
import json
import random
import time

import numpy as np
import pytest

import feather_db
from feather_db import DB
from feather_db import origin as O

P = importlib.import_module("feather_db.pocket")

WORDS = ["note", "café", "naïve", "日本語", "emoji 😀", "a", "x" * 37, "tab\there"]


def _text(rng):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(0, 30)))


def _value(rng):
    """A FeatherStore value as stored: json.dumps of a dict, sometimes not
    one, sometimes with a text that is not a usable string."""
    v = {"kind": rng.choice(["fact", "note"]), "n": rng.random(),
         "nested": {"text": "not this one", "list": [1, "two", None, True]}}
    r = rng.random()
    if r < 0.5:
        v["text"] = _text(rng)
    elif r < 0.6:
        v["text"] = ""
    elif r < 0.7:
        v["text"] = 42
    elif r < 0.75:
        return json.dumps([v])                              # not an object
    elif r < 0.8:
        return '{"text": "dup"}'[:-1] + ', "text": "second wins"}'
    elif r < 0.85:
        return json.dumps(v)[:-1]                           # malformed
    s = json.dumps(v, ensure_ascii=rng.random() < 0.5)
    return s


def _populate(db, rng, scopes, n):
    now = time.time()
    b = feather_db.WriteBatch()
    keys = [f"k{i}" for i in range(n // 2)]                 # shared across scopes: shadowing
    for i in range(n):
        ns = ".".join(rng.choice(scopes))
        m = feather_db.Metadata()
        m.timestamp = int(now - rng.uniform(0, 200 * 86400))
        if rng.random() < 0.3:
            m.valid_at = int(now - rng.uniform(0, 400 * 86400))
        m.namespace_id = ns
        m.key = rng.choice(keys) + ("" if rng.random() < 0.8 else f"-{i}")
        m.source = rng.choice(["pocket"] * 8 + ["langgraph", "_pocket_index", "_pocket_cache",
                                                "_pocket_job"])
        m.content = _text(rng)
        m.importance = rng.choice([1.0, 0.5, rng.random(), 0.0, 1.5, -0.2])
        r = rng.random()
        if r < 0.1:
            m.set_attribute(P._PINNED, "true")
        elif r < 0.2:
            m.set_attribute(P.DUE_ATTR, repr(now + rng.uniform(-5, 5) * 86400))
            m.set_attribute(P.INTENTION_ATTR, "true")
        elif r < 0.22:
            m.set_attribute(P.DUE_ATTR, rng.choice(["nan", "soon", " 12.5 "]))
        if rng.random() < 0.5:
            m.set_attribute(O.ATTR, rng.choice(list(O.ORIGINS) + ["bogus"]))
        if m.source == "langgraph" or rng.random() < 0.1:
            m.set_attribute(P._LG_VALUE, _value(rng))
        b.upsert(m, np.asarray([rng.random() for _ in range(8)], dtype=np.float32))
    db.write(b)
    ids = [i for sc in scopes for i in db.ids_in_namespace(".".join(sc))]
    for i in rng.sample(ids, len(ids) // 3):                # recalls, some long ago
        for _ in range(rng.randint(1, 6)):
            db.touch(i)


def _same(a, b):
    assert [(i.id, i.key, i.heat, i.tokens, i.scope, i.origin) for i in a] == \
           [(i.id, i.key, i.heat, i.tokens, i.scope, i.origin) for i in b]


@pytest.mark.parametrize("seed", range(12))
def test_the_engine_picks_what_the_python_scan_picks(tmp_path, seed):
    rng = random.Random(seed)
    db = DB.open(str(tmp_path / "h.feather"), dim=8)
    scopes = [("org",), ("org", "brand"), ("org", "brand", "agent")]
    _populate(db, rng, scopes, 400)
    for budget in (50, 400, 4000, 10**6):
        for trusted in ({"user"}, {"user", "agent"}, set(O.ORIGINS)):
            for inherit in (True, False):
                pkt = P.Pocket(db, scopes[-1], budget_tokens=budget, inherit=inherit,
                               trusted_origins=trusted, half_life_days=rng.choice([1, 7, 30]))
                now = time.time() + rng.uniform(0, 30 * 86400)
                engine, _ = pkt._hot_select(budget, now)
                python, _ = pkt._hot_scan(budget, now)
                _same(engine, python)
                assert sum(i.tokens for i in engine) <= budget
    db.close()


def test_hot_uses_the_engine_and_still_answers_without_it(tmp_path, monkeypatch):
    rng = random.Random(7)
    db = DB.open(str(tmp_path / "h.feather"), dim=8)
    _populate(db, rng, [("a",), ("a", "b")], 200)
    pkt = P.Pocket(db, ("a", "b"))
    t = time.time()
    monkeypatch.setattr(P._clock, "now", lambda db: t)       # both answers at one moment
    calls = []
    real = type(db).heat_select
    monkeypatch.setattr(type(db), "heat_select", lambda self, *a, **k:
                        calls.append(1) or real(self, *a, **k))
    first = pkt.hot()
    assert calls == [1]
    monkeypatch.delattr(type(db), "heat_select")            # a DB without it
    P._bump(db)
    _same(pkt.hot(), first)
    db.close()


def test_a_store_items_text_is_measured_as_python_measures_it(tmp_path):
    db = DB.open(str(tmp_path / "t.feather"), dim=8)
    cases = ['{"text": "\\ud83d\\ude00 ok"}', '{"text": "\\ud83d alone"}',
             '{"text": "a\\u00e9b", "text": "last"}', '{"a": [1, 2.5e3, -0, {"b": null}], '
             '"text": "x"}', '{"text": "x"} trailing', '{"text": "tab\there"}',
             '  {"text" : "spaced" }  ', '{"text": "1."}', '[{"text": "no"}]',
             '{"text": NaN}', '{"x": -Infinity, "text": "inf ok"}', '{"x": 1e, "text": "y"}',
             '{"text": "\\x"}', '{"x": 01, "text": "y"}', '{"text": "\\/\\b\\f\\n\\r\\t"}']
    b = feather_db.WriteBatch()
    for i, raw in enumerate(cases):
        m = feather_db.Metadata()
        m.timestamp, m.namespace_id, m.key = int(time.time()), "s", f"c{i}"
        m.source, m.content = "langgraph", "content " * (i + 1)
        m.set_attribute(P._LG_VALUE, raw)
        b.upsert(m)
    db.write(b)
    pkt = P.Pocket(db, ("s",), budget_tokens=10**6, min_heat=0.0)
    t = time.time()
    picked = pkt._hot_select(10**6, t)[0]
    assert len(picked) == len(cases)
    _same(picked, pkt._hot_scan(10**6, t)[0])
    db.close()


def test_a_remote_hot_is_one_request_and_the_same_answer(tmp_path, monkeypatch):
    from tests.test_contracts import _remote
    with _remote(tmp_path, monkeypatch) as rdb:
        pkt = P.Pocket(rdb, ("org", "agent"))
        for i in range(60):
            pkt.remember(f"k{i}", f"note {i} " + "word " * (i % 7), pinned=i % 13 == 0)
        t = time.time()
        monkeypatch.setattr(P._clock, "now", lambda db: t)
        sent = []
        real = rdb._session.request
        rdb._session.request = lambda method, url, **kw: sent.append(url) or \
            real(method, url, **kw)
        P._bump(rdb)
        got = pkt.hot()
        # heat_select, then the picks' metadata: whatever the scope's size.
        assert len(sent) == 2 and sent[0].endswith("/heat_select"), sent
        _same(got, pkt._hot_scan(pkt.budget, t)[0])
