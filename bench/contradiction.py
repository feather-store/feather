"""Contradiction eval: when a fact about someone changed, does Feather put the
CURRENT value in the context, and how often do stale values come with it?

    python -m bench.contradiction                     # keyword search, ~3 s, no keys
    python -m bench.contradiction --embed gemini      # dense search, cached vectors
    python -m bench.contradiction --model openrouter:openai/gpt-4o-mini   # + a reader

Each trial invents `--subjects` people, each with five attributes that changed
over time (city, employer, manager, diet, phone plan): 2-4 successive values,
the newest 1-90 days ago, each earlier one 30-365 days before the next, every
value a statement in one of four phrasings ("Alice moved to Denver", "Alice's
home city is Denver" ...). Plus `--distractors` benign memories per person,
some built to compete for the same queries ("Alice's sister lives in Austin",
"Alice flew to Lisbon for a conference"), none naming any value of that
person's attributes.

Ingestion order alternates by trial. `in order`: written oldest first, each
record stamped with its valid time. `shuffled`: the same statements written in
random order within the last few hours, as a backfill or an import would.

Each person's memories go in a scope of their own, written five ways, each on
a fresh DB:

    notes             Pocket.remember, a distinct key per statement, backdated
                      with valid time (Metadata.valid_at = when it became true)
    notes, no valid   the same without valid_at: only the write time, which in
      time            shuffled trials is not the order things happened
    keyed facts       AgentMemory(store, org, user=person).facts.state(attr,
                      text, at=valid time) over a FeatherStore: one record per
                      attribute, superseded values kept in its `supersedes`
    items + valid_at  FeatherStore.put(ns, key, {"text", "valid_at"}), a
                      distinct key per statement
    notes + keys      Pocket.remember(index=[IndexEntry(text, key="user|<attribute>",
                      value)]): the notes, each statement of an attribute in
                      that attribute's supersession chain (the bench's own
                      attribute: an oracle key). PacketBuilder shows a chain
                      once, newest first, older values as `earlier:` lines
                      (Pocket.group_superseded)
    notes + extracted Pocket.remember(index=facts), the facts a
      keys            SessionFactExtractor (--extract MODEL) drew from each
                      note as it was written, told the person's name and the
                      keys already on file (Pocket.known_keys): facts as
                      index, notes as payload, extracted supersession keys

Then one query per (person, attribute), "Which city does Alice live in now?",
through PacketBuilder(Pocket(scope)).build(query=..., k=--k) at --budget
tokens: keyword search (BM25) by default, dense search with --embed (768-d
gemini-embedding-001, every text embedded once into ~/.cache/feather/
contradiction/). Under --embed the FeatherStore variants index the `text`
field only (index_fields=["text"]): the default indexes the whole JSON value,
timestamps included, which no cache can hold, and it is also what renders a
keyed fact's history into its packet. Metrics, per query:

    current in     the current value appears in the packet
    current first  ... and before every stale value (packets render best first)
    stale          distinct stale values anywhere in the packet, history included
    stale records  packet records whose own statement is a stale value, i.e.
                   stale values presented as a statement rather than as history
    stale only     current missing, a stale value present: the packet argues
                   for the wrong answer
    held current   (keyed facts) the record's `text` is the newest value: did
                   facts.state put an out-of-order write in its history?
    tokens         packet size

With --model, packets are sent to a reader with the question (`--model-packets`
per variant and ordering): `correct` = the answer names the current value and no
stale one; `stale` = it names a stale value and not the current one.
`--repeats N` asks each packet N times (the reader is not deterministic).

Split (bench/protocol.py). `--split dev` (default) builds worlds from `--seed`
with the lexicon above: the worlds the 0.2.0-0.3.1 extractor prompts were
tuned on, so dev and never held out. `--split test` builds them from
TEST_SEED_BASE + seed with a held-out lexicon: five other attributes (car,
gym, course, bank, partner), other names, phrasings and distractors, which no
prompt was tuned on. New seeds alone would not be a test: a prompt tuned on
these phrasings fits any world drawn from them. Test needs a committed tree.

Every query is also a record in the report (`records`, and the reader's
`outcomes`), so two runs or two variants can be compared question by
question: `python -m bench.compare contradiction --a A.json --b B.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from feather_db import DB, keys
from feather_db.packet import PacketBuilder
from feather_db.pocket import Pocket, pocket

DAY = 86400.0
ORG = "acme"
NAMES = ["Alice", "Bruno", "Chen", "Dana", "Emeka", "Farah", "Goran", "Hana",
         "Ivan", "Jun", "Kofi", "Lena", "Mateo", "Noor", "Oskar", "Priya"]
OTHERS = ["Sam", "Rita", "Theo", "Uma", "Vik", "Wren"]

ATTRS = {
    "city": {
        "values": ["Denver", "Austin", "Lisbon", "Osaka", "Nairobi", "Toronto",
                   "Bergen", "Cusco", "Tbilisi", "Hobart", "Porto", "Leeds"],
        "say": ["{s} lives in {v}", "{s} moved to {v}",
                "{s} relocated to {v} and lives there now", "{s}'s home city is {v}"],
        "ask": "Which city does {s} live in now?",
        "ask_previous": "Which city did {s} live in before {v}?",
    },
    "employer": {
        "values": ["Acme Robotics", "Bluefin Labs", "Corvid Health", "Dunmore Bank",
                   "Elmstead Foods", "Fathom Analytics", "Granite Freight",
                   "Halcyon Games", "Ironleaf Energy", "Juniper Legal"],
        "say": ["{s} works at {v}", "{s} started a new job at {v}", "{s} joined {v}",
                "{s}'s employer is {v}"],
        "ask": "Where does {s} work now?",
        "ask_previous": "Where did {s} work before {v}?",
    },
    "manager": {
        "values": ["Priya Natarajan", "Tom Okafor", "Lucia Ferraro", "Ken Watanabe",
                   "Sara Lindqvist", "Omar Haddad", "Maya Cohen", "Rui Costa",
                   "Ines Moreau", "Dev Malhotra"],
        "say": ["{s}'s manager is {v}", "{s} now reports to {v}",
                "{v} became {s}'s manager", "{s} reports to {v}"],
        "ask": "Who is {s}'s manager now?",
        "ask_previous": "Who was {s}'s manager before {v}?",
    },
    "diet": {
        "values": ["vegan", "vegetarian", "pescatarian", "keto", "gluten-free",
                   "halal", "low-FODMAP", "paleo"],
        "say": ["{s} eats {v}", "{s} went {v}", "{s} follows a {v} diet", "{s} is {v} now"],
        "ask": "What diet does {s} follow now?",
        "ask_previous": "What diet did {s} follow before {v}?",
    },
    "phone": {
        "values": ["Verizon Unlimited Plus", "T-Mobile Go5G", "Mint 15GB", "Visible Plus",
                   "Google Fi Flexible", "AT&T Value", "Cricket Core", "US Mobile Starter"],
        "say": ["{s} is on the {v} phone plan", "{s} switched to {v}",
                "{s}'s phone plan is {v}", "{s} signed up for {v}"],
        "ask": "Which phone plan is {s} on now?",
        "ask_previous": "Which phone plan was {s} on before {v}?",
    },
}
# ── held-out lexicon (--split test): nothing above is reused ─────────────
NAMES_TEST = ["Aisha", "Bjorn", "Carmen", "Dmitri", "Elif", "Felipe", "Gita", "Hiro",
              "Ingrid", "Jamal", "Keiko", "Luis", "Mirela", "Nnamdi", "Olga", "Pablo"]
OTHERS_TEST = ["Bea", "Cyrus", "Dora", "Eli", "Fen", "Gus"]
ATTRS_TEST = {
    "car": {
        "values": ["Volvo XC40", "Kia Niro", "Subaru Outback", "Mazda CX-5", "Tesla Model 3",
                   "Honda Civic", "Ford Maverick", "Toyota Prius", "Hyundai Ioniq 5",
                   "Skoda Octavia"],
        "say": ["{s} drives a {v}", "{s} bought a {v}", "{s}'s car is a {v}",
                "{s} traded up to a {v}"],
        "ask": "What car does {s} drive now?",
        "ask_previous": "What car did {s} drive before the {v}?",
    },
    "gym": {
        "values": ["Peak Fitness", "IronWorks Gym", "Harbor Climbing", "Northside YMCA",
                   "Studio Flow Yoga", "Granite CrossFit", "Riverside Pool", "Summit Boxing"],
        "say": ["{s} trains at {v}", "{s} got a membership at {v}", "{s}'s gym is {v}",
                "{s} switched gyms to {v}"],
        "ask": "Which gym does {s} go to now?",
        "ask_previous": "Which gym did {s} go to before {v}?",
    },
    "course": {
        "values": ["Data Science Bootcamp", "Evening MBA", "Nursing Degree", "Pottery Course",
                   "UX Design Certificate", "Law Conversion Course", "Welding Apprenticeship",
                   "Japanese Language Course"],
        "say": ["{s} is enrolled in the {v}", "{s} started the {v}",
                "{s} signed up for the {v}", "{s} is taking the {v}"],
        "ask": "Which course is {s} taking now?",
        "ask_previous": "Which course was {s} taking before the {v}?",
    },
    "bank": {
        "values": ["Ally Bank", "Chase", "Monzo", "Revolut", "Schwab", "Wells Fargo",
                   "Capital One", "Starling"],
        "say": ["{s} banks with {v}", "{s} moved accounts to {v}", "{s}'s bank is {v}",
                "{s} opened an account at {v}"],
        "ask": "Which bank does {s} use now?",
        "ask_previous": "Which bank did {s} use before {v}?",
    },
    "partner": {
        "values": ["Rosa Diaz", "Ethan Park", "Mina Sato", "Leo Grant", "Ana Silva",
                   "Kai Moreno", "Zoe Clarke", "Ravi Shah"],
        "say": ["{s} is dating {v}", "{s} started seeing {v}", "{s}'s partner is {v}",
                "{s} is in a relationship with {v}"],
        "ask": "Who is {s}'s partner now?",
        "ask_previous": "Who was {s}'s partner before {v}?",
    },
}
DISTRACTORS_TEST = [
    ("{s}'s brother drives a {v}", "car"), ("{s} test-drove a {v} last week", "car"),
    ("{s} took a trial class at {v}", "gym"),
    ("{s}'s friend {o} is enrolled in the {v}", "course"),
    ("{s} read reviews of {v}", "bank"),
    ("{s}'s cousin is dating {v}", "partner"),
    ("{s} plays the cello", None), ("{s} grows tomatoes on the balcony", None),
    ("{s} volunteers at a food bank", None), ("{s} collects vinyl records", None),
    ("{s}'s favorite season is autumn", None), ("{s} prefers tea to coffee", None),
    ("{s} is saving for a trip to Iceland", None),
]
TEST_SEED_BASE = 1_000_000

# (template, value pool or None): benign, some built to compete for a query.
DISTRACTORS = [
    ("{s}'s sister lives in {v}", "city"), ("{s} flew to {v} for a conference", "city"),
    ("{s} had coffee with a recruiter from {v}", "employer"),
    ("{s}'s friend {o} works at {v}", "employer"),
    ("{s} met {v} at a workshop", "manager"),
    ("{s} cooked a {v} dinner for guests", "diet"),
    ("{s} asked about the {v} phone plan for a parent", "phone"),
    ("{s} likes jazz and long hikes", None), ("{s} adopted a cat named Miso", None),
    ("{s} is training for a half marathon", None), ("{s} is learning Portuguese", None),
    ("{s}'s birthday is in March", None), ("{s} prefers email to phone calls", None),
]
DEV_ATTRS = tuple(ATTRS)
# split -> (names, others, attributes, distractors). ATTRS holds both sets of
# attributes, so every lookup by attribute name works for either split.
LEXICON = {"dev": (NAMES, OTHERS, DEV_ATTRS, DISTRACTORS),
           "test": (NAMES_TEST, OTHERS_TEST, tuple(ATTRS_TEST), DISTRACTORS_TEST)}
ATTRS.update(ATTRS_TEST)

VARIANTS = ("notes", "notes, no valid time", "keyed facts", "items + valid_at",
            "notes + keys", "notes + extracted keys")
KEYED_NOTES = ("notes + keys", "notes + extracted keys")
ORDERS = ("in order", "shuffled")


# ── world ────────────────────────────────────────────────────────────────

def make_world(rng: random.Random, n_subjects: int, n_distract: int, now: float,
               split: str = "dev"):
    """(facts, writes). facts: [(subject, attr, [(value, valid_time, text)] oldest
    first)]. writes: every statement and distractor, as dicts. `split` picks
    the lexicon (LEXICON)."""
    names, others, attrs, distractors = LEXICON[split]
    facts, writes = [], []
    for s in rng.sample(names, n_subjects):
        used = set()
        for attr in attrs:
            spec = ATTRS[attr]
            n = rng.randint(2, 4)
            vals = rng.sample(spec["values"], n)
            used.update(vals)
            t = now - rng.uniform(1, 90) * DAY
            times = [t]
            for _ in range(n - 1):
                t -= rng.uniform(30, 365) * DAY
                times.append(t)
            hist = [(v, vt, rng.choice(spec["say"]).format(s=s, v=v))
                    for v, vt in zip(vals, reversed(times))]
            facts.append((s, attr, hist))
            writes += [{"subject": s, "attr": attr, "value": v, "vt": vt, "text": text}
                       for v, vt, text in hist]
        picks = []
        while len(picks) < n_distract:          # each template once before any repeats
            picks += rng.sample(distractors, len(distractors))
        for tmpl, pool in picks[:n_distract]:
            v = rng.choice([x for x in ATTRS[pool]["values"] if x not in used]) if pool else ""
            writes.append({"subject": s, "attr": None, "value": None,
                           "vt": now - rng.uniform(0, 400) * DAY,
                           "text": tmpl.format(s=s, v=v, o=rng.choice(others))})
    return facts, writes


def order_writes(rng: random.Random, writes: list, order: str, now: float) -> list:
    """Write order, each with `ts`, the time it was written."""
    if order == "in order":
        out = sorted(writes, key=lambda w: w["vt"])
        return [{**w, "ts": w["vt"]} for w in out]
    out = list(writes)
    rng.shuffle(out)
    return [{**w, "ts": now - (len(out) - i) * 30} for i, w in enumerate(out)]


# ── the four ways to write ───────────────────────────────────────────────

def _scope(variant: str, subject: str) -> tuple:
    kind = {"keyed facts": "facts", "items + valid_at": "items",
            "notes + keys": "keys", "notes + extracted keys": "xkeys"}.get(variant, "notes")
    return (ORG, subject, kind)


def write(variant: str, path: str, writes: list, dim: int, embed,
          extracted: dict | None = None):
    """Write `writes` the variant's way. Returns (db, store or None).
    `extracted`: (subject, text, valid time) -> the facts drawn from it."""
    if variant in KEYED_NOTES:
        from feather_db.pocket import IndexEntry
        db = DB.open(path, dim=dim)
        pockets = {}
        for i, w in enumerate(writes):
            sc = _scope(variant, w["subject"])
            pk = pockets.get(sc) or pockets.setdefault(sc, pocket(db, sc, embed=embed))
            if variant == "notes + keys":
                index = ([IndexEntry(w["text"], key=f"user|{w['attr']}", value=w["value"])]
                         if w["attr"] else [])
            else:
                index = extracted[(w["subject"], w["text"], int(w["vt"]))]
            pk.remember(f"n{i}", w["text"], valid_at=w["vt"], index=index)
            rid, m = keys.get(db, pk.namespace, f"n{i}")
            m.timestamp = int(w["ts"])
            db.update_metadata(rid, m)
        return db, None
    if variant.startswith("notes"):
        db = DB.open(path, dim=dim)
        pockets = {}
        for i, w in enumerate(writes):
            sc = _scope(variant, w["subject"])
            pk = pockets.get(sc) or pockets.setdefault(sc, pocket(db, sc, embed=embed))
            key = f"n{i}"
            pk.remember(key, w["text"])
            rid, m = keys.get(db, pk.namespace, key)
            m.timestamp = int(w["ts"])
            if variant == "notes":
                m.valid_at = int(w["vt"])
            db.update_metadata(rid, m)
        return db, None

    from feather_db.integrations import AgentMemory
    from feather_db.integrations.langgraph_store import FeatherStore
    # With --embed, index (and embed) the statement text only: the default
    # indexes the whole JSON value, timestamps included, which no cache can
    # hold. Without --embed the default is kept, and that default is what puts
    # a keyed fact's history into the packet.
    store = FeatherStore(path, dim=dim, embed=embed, auto_save=False,
                         index_fields=["text"] if embed else None)
    mems = {}
    for i, w in enumerate(writes):
        s = w["subject"]
        if variant == "keyed facts":
            mem = mems.get(s) or mems.setdefault(s, AgentMemory(store, ORG, user=s))
            mem.facts.state(w["attr"] or f"note_{i}", w["text"], at=w["vt"])
        else:
            store.put(_scope(variant, s), f"n{i}", {"text": w["text"], "valid_at": int(w["vt"])})
    return store.db, store


# ── measuring ────────────────────────────────────────────────────────────

def _pos(text: str, value: str):
    m = re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", text, re.I)
    return m.start() if m else None


def _stated(hist, text: str):
    """Which of hist's values a statement names (None for a distractor)."""
    for v, _, _ in hist:
        if _pos(text, v) is not None:
            return v
    return None


def _blank():
    return {"n": 0, "current_in": 0, "current_first": 0, "stale": 0, "stale_records": 0,
            "stale_only": 0, "held_current": 0, "held_n": 0, "tokens": 0}


def measure(variant: str, db, store, facts, *, k: int, budget: int, embed,
            rng: random.Random, out: dict, texts: list | None,
            question: str = "now", history: str = "inline",
            records: list | None = None, trial: int = 0, order: str = "",
            select: str = "rank"):
    """`question` "now" asks for the current value; "previous" for the one
    before it ("Which city did Alice live in before Leeds?"), which is then
    the value counted as right, and the older ones as wrong."""
    queries = list(facts)
    rng.shuffle(queries)
    for s, attr, hist in queries:
        sc = _scope(variant, s)
        pk = Pocket(db, sc, budget_tokens=budget, embed=embed)
        if question == "previous":
            ask = ATTRS[attr]["ask_previous"].format(s=s, v=hist[-1][0])
            hist = hist[:-1]              # the previous value is now the answer
        else:
            ask = ATTRS[attr]["ask"].format(s=s)
        p = PacketBuilder(pk, budget_tokens=budget, history=history,
                          select=select).build(query=ask, k=k)
        cur, stale = hist[-1][0], [v for v, _, _ in hist[:-1]]
        text, tokens = p.text, p.tokens_used
        if variant in KEYED_NOTES:
            # A stale value as a statement, not as an `earlier:` line.
            recs = sum(_stated(hist, line) in stale for line in text.splitlines()
                       if line.startswith("- "))
        pc = _pos(text, cur)
        ps = [x for v in stale if (x := _pos(text, v)) is not None]
        if variant not in KEYED_NOTES:
            recs = 0
            for r in p.refs:
                if store is not None:
                    item = store.get(tuple(r.scope), r.key)
                    content = item.value.get("text", "") if item else ""
                else:
                    content = keys.get(db, ".".join(r.scope), r.key)[1].content
                recs += _stated(hist, content) in stale
        m = out
        qid = f"t{trial}|{s}|{attr}"
        if records is not None:
            records.append({"qid": qid, "variant": variant, "order": order, "attr": attr,
                            "current_in": pc is not None,
                            "current_first": pc is not None and all(pc < x for x in ps),
                            "stale_only": pc is None and bool(ps),
                            "stale_values": len(ps), "tokens": tokens})
        m["n"] += 1
        m["current_in"] += pc is not None
        m["current_first"] += pc is not None and all(pc < x for x in ps)
        m["stale"] += len(ps)
        m["stale_records"] += recs
        m["stale_only"] += pc is None and bool(ps)
        m["tokens"] += tokens
        if variant == "keyed facts":
            item = store.get(sc, attr)
            m["held_n"] += 1
            m["held_current"] += bool(item) and _stated(hist, item.value["text"]) == cur
        if texts is not None:
            texts.append({"qid": qid, "variant": variant, "ask": ask, "text": text,
                          "current": cur, "stale": stale})


# ── embeddings (optional, cached) ────────────────────────────────────────

def _embedder(name: str, all_texts: list[str]):
    """embed(text) over a disk cache; every text is embedded up front, in
    batches, so the trials make no calls."""
    if name != "gemini":
        raise SystemExit(f"--embed {name}: only gemini is wired")
    from .providers_extra import GeminiEmbedder
    dim = 768
    path = (Path(os.environ.get("FEATHER_CACHE", Path.home() / ".cache" / "feather"))
            / "contradiction" / f"gemini-embedding-001_d{dim}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    h = lambda t: hashlib.sha1(t.encode()).hexdigest()           # noqa: E731
    cache = dict(np.load(path)) if path.exists() else {}
    missing = sorted({t for t in all_texts if h(t) not in cache})
    if missing:
        vecs = GeminiEmbedder(dim=dim).embed_batch(missing)
        cache.update({h(t): v for t, v in zip(missing, vecs)})
        np.savez(path, **cache)
        print(f"embedded {len(missing)} new texts (~{sum(map(len, missing)) // 4} tokens)",
              file=sys.stderr)

    def embed(text: str):
        v = cache.get(h(text))
        if v is None:
            raise KeyError(f"not pre-embedded: {text[:60]!r}")
        return v
    return embed, dim


# ── extraction (notes + extracted keys) ──────────────────────────────────

def extract_notes(trials: list, model: str, workers: int = 12) -> tuple[dict, dict]:
    """((subject, text, valid time) -> [Fact], usage) for every note, each
    extracted as it is written: per person in write order, one call per note,
    told the person's name and the one-value keys extracted for them so far
    (what Pocket.known_keys() returns once those notes are written). Cached
    on disk per (note, person, keys so far), so a rerun pays only where the
    model answered differently before."""
    import datetime as _dt
    from feather_db.extractors.base import Fact
    from feather_db.extractors.session_facts import SessionFactExtractor, fact_key, replaces
    from feather_db.pocket import _norm_key
    from .locomo import _Chat
    chat = _Chat(model, "extractor")

    class _Provider:
        def complete(self, messages, max_tokens=3000, temperature=0.0):
            return chat(messages, max_tokens=max_tokens, temperature=temperature)

    ex = SessionFactExtractor(_Provider())
    path = (Path(os.environ.get("FEATHER_CACHE", Path.home() / ".cache" / "feather"))
            / "contradiction" / f"{ex.name}-{ex.version}-{chat.model.replace('/', '_')}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if path.exists():
        for line in open(path):
            try:
                row = json.loads(line)
                cache[row["k"]] = row["facts"]
            except (ValueError, KeyError):
                continue
    lock = __import__("threading").Lock()

    def one(subject, text, vt, known):
        day = _dt.datetime.utcfromtimestamp(vt).strftime("%Y/%m/%d")
        ck = hashlib.sha1(json.dumps([text, subject, day, known]).encode()).hexdigest()
        if ck in cache:
            return [Fact(**f) for f in cache[ck]]
        errors = []
        facts = ex.extract_session([{"role": "user", "content": text}], day,
                                   context={"user_name": subject, "known_keys": known,
                                            "on_error": errors.append})
        if not errors:
            with lock, open(path, "a") as fh:
                fh.write(json.dumps({"k": ck, "facts": [f.__dict__ for f in facts]}) + "\n")
        return facts

    def person(seq):
        known, out = set(), {}
        for w in seq:
            facts = one(w["subject"], w["text"], w["vt"], sorted(known))
            known |= {_norm_key(fact_key(f)) for f in facts if replaces(f)}
            out[(w["subject"], w["text"], int(w["vt"]))] = facts
        return out

    seqs = []
    for ordered in trials:
        by = {}
        for w in ordered:
            by.setdefault(w["subject"], []).append(w)
        seqs += list(by.values())
    got = {}
    with ThreadPoolExecutor(workers) as pool:
        for part in pool.map(person, seqs):
            got.update(part)
    return got, {"model": chat.model, "extractor": f"{ex.name} {ex.version}",
                 "calls_tokens": dict(chat.tokens), "usd": chat.usd()}


# ── reader (optional) ────────────────────────────────────────────────────

READER_SYSTEM = "Answer the question using only the memory below. Give the value only."


def reader(texts: list, model: str, per: int, workers: int = 8, repeats: int = 1) -> dict:
    from bench.judges_llm import _provider_from_name, _retry
    name, _, mdl = model.partition(":")
    llm = _provider_from_name(name, mdl or None)
    chosen, taken = [], {}
    for t in texts:
        key = (t["variant"], t["order"])
        if taken.get(key, 0) < per:
            taken[key] = taken.get(key, 0) + 1
            chosen.append(t)

    def ask(t):
        reply = _retry(lambda: llm.complete(
            [{"role": "system", "content": READER_SYSTEM},
             {"role": "user", "content": f"Memory:\n{t['text'] or '(empty)'}\n\n"
                                         f"Question: {t['ask']}"}],
            max_tokens=30, temperature=0.0), label="reader")
        has_cur = _pos(reply or "", t["current"]) is not None
        has_stale = any(_pos(reply or "", v) is not None for v in t["stale"])
        return t, reply, has_cur and not has_stale, has_stale and not has_cur

    res, examples, outcomes = {}, {}, []
    with ThreadPoolExecutor(workers) as ex:
        for rep in range(1, repeats + 1):
            for t, reply, ok, stale in ex.map(ask, chosen):
                outcomes.append({"qid": t["qid"], "variant": t["variant"],
                                 "order": t["order"], "rep": rep,
                                 "correct": ok, "stale": stale})
                if rep > 1:
                    continue            # the totals are the first run's
                r = res.setdefault(t["variant"], {}).setdefault(
                    t["order"], {"sent": 0, "correct": 0, "stale": 0})
                r["sent"] += 1
                r["correct"] += ok
                r["stale"] += stale
                if stale:
                    examples.setdefault(f"{t['variant']}/{t['order']}",
                                        {"ask": t["ask"], "reply": reply, "packet": t["text"]})
    return {"model": model, "per": per, "repeats": repeats, "results": res,
            "stale_examples": examples, "outcomes": outcomes}


# ── run ──────────────────────────────────────────────────────────────────

def run(*, trials: int = 20, subjects: int = 8, distractors: int = 12, k: int = 10,
        budget: int = 1500, seed: int = 0, embed: str | None = None,
        model: str | None = None, model_packets: int = 40,
        variants: tuple = VARIANTS, extract: str | None = None,
        question: str = "now", history: str = "section",
        now: float | None = None, split: str = "dev", repeats: int = 1,
        select: str = "rank") -> dict:
    if split not in LEXICON:
        raise SystemExit(f"split must be dev or test, not {split!r}")
    if split == "test":
        from .protocol import require_committed
        require_committed("contradiction --split test")
    # Test worlds come from a seed range no dev run draws from.
    base = seed + (TEST_SEED_BASE if split == "test" else 0)
    if not embed:
        # FeatherStore falls back to FEATHER_EMBED_PROVIDER when embed=None;
        # keyword mode means keyword mode.
        os.environ.pop("FEATHER_EMBED_PROVIDER", None)
    # `now` anchors every date: the same `now` (and seed) rebuilds the same
    # worlds, so a rerun reuses the cached extractions, which are keyed by
    # each note's date.
    now = now if now is not None else time.time()
    worlds = [make_world(random.Random(base + t), subjects, distractors, now, split)
              for t in range(trials)]
    orders = [order_writes(random.Random(base + 1000 + t), writes, ORDERS[t % 2], now)
              for t, (_, writes) in enumerate(worlds)]
    extracted, extraction = None, None
    if "notes + extracted keys" in variants:
        if not extract:
            raise SystemExit("notes + extracted keys needs --extract provider:model")
        extracted, extraction = extract_notes(orders, extract)
    fn, dim = (None, 16)
    if embed:
        all_texts = [w["text"] for _, ws in worlds for w in ws] + [
            ATTRS[a]["ask"].format(s=s) for fs, _ in worlds for s, a, _ in fs] + [
            ATTRS[a]["ask_previous"].format(s=s, v=h[-1][0])
            for fs, _ in worlds for s, a, h in fs] + [
            f.metadata["statement"] for fs in (extracted or {}).values() for f in fs]
        fn, dim = _embedder(embed, all_texts)

    totals = {v: {o: _blank() for o in ORDERS} for v in variants}
    texts = [] if model else None
    records: list = []
    with tempfile.TemporaryDirectory() as d:
        for t, (facts, writes) in enumerate(worlds):
            order = ORDERS[t % 2]
            ordered = orders[t]
            for v in variants:
                path = os.path.join(d, f"t{t}_{VARIANTS.index(v)}.feather")
                db, store = write(v, path, ordered, dim, fn, extracted)
                try:
                    batch = [] if texts is not None else None
                    measure(v, db, store, facts, k=k, budget=budget, embed=fn,
                            rng=random.Random(base + 2000 + t), out=totals[v][order],
                            texts=batch, question=question, history=history,
                            records=records, trial=t, order=order, select=select)
                    if batch is not None:
                        texts += [{**x, "order": order} for x in batch]
                finally:
                    (store.close() if store is not None else db.close())
    report = {"params": {"trials": trials, "subjects": subjects, "distractors": distractors,
                         "k": k, "budget": budget, "seed": seed, "split": split,
                         "world_seed_base": base,
                         "attributes": list(LEXICON[split][2]),
                         "search": f"dense ({embed})" if embed else "keyword (BM25)",
                         "variants": list(variants), "question": question,
                         "history": history, "select": select, "now": now},
              "results": totals, "records": records}
    if extraction:
        report["extraction"] = extraction
    if model:
        report["reader"] = reader(texts, model, model_packets, repeats=repeats)
    return report


def table(report: dict) -> str:
    pct = lambda a, n: f"{100 * a / n:.0f}%" if n else "–"     # noqa: E731
    p = report["params"]
    rows = [f"search: {p['search']}, k={p['k']}, budget={p['budget']} tokens, "
            f"{p['trials']} trials x {p['subjects']} people x "
            f"{len(p.get('attributes') or DEV_ATTRS)} attributes, split {p.get('split', 'dev')}",
            "",
            "| variant | ingestion | queries | current in | current first | stale / packet "
            "| stale records / packet | stale only | held current | tokens / packet |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v, orders in report["results"].items():
        for o, m in orders.items():
            n = m["n"]
            rows.append(
                f"| {v} | {o} | {n} | {pct(m['current_in'], n)} | {pct(m['current_first'], n)} "
                f"| {m['stale'] / n:.2f} | {m['stale_records'] / n:.2f} "
                f"| {pct(m['stale_only'], n)} "
                f"| {pct(m['held_current'], m['held_n']) if m['held_n'] else 'n/a'} "
                f"| {m['tokens'] / n:.0f} |")
    rd = report.get("reader")
    if rd:
        rows += ["", f"Reader {rd['model']} ({rd['per']} packets per variant and ordering; "
                     "correct = names the current value and no stale one):", "",
                 "| variant | ingestion | sent | correct | stale answer |", "|---|---|---:|---:|---:|"]
        for v, orders in rd["results"].items():
            for o, r in orders.items():
                rows.append(f"| {v} | {o} | {r['sent']} | {pct(r['correct'], r['sent'])} "
                            f"| {pct(r['stale'], r['sent'])} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trials", type=int, default=20, help="alternating in order / shuffled")
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--distractors", type=int, default=12, help="benign memories per person")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed", default=None, help="gemini: dense search over cached vectors")
    ap.add_argument("--model", default=None,
                    help="provider:model reader (bench.judges_llm names), e.g. "
                         "openrouter:openai/gpt-4o-mini")
    ap.add_argument("--model-packets", type=int, default=40,
                    help="packets sent per variant and ordering")
    ap.add_argument("--variants", default=None,
                    help="comma-separated variants to run (default: all but "
                         "'notes + extracted keys' unless --extract is given)")
    ap.add_argument("--extract", default=None,
                    help="provider:model for 'notes + extracted keys' "
                         "(SessionFactExtractor), e.g. openrouter:openai/gpt-4o-mini")
    ap.add_argument("--question", choices=("now", "previous"), default="now",
                    help="ask for the current value, or the one before it")
    ap.add_argument("--now", type=float, default=None,
                    help="unix seconds every date is relative to (default: now); the "
                         "`now` of an earlier report reuses its cached extractions")
    ap.add_argument("--history", choices=("inline", "section", "none"), default="section",
                    help="PacketBuilder(history=...): where superseded memories go")
    ap.add_argument("--select", choices=("rank", "coverage"), default="rank",
                    help="PacketBuilder(select=...): fill the budget by rank, or by "
                         "information covered (feather_db.coverage)")
    ap.add_argument("--split", choices=("dev", "test"), default="dev",
                    help="dev: the tuned worlds and lexicon; test: held-out seeds and "
                         "lexicon (committed tree only)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="with --model: ask each packet N times (3-5 to compare)")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    a = ap.parse_args(argv)
    t0 = time.time()
    report = run(trials=a.trials, subjects=a.subjects, distractors=a.distractors, k=a.k,
                 budget=a.budget, seed=a.seed, embed=a.embed, model=a.model,
                 model_packets=a.model_packets, extract=a.extract,
                 question=a.question, history=a.history, now=a.now,
                 split=a.split, repeats=a.repeats, select=a.select,
                 variants=tuple(v.strip() for v in a.variants.split(","))
                 if a.variants else tuple(v for v in VARIANTS if a.extract
                                          or v != "notes + extracted keys"))
    report["wall_seconds"] = round(time.time() - t0, 2)
    print(table(report))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
