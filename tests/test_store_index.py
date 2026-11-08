"""Facts as index and supersession through FeatherStore and AgentMemory
(FeatherStore.index, AgentMemory.notes, AgentMemory.ingest_session), and
PacketBuilder(history=...)."""
import json
import sys
import time
import zlib

import numpy as np
import pytest

pytest.importorskip("langgraph", reason="needs langgraph")

from feather_db.extractors import SessionFactExtractor                  # noqa: E402
from feather_db.integrations.agent_memory import AgentMemory           # noqa: E402
from feather_db.integrations.langgraph_store import FeatherStore       # noqa: E402

P = sys.modules["feather_db.pocket"]
DAY = 86400


def embed(text: str, dim: int = 64):
    """Bag of words with a stable hash: overlap decides the ranking."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in str(text).lower().replace("?", " ").replace(".", " ").split():
        v[zlib.crc32(tok.encode()) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture
def store(tmp_path):
    s = FeatherStore(str(tmp_path / "s.feather"), dim=64, embed=embed, auto_save=False)
    yield s
    s.close()


NS = ("acme", "alice", "notes")


def _city(store, key, text, value, days_ago, now):
    store.put(NS, key, {"text": text, "valid_at": now - days_ago * DAY})
    store.index(NS, key, [P.IndexEntry(text, key="user|home city", value=value,
                                       valid_at=now - days_ago * DAY)])


def test_an_index_hit_returns_the_item_and_index_records_are_not_items(store):
    store.put(NS, "t1", {"text": "Gonna see the doc about my knee tomorrow"})
    store.index(NS, "t1", ["Alice has a knee injury and sees an orthopedist"])
    store.put(NS, "t2", {"text": "The kids loved the pottery class"})
    hits = store.search(NS, query="orthopedist injury", limit=1)
    assert [h.key for h in hits] == ["t1"]
    assert hits[0].value["text"].startswith("Gonna see the doc")
    # listing, count, newest and get see items only
    assert store.count(NS) == 2
    assert {i.key for i in store.search(NS, limit=10)} == {"t1", "t2"}
    assert {i.key for i in store.newest(NS, 10)} == {"t1", "t2"}
    assert store.known_keys(NS) == []


def test_store_search_groups_a_chain_with_history_in_earlier(store):
    now = time.time()
    _city(store, "n1", "Alice lives in Bergen", "Bergen", 300, now)
    _city(store, "n2", "Alice's home city is Leeds", "Leeds", 10, now)
    hits = store.search(NS, query="Bergen", limit=1)
    assert [h.key for h in hits] == ["n2"]                 # the current one, brought in
    assert [e["text"] for e in hits[0].value["earlier"]] == ["Alice lives in Bergen"]
    assert store.known_keys(NS) == ["user|home city"]


def test_rewrite_keeps_and_delete_takes_the_index(store):
    store.put(NS, "t1", {"text": "first"})
    store.index(NS, "t1", ["zebra crossing"])
    store.put(NS, "t1", {"text": "second"})                 # index kept
    assert [h.value["text"] for h in store.search(NS, query="zebra crossing", limit=1)] \
        == ["second"]
    store.delete(NS, "t1")
    assert store.search(NS, query="zebra crossing", limit=3) == []
    assert store.count(NS) == 0
    with pytest.raises(KeyError):
        store.index(NS, "missing", ["x"])


def test_notes_keep_the_words_and_recall_finds_them_by_their_facts(store):
    mem = AgentMemory(store, org="acme", user="alice")
    k = mem.notes.add("ugh, the knee again. physio says six more weeks",
                      index=["Alice is doing physiotherapy for a knee injury"])
    got = mem.recall("physiotherapy knee injury", kinds=["notes"], limit=1)
    assert got[0]["text"].startswith("ugh, the knee again") and got[0]["kind"] == "notes"
    assert mem.summary()["notes"] == 1 and mem.notes.get(k) is not None


def test_ingest_session_stores_turns_indexed_by_their_facts(store):
    reply = json.dumps([
        {"statement": "On 2023/05/14 Alice moved to Denver.", "date": "2023/05/14",
         "subject": "user", "attribute": "home city", "value": "Denver",
         "one_value": True, "turn": 0, "kind": "state"},
        {"statement": "On 2023/05/14 Alice likes hiking.", "date": "2023/05/14",
         "subject": "user", "attribute": "hobby", "value": "hiking", "turn": 2,
         "kind": "preference"}])
    asked = []

    class Provider:
        def complete(self, messages, max_tokens=3000, temperature=0.0):
            asked.append(messages[1]["content"])
            return reply

    mem = AgentMemory(store, org="acme", user="alice")
    turns = [{"role": "user", "content": "we finally made it to the new place!"},
             {"role": "assistant", "content": "Congrats! How is it?"},
             {"role": "user", "content": "great trails nearby"}]
    got = mem.ingest_session(turns, "2023/05/14", SessionFactExtractor(Provider()),
                             session_id="s1", user_name="Alice")
    assert got == {"turns": 3, "facts": 2, "one_value": 1}
    assert "The user is Alice." in asked[0]
    assert mem.notes.known_keys() == ["user|home city"]
    hit = mem.recall("moved to Denver", kinds=["notes"], limit=1)[0]
    assert hit["text"] == "user: we finally made it to the new place!"
    assert mem.notes.get("s1#1")["origin"] == "agent"
    # a second session is told the keys on file; re-ingesting rewrites
    mem.ingest_session(turns, "2023/06/01", SessionFactExtractor(Provider()),
                       session_id="s1")
    assert "- user | home city" in asked[1]
    assert mem.summary()["notes"] == 3


def test_packet_history_inline_section_or_none(tmp_path):
    import feather_db
    from feather_db.packet import PacketBuilder
    db = feather_db.DB.open(str(tmp_path / "p.feather"), dim=8)
    pkt, now = P.Pocket(db, ("org", "alice")), time.time()
    for key, text, value, ago in (("n1", "Alice lives in Bergen", "Bergen", 300),
                                  ("n2", "Alice's home city is Leeds", "Leeds", 10)):
        pkt.remember(key, text, valid_at=now - ago * DAY,
                     index=[P.IndexEntry(text, key="user|home city", value=value)])
    inline = PacketBuilder(pkt, budget_tokens=400, history="inline").build(
        query="Alice Bergen").text
    assert "    earlier: [" in inline
    section = PacketBuilder(pkt, budget_tokens=400).build(query="Alice Bergen").text
    assert "earlier:" not in section
    assert section.index("EARLIER VALUES") > section.index("Leeds") \
        and section.index("Bergen") > section.index("EARLIER VALUES")
    none = PacketBuilder(pkt, budget_tokens=400, history="none").build(
        query="Alice Bergen").text
    assert "Leeds" in none and "Bergen" not in none
    with pytest.raises(ValueError):
        PacketBuilder(pkt, history="sideways")
    db.close()
