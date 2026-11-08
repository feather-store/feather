"""Facts as index, raw memory as payload, and supersession.

Store what was actually said; let extracted facts FIND it, and let one-value
facts mark what is current. Runs offline (keyword search, hand-written
entries); with an LLM, the entries come from SessionFactExtractor:

    from feather_db.extractors import SessionFactExtractor
    ex = SessionFactExtractor(provider)
    facts = ex.extract_session([{"role": "user", "content": text}], date,
                               context={"user_name": "Alice",
                                        "known_keys": pkt.known_keys()})
    pkt.remember(key, text, valid_at=t, index=facts)

    python3 examples/facts_as_index.py
"""
import os
import tempfile
import time

import feather_db
from feather_db import IndexEntry
from feather_db.packet import PacketBuilder

DAY = 86400
now = time.time()

with tempfile.TemporaryDirectory() as d:
    db = feather_db.DB.open(os.path.join(d, "alice.feather"), dim=8)
    pkt = feather_db.Pocket(db, ("acme", "alice"))

    # Raw turns are the payload. The index entry is what an extractor drew
    # from each: searchable, never shown. A `key` marks a one-value attribute.
    pkt.remember("t1", "Alice: we finally unpacked, Bergen is gorgeous in spring",
                 valid_at=now - 300 * DAY,
                 index=[IndexEntry("Alice moved to Bergen", key="user|home city",
                                   value="Bergen")])
    pkt.remember("t2", "Alice: new flat in Leeds, closer to my sister",
                 valid_at=now - 20 * DAY,
                 index=[IndexEntry("Alice lives in Leeds", key="user|home city",
                                   value="Leeds")])
    pkt.remember("t3", "Alice: the kids want a dog named Biscuit",
                 valid_at=now - 5 * DAY,
                 index=["Alice's kids want a dog"])          # findable, no key

    # "moved" matches only the old turn's index entry; recall brings the
    # current value in and shows the old one as its history.
    for item in pkt.recall("where did Alice move", k=3):
        print(item.key, "->", item.content)
        for e in item.earlier:
            print("   earlier:", e.key, "->", e.content)

    print()
    print(PacketBuilder(pkt, budget_tokens=400).build(query="where did Alice move").text)
    print()
    print("keys on file:", pkt.known_keys())
    db.close()
