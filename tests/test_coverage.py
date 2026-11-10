"""Coverage selection (feather_db.coverage, PacketBuilder(select="coverage")):
what is chosen is information, so a fact told twice is chosen once."""
import itertools
import random

import pytest

import feather_db
from feather_db import coverage as C
from feather_db.packet import PacketBuilder
from feather_db.pocket import IndexEntry, Pocket


def _value(rel, units, chosen, tau=0.7):
    """F(S) computed from scratch, the way the docstring defines it."""
    reps, groups = [], []
    for us in units:
        g = set()
        for u in us:
            for c, r in enumerate(reps):
                if C._jaccard(u, r) >= tau:
                    break
            else:
                reps.append(u)
                c = len(reps) - 1
            g.add(c)
        groups.append(g)
    w = [0.0] * len(reps)
    for i, g in enumerate(groups):
        for c in g:
            w[c] = max(w[c], rel[i] / len(g))
    return sum(w[c] for c in set().union(*[groups[i] for i in chosen])) if chosen else 0.0


def test_a_repeated_fact_is_chosen_once():
    texts = ["Alice lives in Denver", "alice lives in denver!", "Bob owns a red bike",
             "Carol plays chess on Sundays"]
    keep = C.select([1.0, 0.95, 0.5, 0.4], [1, 1, 1, 1],
                    [C.units_of([t]) for t in texts], budget=10, max_items=3)
    assert keep == [0, 2, 3]                    # the duplicate gave way to new facts


def test_rank_order_is_kept_and_limits_hold():
    rng = random.Random(1)
    words = [f"w{i}" for i in range(30)]
    for _ in range(200):
        n = rng.randint(1, 12)
        units = [[frozenset(rng.sample(words, rng.randint(1, 5)))
                  for _ in range(rng.randint(1, 3))] for _ in range(n)]
        rel = [rng.random() for _ in range(n)]
        cost = [rng.randint(1, 20) for _ in range(n)]
        budget, k = rng.randint(5, 80), rng.randint(1, 6)
        keep = C.select(rel, cost, units, budget=budget, max_items=k)
        assert keep == sorted(keep) and len(keep) <= k
        assert sum(cost[i] for i in keep) <= budget


def test_the_greedy_is_near_the_optimum():
    """Khuller-Moss-Naor: the better of density greedy and the best single
    item is within (1 - 1/e)/2 of the optimum; checked exhaustively."""
    rng = random.Random(2)
    words = [f"w{i}" for i in range(12)]
    worst = 1.0
    for _ in range(300):
        n = rng.randint(2, 8)
        units = [[frozenset(rng.sample(words, rng.randint(1, 3)))
                  for _ in range(rng.randint(1, 3))] for _ in range(n)]
        rel = [rng.random() for _ in range(n)]
        cost = [rng.randint(1, 10) for _ in range(n)]
        budget = rng.randint(3, 25)
        got = _value(rel, units, C.select(rel, cost, units, budget=budget, max_items=n))
        best = max(_value(rel, units, s) for r in range(n + 1)
                   for s in itertools.combinations(range(n), r)
                   if sum(cost[i] for i in s) <= budget)
        if best > 0:
            worst = min(worst, got / best)
    assert worst >= (1 - 1 / 2.718281828) / 2, worst


@pytest.fixture
def pkt(tmp_path):
    db = feather_db.DB.open(str(tmp_path / "c.feather"), dim=8)
    p = Pocket(db, ("org", "u1"), budget_tokens=4000)
    yield p
    db.close()


def test_a_packet_by_coverage_leaves_out_what_is_already_said(pkt):
    for s in range(3):                           # one fact, three sessions
        pkt.remember(f"s{s}", f"session {s}: chatted. Alice lives in Denver now.",
                     index=[IndexEntry("Alice lives in Denver")])
    pkt.remember("bike", "Alice bought a bike.", index=[IndexEntry("Alice owns a bike")])
    pkt.remember("job", "Alice works at a bakery.", index=[IndexEntry("Alice works at a bakery")])
    ask = "where does Alice live, Denver?"
    rank = PacketBuilder(pkt).build(query=ask, k=3)
    cover = PacketBuilder(pkt, select="coverage").build(query=ask, k=3)
    said = lambda p: sum("lives in Denver" in line for line in p.text.splitlines())
    assert said(rank) == 3                       # three slots for one fact
    assert said(cover) == 1                      # once, and room for the rest
    assert "bike" in cover.text and "bakery" in cover.text
    assert cover.tokens_used <= rank.tokens_used


def test_select_is_checked(pkt):
    with pytest.raises(ValueError):
        PacketBuilder(pkt, select="best")


def test_locomo_coverage_gives_a_repeated_turn_one_slot():
    import datetime as dt
    import numpy as np
    from bench import locomo
    day = dt.datetime(2023, 5, 1)
    said = "I adopted a dog named Rex last week."
    turns = ([{"content": said, "speaker": "A", "dia_id": f"D1:{i}", "dt": day} for i in range(4)]
             + [{"content": f"Unrelated chat number {i} about the weather.", "speaker": "B",
                 "dia_id": f"D2:{i}", "dt": day} for i in range(6)]
             + [{"content": "My dog Rex loves the park by the river.", "speaker": "A",
                 "dia_id": "D3:1", "dt": day}])
    qs = [{"question": "What is the name of the dog adopted?"}]
    # Random vectors carry no meaning, so BM25 must find the turns. In 8
    # dimensions their chance cosines (sd ~0.35) outweigh it under score
    # fusion; at 1024 (sd ~0.03) they are the noise a real embedder's are not.
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((len(turns) + 1, 1024)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sample = {"sample_id": "s1"}
    rank = locomo.retrieve(sample, turns, qs, vecs, 3)[0]
    cover = locomo.retrieve(sample, turns, qs, vecs, 3, select="coverage")[0]
    assert sum(turns[i]["content"] == said for i in rank) >= 2
    assert sum(turns[i]["content"] == said for i in cover) == 1
    assert len(cover) == 3
