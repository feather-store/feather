"""scoring= ranks by cosine x freshness x importance, and recalls fade.

The previous score was (1-w) * 1/(1+L2^2) + w * recency. For unit vectors
1/(1+L2^2) is 1/(3-2cos), so with the defaults a record at cos 0.5 written
today tied one at cos 0.8 from 30 days ago, and for vectors that are not unit
length the similarity term was ~0 and recency ranked alone. And because
ln(1+recall_count) never decreased, a record recalled 100 times a year ago
stayed ~1000x hotter than an unused one of the same age, for good.
"""
import math
import time

import numpy as np
import pytest

import feather_db
from feather_db import DB
from feather_db.pocket import Pocket, effective_recalls

DIM = 8
DAY = 86400
CFG = feather_db.ScoringConfig(half_life=30.0, weight=0.3, min=0.0)


def at_cos(c, axis=1, scale=1.0):
    """A vector at cosine `c` to e0, leaning toward `axis`."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[axis] = c, math.sqrt(1 - c * c)
    return v * scale


def query(scale=1.0):
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = scale
    return q


def meta(age_days, recalls=0, recalled_days_ago=None, importance=1.0):
    m = feather_db.Metadata()
    now = int(time.time())
    m.timestamp = now - int(age_days * DAY)
    m.importance = importance
    m.recall_count = recalls
    if recalled_days_ago is not None:
        m.last_recalled_at = now - int(recalled_days_ago * DAY)
    return m


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "s.feather"), dim=DIM)
    yield d
    d.close()


def ranked(db, q, **kw):
    return [r.id for r in db.search(q, k=10, scoring=CFG, record_salience=False, **kw)]


def test_a_month_of_age_does_not_outweigh_relevance(db):
    db.add(1, at_cos(0.5, axis=1), meta(age_days=0))
    db.add(2, at_cos(0.8, axis=2), meta(age_days=30))
    assert ranked(db, query()) == [2, 1]          # tied at 0.65 before


def test_the_score_is_cosine_times_freshness_times_importance(db):
    db.add(1, at_cos(0.8), meta(age_days=30, importance=0.5))
    (r,) = db.search(query(), k=1, scoring=CFG, record_salience=False)
    freshness = 0.7 + 0.3 * 0.5                    # one half-life old
    assert r.score == pytest.approx(0.8 * freshness * 0.5, rel=1e-3)


def test_vector_length_does_not_change_the_ranking(db, tmp_path):
    """Same records at 10x length: same cosines, same scores, same order."""
    other = DB.open(str(tmp_path / "scaled.feather"), dim=DIM)
    try:
        for d, scale in ((db, 1.0), (other, 10.0)):
            d.add(1, at_cos(0.9, axis=1, scale=scale), meta(age_days=60))
            d.add(2, at_cos(0.6, axis=2, scale=scale), meta(age_days=0))
            d.add(3, at_cos(0.3, axis=3, scale=scale), meta(age_days=0))
        unit = db.search(query(), k=3, scoring=CFG, record_salience=False)
        long = other.search(query(10.0), k=3, scoring=CFG, record_salience=False)
        assert [r.id for r in unit] == [r.id for r in long] == [1, 2, 3]
        assert [r.score for r in long] == pytest.approx([r.score for r in unit], rel=1e-4)
    finally:
        other.close()


def test_an_unrelated_record_is_not_lifted_by_recency_or_importance(db):
    db.add(1, at_cos(-0.5), meta(age_days=0, importance=1.0))
    db.add(2, at_cos(0.1), meta(age_days=3650, importance=0.2))
    assert ranked(db, query()) == [2, 1]


def test_filtered_and_hybrid_search_score_the_same_way(db):
    for i, (c, age) in enumerate([(0.5, 0), (0.8, 30)], start=1):
        m = meta(age_days=age)
        m.namespace_id, m.content = "ns", "alpha"
        db.add(i, at_cos(c, axis=i), m)
    f = feather_db.FilterBuilder().namespace("ns").build()
    assert ranked(db, query(), filter=f) == [2, 1]
    hits = db.hybrid_search(query(), "alpha", k=2, scoring=CFG)
    assert [h.id for h in hits] == [2, 1]


# ── recalls fade ─────────────────────────────────────────────────────────

def test_recalls_a_year_old_no_longer_count(db):
    """Same cosine, same age: one recalled 100 times a year ago, one never."""
    db.add(1, at_cos(0.8, axis=1), meta(age_days=365, recalls=100, recalled_days_ago=365))
    db.add(2, at_cos(0.8, axis=2), meta(age_days=365))
    db.add(3, at_cos(0.8, axis=3), meta(age_days=365, recalls=100, recalled_days_ago=1))
    scores = {r.id: r.score for r in db.search(query(), k=3, scoring=CFG,
                                                record_salience=False)}
    assert scores[1] == pytest.approx(scores[2], rel=1e-3)
    assert scores[3] > scores[2] + 0.05            # recent use still counts


def test_pocket_heat_fades_old_recalls_too(tmp_path):
    db = DB.open(str(tmp_path / "h.feather"), dim=DIM)
    try:
        p = Pocket(db, "agent")
        stale = meta(age_days=365, recalls=100, recalled_days_ago=365)
        unused = meta(age_days=365)
        recent = meta(age_days=365, recalls=100, recalled_days_ago=1)
        assert p.heat(stale) < p.min_heat          # not carried (raw count: 0.0016)
        assert p.heat(stale) < 3 * p.heat(unused)  # raw count: ~10^13 x
        assert p.heat(recent) > 100 * p.heat(unused)
    finally:
        db.close()


def test_a_count_without_a_recall_time_fades_from_the_record_time():
    m = meta(age_days=60, recalls=8)
    assert effective_recalls(m, time.time()) == pytest.approx(2.0, rel=1e-3)   # H = 30 d


# ── the count itself is a moving count ───────────────────────────────────

def _touched(db, m):
    db.add(1, at_cos(0.8), m)
    db.touch(1)
    return db.get_metadata(1).recall_count


def test_one_new_recall_does_not_bring_back_an_old_count(db):
    """100 recalls a year ago + 1 now is ~1 recent recall (exact 1.02), not
    101 faded from now as if all had happened now (~99)."""
    assert _touched(db, meta(age_days=400, recalls=100, recalled_days_ago=365)) == 1


def test_a_recall_fades_the_count_by_the_gap_before_adding_one(db):
    # round(100 * 0.5^(1/30)) + 1 = round(97.7) + 1
    assert _touched(db, meta(age_days=30, recalls=100, recalled_days_ago=1)) == 99


def test_a_count_without_a_recall_time_fades_from_the_record_time_on_recall(db):
    # round(8 * 0.5^(60/30)) + 1
    assert _touched(db, meta(age_days=60, recalls=8)) == 3


def test_recalls_close_together_count_one_each(db):
    db.add(1, at_cos(0.8), meta(age_days=0))
    for _ in range(50):
        db.touch(1)
    assert db.get_metadata(1).recall_count == 50


@pytest.mark.parametrize("refine", [0, 2])
def test_an_int8_modality_scores_like_float(db, tmp_path, refine):
    """|x|^2 comes from the int8 row: the cosine is off by quantization only."""
    q8 = DB.open(str(tmp_path / "q8.feather"), dim=DIM)
    try:
        rows = [(1, at_cos(0.9, axis=1), 60), (2, at_cos(0.6, axis=2), 0),
                (3, at_cos(0.3, axis=3), 0), (4, at_cos(0.75, axis=4), 10)]
        for d in (db, q8):
            for i, v, age in rows:
                d.add(i, v, meta(age_days=age))
        q8.set_int8_ram("text", 1.0, refine)        # fixed clip: unit vectors
        assert q8.is_int8_ram("text")
        f = db.search(query(), k=4, scoring=CFG, record_salience=False)
        i8 = q8.search(query(), k=4, scoring=CFG, record_salience=False)
        assert [r.id for r in i8] == [r.id for r in f]
        assert [r.score for r in i8] == pytest.approx([r.score for r in f], abs=2e-2)
    finally:
        q8.close()


# ── the count is a moving count ──────────────────────────────────────────

def test_one_new_recall_does_not_revive_an_old_count(db):
    """100 recalls a year ago plus one today is about one recent recall, not
    101: each recall fades the history before adding itself."""
    db.add(1, at_cos(0.8, axis=1), meta(age_days=400, recalls=100, recalled_days_ago=365))
    db.touch(1)
    assert db.get_metadata(1).recall_count == 1


def test_recalls_close_together_each_count(db):
    db.add(1, at_cos(0.8), meta(age_days=1))
    for _ in range(5):
        db.touch(1)
    assert db.get_metadata(1).recall_count == 5


def test_the_moving_count_survives_a_reopen(tmp_path):
    path = str(tmp_path / "m.feather")
    d = DB.open(path, dim=DIM)
    d.add(1, at_cos(0.8), meta(age_days=1))
    d.touch(1); d.touch(1)
    before = d.get_metadata(1)
    d.close()
    d = DB.open(path, dim=DIM)
    try:
        after = d.get_metadata(1)
        assert (after.recall_count, after.last_recalled_at) == \
               (before.recall_count, before.last_recalled_at)
    finally:
        d.close()


def test_context_chain_fades_old_recalls(db):
    db.add(1, at_cos(0.8, axis=1), meta(age_days=365, recalls=100, recalled_days_ago=365))
    db.add(2, at_cos(0.8, axis=2), meta(age_days=365))
    nodes = {n.id: n.score for n in db.context_chain(query(), k=2, hops=0).nodes}
    assert nodes[1] == pytest.approx(nodes[2], rel=1e-2)


# ── age is measured from when it happened ────────────────────────────────

def test_freshness_comes_from_valid_time(db):
    """Both written today; one is about something a year old."""
    old_news = meta(age_days=0)
    old_news.valid_at = int(time.time()) - 365 * DAY
    db.add(1, at_cos(0.8, axis=1), old_news)
    db.add(2, at_cos(0.8, axis=2), meta(age_days=0))
    assert ranked(db, query()) == [2, 1]


def test_pocket_heat_ages_from_valid_time_too(tmp_path):
    db = DB.open(str(tmp_path / "v.feather"), dim=DIM)
    try:
        p = Pocket(db, "agent")
        old_news = meta(age_days=0)
        old_news.valid_at = int(time.time()) - 365 * DAY
        assert p.heat(old_news) < p.heat(meta(age_days=0)) / 100
    finally:
        db.close()


# ── the scored pool widens until the top k is exact ───────────────────────

def _stale_near_fresh_far(db):
    """Twelve stale records nearest the query, one fresh record just behind
    them: the fresh one has the best score but is 13th by distance, outside
    the old fixed pools for k=1 (3 for search, 9 for hybrid's vector half)."""
    for i in range(12):
        db.add(i + 1, at_cos(0.95, axis=1 + i % 6), meta(age_days=365))
    db.add(99, at_cos(0.90, axis=7), meta(age_days=0))


def test_scored_search_reaches_past_the_first_pool(db):
    _stale_near_fresh_far(db)
    (top,) = db.search(query(), k=1, scoring=CFG, record_salience=False)
    assert top.id == 99


def test_scored_hybrid_reaches_past_the_first_pool(db):
    _stale_near_fresh_far(db)
    assert db.hybrid_search(query(), "", k=1, scoring=CFG)[0].id == 99


def test_one_salience_half_life_for_every_reader(tmp_path):
    """Pocket (recency half-life 7 d) and scored search (30 d) credit a record's
    use alike: recalls fade at the engine's one H whoever reads them."""
    import feather_db.core as core
    m = meta(age_days=10, recalls=40, recalled_days_ago=15)
    expect = 40 * 0.5 ** (15 / core.RECALL_HALF_LIFE_DAYS)
    assert effective_recalls(m, time.time()) == pytest.approx(expect, rel=1e-3)
    db = DB.open(str(tmp_path / "h.feather"), dim=DIM)
    try:
        assert Pocket(db, "a", half_life_days=7.0).half_life == 7.0   # recency only
    finally:
        db.close()


# ── relevance is measured from the model's background cosine ─────────────

def test_the_background_is_the_median_of_random_pairs(tmp_path):
    rng = np.random.default_rng(3)
    dim = 64
    base = np.zeros(dim, dtype=np.float32); base[0] = 1.0
    v = base + rng.standard_normal((300, dim)).astype(np.float32) * 0.13
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    db = DB.open(str(tmp_path / "bg.feather"), dim=dim)
    try:
        db.add_batch(list(range(1, 301)), v)
        sims = v @ v.T
        truth = float(np.median(sims[np.triu_indices(300, 1)]))
        assert db.background_cosine() == pytest.approx(truth, abs=0.02)
        assert db.background_cosine("no-such-modality") == 0.0
    finally:
        db.close()


def test_a_small_collection_scores_by_plain_cosine(db):
    """Under 64 vectors the background cannot be told: floor 0, plain cosine."""
    db.add(1, at_cos(0.8), meta(age_days=30))
    assert db.background_cosine() == 0.0
    (r,) = db.search(query(), k=1, scoring=CFG, record_salience=False)
    assert r.score == pytest.approx(0.8 * (0.7 + 0.3 * 0.5), rel=1e-3)


def test_an_explicit_floor_is_used_as_given(db):
    db.add(1, at_cos(0.8), meta(age_days=0))
    cfg = feather_db.ScoringConfig(30.0, 0.3, 0.0, relevance_floor=0.5)
    (r,) = db.search(query(), k=1, scoring=cfg, record_salience=False)
    assert r.score == pytest.approx((0.8 - 0.5) / 0.5, rel=1e-3)
