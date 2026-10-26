"""`confidence_gte` — filtering on how certain a record is.

`Metadata.confidence` has been stored and persisted since Phase 2, but was never
in `SearchFilter`, so a caller could record how sure it was about a fact and then
had no way to ask for only the sure ones. Findings need exactly that: a claim at
0.3 must not come back ranked beside one at 0.95.

Kept separate from `importance` on purpose. Importance is how much a record
matters; confidence is how likely it is to be true. A critical rule believed
weakly and a trivial fact known for certain are not the same thing.
"""
import numpy as np
import pytest

import feather_db
from feather_db import DB, FilterBuilder

DIM = 32


def seed(db):
    """Nine records across the confidence range, importance deliberately
    uncorrelated so a test cannot pass by filtering the wrong field."""
    rows = [
        (1, 0.95, 0.2, "settled: spend rose 12 percent last quarter"),
        (2, 0.90, 1.0, "settled: the account uses GBP"),
        (3, 0.75, 0.5, "likely: the ugc format outperforms static"),
        (4, 0.60, 0.9, "likely: the audience skews urban"),
        (5, 0.50, 0.1, "uncertain: the hook may have driven the lift"),
        (6, 0.30, 1.0, "weak: a competitor may be bidding on our terms"),
        (7, 0.10, 0.8, "speculative: the drop could be seasonal"),
        (8, 1.00, 0.3, "certain: the campaign ended on 30 September"),
        (9, 0.45, 0.6, "uncertain: creative fatigue is a possible cause"),
    ]
    for rid, conf, imp, text in rows:
        m = feather_db.Metadata()
        m.content = text
        m.confidence = conf
        m.importance = imp
        m.namespace_id = "brand.b1.shared.findings"
        db.add(id=rid, vec=np.random.rand(DIM).astype(np.float32), meta=m)
    return rows


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "c.feather"), dim=DIM)
    seed(d)
    yield d
    d.close()


def ids(results):
    return sorted(r.id for r in results)


# ── the filter ────────────────────────────────────────────────────────────

def test_it_keeps_only_records_at_or_above_the_threshold(db):
    q = np.random.rand(DIM).astype(np.float32)
    f = FilterBuilder().min_confidence(0.75).build()
    got = db.search(q, k=20, filter=f)

    assert ids(got) == [1, 2, 3, 8]
    for r in got:
        assert r.metadata.confidence >= 0.75


def test_the_threshold_is_inclusive(db):
    """`_gte`, matching `importance_gte`. A record at exactly the threshold is
    included — otherwise `min_confidence(0.5)` silently drops the 0.5 case."""
    q = np.random.rand(DIM).astype(np.float32)
    got = db.search(q, k=20, filter=FilterBuilder().min_confidence(0.5).build())
    assert 5 in ids(got), "the record at exactly 0.5 was excluded"


def test_a_zero_threshold_keeps_everything(db):
    q = np.random.rand(DIM).astype(np.float32)
    got = db.search(q, k=20, filter=FilterBuilder().min_confidence(0.0).build())
    assert len(got) == 9


def test_an_unreachable_threshold_returns_nothing_rather_than_the_best_available(db):
    """A filter that matches nothing must return nothing. Degrading to
    "closest anyway" would hand a caller speculation it explicitly excluded."""
    q = np.random.rand(DIM).astype(np.float32)
    got = db.search(q, k=20, filter=FilterBuilder().min_confidence(1.01).build())
    assert got == [] or len(got) == 0


# ── it is genuinely a different axis from importance ──────────────────────

def test_confidence_and_importance_filter_independently(db):
    """The seeded data deliberately uncorrelates them, so a test cannot pass by
    filtering the wrong field."""
    q = np.random.rand(DIM).astype(np.float32)

    conf_only = ids(db.search(q, k=20, filter=FilterBuilder().min_confidence(0.9).build()))
    imp_only = ids(db.search(q, k=20, filter=FilterBuilder().min_importance(0.9).build()))

    assert conf_only == [1, 2, 8]
    assert imp_only == [2, 4, 6]
    assert conf_only != imp_only


def test_both_compose_as_an_and(db):
    q = np.random.rand(DIM).astype(np.float32)
    f = FilterBuilder().min_confidence(0.9).min_importance(0.9).build()
    assert ids(db.search(q, k=20, filter=f)) == [2]      # only record 2 clears both


def test_it_composes_with_a_scope(db):
    """The combination findings actually need: this brand's shared findings,
    only the ones worth believing."""
    m = feather_db.Metadata()
    m.content = "settled: a fact in a different brand"
    m.confidence = 1.0
    m.namespace_id = "brand.b2.shared.findings"
    db.add(id=99, vec=np.random.rand(DIM).astype(np.float32), meta=m)

    q = np.random.rand(DIM).astype(np.float32)
    f = (FilterBuilder()
         .namespace("brand.b1.shared.findings")
         .min_confidence(0.9)
         .build())
    assert ids(db.search(q, k=20, filter=f)) == [1, 2, 8]


# ── persistence ───────────────────────────────────────────────────────────

def test_confidence_survives_save_and_reload(tmp_path):
    """The field was already persisted; this guards the filter against a format
    regression rather than the storage."""
    path = str(tmp_path / "p.feather")
    first = DB.open(path, dim=DIM)
    seed(first)
    first.close()

    reopened = DB.open(path, dim=DIM)
    try:
        q = np.random.rand(DIM).astype(np.float32)
        f = FilterBuilder().min_confidence(0.75).build()
        assert ids(reopened.search(q, k=20, filter=f)) == [1, 2, 3, 8]
        assert reopened.get_metadata(7).confidence == pytest.approx(0.10)
    finally:
        reopened.close()


def test_the_default_confidence_is_one_so_unset_records_are_not_filtered_out(tmp_path):
    """Records written before anyone set confidence must not vanish the moment
    someone starts filtering on it."""
    d = DB.open(str(tmp_path / "d.feather"), dim=DIM)
    try:
        m = feather_db.Metadata()
        m.content = "written without touching confidence"
        d.add(id=1, vec=np.random.rand(DIM).astype(np.float32), meta=m)

        q = np.random.rand(DIM).astype(np.float32)
        got = d.search(q, k=5, filter=FilterBuilder().min_confidence(1.0).build())
        assert ids(got) == [1]
    finally:
        d.close()
