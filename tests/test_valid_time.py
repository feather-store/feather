"""Valid time (Metadata.valid_at): when what a record says happened, as
opposed to `timestamp`, when it was recorded. A session on 05/14 saying
"last Saturday I went to a wedding" is recorded 05/14 and valid 05/06; an
agent asking "what happened in early May" or "what happened most
recently" means the second."""
import os

import numpy as np
import pytest

import feather_db
from feather_db import FilterBuilder

DAY = 86_400
T0 = 1_683_000_000


def _meta(ns, ts, valid=0, content=""):
    m = feather_db.Metadata()
    m.namespace_id, m.timestamp, m.valid_at = ns, ts, valid
    m.content = content
    return m


def _vec(i, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


@pytest.fixture
def db(tmp_path):
    d = feather_db.DB.open(str(tmp_path / "v.feather"), dim=8)
    yield d
    d.close()


def _fill(db):
    # Recorded in order 1, 2, 3; happened in order 3, 1, 2.
    db.add(1, _vec(1), _meta("u", T0 + 1 * DAY, T0 - 10 * DAY, "wedding"))
    db.add(2, _vec(2), _meta("u", T0 + 2 * DAY, T0 - 5 * DAY, "bake"))
    db.add(3, _vec(3), _meta("u", T0 + 3 * DAY, T0 - 20 * DAY, "bike"))
    db.add(4, _vec(4), _meta("u", T0 + 4 * DAY))          # no valid time given


def test_valid_time_defaults_to_the_timestamp():
    m = _meta("u", 123)
    assert m.valid_at == 0 and m.valid_time == 123
    m.valid_at = 99
    assert m.valid_time == 99


def test_newest_by_valid_orders_by_when_it_happened(db):
    _fill(db)
    assert db.newest("u", 10) == [4, 3, 2, 1]                    # recorded
    assert db.newest("u", 10, by_valid=True) == [4, 2, 1, 3]     # happened
    assert db.newest("u", 2, by_valid=True,
                     before=(T0 - 5 * DAY, 2)) == [1, 3]         # pages


def test_filter_by_valid_time_is_inclusive(db):
    _fill(db)
    f = FilterBuilder().namespace("u").valid_between(
        T0 - 10 * DAY, T0 - 5 * DAY).build()
    got = {r.id for r in db.search(_vec(0), k=10, filter=f)}
    assert got == {1, 2}
    f = FilterBuilder().namespace("u").valid_between(after=T0 + 4 * DAY).build()
    assert {r.id for r in db.search(_vec(0), k=10, filter=f)} == {4}


def test_valid_time_survives_save_reopen_and_the_wal(tmp_path):
    path = str(tmp_path / "v.feather")
    db = feather_db.DB.open(path, dim=8)
    _fill(db)
    db.save()
    db.update_metadata(2, _meta("u", T0 + 2 * DAY, T0 - 30 * DAY, "bake"))
    db.close(save=False)                         # the update lives in the WAL
    db = feather_db.DB.open(path, dim=8)
    assert db.get_metadata(1).valid_at == T0 - 10 * DAY
    assert db.get_metadata(2).valid_at == T0 - 30 * DAY
    assert db.get_metadata(4).valid_at == 0
    assert "\0valid_at" not in db.get_metadata(1).attributes
    assert db.newest("u", 10, by_valid=True) == [4, 1, 3, 2]
    db.close()


def test_an_update_moves_the_record_in_the_valid_index(db):
    _fill(db)
    db.update_metadata(3, _meta("u", T0 + 3 * DAY, T0 + 9 * DAY, "bike"))
    assert db.newest("u", 1, by_valid=True) == [3]
    db.forget(3)
    assert db.newest("u", 10, by_valid=True) == [4, 2, 1]


def test_a_build_without_valid_time_writes_it_back_intact(tmp_path):
    # Such a build reads the reserved attribute as an ordinary attribute
    # and writes it back as one; this build reads it as valid_at again.
    path = str(tmp_path / "v.feather")
    db = feather_db.DB.open(path, dim=8)
    m = _meta("u", T0)
    m.set_attribute("\0valid_at", str(T0 - DAY))
    db.add(1, _vec(1), m)
    bad = _meta("u", T0)
    bad.set_attribute("\0valid_at", "not a number")
    db.add(2, _vec(2), bad)
    db.close()
    db = feather_db.DB.open(path, dim=8)
    assert db.get_metadata(1).valid_at == T0 - DAY
    got = db.get_metadata(2)
    assert got.valid_at == 0 and got.get_attribute("\0valid_at") == "not a number"
    db.close()
