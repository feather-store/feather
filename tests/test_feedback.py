"""Tests for the feedback module — events, log, decay modifier."""
from __future__ import annotations
import os
import tempfile
import time

import pytest

from feather_db.feedback import (
    FeedbackEvent,
    FeedbackLog,
    feedback_decay_modifier,
    FeedbackWeights,
    KIND_FACT_CORRECTED,
    KIND_FACT_ENDORSED,
    KIND_FACT_RETRACTED,
    KIND_RETRIEVAL_UP,
    KIND_RETRIEVAL_DOWN,
    KIND_CONTRADICTION_RESOLVED,
)


@pytest.fixture
def log_path():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


# ── FeedbackEvent ────────────────────────────────────────────────────

def test_event_id_is_deterministic():
    e1 = FeedbackEvent(kind=KIND_FACT_ENDORSED, target_id=42,
                       namespace_id="acme", timestamp=1_700_000_000,
                       user_id="u_1")
    e2 = FeedbackEvent(kind=KIND_FACT_ENDORSED, target_id=42,
                       namespace_id="acme", timestamp=1_700_000_000,
                       user_id="u_1")
    assert e1.event_id == e2.event_id
    assert len(e1.event_id) == 12


def test_event_id_differs_on_different_payload():
    e1 = FeedbackEvent(kind=KIND_FACT_CORRECTED, target_id=42,
                       namespace_id="acme", timestamp=1_700_000_000,
                       payload={"old_object": "4.5%", "new_object": "4.2%"})
    e2 = FeedbackEvent(kind=KIND_FACT_CORRECTED, target_id=42,
                       namespace_id="acme", timestamp=1_700_000_000,
                       payload={"old_object": "4.5%", "new_object": "5.0%"})
    assert e1.event_id != e2.event_id


def test_event_default_timestamp_is_now():
    before = int(time.time())
    e = FeedbackEvent(kind=KIND_FACT_ENDORSED, target_id=1,
                      namespace_id="ns")
    after = int(time.time())
    assert before <= e.timestamp <= after


def test_event_round_trip_json():
    e = FeedbackEvent(
        kind=KIND_FACT_CORRECTED, target_id=42, namespace_id="acme",
        timestamp=1_700_000_000, user_id="u_1",
        payload={"old_object": "Bob", "new_object": "Alice"},
        notes="seen in earnings call",
    )
    s = e.to_json()
    e2 = FeedbackEvent.from_json(s)
    assert e2.kind == e.kind
    assert e2.target_id == e.target_id
    assert e2.payload == e.payload
    assert e2.notes == e.notes
    assert e2.event_id == e.event_id


# ── FeedbackLog ──────────────────────────────────────────────────────

def test_log_append_and_iter(log_path):
    log = FeedbackLog(log_path)
    e = FeedbackEvent(kind=KIND_FACT_ENDORSED, target_id=42,
                      namespace_id="acme", user_id="u_1")
    eid = log.append(e)
    assert eid == e.event_id
    events = list(log.iter_events())
    assert len(events) == 1
    assert events[0].kind == KIND_FACT_ENDORSED
    assert events[0].target_id == 42


def test_log_rejects_unknown_kind(log_path):
    log = FeedbackLog(log_path)
    e = FeedbackEvent(kind="bogus_kind", target_id=1, namespace_id="ns")
    with pytest.raises(ValueError, match="Unknown feedback kind"):
        log.append(e)
    # Nothing was written
    assert not os.path.exists(log_path) or os.path.getsize(log_path) == 0


def test_log_validate_off_accepts_unknown_kind(log_path):
    log = FeedbackLog(log_path, validate_kinds=False)
    e = FeedbackEvent(kind="legacy_kind", target_id=1, namespace_id="ns")
    log.append(e)
    assert log.count() == 1


def test_log_iter_filters(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 1, "acme",
                             timestamp=100))
    log.append(FeedbackEvent(KIND_FACT_CORRECTED, 1, "acme",
                             timestamp=200,
                             payload={"old_object": "x", "new_object": "y"}))
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 2, "acme",
                             timestamp=300))
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 1, "nike",
                             timestamp=400))

    assert log.count(target_id=1) == 3
    assert log.count(namespace_id="acme") == 3
    assert log.count(kind=KIND_FACT_ENDORSED) == 3
    assert log.count(target_id=1, namespace_id="acme") == 2
    assert log.count(since=250) == 2
    assert log.count(until=150) == 1


def test_log_skips_corrupt_lines(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 1, "acme"))
    # Hand-corrupt: append garbage line
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("this is not json\n")
        fh.write("\n")  # empty line, also skipped
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 2, "acme"))
    events = list(log.iter_events())
    assert len(events) == 2  # 2 valid, 1 corrupt skipped, 1 empty skipped


def test_log_iter_on_missing_file():
    log = FeedbackLog("/nonexistent/path/to/log.jsonl")
    assert list(log.iter_events()) == []
    assert log.count() == 0


def test_log_aggregate_for_target(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 42, "acme", timestamp=100))
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 42, "acme", timestamp=200))
    log.append(FeedbackEvent(KIND_FACT_CORRECTED, 42, "acme", timestamp=300,
                             payload={"old_object": "x", "new_object": "y"}))
    log.append(FeedbackEvent(KIND_RETRIEVAL_UP, 42, "acme", timestamp=400,
                             payload={"query": "q", "rank": 1}))
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 99, "acme",
                             timestamp=500))  # different target

    summary = log.aggregate_for_target(42)
    assert summary["endorsed"] == 2
    assert summary["corrected"] == 1
    assert summary["retrieval_up"] == 1
    assert summary["retracted"] == 0
    assert summary["last_event_ts"] == 400


def test_log_per_target_counts(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 1, "acme"))
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 1, "acme"))
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 2, "acme"))
    log.append(FeedbackEvent(KIND_FACT_CORRECTED, 2, "acme",
                             payload={"old_object": "x", "new_object": "y"}))
    counts = log.per_target_counts(kind=KIND_FACT_ENDORSED)
    assert counts == {1: 2, 2: 1}


# ── feedback_decay_modifier ──────────────────────────────────────────

def test_decay_no_feedback_returns_one(log_path):
    log = FeedbackLog(log_path)
    assert feedback_decay_modifier(target_id=42, log=log) == 1.0


def test_decay_retraction_zeros_weight(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_RETRACTED, 42, "acme",
                             payload={"reason": "wrong source"}))
    assert feedback_decay_modifier(42, log) == 0.0


def test_decay_endorsement_boosts(log_path):
    log = FeedbackLog(log_path)
    for _ in range(3):
        log.append(FeedbackEvent(KIND_FACT_ENDORSED, 42, "acme",
                                  timestamp=int(time.time())))
        time.sleep(0.001)  # small jitter so event_ids differ
    # 3 endorsements: 1 + 0.2*3 = 1.6
    mod = feedback_decay_modifier(42, log)
    assert abs(mod - 1.6) < 1e-9


def test_decay_endorsement_capped(log_path):
    log = FeedbackLog(log_path)
    for i in range(20):
        log.append(FeedbackEvent(KIND_FACT_ENDORSED, 42, "acme",
                                  timestamp=1_700_000_000 + i))
    # Cap at 5: 1 + 0.2*5 = 2.0
    mod = feedback_decay_modifier(42, log)
    assert abs(mod - 2.0) < 1e-9


def test_decay_correction_penalizes(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_CORRECTED, 42, "acme",
                             timestamp=1, payload={"old_object": "x",
                                                    "new_object": "y"}))
    # 1 correction: 0.5^1 = 0.5
    mod = feedback_decay_modifier(42, log)
    assert abs(mod - 0.5) < 1e-9


def test_decay_combined_endorse_and_correct(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 42, "acme", timestamp=1))
    log.append(FeedbackEvent(KIND_FACT_CORRECTED, 42, "acme", timestamp=2,
                             payload={"old_object": "x", "new_object": "y"}))
    # endorsement: *1.2 ; correction: *0.5  → 0.6
    mod = feedback_decay_modifier(42, log)
    assert abs(mod - 0.6) < 1e-9


def test_decay_namespace_isolation(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_RETRACTED, 42, "acme", timestamp=1,
                             payload={"reason": "x"}))
    # Same target_id in a different namespace shouldn't see the retract
    mod_acme = feedback_decay_modifier(42, log, namespace_id="acme")
    mod_nike = feedback_decay_modifier(42, log, namespace_id="nike")
    assert mod_acme == 0.0
    assert mod_nike == 1.0


def test_decay_custom_weights(log_path):
    log = FeedbackLog(log_path)
    log.append(FeedbackEvent(KIND_FACT_ENDORSED, 42, "acme", timestamp=1))
    weights = FeedbackWeights(endorsement_boost=0.5, endorsement_cap=2)
    mod = feedback_decay_modifier(42, log, weights=weights)
    assert abs(mod - 1.5) < 1e-9
