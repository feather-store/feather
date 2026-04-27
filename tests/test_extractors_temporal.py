"""Tests for feather_db.extractors.TemporalParser."""
from __future__ import annotations
import datetime as dt

import pytest

from feather_db.extractors import TemporalParser, ExtractedTimestamp


# Fixed anchor for deterministic test output
ANCHOR = dt.datetime(2026, 4, 27, 12, 0, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def parser():
    return TemporalParser(anchor=ANCHOR)


# ── ISO dates ─────────────────────────────────────────────────────────────

def test_iso_date(parser):
    result = parser.extract("The campaign launched on 2024-03-15.")
    assert len(result) == 1
    ts = result[0]
    assert isinstance(ts, ExtractedTimestamp)
    assert ts.surface_form == "2024-03-15"
    assert ts.iso.startswith("2024-03-15")
    assert ts.kind == "absolute"
    assert ts.confidence == 1.0


def test_iso_date_invalid_does_not_crash(parser):
    result = parser.extract("invalid date 2024-13-45 here")
    assert result == []


# ── Numeric dates ─────────────────────────────────────────────────────────

def test_numeric_date_us_default(parser):
    result = parser.extract("Run on 3/15/2024.")
    assert len(result) == 1
    assert result[0].iso.startswith("2024-03-15")


def test_numeric_date_eu_format():
    parser = TemporalParser(anchor=ANCHOR, date_format="eu")
    result = parser.extract("Run on 15/3/2024.")
    assert len(result) == 1
    assert result[0].iso.startswith("2024-03-15")


# ── Month + year ──────────────────────────────────────────────────────────

def test_month_year(parser):
    result = parser.extract("We pivoted in March 2024.")
    assert len(result) == 1
    ts = result[0]
    assert ts.iso.startswith("2024-03-01")
    assert ts.range_end_iso.startswith("2024-04-01")
    assert ts.kind == "range_start"


def test_month_year_december_rolls_year(parser):
    result = parser.extract("Sale ran in December 2024.")
    ts = result[0]
    assert ts.iso.startswith("2024-12-01")
    assert ts.range_end_iso.startswith("2025-01-01")


# ── Quarter ───────────────────────────────────────────────────────────────

def test_quarter(parser):
    result = parser.extract("Plan covers Q3 2024 budget.")
    ts = result[0]
    assert ts.iso.startswith("2024-07-01")
    assert ts.range_end_iso.startswith("2024-10-01")


# ── Relative "ago" ────────────────────────────────────────────────────────

def test_relative_3_days_ago(parser):
    result = parser.extract("Three days ago we shipped.")
    assert len(result) == 1
    ts = result[0]
    expected = (ANCHOR - dt.timedelta(days=3)).isoformat()
    assert ts.iso == expected
    assert ts.kind == "relative"


def test_relative_2_weeks_ago(parser):
    result = parser.extract("Started 2 weeks ago.")
    ts = result[0]
    expected = (ANCHOR - dt.timedelta(days=14)).isoformat()
    assert ts.iso == expected


# ── last / next week | month | year ───────────────────────────────────────

def test_last_week(parser):
    result = parser.extract("Last week's CTR was 4.5%.")
    ts = result[0]
    expected = (ANCHOR - dt.timedelta(days=7)).isoformat()
    assert ts.iso == expected


def test_next_quarter(parser):
    result = parser.extract("Next quarter's plan is locked.")
    ts = result[0]
    expected = (ANCHOR + dt.timedelta(days=91)).isoformat()
    assert ts.iso == expected


# ── yesterday / today / tomorrow ──────────────────────────────────────────

def test_yesterday(parser):
    result = parser.extract("yesterday's metrics")
    ts = result[0]
    expected = (ANCHOR - dt.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).isoformat()
    assert ts.iso == expected


# ── Multiple expressions in one call ──────────────────────────────────────

def test_multiple_expressions(parser):
    text = "On 2024-03-15 we launched. Yesterday we paused. Next month we resume."
    result = parser.extract(text)
    surfaces = {r.surface_form for r in result}
    assert "2024-03-15" in surfaces
    assert any(s.lower() == "yesterday" for s in surfaces)
    assert any(s.lower() == "next month" for s in surfaces)


# ── Per-call context override ─────────────────────────────────────────────

def test_context_anchor_override(parser):
    custom_anchor = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    result = parser.extract("yesterday", context={"anchor": custom_anchor})
    expected = (custom_anchor - dt.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).isoformat()
    assert result[0].iso == expected


# ── Invalid format kwarg ──────────────────────────────────────────────────

def test_invalid_date_format_raises():
    with pytest.raises(ValueError):
        TemporalParser(date_format="iso")


# ── Empty / no-match cases ────────────────────────────────────────────────

def test_empty_text_returns_empty(parser):
    assert parser.extract("") == []


def test_no_temporal_returns_empty(parser):
    assert parser.extract("just a regular sentence with no time") == []
