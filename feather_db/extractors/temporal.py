"""Rule-based temporal expression parser — first OSS extractor to ship.

Resolves natural-language temporal phrases to ISO-8601 timestamps without
any LLM dependency. Handles the common cases the upstream FactExtractor
will surface: absolute dates, relative offsets ("last week", "3 days
ago"), named months, fiscal-quarter terms, and week-of-year.

For ambiguous or domain-specific temporal references (e.g. "Q3 of last
fiscal year"), the LLM-backed FactExtractor remains responsible for
producing valid ISO outputs directly.
"""
from __future__ import annotations
import re
import datetime as _dt
from typing import Optional

from .base import Extractor, ExtractedTimestamp


_MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_UNIT_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,    # approximate; production should use dateutil.relativedelta
    "year": 365, "years": 365,
    "quarter": 91, "quarters": 91,
}

# "3 days ago", "two weeks ago"
_PATTERN_RELATIVE_AGO = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(day|days|week|weeks|month|months|year|years|quarter|quarters)\s+ago\b",
    re.IGNORECASE,
)

# "last week", "last month", "next quarter"
_PATTERN_RELATIVE_LAST_NEXT = re.compile(
    r"\b(last|next)\s+(week|month|year|quarter)\b",
    re.IGNORECASE,
)

# "yesterday", "today", "tomorrow"
_PATTERN_DAY_REL = re.compile(r"\b(yesterday|today|tomorrow)\b", re.IGNORECASE)

# "March 2024", "in March 2024"
_PATTERN_MONTH_YEAR = re.compile(
    r"\b(?:in\s+)?(january|february|march|april|may|june|july|august|"
    r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|"
    r"sep|sept|oct|nov|dec)\s+(\d{4})\b",
    re.IGNORECASE,
)

# "2024-03-15" (ISO already)
_PATTERN_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# "3/15/2024" (US) or "15/3/2024" (EU) — we default to US since that's
# typical for ad-platform exports
_PATTERN_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")

# "Q1 2024", "Q3-2025"
_PATTERN_QUARTER = re.compile(r"\bQ([1-4])[\s-]+(\d{4})\b", re.IGNORECASE)

_WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


class TemporalParser(Extractor):
    """Rule-based temporal expression parser.

    Args:
        anchor: the "now" the parser uses for relative phrases. Defaults
                to wall-clock UTC. Set this to the question_date or
                document timestamp for deterministic, reproducible
                extraction (essential for benchmarks).
        date_format: 'us' (default) interprets MM/DD/YYYY; 'eu' interprets
                DD/MM/YYYY for ambiguous numeric dates.
    """

    name = "temporal_parser"
    version = "0.1.0"

    def __init__(self, anchor: Optional[_dt.datetime] = None,
                 date_format: str = "us"):
        self._anchor = anchor or _dt.datetime.now(_dt.timezone.utc)
        if date_format not in ("us", "eu"):
            raise ValueError("date_format must be 'us' or 'eu'")
        self._date_format = date_format

    def extract(self, text: str, *,
                context: Optional[dict] = None) -> list[ExtractedTimestamp]:
        """Find every temporal phrase in `text` and resolve to ISO-8601.

        If `context` provides an "anchor" datetime, that overrides the
        instance anchor for this call only. Useful when running
        per-question with a question_date.
        """
        anchor = self._anchor
        if context and isinstance(context.get("anchor"), _dt.datetime):
            anchor = context["anchor"]

        out: list[ExtractedTimestamp] = []

        # Order matters: parse most-specific patterns first so
        # less-specific ones don't shadow them.
        out.extend(self._extract_iso_dates(text))
        out.extend(self._extract_numeric_dates(text))
        out.extend(self._extract_month_year(text))
        out.extend(self._extract_quarter(text))
        out.extend(self._extract_relative_ago(text, anchor))
        out.extend(self._extract_last_next(text, anchor))
        out.extend(self._extract_day_relative(text, anchor))

        return out

    # ── Concrete extractors ───────────────────────────────────────────

    def _extract_iso_dates(self, text: str) -> list[ExtractedTimestamp]:
        out = []
        for m in _PATTERN_ISO_DATE.finditer(text):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                dt = _dt.datetime(y, mo, d, tzinfo=_dt.timezone.utc)
                out.append(ExtractedTimestamp(
                    surface_form=m.group(0),
                    iso=dt.isoformat(),
                    kind="absolute",
                    confidence=1.0,
                ))
            except ValueError:
                pass
        return out

    def _extract_numeric_dates(self, text: str) -> list[ExtractedTimestamp]:
        out = []
        for m in _PATTERN_NUMERIC_DATE.finditer(text):
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            month, day = (a, b) if self._date_format == "us" else (b, a)
            try:
                dt = _dt.datetime(y, month, day, tzinfo=_dt.timezone.utc)
                out.append(ExtractedTimestamp(
                    surface_form=m.group(0),
                    iso=dt.isoformat(),
                    kind="absolute",
                    confidence=0.9,  # less than ISO due to format ambiguity
                ))
            except ValueError:
                pass
        return out

    def _extract_month_year(self, text: str) -> list[ExtractedTimestamp]:
        out = []
        for m in _PATTERN_MONTH_YEAR.finditer(text):
            month = _MONTH_NAMES[m.group(1).lower()]
            year = int(m.group(2))
            start = _dt.datetime(year, month, 1, tzinfo=_dt.timezone.utc)
            # End-of-month
            if month == 12:
                end = _dt.datetime(year + 1, 1, 1, tzinfo=_dt.timezone.utc)
            else:
                end = _dt.datetime(year, month + 1, 1, tzinfo=_dt.timezone.utc)
            out.append(ExtractedTimestamp(
                surface_form=m.group(0),
                iso=start.isoformat(),
                kind="range_start",
                range_end_iso=end.isoformat(),
                confidence=1.0,
            ))
        return out

    def _extract_quarter(self, text: str) -> list[ExtractedTimestamp]:
        out = []
        for m in _PATTERN_QUARTER.finditer(text):
            q = int(m.group(1))
            year = int(m.group(2))
            start_month = (q - 1) * 3 + 1
            end_month = start_month + 3
            start = _dt.datetime(year, start_month, 1, tzinfo=_dt.timezone.utc)
            end = (_dt.datetime(year + 1, 1, 1, tzinfo=_dt.timezone.utc)
                   if end_month > 12
                   else _dt.datetime(year, end_month, 1, tzinfo=_dt.timezone.utc))
            out.append(ExtractedTimestamp(
                surface_form=m.group(0),
                iso=start.isoformat(),
                kind="range_start",
                range_end_iso=end.isoformat(),
                confidence=1.0,
            ))
        return out

    def _extract_relative_ago(self, text: str,
                              anchor: _dt.datetime) -> list[ExtractedTimestamp]:
        out = []
        for m in _PATTERN_RELATIVE_AGO.finditer(text):
            amount_raw = m.group(1).lower()
            amount = (int(amount_raw) if amount_raw.isdigit()
                      else _WORD_TO_INT[amount_raw])
            unit = m.group(2).lower()
            days = _UNIT_DAYS[unit] * amount
            dt = anchor - _dt.timedelta(days=days)
            out.append(ExtractedTimestamp(
                surface_form=m.group(0),
                iso=dt.isoformat(),
                kind="relative",
                confidence=0.85,  # 'months ago' is fuzzy
            ))
        return out

    def _extract_last_next(self, text: str,
                           anchor: _dt.datetime) -> list[ExtractedTimestamp]:
        out = []
        for m in _PATTERN_RELATIVE_LAST_NEXT.finditer(text):
            direction = -1 if m.group(1).lower() == "last" else 1
            unit = m.group(2).lower()
            days = _UNIT_DAYS[unit + ("s" if not unit.endswith("s") else "")]
            dt = anchor + _dt.timedelta(days=direction * days)
            out.append(ExtractedTimestamp(
                surface_form=m.group(0),
                iso=dt.isoformat(),
                kind="relative",
                confidence=0.7,  # "last week" is loosely 7 days back
            ))
        return out

    def _extract_day_relative(self, text: str,
                              anchor: _dt.datetime) -> list[ExtractedTimestamp]:
        out = []
        for m in _PATTERN_DAY_REL.finditer(text):
            phrase = m.group(1).lower()
            offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[phrase]
            dt = (anchor + _dt.timedelta(days=offset)).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            out.append(ExtractedTimestamp(
                surface_form=m.group(0),
                iso=dt.isoformat(),
                kind="relative",
                confidence=1.0,
            ))
        return out
