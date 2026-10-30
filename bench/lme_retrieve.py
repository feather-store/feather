"""LongMemEval retrieval: the dated query, and multi-round retrieval for
count and comparison questions.

Multi-round (after EverOS): an LLM reads a digest of what the first
search found, names what is missing, and writes 1–3 follow-up queries,
each optionally limited to a date window. Their new sessions are added
to the context after the first round's top-k. A count needs every
instance and a comparison needs every item compared; one query worded
like the question reaches the instances that share its words and misses
the rest.

Shared by bench/ and benchmarks/longmemeval_qa.py; each runs its own
searches through the helpers here.
"""
from __future__ import annotations
import json
import re
from typing import Callable, Iterable, Optional

from .scenarios.lme_context import Turn, parse_lme_date, session_header

Chat = Callable[[str, str], str]

MAX_FOLLOWUPS = 3
NEW_SESSIONS_PER_FOLLOWUP = 3
DIGEST_CHARS = 12_000
_DIGEST_TURN_CHARS = 300

_COUNT_RE = re.compile(
    r"\b(how many|how much|how often|total|in total|number of|times|"
    r"altogether|combined|in all|so far|sum)\b", re.I)
_COMPARE_RE = re.compile(
    r"\b(first|earlier|later|before|after|more|less|fewer|most|least|"
    r"longer|shorter|bigger|smaller|compared?|difference|between|order|"
    r"or the|either|both)\b", re.I)


def dense_query(question: str, question_date: str) -> str:
    """The text embedded for the vector half of hybrid search. The date
    lets "last week" / "this month" land near sessions of that time; the
    keyword half keeps the bare question (date tokens are noise to BM25)."""
    return (f"[Current date: {question_date}] {question}" if question_date
            else question)


def needs_multi_round(question: str) -> bool:
    """Count and comparison questions: the ones a single query misses."""
    return bool(_COUNT_RE.search(question) or _COMPARE_RE.search(question))


PLANNER_SYSTEM = """\
You plan retrieval for an assistant that answers questions from a user's \
past chat sessions. You get the question, the current date, and excerpts \
of the sessions a first search found (each headed by its date).

Decide whether the excerpts hold everything the question needs:
- a count needs EVERY instance, across all sessions;
- a comparison or ordering needs EACH item compared, with its date;
- a time span needs both events.

If something is missing, write up to 3 search queries, one per missing \
thing. Word each one the way the user would have mentioned the thing in \
conversation (e.g. "I just got back from my trip to Denver"), not the way \
the question asks about it. Use "after"/"before" (YYYY/MM/DD) only when \
the question fixes a time window.

Reply with JSON only:
{"missing": "<what is missing, or empty>", "queries": [{"query": "...", \
"after": null, "before": null}]}
Use an empty "queries" list when nothing is missing."""

PLANNER_USER_TMPL = """\
Current date: {question_date}
Question: {question}

Excerpts from the first search:

{digest}"""


def digest(ranked_hits: list[Turn], question_date: str,
           max_sessions: int, max_chars: int = DIGEST_CHARS) -> str:
    """What the planner reads: the first round's sessions in rank order,
    their retrieved turns clipped. Enough to see what was found, at a
    fraction of the answer context."""
    by_sid: dict[str, list[Turn]] = {}
    for t in ranked_hits:
        if t.session_id not in by_sid and len(by_sid) >= max_sessions:
            continue
        by_sid.setdefault(t.session_id, []).append(t)
    blocks, used = [], 0
    for turns in by_sid.values():
        lines = [session_header(turns[0].session_date, question_date)]
        for t in sorted(turns, key=lambda t: t.order):
            text = " ".join(t.content.split())
            if len(text) > _DIGEST_TURN_CHARS:
                text = text[:_DIGEST_TURN_CHARS] + " […]"
            lines.append(f"[{t.role or 'unknown'}] {text}")
        block = "\n".join(lines)
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def _first_json_object(raw: str) -> Optional[dict]:
    start = (raw or "").find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(raw)):
            depth += {"{": 1, "}": -1}.get(raw[i], 0)
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    break
        start = raw.find("{", start + 1)
    return None


def _date_or_none(v) -> Optional[str]:
    if not isinstance(v, str) or not v.strip():
        return None
    return v.strip() if parse_lme_date(v.strip()) else None


def plan_followups(chat: Chat, question: str, question_date: str,
                   digest_text: str) -> tuple[list[dict], str]:
    """(follow-up queries, what the planner said is missing). Any failure
    to call or parse is no follow-ups: round one stands."""
    try:
        raw = chat(PLANNER_SYSTEM, PLANNER_USER_TMPL.format(
            question=question, question_date=question_date or "unknown",
            digest=digest_text or "(nothing found)"))
    except Exception as e:
        return [], f"[planner error: {e}]"
    obj = _first_json_object(raw) or {}
    queries = []
    for q in obj.get("queries") or []:
        if isinstance(q, str):
            q = {"query": q}
        if not isinstance(q, dict) or not str(q.get("query") or "").strip():
            continue
        queries.append({"query": str(q["query"]).strip(),
                        "after": _date_or_none(q.get("after")),
                        "before": _date_or_none(q.get("before"))})
        if len(queries) == MAX_FOLLOWUPS:
            break
    return queries, str(obj.get("missing") or "")


def in_window(session_date: str, after: Optional[str],
              before: Optional[str]) -> bool:
    """Whole-day bounds, inclusive. An undated session is kept."""
    d = parse_lme_date(session_date)
    if d is None:
        return True
    a, b = parse_lme_date(after or ""), parse_lme_date(before or "")
    return (a is None or d.date() >= a.date()) and \
           (b is None or d.date() <= b.date())


def merge_sessions(primary: list[str], followups: Iterable[list[str]],
                   k: int, per_followup: int = NEW_SESSIONS_PER_FOLLOWUP
                   ) -> list[str]:
    """Round one's top-k, then each follow-up's best sessions not already
    in. Follow-ups add slots rather than compete for them: their sessions
    rank low for the original question, which is why they were missed."""
    out = list(dict.fromkeys(primary))[:k]
    seen = set(out)
    for ranked in followups:
        added = 0
        for sid in ranked:
            if added == per_followup:
                break
            if sid not in seen:
                out.append(sid)
                seen.add(sid)
                added += 1
    return out
