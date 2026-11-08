"""SessionFactExtractor — dated memories from a whole conversation session.

FactExtractor reads one text at a time and is never told when it was
said, so "last Tuesday" stays relative and a change ("I switched to Y")
has nothing to supersede. This extractor reads one SESSION per call —
cheaper than one call per turn, and each turn is read in the context of
the others — anchors every date to the session date, and gives each fact
a (subject, attribute) key so later values can be linked to earlier ones.

`link_validity()` then chains facts that share a key: each keeps its own
valid-from date and gains a valid-until (the next value's date). Nothing
is overwritten; "what was it in March" and "what is it now" both stay
answerable.

The facts are an index, not a replacement: each cites the turn it came
from, so a caller can retrieve by fact and hand the reader the session.
"""
from __future__ import annotations
import datetime as _dt
import re
import time
from typing import Optional

from .base import Extractor, Fact
from ._jsonparse import extract_json

SESSION_FACT_SYSTEM_PROMPT = """\
You extract dated memories from one chat session between a user and an \
assistant. The session date is given; turns are numbered.

Extract short facts worth remembering later: what the user did, bought, \
owns, attended, plans, prefers or dislikes; people, places and names they \
mention; numbers (counts, amounts, prices, durations, distances); and \
anything that changed. Include what the assistant recommended or listed \
only when the user is likely to ask about it again.

Rules:
- One fact per item. "statement" is one sentence that starts with the \
date: "On 2023/05/14 the user switched from yoga to pilates."
- Every date is absolute (YYYY/MM/DD). Resolve "yesterday", "last \
Tuesday", "two weeks ago", "next month" against the session date. When \
the text gives no time, use the session date.
- A change ("I switched from X to Y", "I now have 3", "we moved to Z") is \
stated with its NEW value; the old value is kept elsewhere.
- "subject" and "attribute" name what the fact is about, short and \
lowercase, so two facts about the same thing match across sessions. The \
subject is who or what the attribute belongs to: "Tom became my coach" is \
("user", "coach") with value "Tom". Name the attribute by WHAT the value is, \
a noun, never by the verb that said it: "drives a Civic", "bought a Civic" \
and "my car is a Civic" are all ("user", "car"); "goes to Lincoln High" and \
"transferred to Lincoln High" are both ("user", "school"); ("user's car", \
"mileage"), ("user", "gym routine").
- When attributes already on file are listed, a fact about one of those \
things uses that exact subject and attribute, not a new name for it.
- "one_value": true only when a later, different value would make this fact \
false: the attribute is a current state with exactly one value at a time, \
such as where someone lives, their job or employer, school, car, \
relationship status, a pet's name, a total or count. That holds whether or \
not this fact is a change ("moved to", "switched to", "joined" are \
one-value). False for everything else: anything that happened (an event, \
incident, trip or visit), activities, likes, hobbies, feelings, plans, \
goals, projects, and anything several values of which can be true at \
once. When unsure, false.
- "turn" is the number of the turn that supports the fact.
- "kind": "event" for something that happened once on a date (attended, bought, baked, visited, paid, was born); "state" for a value that holds until it changes (job, address, count owned, plan); "preference" for a like, dislike, habit, or something the user owns or uses that should shape advice; otherwise "other".
- "entity": for an event, the specific instance, named the same way every time it comes up, so a second mention of the same event matches the first: "wedding of rachel and mike", "sourdough loaf for book club", "bike chain replacement". Lowercase.
- "amount": a number the fact carries, with its unit ("$40", "3 plants", "250 pages"), else "".
- "own": true when the user did, has or attended it themselves; false when it is someone else's, or only asked about, considered, planned or suggested by the assistant.
- At most {max_facts} facts. No facts about the assistant itself, and none \
that are general knowledge rather than about this user.

Output a JSON array:
[{{"statement": "...", "date": "YYYY/MM/DD", "subject": "...", \
"attribute": "...", "value": "...", "one_value": false, "turn": 0, \
"kind": "event", "entity": "...", "amount": "", "own": true}}]
Return [] when there is nothing worth remembering."""

_DATE_RE = re.compile(r"^\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})")


def _parse_day(s) -> Optional[_dt.datetime]:
    m = _DATE_RE.match(str(s or ""))
    if not m:
        return None
    try:
        return _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


KINDS = ("event", "state", "preference", "other")


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())


def _kind(v) -> str:
    v = str(v or "").strip().lower()
    return v if v in KINDS else "other"


def fact_key(f: Fact) -> str:
    """The identity two facts share when one supersedes the other."""
    return f"{_norm(f.subject)}|{_norm(f.predicate)}"


def replaces(f: Fact) -> bool:
    """Does a later value of this fact's key end this one? The extractor
    says so per fact (0.3.0: `replaces`); a fact without the field (an older
    extraction, a hand-built Fact) keeps the old rule, which was yes.

    A shared key alone is not enough: on LoCoMo 117 extracted keys held
    different values, mostly ones that add up ("hobbies", "plan",
    "preference"), and treating them as replaced told the reader that
    "Caroline will continue her education" was over once she "plans to
    adopt" (bench.locomo --supersede)."""
    if _norm(f.predicate) in _CATEGORY_ATTRS:
        return False
    return bool((f.metadata or {}).get("replaces", True))


# Attribute names that name a category, not a thing: "user|event" chained
# a Tony Awards viewing, a Napa trip and a birthday party as one value
# replacing the next. 0.3.1 flagged 121 LongMemEval facts (27 contradiction,
# 10 LoCoMo) that way despite the prompt's rule to name what the value is.
_CATEGORY_ATTRS = frozenset({"event", "events"})


def known_keys_text(keys) -> str:
    """The prompt lines that list attributes already on file, one
    'subject | attribute' each: what lets separate calls name one thing one
    way ("address" in one session, "home city" in the next split a chain)."""
    rows = sorted({k if isinstance(k, str) else "|".join(k) for k in keys or ()})
    return "\n".join("- " + " | ".join(p.strip() for p in r.split("|", 1)) for r in rows)


class SessionFactExtractor(Extractor):
    """One LLM call per session → dated Facts.

    Each Fact has subject / predicate (= the attribute) / object (= the
    value), `valid_at` (unix seconds of the event date, local midnight),
    and metadata: statement, date (YYYY/MM/DD), turn, session_id, kind,
    entity, amount, own, and `replaces` (0.3.0): whether a later value of
    the same key ends this one (`replaces()`; the model's `one_value`).

    `context` may carry `user_name` (who "the user" is: a subject naming
    them becomes "user") and `known_keys` ('subject|attribute' strings
    already on file for this user), which the model reuses for a fact about
    the same thing. Without them each call names attributes on its own.

    Args:
        provider:            any feather_db.providers.LLMProvider.
        max_facts:           cap per session (default 30).
        max_turn_chars:      assistant turns are clipped to this many
                             characters (user turns are kept whole): the
                             facts index the session, the session itself
                             still carries the detail.
        max_session_chars:   hard cap on the session text sent.
    """

    name = "session_fact_extractor"
    version = "0.3.1"

    def __init__(self, provider, *, max_facts: int = 30,
                 max_turn_chars: int = 1200,
                 max_session_chars: int = 40_000,
                 max_tokens: int = 3000, max_retries: int = 2):
        self._provider = provider
        self._max_facts = max_facts
        self._max_turn_chars = max_turn_chars
        self._max_session_chars = max_session_chars
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def fingerprint(self) -> str:
        """What else, besides `version`, changes what this extractor returns:
        its prompt, its limits and its model. An index records it
        (feather_db.ingest), so a prompt edited without a version bump still
        marks the indexes the old one made as stale."""
        import hashlib
        p = self._provider
        model = (getattr(p, "_model", None) or getattr(p, "model", None)
                 or type(p).__name__)
        raw = "\x1f".join(map(str, (SESSION_FACT_SYSTEM_PROMPT, self._max_facts,
                                    self._max_turn_chars, self._max_session_chars, model)))
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def session_text(self, turns: list[dict]) -> str:
        lines, used = [], 0
        for i, t in enumerate(turns):
            role = t.get("role") or "unknown"
            text = " ".join(str(t.get("content") or "").split())
            if role != "user" and len(text) > self._max_turn_chars:
                text = text[: self._max_turn_chars] + " […]"
            line = f"[{i}] {role}: {text}"
            if used + len(line) > self._max_session_chars:
                lines.append(f"[{i}…] (rest of session omitted)")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def extract(self, text: str, *, context: Optional[dict] = None) -> list[Fact]:
        """Extractor protocol: `text` is treated as a one-turn user session."""
        ctx = context or {}
        return self.extract_session([{"role": "user", "content": text}],
                                    ctx.get("session_date"), context=ctx)

    def extract_session(self, turns: list[dict], session_date, *,
                        context: Optional[dict] = None) -> list[Fact]:
        """turns: [{"role", "content"}]; session_date: datetime or a
        'YYYY/MM/DD…' string. Returns [] on an unrecoverable failure
        (reported through context["on_error"] if given)."""
        ctx = context or {}
        on_error = ctx.get("on_error")
        if not any(str(t.get("content") or "").strip() for t in turns):
            return []
        day = (session_date if isinstance(session_date, _dt.datetime)
               else _parse_day(session_date))
        day_str = day.strftime("%Y/%m/%d (%a)") if day else "unknown"
        user_name = str(ctx.get("user_name") or "").strip()
        head = [f"Session date: {day_str}"]
        if user_name:
            head.append(f"The user is {user_name}.")
        known = known_keys_text(ctx.get("known_keys"))
        if known:
            head.append("Attributes already on file (subject | attribute):\n" + known)
        messages = [
            {"role": "system", "content": SESSION_FACT_SYSTEM_PROMPT.format(
                max_facts=self._max_facts)},
            {"role": "user", "content":
                "\n".join(head) + f"\n\n{self.session_text(turns)}"},
        ]
        raw, last_err = None, None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._provider.complete(messages, max_tokens=self._max_tokens,
                                              temperature=0.0)
                break
            except Exception as e:
                last_err = e
                if attempt < self._max_retries:
                    time.sleep(1.5 ** (attempt + 1))
        if raw is None:
            if on_error:
                on_error(f"SessionFactExtractor LLM call failed: {last_err}")
            return []
        parsed, ok, _ = extract_json(raw)
        if ok and isinstance(parsed, dict):      # {"facts": [...]} in JSON mode
            parsed = next((v for v in parsed.values() if isinstance(v, list)), None)
        if not ok or not isinstance(parsed, list):
            if on_error:
                on_error(f"SessionFactExtractor JSON parse failed; raw[:200]={raw[:200]!r}")
            return []

        now_ts = int(time.time())
        facts: list[Fact] = []
        for item in parsed[: self._max_facts]:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement") or "").strip()
            if not statement:
                continue
            when = _parse_day(item.get("date")) or day
            try:
                turn = int(item.get("turn"))
            except (TypeError, ValueError):
                turn = None
            if turn is not None and not 0 <= turn < len(turns):
                turn = None
            subject = str(item.get("subject") or "user").strip()
            if user_name and _norm(subject) == _norm(user_name):
                subject = "user"
            facts.append(Fact(
                subject=subject,
                predicate=str(item.get("attribute") or statement).strip(),
                object=str(item.get("value") or "").strip(),
                provenance=ctx.get("source_id"),
                extracted_at=now_ts,
                valid_at=int(when.timestamp()) if when else None,
                extractor_name=self.name,
                extractor_version=self.version,
                metadata={"statement": statement,
                          "date": when.strftime("%Y/%m/%d") if when else "",
                          "turn": turn,
                          "session_id": ctx.get("session_id"),
                          "kind": _kind(item.get("kind")),
                          "replaces": (item.get("one_value") is True
                                       or item.get("replaces") is True),
                          "entity": _norm(str(item.get("entity") or "")),
                          "amount": str(item.get("amount") or "").strip(),
                          "own": item.get("own") is not False},
            ))
        return facts


def link_validity(facts: list[Fact]) -> dict[str, list[Fact]]:
    """Chain facts that share fact_key() in date order, in place: each gets
    metadata["valid_from"], and every one that a later value replaces also
    metadata["valid_until"] (the next value's date) and
    metadata["superseded_by"] (its statement). Returns {key: chain}.

    Only facts that `replaces()` are closed: a like or a plan stays open
    when another of the same key comes along. Facts on the same day with
    the same value are one fact said twice and do not supersede each
    other."""
    chains: dict[str, list[Fact]] = {}
    for f in facts:
        chains.setdefault(fact_key(f), []).append(f)
    for chain in chains.values():
        chain.sort(key=lambda f: (f.valid_at is None, f.valid_at or 0))
        for f in chain:
            f.metadata["valid_from"] = f.metadata.get("date", "")
            f.metadata.pop("valid_until", None)
            f.metadata.pop("superseded_by", None)
        for i, f in enumerate(chain):
            if not replaces(f):
                continue
            # Closed by the first later fact that is not a repeat of it.
            nxt = next((g for g in chain[i + 1:] if g.metadata.get("date") and replaces(g)
                        and not (
                g.valid_at == f.valid_at and
                g.object.strip().lower() == f.object.strip().lower())), None)
            if nxt is not None:
                f.metadata["valid_until"] = nxt.metadata["date"]
                f.metadata["superseded_by"] = nxt.metadata["statement"]
    return chains


def timeline(chains: dict[str, list[Fact]], keys, max_keys: int = 15) -> str:
    """Lines for the keys whose value changed, oldest value first:
    'user / job: junior dev (2023/01/10 – 2023/05/14) → senior dev (2023/05/14 – now)'."""
    lines = []
    for key in dict.fromkeys(keys):
        chain = [f for f in chains.get(key, []) if f.object and replaces(f)]
        values = []
        for f in chain:
            if values and values[-1][0].lower() == f.object.lower():
                continue
            values.append((f.object, f.metadata.get("valid_from", "")))
        if len(values) < 2:
            continue
        subj, attr = chain[0].subject, chain[0].predicate
        spans = [f"{v} ({start or '?'} – "
                 f"{values[i + 1][1] if i + 1 < len(values) else 'now'})"
                 for i, (v, start) in enumerate(values)]
        lines.append(f"- {subj} / {attr}: " + " → ".join(spans))
        if len(lines) == max_keys:
            break
    return "\n".join(lines)


def ledger(facts: list[Fact], *, own_only: bool = False,
           max_rows: int = 60) -> str:
    """Events as one line per distinct event, oldest first, deduplicated on
    the canonical entity: a wedding mentioned in three sessions is one row
    with three dates, not three weddings. Rows the user did not do
    themselves are kept and marked, unless own_only, because "is it the
    user's own" is the error a reader makes when counting:
      '- 2023/05/02 | wedding of rachel and mike | own | 2023/05/02, 2023/06/11'
    """
    rows: dict[str, list[Fact]] = {}
    for f in facts:
        if f.metadata.get("kind") != "event":
            continue
        if own_only and not f.metadata.get("own", True):
            continue
        key = f.metadata.get("entity") or _norm(f.metadata.get("statement", ""))
        rows.setdefault(key, []).append(f)
    out = []
    for key, group in rows.items():
        group.sort(key=lambda f: f.metadata.get("date") or "9999")
        first = group[0].metadata
        dates = list(dict.fromkeys(g.metadata.get("date") or "?" for g in group))
        amount = next((g.metadata["amount"] for g in group
                       if g.metadata.get("amount")), "")
        own = any(g.metadata.get("own", True) for g in group)
        out.append((first.get("date") or "9999", " | ".join(
            x for x in (first.get("date") or "?", key, amount,
                        "own" if own else "NOT the user's own",
                        "mentioned " + ", ".join(dates) if len(dates) > 1
                        else "") if x)))
    out.sort()
    return "\n".join(f"- {line}" for _, line in out[:max_rows])


def profile(chains: dict[str, list[Fact]], max_rows: int = 40) -> str:
    """The user's preferences, one line per (subject, attribute) at its
    latest value: what advice should build on whatever the question."""
    rows = []
    for chain in chains.values():
        prefs = [f for f in chain if f.metadata.get("kind") == "preference"]
        if prefs:
            f = prefs[-1]          # chains are date-ordered by link_validity
            rows.append((f.metadata.get("date") or "",
                         f"- {f.metadata['statement']}"))
    rows.sort(reverse=True)        # newest first, then cut
    return "\n".join(line for _, line in rows[:max_rows])
