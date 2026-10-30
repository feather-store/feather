"""LongMemEval answering: the prompt, and a date-difference tool.

Shared by bench/ (LLMJudge) and benchmarks/longmemeval_qa.py. Each
procedure in the prompt targets one weak category; the tool exists
because models guess date arithmetic and temporal questions are 133 of
500.

The tool is a text protocol, not provider tool-calling: the model writes
`DAYS_BETWEEN(2023/04/10, 2023/05/20)` lines and stops, the harness
appends the results and asks it to continue. That works the same on every
provider, including ones that take only (system, user).
"""
from __future__ import annotations
import datetime
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

Chat = Callable[[str, str], str]   # (system, user) -> reply

ANSWER_SYSTEM = """\
You answer questions about a user from their past chat sessions with an assistant.

The context lists sessions oldest first. Each is headed like
  [Session 2023/05/14 (Sun) 17:50 · 16 days before current date]
and every turn is marked [user] or [assistant]. The current date is given
with the question, followed by the two weeks before it with their
weekdays. A "Changes over time" list may come first: values that
changed, each with the dates it held, extracted from these sessions. Use
it to find the latest value and when it changed; where it and a session
disagree, the session is right.

To fit, some sessions are shortened: assistant replies are cut ("[…]"
marks a cut), user turns are kept whole.

Use the procedure that fits the question (more than one may apply):

COUNTING ("how many", "how much in total", "list all"):
  Write this ledger in your reply; the answer is read off it, not recalled.
  1. WINDOW: if the question limits time ("in the past two weeks", "this
     year", "last month", "before X"), write its exact first and last date.
  2. CANDIDATES: go through the sessions ONE BY ONE, oldest first. For
     every possible instance write one line:
       <date> | <item> | IN or OUT: <reason>
     An instance mentioned in passing, or in a session mostly about
     something else, is still a candidate. Mark it OUT when it is
       - outside the window (date it by the session, or by a date stated
         in the text);
       - not the user's own: something the assistant suggested or listed,
         or the user only asked about, considered or planned (unless the
         question asks about plans);
       - the same item as an earlier line (the same concert, book or
         repair mentioned again), or a different kind than asked.
  3. If the user states a running total ("that's my third visit this year"), the
     latest such statement is the count; do not add instances to it.
  4. For sums of money or quantities, write each IN amount, then add.
  5. The answer is the number of IN lines (or the sum).

TIME SPANS ("how long ago", "how many days/weeks/months between", "when",
"which came first", "in what order"):
  1. Event date: a session date, or a date stated in the text. A relative
     date ("last Tuesday", "two weeks ago") counts back from the date of
     the session it was said in.
  2. Reference date: the other event, or the current date.
  3. Difference: call DAYS_BETWEEN for every difference your answer rests
     on, even one that looks easy. Never do date arithmetic in your head;
     an answer without the call is sent back.
  "Last Saturday", "the past weekend", "last week": read the date off the
  calendar under the current date.

KNOWLEDGE UPDATES (a value that may have changed: current job, latest
number, where they live now, a plan that was revised):
  The most recent statement wins. Give the current value, and say when it
  changed if the sessions show it.

PREFERENCES (recommendations, suggestions, tips, "what should I…",
"any ideas…"):
  Never answer "unknown": a request for suggestions always gets them.
  Find what the user said they like, use, own, avoid or plan, including in
  related contexts (a hotel they loved in another city, a hobby, a past
  purchase), and carry those preferences over. Name them, and give
  specific suggestions that fit them; no generic advice. Here the final
  answer is the full personalised recommendation, several sentences.

ABSTENTION (factual questions about the user's past, not requests for
suggestions):
  If what the question needs is not in the sessions, say it is unknown
  (it was not mentioned). Do not guess, and do not substitute a similar
  item. But when the facts are present and only need combining, counting
  or arithmetic, answer.
  Before answering "unknown", search every session again for the thing
  under other words: the user names the specific item, the question the
  category (a sedan is a vehicle, a sonnet is a poem, "my Kindle"
  is an e-reader), and a date may be relative to its session
  ("yesterday" in a session of 2023/03/05 is 2023/03/04).

PREMISE CHECK (every factual question, before its final answer):
  A question can presuppose something that never happened. List what it
  takes as given: each named person, place, item, role and event, and
  each thing it says the user did. For each, point to where the USER
  says it. A premise fails when
    - it appears nowhere, or only in an assistant reply (the assistant
      listing gym prices does not mean the user joined a gym);
    - the sessions name something different in its place (a "team lead"
      role is not a "director" one; a car bought from a dealer is not a
      car bought from the user's brother);
    - one of the quantities a sum or difference needs is missing.
  A category word in the question may match a specific item (see above);
  a NAME in the question must match the same name. If a premise fails,
  the answer is that it was not mentioned: say which premise is missing
  and what the sessions do say instead. Do not answer around it.

TOOL:
  DAYS_BETWEEN(YYYY/MM/DD, YYYY/MM/DD) returns the exact days, weeks and
  months between two dates. To use it, write each call on its own
  line and end your message there; the results will be sent back.

Reason step by step, then end with:
FINAL ANSWER: <the answer: concise for facts, counts and dates; the full
recommendation for preference questions>"""

CHANGES_HEADER = ("[Changes over time — extracted from the sessions below; "
                  "where they disagree, the sessions win]")
# Written by the extractor once per session, so the reader is not the
# first to resolve "last Saturday" or to notice a wedding mentioned twice.
LEDGER_HEADER = (
    "[Event ledger — extracted from the sessions below, one row per "
    "distinct event: date | event | amount | whose | the dates it was "
    "mentioned on. For a count or a total, start from these rows: a row "
    "mentioned on several dates is still one event, and a row marked NOT "
    "the user's own does not count for the user. Check each row you use "
    "against its session; the sessions win.]")
PROFILE_HEADER = (
    "[About the user — preferences, habits and things they own or use, "
    "latest first. A recommendation should build on these.]")

ANSWER_USER_TMPL = """\
Sessions (oldest first):

{context}

Current date: {question_date}{calendar}
Question: {question}"""

_CALL_RE = re.compile(
    r"DAYS_BETWEEN\(\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*,\s*"
    r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*\)")
_FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE | re.DOTALL)
MAX_TOOL_ROUNDS = 3
CALENDAR_DAYS = 14

# A question whose answer is a date difference. Only 4 of 12 temporal
# failures had called DAYS_BETWEEN; the rest did the arithmetic unaided.
_SPAN_Q_RE = re.compile(
    r"\bhow (?:many|much) (?:\w+ )?(?:days?|weeks?|months?|years?)\b|"
    r"\bhow long\b|\bago\b|\bhow old\b|\b(?:had )?passed\b", re.I)
TOOL_NUDGE = ("You answered without calling DAYS_BETWEEN. Write the "
              "DAYS_BETWEEN call for each date difference your answer rests "
              "on, one per line, and end your message there.")


def _date(s: str) -> datetime.date:
    y, m, d = (int(x) for x in re.split(r"[/-]", s))
    return datetime.date(y, m, d)


def days_between(a: str, b: str) -> str:
    """'DAYS_BETWEEN(a, b) = …' for two YYYY/MM/DD dates, either order."""
    try:
        d1, d2 = _date(a), _date(b)
    except ValueError as e:
        return f"DAYS_BETWEEN({a}, {b}) = error: {e}"
    lo, hi = sorted((d1, d2))
    days = (hi - lo).days
    months = (hi.year - lo.year) * 12 + hi.month - lo.month - (hi.day < lo.day)
    sign = "" if d2 >= d1 else " (the second date is earlier)"
    # "How many months" is usually asked loosely: Dec 20 -> Jun 10 is 5
    # full months but "about 6".
    return (f"DAYS_BETWEEN({a}, {b}) = {days} days = {days // 7} weeks "
            f"{days % 7} days = {months} full calendar months "
            f"(≈{days / 30.44:.1f} months, ≈{days / 365.25:.1f} years){sign}")


def is_span_question(question: str) -> bool:
    return bool(_SPAN_Q_RE.search(question or ""))


def calendar(question_date: str, days: int = CALENDAR_DAYS) -> str:
    """The days before the current date with their weekdays, so "last
    Saturday" is read off rather than computed. "" when undated."""
    m = re.match(r"\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})", question_date or "")
    if not m:
        return ""
    try:
        today = _date(m.group(1))
    except ValueError:
        return ""
    back = (today - datetime.timedelta(days=i) for i in range(1, days + 1))
    return "\nThe days before it: " + ", ".join(
        d.strftime("%a %Y/%m/%d") for d in back)


def final_answer(reply: str) -> str:
    """The text after the last FINAL ANSWER:, else the whole reply."""
    hits = list(_FINAL_RE.finditer(reply or ""))
    return hits[-1].group(1).strip() if hits else (reply or "").strip()


def answer(chat: Callable[[str, str], str], question: str, context: str,
           question_date: str) -> str:
    """Run the answer prompt, serving DAYS_BETWEEN calls. Returns the
    model's last reply; use final_answer() on it."""
    user = ANSWER_USER_TMPL.format(context=context, question=question,
                                   question_date=question_date or "unknown",
                                   calendar=calendar(question_date))
    return _with_tools(chat, ANSWER_SYSTEM, user, is_span_question(question))


def _with_tools(chat: Chat, system: str, user: str, nudge: bool) -> str:
    """One exchange, serving DAYS_BETWEEN calls for up to MAX_TOOL_ROUNDS.
    nudge: a reply that called no tool is sent back once."""
    transcript: list[str] = []
    served: set[tuple[str, str]] = set()
    for _ in range(MAX_TOOL_ROUNDS + 1):
        reply = chat(system, user if not transcript else
                     user + "\n\n" + "\n\n".join(transcript))
        transcript.append(f"Your reasoning so far:\n{reply.strip()}")
        # A reply that guessed a FINAL ANSWER alongside its calls still
        # gets the results: the guess is what the tool is there to replace.
        calls = [c for c in dict.fromkeys(_CALL_RE.findall(reply))
                 if c not in served]
        if not calls:
            # A date difference answered without the tool goes back once.
            if nudge and not served:
                nudge = False
                transcript.append(TOOL_NUDGE)
                continue
            break
        served.update(calls)
        results = "\n".join(days_between(a, b) for a, b in calls)
        transcript.append(f"Tool results:\n{results}\n\nContinue, then end "
                          "with FINAL ANSWER:")
    return reply


# ── Self-consistency for counts ───────────────────────────────────────────
# A count is where one reading most often slips (an instance missed in a
# long session). Sampling several readings and taking the majority answer
# smooths that out without seeing the gold answer.

_COUNT_Q_RE = re.compile(
    r"\b(how many|how much|how often|total|in total|number of|altogether|"
    r"combined|in all)\b", re.I)
_NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    .split())}
_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?|\b(?:" + "|".join(_NUM_WORDS) + r")\b",
                     re.I)


def is_count_question(question: str) -> bool:
    return bool(_COUNT_Q_RE.search(question or ""))


def vote_key(final: str) -> str:
    """What two answers must share to count as the same answer: the first
    number (digits or a word up to twenty), else the normalised text."""
    m = _NUM_RE.search(final or "")
    if m:
        tok = m.group(0).lower().lstrip("$").replace(",", "")
        return str(_NUM_WORDS.get(tok, tok))
    return " ".join(re.sub(r"[^\w\s]", " ", (final or "").lower()).split())


def answer_vote(chats: list[Chat], question: str, context: str,
                question_date: str) -> tuple[str, str]:
    """Run answer() once per chat (the first should be temperature 0) in
    parallel; return (final answer, transcript) of the first reading whose
    vote_key has the most votes. Ties go to the earliest chat."""
    with ThreadPoolExecutor(max_workers=len(chats)) as pool:
        replies = list(pool.map(
            lambda c: _safe_answer(c, question, context, question_date), chats))
    finals = [final_answer(r) for r in replies]
    keys = [vote_key(f) for f in finals]
    counts = Counter(k for k, r in zip(keys, replies) if r)
    if not counts:
        return finals[0], replies[0]
    best = max(counts.values())
    winner = next(i for i, k in enumerate(keys) if counts.get(k) == best and replies[i])
    tally = ", ".join(f"{k!r}×{n}" for k, n in counts.most_common())
    return finals[winner], f"[vote: {tally}]\n\n{replies[winner]}"


def _safe_answer(chat: Chat, question: str, context: str,
                 question_date: str) -> Optional[str]:
    try:
        return answer(chat, question, context, question_date)
    except Exception:
        return ""


# ── Map-reduce for counts ─────────────────────────────────────────────────
# A count of k is right only if every instance is spotted (~r**k) and no
# look-alike is let in. One read of a 160k-char context does both at once,
# over a context the budget has already shortened. Here each retrieved
# session is read whole by its own cheap call, which lists every candidate
# with its date and status and decides nothing; one reduce call reads only
# those lists and applies the rules (window, the user's own, duplicates,
# running totals). Maps are for recall, the reduce for precision: a map
# that filtered would make 20 precision calls that cannot see each other.

MAP_WORKERS = 8

MAP_SYSTEM = """\
You read ONE past chat session between a user and an assistant and list \
everything in it that could bear on a question about the user. You do not \
answer the question.

List every candidate instance, even one mentioned in passing or in a \
session mostly about something else, and even if you doubt it counts: a \
later step decides. One line each:
<YYYY/MM/DD> | <the item, specific: names, amounts, quantities> | <status> | "<short quote>"

<YYYY/MM/DD>: when it happened. The session date unless the text dates \
it; make relative dates absolute from the session date ("last Tuesday" in \
a session of 2023/05/18 (Thu) is 2023/05/16). Write "undated" if unknown.
<status>: one of
  user did/has     - the user did, owns, experienced or attended it
  user plans       - the user intends or considers it, not yet done
  user asked       - the user only asked about it
  assistant said   - the assistant suggested or listed it, the user did not take it up
  running total    - the user states a total so far ("my third visit this year")

If nothing in the session bears on the question, reply exactly: NONE"""

MAP_USER_TMPL = """\
Current date: {question_date}
Question (do not answer it): {question}

{session}"""

REDUCE_SYSTEM = """\
You answer a counting or totalling question about a user. Every relevant \
past session has been read separately, and each listed its candidate \
instances as
  <date> | <item> | <status> | "<quote>"
You see those lists, session by session, oldest first. Sessions that \
listed nothing are not shown.

Write this ledger, then answer from it:
1. WINDOW: if the question limits time ("in the past two weeks", "this
   year", "last month", "before X"), write its exact first and last date.
2. One line per candidate: <date> | <item> | IN or OUT: <reason>. OUT if
   - outside the window;
   - not the user's own: status "assistant said" or "user asked", or
     "user plans" unless the question asks about plans;
   - the same item as an earlier line (the same concert, book or
     repair mentioned in several sessions), or not the kind asked.
3. If there is a "running total", the latest one is the count; do not
   add instances to it.
4. For sums of money or quantities, write each IN amount, then add.
5. The answer is the number of IN lines (or the sum). If no list holds
   what the question needs, the answer is that it was not mentioned.

TOOL:
  DAYS_BETWEEN(YYYY/MM/DD, YYYY/MM/DD) returns the exact days, weeks and
  months between two dates. To use it, write each call on its own line
  and end your message there; the results will be sent back.

End with:
FINAL ANSWER: <the count or total, with the items it counts>"""

# The same reduce for a comparison or an order: which is the most / least,
# the order several things happened in. The IN/OUT ledger is what keeps a
# look-alike or a repeat from tipping a comparison, as it does a count.
AGG_REDUCE_SYSTEM = """\
You answer a question that compares or orders things about a user: which \
is the most or least, the largest or smallest, or the order several \
things happened in. Every relevant past session has been read \
separately, and each listed its candidate instances as
  <date> | <item> | <status> | "<quote>"
You see those lists, session by session, oldest first. Sessions that \
listed nothing are not shown.

Write this ledger, then answer from it:
1. WINDOW: if the question limits time ("in the past month", "this
   year", "before X"), write its exact first and last date.
2. One line per candidate: <date> | <item> | <amount if any> | IN or
   OUT: <reason>. OUT if
   - outside the window;
   - not the user's own: status "assistant said" or "user asked", or
     "user plans" unless the question asks about plans;
   - the same instance as an earlier line (one purchase mentioned in
     several sessions), or not the kind asked.
3. Group the IN lines by what the question compares (store, person,
   item) and add each group's amounts; or, for an order, sort the IN
   lines by date, using DAYS_BETWEEN where two dates are close.
4. The answer is the group that wins, or the order. If no list holds
   what the question needs, the answer is that it was not mentioned.

TOOL:
  DAYS_BETWEEN(YYYY/MM/DD, YYYY/MM/DD) returns the exact days, weeks and
  months between two dates. To use it, write each call on its own line
  and end your message there; the results will be sent back.

End with:
FINAL ANSWER: <the answer, with the amounts or dates it rests on>"""

# Ledger rows (lme_facts / feather_db.extractors.ledger) come first when
# there are any: extracted once per session, already deduplicated, and a
# second source for an instance a map read missed.
LEDGER_BLOCK = """\
Event ledger (extracted from every session, one row per distinct event: \
date | event | amount | whose | the dates it was mentioned on). Treat each \
row as one more candidate; an event both here and in a list is one \
instance:
{ledger}

"""

REDUCE_USER_TMPL = """\
{ledger}Candidates from {n_listed} of {n_read} sessions read{failed}:

{lists}

Current date: {question_date}{calendar}
Question: {question}"""


def is_map_reduce_question(question: str) -> bool:
    """Counts and totals; a date span ("how many days ago") is not one."""
    return is_count_question(question) and not is_span_question(question)


def _map_lines(reply: str) -> list[str]:
    """The candidate lines of a map reply; [] for NONE or no lines."""
    lines = [ln.strip().lstrip("-* ").strip() for ln in (reply or "").splitlines()]
    return [ln for ln in lines if ln.count("|") >= 2]


def map_session(chat: Chat, question: str, session: str,
                question_date: str) -> list[str]:
    return _map_lines(chat(MAP_SYSTEM, MAP_USER_TMPL.format(
        question=question, question_date=question_date or "unknown",
        session=session)))


def count_map_reduce(map_chat: Chat, reduce_chat: Chat, question: str,
                     sessions: list[str], question_date: str,
                     workers: int = MAP_WORKERS,
                     ledger: str = "") -> tuple[str, str, dict]:
    """(final answer, transcript, stats). sessions: whole sessions, each
    rendered with its dated header, oldest first. ledger: event ledger
    rows, shown to the reduce as more candidates. A question that is not
    a count or total (a "most", an order) is reduced by AGG_REDUCE_SYSTEM.
    Raises RuntimeError if every map call failed, so the caller can fall
    back to one read."""
    def one(session: str) -> Optional[list[str]]:
        try:
            return map_session(map_chat, question, session, question_date)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(sessions)))) as pool:
        found = list(pool.map(one, sessions))
    failed = sum(f is None for f in found)
    if sessions and failed == len(sessions):
        raise RuntimeError("every map call failed")
    blocks = [session.split("\n", 1)[0] + "\n" + "\n".join(lines)
              for session, lines in zip(sessions, found) if lines]
    lists = "\n\n".join(blocks) or "(no session listed anything)"
    user = REDUCE_USER_TMPL.format(
        ledger=LEDGER_BLOCK.format(ledger=ledger) if ledger else "",
        n_listed=len(blocks), n_read=len(sessions) - failed,
        failed=f" ({failed} could not be read)" if failed else "",
        lists=lists, question=question,
        question_date=question_date or "unknown",
        calendar=calendar(question_date))
    system = (REDUCE_SYSTEM if is_map_reduce_question(question)
              else AGG_REDUCE_SYSTEM)
    reply = _with_tools(reduce_chat, system, user, nudge=False)
    stats = {"sessions_read": len(sessions) - failed, "map_failed": failed,
             "sessions_listed": len(blocks),
             "candidates": sum(len(f or ()) for f in found)}
    return final_answer(reply), f"[map lists]\n{lists}\n\n[reduce]\n{reply}", stats
