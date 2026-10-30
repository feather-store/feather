"""The routed LongMemEval reader: a question goes to one focused procedure,
and an abstention or an answer gets a check of its own.

lme_answer.ANSWER_SYSTEM gives one reading six procedures at once, and
they pull against each other: the premise check that stops the model
answering a question about something that never happened is the same text
that makes it give up on a question whose answer is there. On the 0.904
run 11 answerable questions came back "not mentioned", 9 of them with
every gold session in the context, while 5 unanswerable ones were
answered anyway. Here:

  1. route     one short call names the question's kind; the reader gets
               the shared preamble and that kind's procedure only.
  2. draft     the routed reading, without the premise check.
  3. verify    a draft that abstains gets a second retrieval (paraphrased
               queries + BM25 on the key nouns, run by the caller) and a
               reading that must QUOTE the turn that answers. The answer
               stands only if the quote is found in what it was shown;
               otherwise the abstention stays.
  4. premises  a draft that answers gets its own premise check, one line
               per premise, SUPPORTED with a quote or UNSUPPORTED. Any
               unsupported premise turns the answer into "not mentioned".

Aggregations (counts, totals, most/least, an order of several things) go
to the caller's map-reduce when it has one: every session read on its own,
one reduce over the lists and the event ledger.

The prompt pieces are sliced from ANSWER_SYSTEM, not retyped, so each
procedure is the text the single reader was measured with.
"""
from __future__ import annotations
import json
import re
from typing import Callable, Optional

from . import lme_answer
from .lme_answer import ANSWER_SYSTEM, Chat

AGGREGATE, TEMPORAL, UPDATE, PREFERENCE, FACT = (
    "aggregate", "temporal", "update", "preference", "fact")
ROUTES = (AGGREGATE, TEMPORAL, UPDATE, PREFERENCE, FACT)

# ── Prompt pieces, sliced from ANSWER_SYSTEM ─────────────────────────────
_HEADINGS = ("COUNTING (", "TIME SPANS (", "KNOWLEDGE UPDATES (",
             "PREFERENCES (", "ABSTENTION (", "PREMISE CHECK (", "TOOL:\n")


def _sections(text: str) -> tuple[str, dict[str, str], str]:
    """(preamble before the procedures, {heading: section}, the tool and
    answer-format tail)."""
    intro = text.index("Use the procedure that fits")
    starts = [text.index(h) for h in _HEADINGS]
    secs = {h: text[a:b] for h, a, b in zip(_HEADINGS, starts, starts[1:])}
    return text[:intro], secs, text[starts[-1]:]


_HEAD, _SECS, _TAIL = _sections(ANSWER_SYSTEM)

_FACT_PROCEDURE = """\
FACTS (what, which, who, where, when, how much — one thing the user said
or did):
  Find the turn that answers. Quote it in your reasoning before you
  answer, and answer from the quote.

"""

PROCEDURES = {
    AGGREGATE:  _SECS["COUNTING ("],
    TEMPORAL:   _SECS["TIME SPANS ("],
    UPDATE:     _SECS["KNOWLEDGE UPDATES ("],
    PREFERENCE: _SECS["PREFERENCES ("],
    FACT:       _FACT_PROCEDURE,
}


def system_for(route: str) -> str:
    """The answer prompt for one route: its procedure, plus the abstention
    rule for every factual route. No premise check: that is its own call."""
    body = PROCEDURES[route]
    if route != PREFERENCE:
        body += _SECS["ABSTENTION ("]
    return _HEAD + "Use this procedure:\n\n" + body + _TAIL


# ── 1. Route ──────────────────────────────────────────────────────────────
CLASSIFY_SYSTEM = """\
Classify a question a user asks about their own past chat sessions. \
Reply with one word:
aggregate   - counts, totals or sums across several things, which one is \
the most/least/largest, or the order of several things ("how many", "how \
much in total", "which did I spend most on", "in what order")
temporal    - dates and durations: when, how long ago, how many days \
between, which of two things came first
update      - a value that may have changed over time: current job, \
latest number, where something is now. Also a "how many" whose answer \
is a number the user states and revises ("how many engineers do I \
lead", "how many times have I worn them"): the latest statement \
answers it, not a count of instances
preference  - asks for a recommendation, suggestion, tips or ideas
fact        - anything else: one thing the user said or did"""

_AGG_RE = re.compile(
    r"\b(most|least|highest|lowest|biggest|largest|smallest|longest|"
    r"shortest|in what order|order of|rank)\b", re.I)
_PREF_RE = re.compile(
    r"\b(recommend|suggest|tips?|ideas?|advice|what should i|any good)\b", re.I)
_UPDATE_RE = re.compile(r"\b(currently|now|latest|current|still|anymore)\b",
                        re.I)


def is_aggregation_question(question: str) -> bool:
    """Counts and totals (lme_answer.is_map_reduce_question), and the
    superlatives and orderings that need every instance just as much."""
    return (lme_answer.is_map_reduce_question(question)
            or bool(_AGG_RE.search(question or "")))


def route_heuristic(question: str) -> str:
    """The route without a model call: the fallback when classify fails."""
    if is_aggregation_question(question):
        return AGGREGATE
    if lme_answer.is_span_question(question):
        return TEMPORAL
    if _PREF_RE.search(question or ""):
        return PREFERENCE
    if _UPDATE_RE.search(question or ""):
        return UPDATE
    return FACT


def classify(chat: Chat, question: str) -> str:
    try:
        reply = chat(CLASSIFY_SYSTEM, f"Question: {question}").lower()
    except Exception:
        return route_heuristic(question)
    found = [r for r in ROUTES if re.search(rf"\b{r}\b", reply)]
    return found[0] if len(found) == 1 else route_heuristic(question)


# ── Abstention, and quotes ────────────────────────────────────────────────
_ABSTAIN_RE = re.compile(
    r"not mentioned|\bunknown\b|no information|not enough information|"
    r"not (?:provided|specified|stated|available|recorded)|"
    r"(?:does|did)(?: not|n't) (?:mention|say|specify)|cannot be determined|"
    r"there is no (?:record|mention)|is(?: not|n't) mentioned", re.I)


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_RE.search(answer or ""))


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def quote_in(quote: str, text: str, shingle: int = 5,
             min_share: float = 0.6) -> bool:
    """Whether `quote` is in `text`, allowing for the trimming and small
    slips a model makes when it copies: verbatim after normalising case,
    punctuation and spacing, or most of its 5-word runs are. A quote under
    four words is too common to prove anything."""
    q, t = _norm(quote), _norm(text)
    words = q.split()
    if len(words) < 4:
        return False
    if q in t:
        return True
    runs = [" ".join(words[i:i + shingle])
            for i in range(len(words) - shingle + 1)]
    return bool(runs) and sum(r in t for r in runs) / len(runs) >= min_share


def _first_json(raw: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


# ── 3. Verify an abstention ───────────────────────────────────────────────
RESEARCH_SYSTEM = """\
A question about a user's past chat sessions was answered "not mentioned". \
Before accepting that, the sessions will be searched again. Write the \
searches. The user may have named the specific thing where the question \
names its category (a sedan is a vehicle, "my Kindle" an e-reader), or \
said it in other words.

Reply with JSON only:
{"queries": ["<3 to 5 paraphrases of the question, each a different way \
the user might have said it>"], "nouns": "<the key nouns and names, space \
separated, for a keyword search>"}"""

VERIFY_SYSTEM = """\
You check an answer of "not mentioned" to a question about a user's past \
chat sessions. Below are the sessions the first reading had, then turns \
found by a second, wider search.

Look for the turn that answers the question, including in other words, \
as a specific item the question names by its category, or as facts that \
only need combining or arithmetic. Then reply in exactly one of two forms.

If a turn answers it:
QUOTE: "<the words of that turn, copied exactly>"
FINAL ANSWER: <the answer>

If no turn answers it:
CLOSEST: "<the turn that comes closest, copied exactly>"
WHY NOT: <what it says instead>
FINAL ANSWER: not mentioned — <what the sessions say instead>

A quote is checked against the text you were shown; a quote that is not \
there is discarded, and so is the answer that rests on it."""

VERIFY_USER_TMPL = """\
Sessions the first reading had:

{context}

Turns found by the second search:

{evidence}

Current date: {question_date}{calendar}
Question: {question}
First answer: {draft}"""

_QUOTE_RE = re.compile(r'QUOTE:\s*"?(.+?)"?\s*$', re.M)


def research_queries(chat: Chat, question: str) -> tuple[list[str], str]:
    """(paraphrased queries, key nouns). Falls back to the question."""
    try:
        got = _first_json(chat(RESEARCH_SYSTEM, f"Question: {question}"))
    except Exception:
        got = None
    queries = [str(q) for q in (got or {}).get("queries") or [] if str(q).strip()]
    nouns = str((got or {}).get("nouns") or "").strip()
    return (queries[:5] or [question]), nouns


def verify_abstention(chat: Chat, question: str, draft: str, context: str,
                      evidence: str, question_date: str) -> tuple[str, dict]:
    """(final answer, trace). The answer replaces the abstention only if
    the verifier quotes a turn that is really in what it was shown."""
    reply = chat(VERIFY_SYSTEM, VERIFY_USER_TMPL.format(
        context=context, evidence=evidence or "(nothing new found)",
        question=question, draft=draft,
        question_date=question_date or "unknown",
        calendar=lme_answer.calendar(question_date)))
    final = lme_answer.final_answer(reply)
    m = _QUOTE_RE.search(reply)
    quote = m.group(1).strip() if m else ""
    found = bool(quote) and quote_in(quote, context + "\n" + evidence)
    trace = {"reply": reply, "quote": quote, "quote_found": found}
    if found and not is_abstention(final):
        return final, trace
    return draft, trace


# ── 4. Premise check ──────────────────────────────────────────────────────
PREMISE_SYSTEM = """\
A question about a user's past chat sessions can presuppose something \
that never happened. You check its premises; you do not answer it.

List what the question takes as given: each named person, place, item, \
role and event, and each thing it says the user did. For each, find \
where the USER says it. One line each:
<premise> | SUPPORTED | "<the user's words, copied exactly>"
<premise> | UNSUPPORTED | <what the sessions say instead, or "nowhere">

A premise is UNSUPPORTED when
  - it appears nowhere, or only in an assistant reply (the assistant \
listing gym prices does not mean the user joined a gym);
  - the sessions name something different in its place (a "team lead" \
role is not a "director" one; a car bought from a dealer is not a car \
bought from the user's brother);
  - one of the quantities a sum or difference needs is missing.
A category word in the question may match a specific item the user named \
(a sedan is a vehicle); a NAME in the question must match the same name. \
Relative dates count back from the session they were said in.

End with:
VERDICT: SUPPORTED or UNSUPPORTED"""

PREMISE_USER_TMPL = """\
Sessions (oldest first):

{context}

Current date: {question_date}
Question: {question}"""

_VERDICT_RE = re.compile(r"VERDICT:\s*(SUPPORTED|UNSUPPORTED)", re.I)
_PREMISE_LINE_RE = re.compile(r"^\s*[-*]?\s*(.+?)\s*\|\s*(SUPPORTED|UNSUPPORTED)"
                              r"\s*\|\s*(.*)$", re.M | re.I)


def check_premises(chat: Chat, question: str, context: str,
                   question_date: str) -> dict:
    """{"verdict": "supported"|"unsupported"|"unclear", "premises":
    [(premise, status, evidence)], "reply"}. Unsupported needs both the
    verdict line and a premise line that says so: one without the other
    is a reply that did not follow the format, not evidence."""
    reply = chat(PREMISE_SYSTEM, PREMISE_USER_TMPL.format(
        context=context, question=question,
        question_date=question_date or "unknown"))
    lines = [(p.strip(), s.upper(), e.strip())
             for p, s, e in _PREMISE_LINE_RE.findall(reply)]
    m = _VERDICT_RE.findall(reply)
    verdict = m[-1].upper() if m else ""
    failed = [x for x in lines if x[1] == "UNSUPPORTED"]
    if verdict == "UNSUPPORTED" and failed:
        out = "unsupported"
    elif verdict == "SUPPORTED" or (lines and not failed):
        out = "supported"
    else:
        out = "unclear"
    return {"verdict": out, "premises": lines, "reply": reply}


def premise_answer(premises: list[tuple[str, str, str]]) -> str:
    failed = [(p, e) for p, s, e in premises if s == "UNSUPPORTED"]
    return "Not mentioned: " + "; ".join(
        f"{p} — the sessions say {e.strip(chr(34)) or 'nothing about it'}"
        for p, e in failed)


# ── The readers ───────────────────────────────────────────────────────────
def _verify(answer_chat: Chat, aux_chat: Chat, question: str, context: str,
            question_date: str, final: str, transcript: str, trace: dict,
            search) -> tuple[str, str]:
    """Re-search and verify an abstaining draft. A check can only revise
    the draft; one that fails (an API error after retries) leaves it
    standing and says so in the trace: failing the question would throw
    away an answer we have."""
    if not is_abstention(final) or search is None:
        return final, transcript
    try:
        queries, nouns = research_queries(aux_chat, question)
        evidence = search(queries, nouns)
        trace["research"] = {"queries": queries, "nouns": nouns,
                             "evidence_chars": len(evidence)}
        verified, vtrace = verify_abstention(
            answer_chat, question, final, context, evidence, question_date)
        trace["verify"] = vtrace
        return verified, transcript + f"\n\n[verify]\n{vtrace['reply']}"
    except Exception as e:
        trace["verify_error"] = f"{type(e).__name__}: {e}"
        return final, transcript


def verified_answer(answer_chat: Chat, aux_chat: Chat, question: str,
                    context: str, question_date: str, *,
                    search: Optional[Callable[[list[str], str], str]] = None
                    ) -> tuple[str, str, dict]:
    """The single reader (lme_answer.ANSWER_SYSTEM, premise check and all)
    plus one safe addition: a draft that abstains is searched again and
    must be overturned by a quote that is really there.

    This is what survived measurement. On all 500 (vs the single reader's
    0.904), routed_answer scored 0.794: its separate premise check
    flipped 81 answers to "not mentioned" and 46 of them had been right
    (LongMemEval's unanswerable questions are built from near-miss
    entities, so no word-overlap guard separates the good flips from the
    bad); map-reduce lost 11 net on aggregations; the sliced per-route
    prompts lost on preferences. The verify step changed one answer, to a
    right one, broke none, and every quote it gave was real."""
    trace: dict = {"route": "single"}
    reply = lme_answer.answer(answer_chat, question, context, question_date)
    final, transcript = lme_answer.final_answer(reply), reply
    trace["draft"] = final
    final, transcript = _verify(answer_chat, aux_chat, question, context,
                                question_date, final, transcript, trace,
                                search)
    return final, transcript, trace


def routed_answer(answer_chat: Chat, aux_chat: Chat, question: str,
                  context: str, question_date: str, *,
                  search: Optional[Callable[[list[str], str], str]] = None,
                  map_reduce: Optional[Callable[[str], tuple[str, str]]] = None,
                  premise_check: bool = True) -> tuple[str, str, dict]:
    """(final answer, transcript, trace). MEASURED WORSE than the single
    reader (0.794 vs 0.904 on all 500; see verified_answer): kept to
    reproduce that result, not to use.

    answer_chat: the answer model (drafts, verification, premises).
    aux_chat:    a short call (route, research queries).
    search(queries, nouns): the caller's second retrieval; returns the
                 turns found, rendered. None: abstentions are kept as is.
    map_reduce(question): the caller's map-reduce for an aggregation;
                 returns (final, transcript) or raises to fall back.
    premise_check: check the premises of every answer that is not an
                 abstention, including one the verifier recovered (it
                 may have found a look-alike).
    """
    trace: dict = {"route": classify(aux_chat, question)}
    route = trace["route"]
    final = transcript = ""
    if route == AGGREGATE and map_reduce is not None:
        try:
            final, transcript = map_reduce(question)
            trace["map_reduce"] = True
        except Exception as e:
            trace["map_reduce"] = f"{type(e).__name__}: {e}"
    if not transcript:
        user = lme_answer.ANSWER_USER_TMPL.format(
            context=context, question=question,
            question_date=question_date or "unknown",
            calendar=lme_answer.calendar(question_date))
        reply = lme_answer._with_tools(answer_chat, system_for(route), user,
                                       lme_answer.is_span_question(question))
        final, transcript = lme_answer.final_answer(reply), reply
    trace["draft"] = final
    if route == PREFERENCE:
        return final, transcript, trace

    final, transcript = _verify(answer_chat, aux_chat, question, context,
                                question_date, final, transcript, trace,
                                search)
    if premise_check and not is_abstention(final):
        try:
            premises = check_premises(answer_chat, question, context,
                                      question_date)
        except Exception as e:
            trace["premise_error"] = f"{type(e).__name__}: {e}"
        else:
            trace["premises"] = {k: premises[k] for k in ("verdict", "premises")}
            transcript += f"\n\n[premises]\n{premises['reply']}"
            if premises["verdict"] == "unsupported":
                final = premise_answer(premises["premises"])
    return final, transcript, trace
