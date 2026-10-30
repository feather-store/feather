"""Context assembly for the LongMemEval scenarios.

The answerer only ever sees the string built here, so it must carry what
the questions need:

  - dates. Temporal-reasoning questions (133 of 500) are date arithmetic
    against the question date: every session is headed with its date and
    its distance from the question date, computed here rather than by the
    model. The question date itself goes to the answerer separately.
  - roles. "What did you recommend" and "what did I say" are different
    questions.
  - sessions, not fragments. A count across sessions cannot be answered
    from a few turns, so `unit="session"` expands every retrieved turn to
    its session and ranks sessions by their best turn.
  - chronological order. Knowledge-update questions want the latest
    value, so sessions are rendered oldest first whatever their rank.

Neither the `has_answer` flag nor the session id is shown: the first is
the gold label, and in the oracle set evidence ids start with "answer_".
"""
from __future__ import annotations
import datetime
from dataclasses import dataclass
from typing import Callable, Optional


def parse_lme_date(s: str) -> Optional[datetime.datetime]:
    """LongMemEval dates are '2023/04/10 (Mon) 23:07', which
    `fromisoformat` rejects. Accepts ISO as well."""
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Turn:
    session_id: str
    session_date: str     # as given by the dataset; shown to the answerer
    role: str
    content: str
    order: int            # position in the haystack, for stable ordering


# Clip lengths for the compact renderings of a session.
_CLIP_OTHER = 300         # an assistant turn neither retrieved nor next to one
_CLIP_HIT = 2000          # a retrieved turn, in the compact renderings

# Renderings of a session, most complete first. User turns are 13% of
# LongMemEval's text and hold 94% of its evidence (842 of 896 turns), so
# assistant prose is shortened first and user turns stay whole in every
# level but the last two, the floors every retrieved session is admitted at.
VERBATIM, CLIP_ASSISTANT, DROP_ASSISTANT, HITS_ONLY, EVIDENCE = 0, 1, 2, 3, 4

# EVIDENCE: a session's best hit and this many turns either side. HITS_ONLY
# grows with the session's hit count: multi-round retrieval merges ~100
# turns per round, a top session carries 10+ of them, and admitting at
# HITS_ONLY spent 160k chars on 12-15 of ~25 sessions, cutting golds
# retrieved at ranks 10-24. EVIDENCE is bounded (~3k chars) whatever the
# hit count, so every retrieved session gets in before any is expanded.
_EVIDENCE_SPAN = 2


def _clip(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + " […]"


def session_header(session_date: str, question_date: str = "") -> str:
    """'[Session 2023/05/14 (Sun) 17:50 · 16 days before current date]'."""
    head = f"[Session {session_date or 'unknown date'}"
    d, q = parse_lme_date(session_date), parse_lme_date(question_date)
    if d and q:
        days = (q.date() - d.date()).days
        if days == 0:
            head += " · same day as current date"
        elif days > 0:
            head += f" · {days} day{'s' * (days != 1)} before current date"
        else:
            head += f" · {-days} days after current date"
    return head + "]"


def _render_session(turns: list[Turn], hits: set[int], level: int,
                    question_date: str = "", best: tuple = ()) -> str:
    """level VERBATIM: every turn whole. CLIP_ASSISTANT: assistant turns
    away from any hit clipped. DROP_ASSISTANT: those dropped, retrieved
    assistant turns clipped. HITS_ONLY: hit turns and their neighbours
    only, clipped. EVIDENCE: the best hit (best[0]) and _EVIDENCE_SPAN
    turns either side, clipped. User turns are whole at every level but
    HITS_ONLY and EVIDENCE."""
    span = 1
    if level == EVIDENCE:
        hits, span = set(best[:1]), _EVIDENCE_SPAN
    near = set()
    for i, t in enumerate(turns):
        if t.order in hits:
            near.update(range(i - span, i + span + 1))
    lines = []
    skipped = False
    for i, t in enumerate(turns):
        body = t.content.strip()
        if level in (HITS_ONLY, EVIDENCE):
            if i not in near:
                skipped = True
                continue
            body = _clip(body, _CLIP_HIT if t.order in hits else _CLIP_OTHER)
        elif level > VERBATIM and t.role != "user":
            if t.order in hits:
                if level == DROP_ASSISTANT:
                    body = _clip(body, _CLIP_HIT)
            elif level == DROP_ASSISTANT:
                skipped = True
                continue
            elif i not in near:
                body = _clip(body, _CLIP_OTHER)
        if skipped:
            lines.append("[…]")
            skipped = False
        lines.append(f"[{t.role or 'unknown'}] {body}")
    if skipped:
        lines.append("[…]")
    return (session_header(turns[0].session_date, question_date) + "\n"
            + "\n".join(lines))


def session_text(turns: list[Turn], question_date: str = "") -> str:
    """One whole session, dated and role-marked, as render() shows it at
    VERBATIM (what a count's map step reads)."""
    return _render_session(turns, set(), VERBATIM, question_date)


def build_context(ranked_hits: list[Turn], all_turns: list[Turn], **kw) -> str:
    """Render retrieved turns as dated sessions, oldest first (see
    render_context for the arguments)."""
    return render_context(ranked_hits, all_turns, **kw)[0]


@dataclass
class Rendered:
    text: str
    retrieved: list[str]          # sessions retrieved, in rank order
    in_context: list[str]         # sessions that made it in, in rank order
    levels: dict[str, int]        # session -> rendering level it got


def render_context(ranked_hits: list[Turn], all_turns: list[Turn], **kw
                   ) -> tuple[str, list[str], list[str]]:
    """(context, sessions retrieved in rank order, sessions that made it
    into the context in rank order); see render() for the arguments.
    Coverage is measured on the third: a session retrieved but cut by the
    budget never reached the reader."""
    r = render(ranked_hits, all_turns, **kw)
    return r.text, r.retrieved, r.in_context


def render(ranked_hits: list[Turn], all_turns: list[Turn], *,
           unit: str = "session", max_chars: int = 160_000,
           max_sessions: Optional[int] = None,
           question_date: str = "",
           max_tokens: Optional[int] = None,
           count_tokens: Optional[Callable[[str], int]] = None) -> Rendered:
    """ranked_hits: retrieved turns, best first.
    all_turns:   the whole haystack (for session expansion).
    unit:        "session" expands hits to their sessions; "turn" shows
                 only the hit turns, still grouped and dated.
    max_chars:   budget, filled breadth first: every session is admitted
                 in rank order at EVIDENCE (its best hit and neighbours),
                 then the
                 sessions are upgraded one level at a time, in rank order,
                 while the budget allows. Admitting the top sessions
                 verbatim first spent 160k chars on ~14 of ~22 retrieved
                 sessions, and a count needs all of them.
    max_tokens:  budget in tokens instead, measured by count_tokens (the
                 answer model's tokenizer). Characters are not what the
                 model reads: a token is ~4.5 chars of English prose and
                 fewer of numbers, dates and code.
    """
    if max_tokens is not None:
        if count_tokens is None:
            raise ValueError("max_tokens needs count_tokens")
        cost, limit = count_tokens, max_tokens
    else:
        cost, limit = len, max_chars
    sep = cost("\n\n")
    if unit not in ("session", "turn"):
        raise ValueError(f"unit must be 'session' or 'turn', not {unit!r}")

    by_session: dict[str, list[Turn]] = {}
    for t in all_turns:
        by_session.setdefault(t.session_id, []).append(t)

    hits_by_session: dict[str, set[int]] = {}
    best_hit: dict[str, int] = {}
    session_rank: list[str] = []
    for t in ranked_hits:
        if t.session_id not in hits_by_session:
            hits_by_session[t.session_id] = set()
            best_hit[t.session_id] = t.order
            session_rank.append(t.session_id)
        hits_by_session[t.session_id].add(t.order)
    retrieved = list(session_rank)
    if max_sessions is not None:
        session_rank = session_rank[:max_sessions]

    if unit == "turn":
        levels = (HITS_ONLY, VERBATIM)
        shown = {sid: [t for t in by_session[sid]
                       if t.order in hits_by_session[sid]]
                 for sid in session_rank}
    else:
        levels = (EVIDENCE, HITS_ONLY, DROP_ASSISTANT, CLIP_ASSISTANT,
                  VERBATIM)
        shown = {sid: by_session[sid] for sid in session_rank}

    def block(sid: str, level: int) -> str:
        return _render_session(shown[sid], hits_by_session[sid], level,
                               question_date, (best_hit[sid],))

    chosen: dict[str, str] = {}
    size: dict[str, int] = {}
    level_of: dict[str, int] = {}
    used = 0
    # Admit at the cheaper floor: EVIDENCE's wider window can cost more
    # than HITS_ONLY on a session with a single hit.
    floors = levels[:2] if levels[0] == EVIDENCE else levels[:1]
    for sid in session_rank:
        n, b, level = min((cost(b), b, lv) for lv in floors
                          for b in (block(sid, lv),))
        if used + n + sep <= limit:
            chosen[sid], size[sid], level_of[sid] = b, n, level
            used += n + sep
    for level in levels[1:]:
        for sid in chosen:
            b = block(sid, level)
            n = cost(b)
            if used + n - size[sid] <= limit:
                used += n - size[sid]
                chosen[sid], size[sid], level_of[sid] = b, n, level

    def when(sid: str):
        d = parse_lme_date(by_session[sid][0].session_date)
        return (d or datetime.datetime.max, by_session[sid][0].order)

    text = "\n\n".join(chosen[sid] for sid in sorted(chosen, key=when))
    return Rendered(text, retrieved, list(chosen), level_of)
