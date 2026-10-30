"""LongMemEval scenario.

For each question:
  1. Open a fresh in-memory Feather DB.
  2. Embed every chat turn in the haystack and add to the DB with
     metadata (timestamp, session_id, role, namespace = question_id).
     The has_answer gold label is not stored.
  3. Embed the question and run hybrid_search.
  4. Build the context (`lme_context.build_context`): retrieved turns
     expanded to their sessions (unit="session", default) or kept as
     turns (unit="turn"), each session headed by its date, every turn
     by its role, sessions oldest first. The question date goes to the
     answerer alongside it.
  5. Score: substring of the context, or LLM answerer + LLM judge.

Decay is off: the CLI refuses --decay-* for this scenario (the arguments
remain for ablations from Python). Temporal and multi-session questions
need old evidence, and decay ranks it lower. Recency reaches the answerer
as session dates and the prompt's "latest wins" rule instead.

Output (see lme_eval.summarize):
    metrics["overall"]               official judge if run, else lenient
    metrics["overall_official"]      / overall_lenient, by_question_type_*,
                                     task_averaged_*, abstention_*
    metrics["judge_agreement"]       official vs lenient
    metrics["failure_buckets"]       retrieval_miss / packing_miss /
                                     reader_wrong /
                                     judge_disagreement / judge_error
    metrics["by_axis"]               {axis: mean_score}, headline judge
    metrics["questions_file"]        per-question JSONL
    metrics["per_q_seconds_p50"]
    metrics["embeddings_emitted"]    total embed() calls
    metrics["history_turns_total"]
"""
from __future__ import annotations
import datetime
import glob
import os
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Optional

import feather_db

from feather_db.extractors import (fact_key, ledger, link_validity, profile,
                                   timeline)
from feather_db.filter import FilterBuilder

from .. import lme_answer, lme_eval, lme_reader, lme_retrieve
from ..datasets.longmemeval import iter_history_turns, question_type_axis
from ..embedders import Embedder
from ..judges import Judge
from .lme_context import Turn, parse_lme_date, render, session_text

# Turns fetched per session wanted when unit="session": a session is
# ranked by its best turn, so the pool must be wide enough to reach every
# session a multi-session count needs.
SESSION_POOL_FACTOR = 5
# Concurrent fact extractions per question (x --workers questions).
FACT_WORKERS = 8
# Facts searched for the event ledger (count questions) and the preference
# profile (every question), by similarity to the question, kind-filtered.
LEDGER_K = 60
PROFILE_K = 30
# The second retrieval behind an abstention (reader="routed"): its own
# budget, on top of the context the first reading had.
RESEARCH_CHARS = 40_000


def _judge_can_answer(judge) -> bool:
    """True if the judge object also exposes `answer` or `answer_full`."""
    return (callable(getattr(judge, "answer", None))
            or callable(getattr(judge, "answer_full", None)))


def _answer(judge, question: str, context: str,
            question_date: str) -> tuple[str, str]:
    """(final answer, transcript)."""
    if callable(getattr(judge, "answer_full", None)):
        return judge.answer_full(question, context, question_date)
    try:
        a = judge.answer(question, context, question_date=question_date)
    except TypeError:   # an answerer that predates question_date
        a = judge.answer(question, context)
    return a, a


def judge_answer(q: dict, predicted: str, judge, official_judge=None
                 ) -> tuple[Optional[dict], dict]:
    """(official verdict or None, lenient verdict) for one answer."""
    qid = q.get("question_id", "")
    question, gold = q.get("question", ""), q.get("answer", "") or ""
    if callable(getattr(judge, "verdict", None)):
        lenient = judge.verdict(question, gold, predicted)
    else:   # substring and other score-only judges
        jr = judge.score(predicted=predicted, gold=gold, question=question)
        lenient = {"correct": jr.score >= 1.0, "raw": jr.rationale}
    official = None
    if official_judge is not None:
        official = official_judge.verdict(
            q.get("question_type", ""), question, gold, predicted,
            lme_eval.is_abstention(qid))
    return official, lenient


def finish(records: list[dict], questions_path) -> dict:
    """Summary metrics for a scenario; writes the per-question JSONL."""
    summary = lme_eval.summarize(records)
    if questions_path:
        lme_eval.write_jsonl(questions_path, records)
        summary["questions_file"] = str(questions_path)
    by_axis = defaultdict(list)
    judge = "official" if "overall_official" in summary else "lenient"
    for r in records:
        if r.get(judge):
            by_axis[question_type_axis(r["question_type"], r["question_id"])
                    ].append(1.0 if r[judge]["correct"] else 0.0)
    summary["overall"] = summary.get(f"overall_{judge}") or 0.0
    summary["headline_judge"] = judge
    summary["by_axis"] = {k: round(statistics.fmean(v), 4)
                          for k, v in sorted(by_axis.items())}
    print("\n" + lme_eval.format_summary(summary), flush=True)
    return summary


def run(questions: list[dict], embedder: Embedder, judge: Judge,
        top_k: int = 20, ef: Optional[int] = None,
        modality: str = "text",
        scoring_half_life: Optional[float] = None,
        scoring_time_weight: Optional[float] = None,
        unit: str = "session",
        context_chars: int = 160_000,
        official_judge=None,
        questions_path=None,
        multi_round: str = "auto",
        fact_cache=None,
        count_map: str = "off",
        context_tokens: Optional[int] = None,
        count_tokens=None,
        workers: int = 1,
        reader: str = "single") -> dict:
    """Execute the scenario and return aggregate metrics.

    Adaptive decay is engaged when scoring_time_weight is provided.
    Because Feather scores against real wall-clock time, we shift each
    question's timestamps so that the question_date maps to "now" —
    preserving the *relative* ages of turns vs. the question.

    multi_round: "auto" runs a second retrieval round (lme_retrieve) for
    count and comparison questions, "always" for every question, "off"
    never. It needs an answerer exposing planner_chat().

    context_tokens: budget the context in tokens (count_tokens = the
    answer model's tokenizer, bench.tokens) instead of context_chars.

    count_map: "retrieved" answers count and total questions by
    map-reduce (lme_answer.count_map_reduce): every retrieved session is
    read whole by its own call, then one call merges the lists and counts.
    "haystack" maps every session of the haystack (no retrieval: an upper
    bound for counts). "off" reads the budgeted context once. Needs an
    answerer exposing map_chat() and answer_chat().

    count_map also takes the aggregations that are not counts: "which
    is the most", "in what order" (lme_reader.is_aggregation_question).

    reader: "single" answers with lme_answer's one prompt; "verified"
    with the same prompt, and an abstaining draft is re-searched and must
    be overturned by a real quote (lme_reader.verified_answer); "routed"
    is lme_reader.routed_answer, measured WORSE (0.794 vs 0.904). The
    last two need an answerer exposing answer_chat() and planner_chat().

    fact_cache (lme_facts.SessionFactCache): extract dated facts per
    session at ingest and index them beside the turns. A fact hit stands
    for the turn it cites, so facts find sessions and the reader still
    gets the sessions; facts whose value changed are shown as a dated
    timeline above them.

    Every question is scored by the lenient judge and, when given, the
    official one; one record per question goes to `questions_path`
    (JSONL, see lme_eval.make_record).
    """
    use_decay = scoring_time_weight is not None and scoring_time_weight > 0
    if use_decay:
        print("[lme] warning: decay ranks old evidence lower; temporal and "
              "multi-session questions need it", flush=True)
    pool_k = top_k * SESSION_POOL_FACTOR if unit == "session" else top_k
    n_q = len(questions)
    if n_q == 0:
        return {"overall": 0.0, "by_axis": {}, "by_question_type": {}, "n_q": 0}
    if count_map not in ("off", "retrieved", "haystack"):
        raise ValueError(f"count_map must be off, retrieved or haystack, "
                         f"not {count_map!r}")
    can_map = (count_map != "off" and callable(getattr(judge, "map_chat", None))
               and callable(getattr(judge, "answer_chat", None)))
    if reader not in ("single", "verified", "routed"):
        raise ValueError(f"reader must be single, verified or routed, "
                         f"not {reader!r}")
    routed = (reader != "single"
              and callable(getattr(judge, "answer_chat", None))
              and callable(getattr(judge, "planner_chat", None)))
    if reader != "single" and not routed:
        raise ValueError(f"reader={reader!r} needs an answerer exposing "
                         "answer_chat() and planner_chat()")

    records: list[dict] = []
    failures: list[dict] = []
    use_llm_answer = _judge_can_answer(judge)
    has_batch = callable(getattr(embedder, "embed_batch", None))
    progress_every = max(1, n_q // 20)  # ~5% increments
    run_t0 = time.perf_counter()

    lock = threading.Lock()
    counters = {"embeddings_emitted": 0, "history_turns_total": 0, "done": 0}

    def bump(name: str, n: int) -> None:
        with lock:
            counters[name] += n

    def progress() -> None:
        with lock:
            counters["done"] += 1
            done = counters["done"]
            if done % progress_every and done != n_q:
                return
            ok = len(records)
            running = statistics.fmean(
                [(r["official"] or r["lenient"])["correct"]
                 for r in records]) if records else 0.0
            nfail = len(failures)
        wall = time.perf_counter() - run_t0
        eta = (wall / done) * (n_q - done) if done else 0
        print(f"[lme] {done}/{n_q}  ok={ok}  fail={nfail}  "
              f"running_score={running:.3f}  elapsed={wall:.0f}s  "
              f"eta={eta:.0f}s", flush=True)

    def process(q_idx: int, q: dict) -> None:
        qid = q.get("question_id") or f"q{q_idx}"
        qtype = q.get("question_type") or "unknown"
        question_text = q.get("question", "") or ""

        path = tempfile.mktemp(suffix=f"_{qid}.feather")
        t0 = time.perf_counter()
        db = None
        try:
            db = feather_db.DB.open(path, dim=embedder.dim)
            if ef is not None:
                db.set_ef(ef)

            # ---------- Collect haystack into a flat list ----------
            # Feather decays against wall-clock time, so shift every
            # timestamp by the same amount to put question_date at "now".
            question_date = q.get("question_date") or ""
            q_dt = parse_lme_date(question_date)
            ts_shift = (int(time.time()) - int(q_dt.timestamp())) if q_dt else 0

            turns: list[Turn] = []
            stamps: list[int] = []
            for sid, date_str, role, content, _has_ans in iter_history_turns(q):
                if not content.strip():
                    continue
                d = parse_lme_date(date_str)
                stamps.append(int(d.timestamp()) + ts_shift if d else 0)
                turns.append(Turn(sid, date_str, role, content, len(turns)))
            bump("history_turns_total", len(turns))

            # ---------- Embed all turns + question (batch if possible) ----------
            if has_batch:
                texts = [t.content for t in turns] + [
                    lme_retrieve.dense_query(question_text, question_date)]
                vecs = embedder.embed_batch(texts)
                turn_vecs, qvec = vecs[:-1], vecs[-1]
            else:
                turn_vecs = [embedder.embed(t.content) for t in turns]
                qvec = embedder.embed(
                    lme_retrieve.dense_query(question_text, question_date))
            bump("embeddings_emitted", len(turns) + 1)

            # ---------- Ingest ----------
            # id = position + 1, so a hit maps straight back to its Turn.
            for t, ts, v in zip(turns, stamps, turn_vecs):
                meta = feather_db.Metadata()
                meta.content = t.content
                meta.timestamp = ts
                meta.namespace_id = qid
                meta.entity_id = t.session_id
                meta.source = t.role
                db.add(id=t.order + 1, vec=v, meta=meta, modality=modality)

            # ---------- Dated facts (one extraction per session) ----------
            # ids after the turns'; each maps to the Turn it cites.
            fact_turn: dict[int, Turn] = {}
            fact_of: dict[int, object] = {}
            chains: dict = {}
            all_facts: list = []
            if fact_cache is not None:
                by_session: dict[str, list[Turn]] = {}
                for t in turns:
                    by_session.setdefault(t.session_id, []).append(t)
                facts = []
                # One extraction call per session, FACT_WORKERS at a time:
                # a question has ~50 sessions and each call waits on an LLM.
                with ThreadPoolExecutor(max_workers=FACT_WORKERS) as fpool:
                    extracted = list(fpool.map(
                        lambda item: fact_cache.facts(
                            item[0], item[1][0].session_date,
                            [{"role": t.role, "content": t.content}
                             for t in item[1]]),
                        by_session.items()))
                for (sid, sturns), got in zip(by_session.items(), extracted):
                    for f in got:
                        i = f.metadata.get("turn")
                        anchor = (sturns[i] if isinstance(i, int)
                                  and 0 <= i < len(sturns) else
                                  next((t for t in sturns if t.role == "user"),
                                       sturns[0]))
                        facts.append((f, anchor))
                all_facts = [f for f, _ in facts]
                chains = link_validity(all_facts)
                texts = [f.metadata["statement"] for f, _ in facts]
                fvecs = (embedder.embed_batch(texts) if has_batch and texts
                         else [embedder.embed(x) for x in texts])
                bump("embeddings_emitted", len(texts))
                for j, ((f, anchor), v) in enumerate(zip(facts, fvecs)):
                    fid = len(turns) + 1 + j
                    meta = feather_db.Metadata()
                    meta.content = f.metadata["statement"]
                    meta.timestamp = (f.valid_at + ts_shift if f.valid_at
                                      else stamps[anchor.order])
                    meta.namespace_id = qid
                    meta.entity_id = anchor.session_id
                    meta.source = "fact"
                    meta.set_attribute("kind", "fact")
                    meta.set_attribute("fact_kind",
                                       f.metadata.get("kind", "other"))
                    meta.set_attribute("valid_from",
                                       f.metadata.get("valid_from", ""))
                    meta.set_attribute("valid_until",
                                       f.metadata.get("valid_until", ""))
                    db.add(id=fid, vec=v, meta=meta, modality=modality)
                    fact_turn[fid], fact_of[fid] = anchor, f
            fact_hits: list = []

            # ---------- Retrieve ----------
            scoring = None
            if use_decay:
                scoring = feather_db.ScoringConfig(
                    half_life=scoring_half_life or 30.0,
                    weight=scoring_time_weight,
                    min=0.0,
                )

            def search(text, vec, after=None, before=None) -> list[Turn]:
                """Hybrid search; a date window is applied to the hits, so
                fetch more to fill the pool after it."""
                k = pool_k * (3 if (after or before) else 1)
                kw = {"scoring": scoring} if scoring is not None else {}
                try:
                    results = db.hybrid_search(vec, text, k=k,
                                               modality=modality, **kw)
                except Exception:
                    try:
                        results = db.search(vec, k=k, modality=modality, **kw)
                    except Exception:
                        results = db.search(vec, k=k, modality=modality)
                found = []
                for r in results:
                    if 1 <= r.id <= len(turns):
                        found.append(turns[r.id - 1])
                    elif r.id in fact_turn:
                        found.append(fact_turn[r.id])
                        fact_hits.append(fact_of[r.id])
                if after or before:
                    found = [t for t in found if lme_retrieve.in_window(
                        t.session_date, after, before)]
                return found[:pool_k]

            hits = search(question_text, qvec)
            followups, missing = [], ""
            round1_sids = list(dict.fromkeys(t.session_id for t in hits))
            planner = getattr(judge, "planner_chat", None)
            if callable(planner) and (multi_round == "always" or (
                    multi_round == "auto"
                    and lme_retrieve.needs_multi_round(question_text))):
                followups, missing = lme_retrieve.plan_followups(
                    planner(), question_text, question_date,
                    lme_retrieve.digest(hits, question_date, top_k))
            if followups:
                rounds = []
                for f in followups:
                    v = embedder.embed(lme_retrieve.dense_query(
                        f["query"], question_date))
                    bump("embeddings_emitted", 1)
                    rounds.append(search(f["query"], v, f["after"],
                                         f["before"]))
                order = lme_retrieve.merge_sessions(
                    round1_sids,
                    [list(dict.fromkeys(t.session_id for t in r))
                     for r in rounds], top_k)
                by_sid: dict[str, list[Turn]] = {}
                for t in hits + [t for r in rounds for t in r]:
                    if t not in by_sid.setdefault(t.session_id, []):
                        by_sid[t.session_id].append(t)
                hits = [t for sid in order for t in by_sid[sid]]
                max_sessions = len(order)
            else:
                max_sessions = top_k
            changes = timeline(chains, [fact_key(f) for f in fact_hits])
            events, prefs = "", ""
            if all_facts:
                def facts_of(kind, k):
                    flt = (FilterBuilder().namespace(qid)
                           .attribute("fact_kind", kind).build())
                    return [fact_of[r.id] for r in db.search(
                        qvec, k=k, modality=modality, filter=flt)
                        if r.id in fact_of]
                if lme_reader.is_aggregation_question(question_text):
                    # Every mention of a matched event, not only the
                    # mentions that ranked: dedup needs all of them.
                    keys = {f.metadata.get("entity") for f in
                            facts_of("event", LEDGER_K)} - {""}
                    events = ledger([f for f in all_facts
                                     if f.metadata.get("entity") in keys])
                prefs = profile({fact_key(f): chains[fact_key(f)]
                                 for f in facts_of("preference", PROFILE_K)})
            preamble = "\n\n".join(
                f"{head}\n{body}" for head, body in (
                    (lme_answer.PROFILE_HEADER, prefs),
                    (lme_answer.LEDGER_HEADER, events),
                    (lme_answer.CHANGES_HEADER, changes)) if body)
            budget = ({"max_tokens": context_tokens - count_tokens(preamble),
                       "count_tokens": count_tokens}
                      if context_tokens else
                      {"max_chars": context_chars - len(preamble)})
            rendered = render(
                hits, turns, unit=unit, max_sessions=max_sessions,
                question_date=question_date, **budget)
            context = rendered.text
            retrieved_sids, context_sids = (rendered.retrieved,
                                            rendered.in_context)
            if preamble:
                context = f"{preamble}\n\n{context}"

            # ---------- Answer ----------
            map_stats = None
            reader_trace = None

            def map_reduce(question: str) -> tuple[str, str]:
                """Every retrieved session read whole, one reduce over the
                lists and the ledger. Raises to fall back to one read."""
                nonlocal map_stats
                by_sid: dict[str, list[Turn]] = {}
                for t in turns:
                    by_sid.setdefault(t.session_id, []).append(t)
                sids = (list(by_sid) if count_map == "haystack" else
                        rendered.retrieved[:max_sessions])
                sids.sort(key=lambda sid: (
                    parse_lme_date(by_sid[sid][0].session_date)
                    or datetime.datetime.max, by_sid[sid][0].order))
                try:
                    final, text, map_stats = lme_answer.count_map_reduce(
                        judge.map_chat(), judge.answer_chat(), question,
                        [session_text(by_sid[sid], question_date)
                         for sid in sids], question_date, ledger=events)
                except Exception as e:
                    map_stats = {"error": f"{type(e).__name__}: {e}"}
                    raise
                map_stats["sessions"] = sids
                return final, text

            def research(queries: list[str], nouns: str) -> str:
                """The second retrieval behind an abstention: each
                paraphrase searched like the question, the key nouns by
                BM25 alone; sessions the context lacks come first."""
                found: list[Turn] = []
                for text in queries:
                    v = embedder.embed(lme_retrieve.dense_query(
                        text, question_date))
                    bump("embeddings_emitted", 1)
                    found += search(text, v)
                if nouns:
                    try:
                        found += [turns[r.id - 1] for r in
                                  db.keyword_search(nouns, pool_k)
                                  if 1 <= r.id <= len(turns)]
                    except Exception:
                        pass
                have = set(context_sids)
                found = ([t for t in found if t.session_id not in have]
                         + [t for t in found if t.session_id in have])
                seen: set[int] = set()
                unique = [t for t in found
                          if not (t.order in seen or seen.add(t.order))]
                return render(unique, turns, unit=unit,
                              max_chars=RESEARCH_CHARS,
                              question_date=question_date).text

            if use_llm_answer and reader == "verified":
                if can_map and lme_reader.is_aggregation_question(
                        question_text):
                    try:
                        predicted, transcript = map_reduce(question_text)
                    except Exception:
                        pass             # fall back to the reader
                if map_stats is None or "error" in map_stats:
                    predicted, transcript, reader_trace = \
                        lme_reader.verified_answer(
                            judge.answer_chat(), judge.planner_chat(),
                            question_text, context, question_date,
                            search=research)
            elif use_llm_answer and routed:
                predicted, transcript, reader_trace = \
                    lme_reader.routed_answer(
                        judge.answer_chat(), judge.planner_chat(),
                        question_text, context, question_date,
                        search=research,
                        map_reduce=map_reduce if can_map else None)
            elif use_llm_answer:
                if can_map and lme_reader.is_aggregation_question(
                        question_text):
                    try:
                        predicted, transcript = map_reduce(question_text)
                    except Exception:
                        pass             # fall back to one read
                if map_stats is None or "error" in map_stats:
                    predicted, transcript = _answer(
                        judge, question_text, context, question_date)
            else:
                predicted, transcript = context, ""

            # ---------- Score ----------
            official, lenient = judge_answer(q, predicted, judge,
                                             official_judge)
            rec = lme_eval.make_record(
                q, retrieved_sessions=retrieved_sids,
                context_sessions=context_sids, context_chars=len(context),
                answer=predicted if use_llm_answer else "",
                transcript=transcript, official=official, lenient=lenient,
                seconds=time.perf_counter() - t0,
                followups=followups, missing=missing,
                round1_sessions=round1_sids[:top_k],
                fact_hits=list(dict.fromkeys(
                    f.metadata["statement"] for f in fact_hits))[:20],
                changes=changes, ledger=events, profile=prefs,
                count_map=map_stats, reader=reader_trace,
                context_tokens=(count_tokens(context) if count_tokens
                                else None),
                evidence_levels={sid: rendered.levels[sid] for sid in
                                 q.get("answer_session_ids") or ()
                                 if sid in rendered.levels})
            with lock:
                records.append(rec)

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            failed = {"correct": False, "raw": "", "error": err}
            with lock:
                failures.append({
                    "question_id": qid,
                    "question_type": qtype,
                    "error": err,
                    "q_index": q_idx,
                })
                # Scored wrong, not dropped: see lme_eval's buckets.
                records.append(lme_eval.make_record(
                    q, retrieved_sessions=[], context_sessions=[],
                    context_chars=0, answer="",
                    official=failed if official_judge is not None else None,
                    lenient=failed, run_error=err))
            print(f"[lme] q{q_idx+1}/{n_q} ({qid}) FAILED: "
                  f"{type(e).__name__}: {str(e)[:120]}",
                  flush=True)

        finally:
            if db is not None:
                db.close(save=False)
            for leftover in glob.glob(path + "*"):     # .wal, .lock, ...
                os.remove(leftover)

            progress()

    # Questions are independent (each has its own DB file) and almost all
    # of a question's time is spent waiting on the embedder and LLMs.
    if workers <= 1:
        for i, q in enumerate(questions):
            process(i, q)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in [pool.submit(process, i, q) for i, q in enumerate(questions)]:
                f.result()
    order = {q.get("question_id"): i for i, q in enumerate(questions)}
    records.sort(key=lambda r: order.get(r["question_id"], 0))
    embeddings_emitted = counters["embeddings_emitted"]
    history_turns_total = counters["history_turns_total"]

    # ---------- Aggregate ----------
    secs = [r["seconds"] for r in records]
    return {
        **finish(records, questions_path),
        "per_q_seconds_p50": statistics.median(secs) if secs else 0.0,
        "per_q_seconds_mean": statistics.fmean(secs) if secs else 0.0,
        "n_questions": n_q,
        "n_failures": len(failures),
        "failures_sample": failures[:10],
        "history_turns_total": history_turns_total,
        "embeddings_emitted": embeddings_emitted,
        "embedder": embedder.name,
        "judge": judge.name,
        "official_judge": getattr(official_judge, "name", None),
        "top_k": top_k,
        "unit": unit,
        "context_chars": context_chars,
        "context_tokens": context_tokens,
        "multi_round": multi_round,
        "count_map": count_map,
        "reader": reader,
        "workers": workers,
        "facts": "session" if fact_cache is not None else "off",
        **(fact_cache.stats() if fact_cache is not None else {}),
        "decay_engaged": use_decay,
        "decay_half_life_days": scoring_half_life if use_decay else None,
        "decay_time_weight": scoring_time_weight if use_decay else None,
    }
