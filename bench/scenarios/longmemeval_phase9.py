"""LongMemEval scenario with Phase 9 ingestion (extracted facts) instead
of raw turns.

For each question:
  1. Open a fresh in-memory Feather DB.
  2. Run IngestPipeline over the haystack turns.
     - FactExtractor produces atomic facts (LLM-backed).
     - EntityResolver canonicalizes mentions.
     - TemporalParser stamps observed_at/valid_at.
     - Source records, facts, entities all stored in DB with edges.
  3. At question time, run hybrid_search over the *facts* + *source
     records* (no separate retrieval modes — the facts join the search
     space alongside raw turns).
  4. Pass retrieved evidence to the answerer LLM, judge as before.
     Retrieved source turns go through `lme_context.build_context`
     (dated sessions, roles, oldest first); retrieved facts follow as a
     dated list.

Cost: this is the EXPENSIVE scenario. Each question now does:
  - N raw-turn embeds (same as baseline)
  - N FactExtractor LLM calls (one per turn)
  - 1 EntityResolver call per turn (over fact subj/obj union)
  - Plus the existing answerer + judge calls.

For LongMemEval_S, expect ~100-150x more LLM tokens than baseline.
Estimate ~$50-100 for full 500-question run with Haiku as the system
LLM. Use --limit to stage runs (10 → 50 → 500).

Output metrics extend the baseline scenario with:
  - facts_extracted_total
  - entities_resolved_total
  - extraction_failures
  - extraction_seconds_p50
"""
from __future__ import annotations
import os
import statistics
import tempfile
import time
from typing import Optional

import feather_db
from feather_db.extractors import FactExtractor, EntityResolver, TemporalParser
from feather_db.pipelines import IngestPipeline, IngestRecord

from .. import lme_eval
from ..datasets.longmemeval import iter_history_turns
from ..embedders import Embedder
from ..judges import Judge
from .lme_context import Turn, parse_lme_date, render_context
from .longmemeval import (SESSION_POOL_FACTOR, _answer, _judge_can_answer,
                          finish, judge_answer)


def _build_context(retrieved, turns: list[Turn], ts_shift: int, *,
                   unit: str, max_chars: int, max_sessions: int,
                   question_date: str = "") -> tuple[str, list[str], list[str]]:
    """Source-record hits become dated sessions; fact hits follow, dated
    by the time they were valid (or observed). Returns render_context's
    (context, retrieved sessions, context sessions); the session lists
    count source turns only, so a question answered from facts alone can
    show as a retrieval miss."""
    import datetime
    hits: list[Turn] = []
    facts: list[str] = []
    for r in retrieved:
        m = r.metadata
        if m is None or not (m.content or "").strip():
            continue
        if m.get_attribute("kind", "") == "source_record":
            try:
                hits.append(turns[int(m.get_attribute("turn_order", ""))])
                continue
            except (ValueError, IndexError):
                pass
        when = ""
        if m.timestamp > 0:
            when = datetime.datetime.fromtimestamp(
                m.timestamp - ts_shift).strftime("%Y/%m/%d (%a) %H:%M")
        facts.append(f"- [{when or 'unknown date'}] {m.content.strip()}")
    sessions, retrieved, in_context = render_context(
        hits, turns, unit=unit, max_chars=max_chars,
        max_sessions=max_sessions, question_date=question_date)
    if not facts:
        return sessions, retrieved, in_context
    block = "### Extracted facts\n" + "\n".join(facts)
    return (f"{sessions}\n\n{block}" if sessions else block,
            retrieved, in_context)


def run(questions: list[dict],
        embedder: Embedder,
        judge: Judge,
        *,
        system_provider,                     # LLM for FactExtractor + EntityResolver
        top_k: int = 20,
        ef: Optional[int] = None,
        modality: str = "text",
        scoring_half_life: Optional[float] = None,
        scoring_time_weight: Optional[float] = None,
        max_facts_per_turn: int = 8,
        unit: str = "session",
        context_chars: int = 160_000,
        official_judge=None,
        questions_path=None) -> dict:
    """Phase 9 ingestion + retrieval scenario. Scoring and the
    per-question JSONL as in longmemeval.run."""
    n_q = len(questions)
    if n_q == 0:
        return {"overall": 0.0, "by_axis": {}, "n_q": 0}

    use_decay = scoring_time_weight is not None and scoring_time_weight > 0
    if use_decay:
        print("[lme9] warning: decay ranks old evidence lower; temporal and "
              "multi-session questions need it", flush=True)
    pool_k = top_k * SESSION_POOL_FACTOR if unit == "session" else top_k
    use_llm_answer = _judge_can_answer(judge)
    has_batch = callable(getattr(embedder, "embed_batch", None))

    fact_extractor = FactExtractor(
        provider=system_provider,
        max_facts_per_call=max_facts_per_turn,
    )
    entity_resolver = EntityResolver(provider=system_provider)
    temporal_parser = TemporalParser()

    records: list[dict] = []
    facts_total = 0
    entities_total = 0
    extraction_failures_total = 0
    extraction_seconds = []
    progress_every = max(1, n_q // 20)
    run_t0 = time.perf_counter()

    for q_idx, q in enumerate(questions):
        qid = q.get("question_id") or f"q{q_idx}"
        question_text = q.get("question", "") or ""

        path = tempfile.mktemp(suffix=f"_{qid}.feather")
        t_q = time.perf_counter()
        try:
            db = feather_db.DB.open(path, dim=embedder.dim)
            if ef is not None:
                db.set_ef(ef)

            # Collect haystack turns into IngestRecords. Shift every
            # timestamp so question_date maps to wall-clock "now".
            question_date = q.get("question_date") or ""
            q_dt = parse_lme_date(question_date)
            ts_shift = (int(time.time()) - int(q_dt.timestamp())) if q_dt else 0

            turns: list[Turn] = []
            records: list[IngestRecord] = []
            for sid, date_str, role, content, _has_ans in iter_history_turns(q):
                if not content.strip():
                    continue
                d = parse_lme_date(date_str)
                ts = int(d.timestamp()) + ts_shift if d else 0
                records.append(IngestRecord(
                    content=content,
                    source_id=f"{qid}::{sid}::{len(records)}",
                    timestamp=ts,
                    metadata={"role": role, "session_id": sid,
                              "session_date": date_str,
                              "turn_order": len(turns)},
                ))
                turns.append(Turn(sid, date_str, role, content, len(turns)))

            # Run the Phase 9 pipeline
            t_extract = time.perf_counter()
            pipeline = IngestPipeline(
                db=db,
                embedder=_PipelineEmbedderAdapter(embedder, has_batch=has_batch),
                fact_extractor=fact_extractor,
                entity_resolver=entity_resolver,
                temporal_parser=temporal_parser,
                namespace=qid,
            )
            stats = pipeline.ingest(records)
            extraction_seconds.append(time.perf_counter() - t_extract)
            facts_total += stats.facts_extracted
            entities_total += stats.entities_resolved
            extraction_failures_total += stats.extraction_failures

            # Retrieve
            qvec = embedder.embed(question_text)
            scoring = None
            if use_decay:
                scoring = feather_db.ScoringConfig(
                    half_life=scoring_half_life or 30.0,
                    weight=scoring_time_weight,
                    min=0.0,
                )
            try:
                if scoring is not None:
                    results = db.hybrid_search(
                        qvec, question_text, k=pool_k,
                        modality=modality, scoring=scoring,
                    )
                else:
                    results = db.hybrid_search(
                        qvec, question_text, k=pool_k, modality=modality,
                    )
            except Exception:
                results = db.search(qvec, k=pool_k, modality=modality)

            context, retrieved_sids, context_sids = _build_context(
                results, turns, ts_shift, unit=unit, max_chars=context_chars,
                max_sessions=top_k, question_date=question_date)
            if use_llm_answer:
                predicted, transcript = _answer(judge, question_text, context,
                                                question_date)
            else:
                predicted, transcript = context, ""
            official, lenient = judge_answer(q, predicted, judge,
                                             official_judge)
            records.append(lme_eval.make_record(
                q, retrieved_sessions=retrieved_sids,
                context_sessions=context_sids, context_chars=len(context),
                answer=predicted if use_llm_answer else "",
                transcript=transcript, official=official, lenient=lenient,
                seconds=time.perf_counter() - t_q))

        except Exception as e:
            print(f"[lme9] q{q_idx+1}/{n_q} ({qid}) FAILED: "
                  f"{type(e).__name__}: {str(e)[:140]}", flush=True)
        finally:
            if os.path.exists(path):
                os.remove(path)
            if (q_idx + 1) % progress_every == 0 or q_idx + 1 == n_q:
                done = q_idx + 1
                ok = len(records)
                running = statistics.fmean(
                    [(r["official"] or r["lenient"])["correct"]
                     for r in records]) if records else 0.0
                wall = time.perf_counter() - run_t0
                eta = (wall / done) * (n_q - done) if done else 0
                print(f"[lme9] {done}/{n_q}  ok={ok}  facts_extracted="
                      f"{facts_total}  running_score={running:.3f}  "
                      f"elapsed={wall:.0f}s  eta={eta:.0f}s",
                      flush=True)

    secs = [r["seconds"] for r in records]
    return {
        **finish(records, questions_path),
        "per_q_seconds_p50": statistics.median(secs) if secs else 0.0,
        "per_q_seconds_mean": statistics.fmean(secs) if secs else 0.0,
        "n_questions": n_q,
        "facts_extracted_total": facts_total,
        "entities_resolved_total": entities_total,
        "extraction_failures": extraction_failures_total,
        "extraction_seconds_p50": (statistics.median(extraction_seconds)
                                   if extraction_seconds else 0.0),
        "embedder": embedder.name,
        "judge": judge.name,
        "official_judge": getattr(official_judge, "name", None),
        "system_llm": getattr(system_provider, "_model", None) or "?",
        "top_k": top_k,
        "unit": unit,
        "context_chars": context_chars,
        "decay_engaged": use_decay,
    }


class _PipelineEmbedderAdapter:
    """Adapt our bench Embedder to whatever IngestPipeline expects.

    IngestPipeline calls .embed(text); both bench Embedders already
    implement that. This thin wrapper exists in case we later want
    batching inside the pipeline.
    """
    def __init__(self, embedder, *, has_batch: bool = False):
        self._embedder = embedder
        self.dim = embedder.dim

    def embed(self, text: str):
        return self._embedder.embed(text)
