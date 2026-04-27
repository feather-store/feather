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
from collections import defaultdict
from typing import Optional

import feather_db
from feather_db.extractors import FactExtractor, EntityResolver, TemporalParser
from feather_db.pipelines import IngestPipeline, IngestRecord

from ..datasets.longmemeval import iter_history_turns, question_type_axis
from ..embedders import Embedder
from ..judges import Judge


def _judge_can_answer(judge):
    return callable(getattr(judge, "answer", None))


def _build_context(retrieved) -> str:
    parts = []
    for r in retrieved:
        c = (r.metadata.content or "").strip() if r.metadata else ""
        if c:
            parts.append(c)
    return "\n---\n".join(parts)


def run(questions: list[dict],
        embedder: Embedder,
        judge: Judge,
        *,
        system_provider,                     # LLM for FactExtractor + EntityResolver
        top_k: int = 10,
        ef: Optional[int] = None,
        modality: str = "text",
        scoring_half_life: Optional[float] = None,
        scoring_time_weight: Optional[float] = None,
        max_facts_per_turn: int = 8) -> dict:
    """Phase 9 ingestion + retrieval scenario."""
    n_q = len(questions)
    if n_q == 0:
        return {"overall": 0.0, "by_axis": {}, "n_q": 0}

    use_decay = scoring_time_weight is not None and scoring_time_weight > 0
    use_llm_answer = _judge_can_answer(judge)
    has_batch = callable(getattr(embedder, "embed_batch", None))

    fact_extractor = FactExtractor(
        provider=system_provider,
        max_facts_per_call=max_facts_per_turn,
    )
    entity_resolver = EntityResolver(provider=system_provider)
    temporal_parser = TemporalParser()

    per_q_seconds = []
    per_q_score = []
    per_q_axis = []
    per_q_type = []
    facts_total = 0
    entities_total = 0
    extraction_failures_total = 0
    extraction_seconds = []
    progress_every = max(1, n_q // 20)
    run_t0 = time.perf_counter()

    for q_idx, q in enumerate(questions):
        qid = q.get("question_id") or f"q{q_idx}"
        qtype = q.get("question_type") or "unknown"
        axis = question_type_axis(qtype)
        gold = q.get("answer", "") or ""
        question_text = q.get("question", "") or ""

        path = tempfile.mktemp(suffix=f"_{qid}.feather")
        scored = False
        t_q = time.perf_counter()
        try:
            db = feather_db.DB.open(path, dim=embedder.dim)
            if ef is not None:
                db.set_ef(ef)

            # Collect haystack turns into IngestRecords
            import datetime
            real_now = int(time.time())
            q_date_ts = 0
            try:
                q_date_ts = int(datetime.datetime.fromisoformat(
                    (q.get("question_date") or "").replace("Z", "+00:00")
                ).timestamp())
            except Exception:
                pass
            ts_shift = (real_now - q_date_ts) if q_date_ts > 0 else 0

            records: list[IngestRecord] = []
            for sid, date_str, role, content, has_ans in iter_history_turns(q):
                if not content.strip():
                    continue
                ts = 0
                try:
                    ts = int(datetime.datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")).timestamp())
                except Exception:
                    pass
                if ts > 0 and ts_shift:
                    ts += ts_shift
                records.append(IngestRecord(
                    content=content,
                    source_id=f"{qid}::{sid}::{len(records)}",
                    timestamp=ts,
                    metadata={"role": role, "session_id": sid,
                              "has_answer": has_ans},
                ))

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
                        qvec, question_text, k=top_k,
                        modality=modality, scoring=scoring,
                    )
                else:
                    results = db.hybrid_search(
                        qvec, question_text, k=top_k, modality=modality,
                    )
            except Exception:
                results = db.search(qvec, k=top_k, modality=modality)

            context = _build_context(results)
            predicted = (judge.answer(question_text, context)
                         if use_llm_answer else context)
            jr = judge.score(predicted=predicted, gold=gold,
                             question=question_text)
            per_q_score.append(jr.score)
            per_q_axis.append(axis)
            per_q_type.append(qtype)
            scored = True

        except Exception as e:
            print(f"[lme9] q{q_idx+1}/{n_q} ({qid}) FAILED: "
                  f"{type(e).__name__}: {str(e)[:140]}", flush=True)
        finally:
            elapsed = time.perf_counter() - t_q
            if scored:
                per_q_seconds.append(elapsed)
            if os.path.exists(path):
                os.remove(path)
            if (q_idx + 1) % progress_every == 0 or q_idx + 1 == n_q:
                done = q_idx + 1
                ok = len(per_q_score)
                running = (statistics.fmean(per_q_score)
                           if per_q_score else 0.0)
                wall = time.perf_counter() - run_t0
                eta = (wall / done) * (n_q - done) if done else 0
                print(f"[lme9] {done}/{n_q}  ok={ok}  facts_extracted="
                      f"{facts_total}  running_score={running:.3f}  "
                      f"elapsed={wall:.0f}s  eta={eta:.0f}s",
                      flush=True)

    by_axis = defaultdict(list)
    by_type = defaultdict(list)
    for s, a, t in zip(per_q_score, per_q_axis, per_q_type):
        by_axis[a].append(s)
        by_type[t].append(s)

    return {
        "overall": (statistics.fmean(per_q_score) if per_q_score else 0.0),
        "by_axis": {k: statistics.fmean(v) for k, v in by_axis.items()},
        "by_question_type": {k: statistics.fmean(v) for k, v in by_type.items()},
        "per_q_seconds_p50": (statistics.median(per_q_seconds)
                              if per_q_seconds else 0.0),
        "per_q_seconds_mean": (statistics.fmean(per_q_seconds)
                               if per_q_seconds else 0.0),
        "n_questions": n_q,
        "n_scored": len(per_q_score),
        "facts_extracted_total": facts_total,
        "entities_resolved_total": entities_total,
        "extraction_failures": extraction_failures_total,
        "extraction_seconds_p50": (statistics.median(extraction_seconds)
                                   if extraction_seconds else 0.0),
        "embedder": embedder.name,
        "judge": judge.name,
        "system_llm": getattr(system_provider, "_model", None) or "?",
        "top_k": top_k,
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
