"""LongMemEval scenario.

For each question:
  1. Open a fresh in-memory Feather DB.
  2. Embed every chat turn in the haystack and add to the DB with
     metadata (timestamp, session_id, role, has_answer flag, namespace
     = question_id).
  3. Embed the question and run hybrid_search top-k.
  4. Build "context" = concatenated content of retrieved memories.
  5. Score: in Phase 1 we simply check whether `gold answer` is a
     substring of `context`. Phase 2 will swap in an LLM answerer +
     LLM judge.

Output:
    metrics["overall"]               mean score across all questions
    metrics["by_axis"]               {axis: mean_score}
    metrics["by_question_type"]      {qtype: mean_score}
    metrics["per_q_seconds_p50"]
    metrics["embeddings_emitted"]    total embed() calls
    metrics["history_turns_total"]
"""
from __future__ import annotations
import os
import statistics
import tempfile
import time
from collections import defaultdict
from typing import Optional

import feather_db

from ..datasets.longmemeval import iter_history_turns, question_type_axis
from ..embedders import Embedder
from ..judges import Judge


def _build_context(retrieved) -> str:
    """Concatenate the content of retrieved SearchResult objects."""
    parts = []
    for r in retrieved:
        c = (r.metadata.content or "").strip() if r.metadata else ""
        if c:
            parts.append(c)
    return "\n---\n".join(parts)


def run(questions: list[dict], embedder: Embedder, judge: Judge,
        top_k: int = 10, ef: Optional[int] = None,
        modality: str = "text") -> dict:
    """Execute the scenario and return aggregate metrics."""
    n_q = len(questions)
    if n_q == 0:
        return {"overall": 0.0, "by_axis": {}, "by_question_type": {}, "n_q": 0}

    per_q_seconds = []
    per_q_score = []
    per_q_axis = []
    per_q_type = []
    embeddings_emitted = 0
    history_turns_total = 0

    for q_idx, q in enumerate(questions):
        qid = q.get("question_id") or f"q{q_idx}"
        qtype = q.get("question_type") or "unknown"
        axis = question_type_axis(qtype)
        gold = q.get("answer", "") or ""
        question_text = q.get("question", "") or ""

        path = tempfile.mktemp(suffix=f"_{qid}.feather")
        t0 = time.perf_counter()
        try:
            db = feather_db.DB.open(path, dim=embedder.dim)
            if ef is not None:
                db.set_ef(ef)

            # ---------- Ingest haystack ----------
            uid = 0
            for sid, date_str, role, content, has_ans in iter_history_turns(q):
                if not content.strip():
                    continue
                uid += 1
                history_turns_total += 1

                # Best-effort timestamp parse — LongMemEval uses ISO-ish strings
                ts = 0
                try:
                    import datetime
                    ts = int(datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00")).timestamp())
                except Exception:
                    pass

                meta = feather_db.Metadata()
                meta.content = content
                meta.timestamp = ts
                meta.namespace_id = qid
                meta.entity_id = sid
                meta.source = role
                meta.set_attribute("has_answer", has_ans)

                vec = embedder.embed(content)
                embeddings_emitted += 1
                db.add(id=uid, vec=vec, meta=meta, modality=modality)

            # ---------- Retrieve ----------
            qvec = embedder.embed(question_text)
            embeddings_emitted += 1

            try:
                results = db.hybrid_search(
                    qvec, question_text, k=top_k, modality=modality,
                )
            except Exception:
                results = db.search(qvec, k=top_k, modality=modality)

            context = _build_context(results)

            # ---------- Score ----------
            jr = judge.score(predicted=context, gold=gold, question=question_text)
            per_q_score.append(jr.score)
            per_q_axis.append(axis)
            per_q_type.append(qtype)

        finally:
            elapsed = time.perf_counter() - t0
            per_q_seconds.append(elapsed)
            if os.path.exists(path):
                os.remove(path)

    # ---------- Aggregate ----------
    by_axis: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, list[float]] = defaultdict(list)
    for s, a, t in zip(per_q_score, per_q_axis, per_q_type):
        by_axis[a].append(s)
        by_type[t].append(s)

    overall = statistics.fmean(per_q_score) if per_q_score else 0.0
    return {
        "overall": overall,
        "by_axis": {k: statistics.fmean(v) for k, v in by_axis.items()},
        "by_question_type": {k: statistics.fmean(v) for k, v in by_type.items()},
        "per_q_seconds_p50": statistics.median(per_q_seconds) if per_q_seconds else 0.0,
        "per_q_seconds_mean": statistics.fmean(per_q_seconds) if per_q_seconds else 0.0,
        "n_questions": n_q,
        "history_turns_total": history_turns_total,
        "embeddings_emitted": embeddings_emitted,
        "embedder": embedder.name,
        "judge": judge.name,
        "top_k": top_k,
    }
