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


def _judge_can_answer(judge) -> bool:
    """True if the judge object also exposes an `answer(question, ctx)` method."""
    return callable(getattr(judge, "answer", None))


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
    failures: list[dict] = []
    use_llm_answer = _judge_can_answer(judge)
    has_batch = callable(getattr(embedder, "embed_batch", None))
    progress_every = max(1, n_q // 20)  # ~5% increments
    run_t0 = time.perf_counter()

    for q_idx, q in enumerate(questions):
        qid = q.get("question_id") or f"q{q_idx}"
        qtype = q.get("question_type") or "unknown"
        axis = question_type_axis(qtype)
        gold = q.get("answer", "") or ""
        question_text = q.get("question", "") or ""

        path = tempfile.mktemp(suffix=f"_{qid}.feather")
        t0 = time.perf_counter()
        scored = False
        try:
            db = feather_db.DB.open(path, dim=embedder.dim)
            if ef is not None:
                db.set_ef(ef)

            # ---------- Collect haystack into a flat list ----------
            import datetime
            turns = []
            for sid, date_str, role, content, has_ans in iter_history_turns(q):
                if not content.strip():
                    continue
                ts = 0
                try:
                    ts = int(datetime.datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")).timestamp())
                except Exception:
                    pass
                turns.append((sid, ts, role, content, has_ans))
            history_turns_total += len(turns)

            # ---------- Embed all turns + question (batch if possible) ----------
            if has_batch:
                texts = [t[3] for t in turns] + [question_text]
                vecs = embedder.embed_batch(texts)
                turn_vecs, qvec = vecs[:-1], vecs[-1]
            else:
                turn_vecs = [embedder.embed(t[3]) for t in turns]
                qvec = embedder.embed(question_text)
            embeddings_emitted += len(turns) + 1

            # ---------- Ingest ----------
            for uid, ((sid, ts, role, content, has_ans), v) in enumerate(
                    zip(turns, turn_vecs), start=1):
                meta = feather_db.Metadata()
                meta.content = content
                meta.timestamp = ts
                meta.namespace_id = qid
                meta.entity_id = sid
                meta.source = role
                meta.set_attribute("has_answer", has_ans)
                db.add(id=uid, vec=v, meta=meta, modality=modality)

            # ---------- Retrieve ----------
            try:
                results = db.hybrid_search(
                    qvec, question_text, k=top_k, modality=modality,
                )
            except Exception:
                results = db.search(qvec, k=top_k, modality=modality)

            context = _build_context(results)

            # ---------- Answer ----------
            predicted = (judge.answer(question_text, context)
                         if use_llm_answer else context)

            # ---------- Score ----------
            jr = judge.score(predicted=predicted, gold=gold, question=question_text)
            per_q_score.append(jr.score)
            per_q_axis.append(axis)
            per_q_type.append(qtype)
            scored = True

        except Exception as e:
            failures.append({
                "question_id": qid,
                "question_type": qtype,
                "error": f"{type(e).__name__}: {e}",
                "q_index": q_idx,
            })
            print(f"[lme] q{q_idx+1}/{n_q} ({qid}) FAILED: "
                  f"{type(e).__name__}: {str(e)[:120]}",
                  flush=True)

        finally:
            elapsed = time.perf_counter() - t0
            if scored:
                per_q_seconds.append(elapsed)
            if os.path.exists(path):
                os.remove(path)

            # Progress print
            if (q_idx + 1) % progress_every == 0 or q_idx + 1 == n_q:
                done = q_idx + 1
                ok = len(per_q_score)
                running = (statistics.fmean(per_q_score) if per_q_score else 0.0)
                wall = time.perf_counter() - run_t0
                eta = (wall / done) * (n_q - done) if done else 0
                print(f"[lme] {done}/{n_q}  ok={ok}  fail={len(failures)}  "
                      f"running_score={running:.3f}  elapsed={wall:.0f}s  "
                      f"eta={eta:.0f}s",
                      flush=True)

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
        "n_scored": len(per_q_score),
        "n_failures": len(failures),
        "failures_sample": failures[:10],
        "history_turns_total": history_turns_total,
        "embeddings_emitted": embeddings_emitted,
        "embedder": embedder.name,
        "judge": judge.name,
        "top_k": top_k,
    }
