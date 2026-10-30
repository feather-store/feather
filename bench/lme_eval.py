"""LongMemEval measurement: the official judge, a lenient judge, per-question
records and failure buckets. Shared by bench/ and benchmarks/longmemeval_qa.py.

Two judges, both reported:
  official  LongMemEval's own per-type check (src/evaluation/evaluate_qa.py,
            `get_anscheck_prompt`, copied verbatim below), including the
            separate abstention prompt. Run it on gpt-4o-2024-08-06 as the
            paper does: one user message, no system prompt, temperature 0,
            max_tokens 10, correct iff 'yes' appears in the reply. This is the
            number to put next to other systems' published scores.
  lenient   one rubric for every type ("lenient about phrasing, strict about
            facts"), the judge these harnesses used before. Kept so earlier
            runs stay comparable, and as a second opinion: a question the two
            disagree on is a judge-noise candidate.

Abstention is decided by '_abs' in the question_id, as the official script
does: question_type never carries it.

Every failure under the official judge lands in one bucket:
  retrieval_miss       a gold evidence session was never retrieved
  packing_miss         every gold session was retrieved, one was cut by the
                       context budget before the reader saw it
  judge_disagreement   evidence covered, lenient judge says correct —
                       read it before believing either judge
  reader_wrong         evidence covered, both judges say wrong
  judge_error          the judge call itself failed; scored wrong, rerun it
  run_error            the question never got an answer (an API error, a
                       crash); scored wrong, rerun it
A question that fails to run is still a record, scored wrong: dropping it
would raise the score by shrinking its denominator, and an outage that
hits the hard questions would look like progress.
Abstention questions are never retrieval misses: their evidence sessions
exist to tempt a wrong answer, not to supply a right one.
"""
from __future__ import annotations
import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Callable, Iterable, Optional

Chat = Callable[[str, str], str]   # (system, user) -> reply; "" = no system

OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"

# ── Official prompts: verbatim from LongMemEval evaluate_qa.py ─────────────
_OFFICIAL_DEFAULT = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_OFFICIAL_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_OFFICIAL_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_OFFICIAL_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_OFFICIAL_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."

# ── Lenient judge (the rubric these harnesses used before) ────────────────
LENIENT_SYSTEM = (
    "You are an expert evaluator for chat-assistant memory benchmarks. "
    "Determine whether the assistant's answer correctly addresses the "
    "user's question, given a reference answer. Be lenient about phrasing "
    "but strict about facts. Output a JSON object: "
    '{"correct": true|false, "reason": "short rationale"}.'
)
LENIENT_USER_TMPL = (
    "Question: {question}\n"
    "Reference answer: {gold}\n"
    "Assistant answer: {predicted}\n\n"
    "Is the assistant answer correct?"
)


def is_abstention(question_id: str) -> bool:
    return "_abs" in (question_id or "")


def official_prompt(qtype: str, question: str, gold: str, response: str,
                    abstention: bool = False) -> str:
    if abstention:
        template = _OFFICIAL_ABSTENTION
    elif qtype in ("single-session-user", "single-session-assistant",
                   "multi-session"):
        template = _OFFICIAL_DEFAULT
    elif qtype == "temporal-reasoning":
        template = _OFFICIAL_TEMPORAL
    elif qtype == "knowledge-update":
        template = _OFFICIAL_KNOWLEDGE_UPDATE
    elif qtype == "single-session-preference":
        template = _OFFICIAL_PREFERENCE
    else:
        raise ValueError(f"no official judge prompt for type {qtype!r}")
    return template.format(question, gold, response)


def official_verdict(chat: Chat, qtype: str, question: str, gold: str,
                     response: str, abstention: bool) -> dict:
    raw = chat("", official_prompt(qtype, question, gold, response,
                                   abstention)).strip()
    return {"correct": "yes" in raw.lower(), "raw": raw}


_JSON_BOOL_RE = re.compile(r'"correct"\s*:\s*(true|false)', re.IGNORECASE)


def parse_lenient(raw: str) -> tuple[bool, str]:
    """(correct, reason) from the lenient judge's reply."""
    if not raw:
        return False, "empty judge response"
    m = _JSON_BOOL_RE.search(raw)
    if m:
        m2 = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
        return m.group(1).lower() == "true", (m2.group(1) if m2 else raw[:120])
    low = raw.lower()
    if "correct: true" in low or "is correct" in low or \
            "correct." in low.replace("incorrect", ""):
        return True, raw[:120]
    if "incorrect" in low or "wrong" in low or "false" in low:
        return False, raw[:120]
    return False, f"unparseable: {raw[:120]}"


def lenient_verdict(chat: Chat, question: str, gold: str,
                    response: str) -> dict:
    raw = chat(LENIENT_SYSTEM, LENIENT_USER_TMPL.format(
        question=question, gold=gold, predicted=response))
    ok, reason = parse_lenient(raw)
    return {"correct": ok, "raw": reason}


# ── Per-question records ──────────────────────────────────────────────────

def coverage(gold_sessions: Iterable[str], context_sessions: Iterable[str]
             ) -> Optional[float]:
    """Fraction of gold evidence sessions present in the context."""
    gold = set(gold_sessions or ())
    if not gold:
        return None
    return len(gold & set(context_sessions or ())) / len(gold)


def bucket(rec: dict) -> Optional[str]:
    """Failure bucket under the official judge (None = correct or unjudged).
    Falls back to the lenient verdict when no official judge ran."""
    if rec.get("run_error") or str(rec.get("answer", "")).startswith(
            "[answerer error"):
        return "run_error"
    primary = rec.get("official") or rec.get("lenient")
    if primary is None or primary["correct"]:
        return None
    if primary.get("error"):
        return "judge_error"
    if not rec["abstention"] and (rec.get("coverage") or 0.0) < 1.0:
        # Records written before retrieval_coverage existed: no way to tell.
        if (rec.get("retrieval_coverage") or 0.0) < 1.0:
            return "retrieval_miss"
        return "packing_miss"
    if rec.get("official") and rec.get("lenient") and rec["lenient"]["correct"]:
        return "judge_disagreement"
    return "reader_wrong"


def make_record(q: dict, *, retrieved_sessions: list[str],
                context_sessions: list[str], context_chars: int,
                answer: str, transcript: str = "",
                official: Optional[dict] = None,
                lenient: Optional[dict] = None,
                seconds: float = 0.0,
                followups: Optional[list] = None, missing: str = "",
                round1_sessions: Optional[list[str]] = None,
                **extra) -> dict:
    """One question's record. round1_sessions: the sessions the first
    retrieval round alone would have put in context, so a multi-round
    run reports what the extra rounds added."""
    qid = q.get("question_id", "")
    gold_sessions = list(q.get("answer_session_ids") or [])
    rec = {
        "question_id": qid,
        "question_type": q.get("question_type", "unknown"),
        "abstention": is_abstention(qid),
        "question": q.get("question", ""),
        "question_date": q.get("question_date", ""),
        "gold": q.get("answer", ""),
        "gold_sessions": gold_sessions,
        "retrieved_sessions": retrieved_sessions,
        "context_sessions": context_sessions,
        "retrieval_coverage": coverage(gold_sessions, retrieved_sessions),
        "coverage": coverage(gold_sessions, context_sessions),
        "context_chars": context_chars,
        "followups": followups or [],
        "missing": missing,
        "round1_coverage": (coverage(gold_sessions, round1_sessions)
                            if round1_sessions is not None else None),
        "answer": answer,
        "transcript": transcript,
        "official": official,
        "lenient": lenient,
        "seconds": round(seconds, 3),
    }
    rec.update(extra)
    rec["bucket"] = bucket(rec)
    return rec


def write_jsonl(path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def _mean(xs):
    return round(statistics.fmean(xs), 4) if xs else None


def summarize(records: list[dict]) -> dict:
    """Both scores overall and per type, abstention separately, the
    retrieval ceiling, judge agreement and the failure buckets."""
    out: dict = {"n_scored": len(records)}
    labels = [bucket(r) for r in records]
    out["run_errors"] = labels.count("run_error")
    out["judge_errors"] = labels.count("judge_error")
    for judge in ("official", "lenient"):
        # Over EVERY record once any has this judge: a record the judge
        # never saw is wrong, not absent (absent would raise the score).
        if not any(r.get(judge) for r in records):
            continue
        rs = records
        ok = lambda r: 1.0 if (r.get(judge) or {}).get("correct") else 0.0   # noqa: E731
        by_type = defaultdict(list)
        for r in rs:
            by_type[r["question_type"]].append(ok(r))
        out[f"overall_{judge}"] = _mean([ok(r) for r in rs])
        out[f"by_question_type_{judge}"] = {t: _mean(v) for t, v in
                                            sorted(by_type.items())}
        # The paper's headline is the mean over questions; the task-averaged
        # one is also common in vendor tables.
        out[f"task_averaged_{judge}"] = _mean(
            [statistics.fmean(v) for v in by_type.values()])
        out[f"abstention_{judge}"] = _mean([ok(r) for r in rs if r["abstention"]])
    both = [r for r in records if r.get("official") and r.get("lenient")]
    if both:
        out["judge_agreement"] = _mean(
            [1.0 if r["official"]["correct"] == r["lenient"]["correct"] else 0.0
             for r in both])
        out["official_only_correct"] = sum(
            r["official"]["correct"] and not r["lenient"]["correct"] for r in both)
        out["lenient_only_correct"] = sum(
            r["lenient"]["correct"] and not r["official"]["correct"] for r in both)
    answerable = [r for r in records
                  if not r["abstention"] and r["coverage"] is not None]
    out["evidence_fully_in_context"] = _mean(
        [1.0 if r["coverage"] >= 1.0 else 0.0 for r in answerable])
    # Evidence that reached the reader shortened (lme_context levels > 0):
    # in context, but possibly without the detail the answer needs.
    leveled = [r for r in answerable if r.get("evidence_levels")]
    if leveled:
        out["evidence_shortened"] = _mean(
            [1.0 if any(v > 0 for v in r["evidence_levels"].values()) else 0.0
             for r in leveled])
    out["evidence_fully_retrieved"] = _mean(
        [1.0 if (r["retrieval_coverage"] or 0) >= 1.0 else 0.0
         for r in answerable])
    multi = [r for r in answerable if r.get("followups")]
    out["multi_round_questions"] = sum(1 for r in records if r.get("followups"))
    if multi:
        # Only over questions that ran follow-ups: what the rounds added.
        out["multi_round_evidence_round1"] = _mean(
            [1.0 if (r["round1_coverage"] or 0) >= 1.0 else 0.0 for r in multi])
        out["multi_round_evidence_final"] = _mean(
            [1.0 if r["coverage"] >= 1.0 else 0.0 for r in multi])
    mapped = [r for r in records if r.get("count_map")]
    if mapped:
        out["count_map_questions"] = sum("error" not in r["count_map"]
                                         for r in mapped)
        out["count_map_fallbacks"] = sum("error" in r["count_map"]
                                         for r in mapped)
    # Recomputed, not read from the record: a run written before a bucket
    # was split keeps the old label in its JSONL.
    labels = [(r["question_type"], bucket(r)) for r in records]
    out["failure_buckets"] = dict(Counter(b for _, b in labels if b))
    by_type_bucket: dict = defaultdict(Counter)
    for qtype, b in labels:
        if b:
            by_type_bucket[qtype][b] += 1
    out["failure_buckets_by_type"] = {t: dict(c) for t, c in
                                      sorted(by_type_bucket.items())}
    return out


def format_summary(s: dict) -> str:
    lines = []
    if s.get("run_errors") or s.get("judge_errors"):
        lines.append(f"!! {s.get('run_errors', 0)} questions failed to run and "
                     f"{s.get('judge_errors', 0)} were not judged; all are "
                     "scored wrong. Rerun before reporting.")
    for judge in ("official", "lenient"):
        if f"overall_{judge}" not in s:
            continue
        lines.append(f"{judge:>8} judge: overall {s[f'overall_{judge}']:.3f}  "
                     f"task-avg {s[f'task_averaged_{judge}']:.3f}  "
                     f"abstention {s[f'abstention_{judge}']}")
        for t, v in s[f"by_question_type_{judge}"].items():
            lines.append(f"    {t:28s} {v:.3f}")
    if "judge_agreement" in s:
        lines.append(f"judge agreement {s['judge_agreement']:.3f}  "
                     f"(official-only correct {s['official_only_correct']}, "
                     f"lenient-only correct {s['lenient_only_correct']})")
    lines.append(f"evidence fully retrieved {s['evidence_fully_retrieved']}  "
                 f"fully in context {s['evidence_fully_in_context']}"
                 + (f"  shortened {s['evidence_shortened']}"
                    if "evidence_shortened" in s else ""))
    if s.get("multi_round_questions"):
        lines.append(f"multi-round on {s['multi_round_questions']} questions: "
                     f"evidence in context {s.get('multi_round_evidence_round1')}"
                     f" after round 1 -> {s.get('multi_round_evidence_final')}")
    if "count_map_questions" in s:
        lines.append(f"count map-reduce on {s['count_map_questions']} "
                     f"questions ({s['count_map_fallbacks']} fell back to "
                     "one read)")
    lines.append(f"failures: {s['failure_buckets']}")
    for t, c in s["failure_buckets_by_type"].items():
        lines.append(f"    {t:28s} {c}")
    return "\n".join(lines)


# ── Comparing two runs ────────────────────────────────────────────────────
# At n=500 one run's score has a standard error of ~1.3 points, so two
# headline numbers a point or two apart say nothing. Pairing by question
# removes the variance of the questions both runs got right (or wrong):
# only the discordant ones carry information, and McNemar's exact test
# asks whether they split more unevenly than a fair coin would.

def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p: b questions only A got right, c only B."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _paired(pairs: list[tuple[bool, bool]]) -> dict:
    n = len(pairs)
    b = sum(a and not z for a, z in pairs)      # A right, B wrong
    c = sum(z and not a for a, z in pairs)      # B right, A wrong
    diff = (c - b) / n if n else 0.0
    # SE of the paired difference in accuracy.
    se = ((b + c) - (c - b) ** 2 / n) ** 0.5 / n if n else 0.0
    return {"n": n, "a": _mean([float(a) for a, _ in pairs]),
            "b": _mean([float(z) for _, z in pairs]),
            "a_only": b, "b_only": c, "diff": round(diff, 4),
            "se": round(se, 4), "p": round(mcnemar_p(b, c), 4)}


def compare(records_a: list[dict], records_b: list[dict],
            judge: str = "official") -> dict:
    """Run B against run A on the questions both scored with `judge`:
    overall and per question type, plus the ids that flipped."""
    a = {r["question_id"]: r for r in records_a if r.get(judge)}
    b = {r["question_id"]: r for r in records_b if r.get(judge)}
    common = [q for q in a if q in b]
    ok = lambda r: bool(r[judge]["correct"])    # noqa: E731
    by_type: dict = defaultdict(list)
    for q in common:
        by_type[a[q]["question_type"]].append((ok(a[q]), ok(b[q])))
    return {
        "judge": judge,
        "overall": _paired([(ok(a[q]), ok(b[q])) for q in common]),
        "by_type": {t: _paired(v) for t, v in sorted(by_type.items())},
        "fixed": [q for q in common if ok(b[q]) and not ok(a[q])],
        "broken": [q for q in common if ok(a[q]) and not ok(b[q])],
        "only_in_a": len(a) - len(common),
        "only_in_b": len(b) - len(common),
    }


def format_compare(c: dict) -> str:
    def row(label, s):
        return (f"{label:28s} n={s['n']:<4d} A {s['a']:.3f}  B {s['b']:.3f}  "
                f"diff {s['diff']:+.3f} ±{s['se']:.3f}  "
                f"(A-only {s['a_only']}, B-only {s['b_only']})  "
                f"McNemar p={s['p']}")
    lines = [f"{c['judge']} judge, B vs A",
             row("overall", c["overall"])]
    lines += [row("  " + t, s) for t, s in c["by_type"].items()]
    lines.append(f"fixed by B: {len(c['fixed'])}  broken by B: "
                 f"{len(c['broken'])}")
    if c["only_in_a"] or c["only_in_b"]:
        lines.append(f"unpaired: {c['only_in_a']} only in A, "
                     f"{c['only_in_b']} only in B (excluded)")
    return "\n".join(lines)


def only_split(records: list[dict], split: str) -> list[dict]:
    """Records of the frozen dev or test split (bench.datasets.lme_split)."""
    if split == "all":
        return records
    from .datasets.lme_split import dev_ids
    dev = dev_ids()
    return [r for r in records if (r["question_id"] in dev) == (split == "dev")]


def check(records: list[dict], expected: int) -> list[str]:
    """What is wrong with a run's questions file as a full result: the
    wrong number of questions, a question twice, failed or unjudged ones,
    or no official verdict. [] = complete."""
    why = []
    ids = [r["question_id"] for r in records]
    if len(set(ids)) != len(ids):
        why.append(f"{len(ids) - len(set(ids))} questions appear twice")
    if len(set(ids)) != expected:
        why.append(f"{len(set(ids))} questions, expected {expected}")
    labels = Counter(bucket(r) for r in records)
    if labels["run_error"]:
        why.append(f"{labels['run_error']} questions failed to run")
    if labels["judge_error"]:
        why.append(f"{labels['judge_error']} questions were not judged")
    missing = sum(not r.get("official") for r in records)
    if missing:
        why.append(f"{missing} questions have no official verdict")
    return why


if __name__ == "__main__":
    # python -m bench.lme_eval A.jsonl B.jsonl [official|lenient] [all|dev|test]
    # python -m bench.lme_eval --check RUN.jsonl EXPECTED_N
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        problems = check(read_jsonl(sys.argv[2]), int(sys.argv[3]))
        for p in problems:
            print(f"INCOMPLETE: {p}", file=sys.stderr)
        sys.exit(1 if problems else 0)
    if len(sys.argv) not in (3, 4, 5):
        sys.exit("usage: python -m bench.lme_eval A.jsonl B.jsonl "
                 "[official|lenient] [all|dev|test]")
    judge = sys.argv[3] if len(sys.argv) > 3 else "official"
    split = sys.argv[4] if len(sys.argv) > 4 else "all"
    a, b = (only_split(read_jsonl(p), split) for p in sys.argv[1:3])
    print(f"split: {split}")
    print(format_compare(compare(a, b, judge)))
