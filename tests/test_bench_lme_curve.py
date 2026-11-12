"""Context-size curve: the embedding cache, the full-context baseline and
the table that compares them."""
import json

import numpy as np

from bench import lme_curve
from bench.embedders import CachedEmbedder, get_embedder
from bench.scenarios import longmemeval as lme


class _Counting:
    """An embedder that counts its paid calls."""
    def __init__(self):
        self.inner = get_embedder("deterministic", dim=8)
        self.dim, self.name, self.calls = 8, "counting_d8", 0

    def embed(self, text):
        self.calls += 1
        return self.inner.embed(text)

    def embed_batch(self, texts):
        self.calls += 1
        return np.stack([self.inner.embed(t) for t in texts])


def test_cache_returns_the_same_vectors_without_paying(tmp_path):
    paid = _Counting()
    a = CachedEmbedder(paid, str(tmp_path)).embed_batch(["x", "y"])
    b = CachedEmbedder(paid, str(tmp_path)).embed_batch(["x", "y"])
    assert paid.calls == 1 and np.array_equal(a, b)
    CachedEmbedder(paid, str(tmp_path)).embed_batch(["x", "y "])   # one char
    assert paid.calls == 2
    CachedEmbedder(paid, str(tmp_path)).embed("q")                  # not cached
    CachedEmbedder(paid, str(tmp_path)).embed("q")
    assert paid.calls == 4


def _question():
    sessions = [("s1", "I bought a road bike today.")] + \
        [(f"d{i}", f"Let's talk about recipe number {i}.") for i in range(8)] + \
        [("s2", "Picked up a mountain bike at the shop.")]
    return {"question_id": "q1", "question_type": "multi-session",
            "question": "How many bikes did I buy? road bike", "answer": "2",
            "question_date": "2023/05/30 (Tue) 12:00",
            "haystack_session_ids": [s for s, _ in sessions],
            "haystack_dates": [f"2023/05/{i + 1:02d} (Mon) 10:00"
                               for i in range(len(sessions))],
            "haystack_sessions": [[{"role": "user", "content": c},
                                   {"role": "assistant", "content": "x" * 3000}]
                                  for _, c in sessions],
            "answer_session_ids": ["s1", "s2"]}


class _Answerer:
    name = "fake"

    def answer_full(self, question, context, question_date=""):
        n = sum(w in context for w in ("road bike", "mountain bike"))
        return str(n), context

    def verdict(self, question, gold, predicted):
        return {"correct": predicted.split()[0] == gold, "raw": ""}


def test_full_context_reads_every_session_verbatim_and_embeds_nothing(tmp_path):
    paid = _Counting()
    m = lme.run([_question()], embedder=paid, judge=_Answerer(), top_k=1,
                context_chars=2000, full_context=True,
                count_tokens=len, questions_path=tmp_path / "q.jsonl")
    rec = json.loads((tmp_path / "q.jsonl").read_text())
    assert paid.calls == 0 and m["embeddings_emitted"] == 0
    assert m["overall"] == 1.0 and len(rec["context_sessions"]) == 10
    assert "[…]" not in rec["transcript"]           # nothing shortened
    assert rec["transcript"].index("road bike") < rec["transcript"].index("mountain")
    assert rec["context_tokens"] == len(rec["transcript"])


def test_curve_rows_pool_reruns_and_compare_with_the_reference(tmp_path, capsys):
    def result(name, params, right):
        qf = tmp_path / f"{name}__questions.jsonl"
        qf.write_text("".join(json.dumps({
            "question_id": f"q{i}", "question_type": "multi-session",
            "official": {"correct": i < right}, "context_tokens": 1000 * (i + 1),
            "context_sessions": ["a", "b"], "coverage": 1.0}) + "\n"
            for i in range(10)))
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"params": params,
                                 "metrics": {"questions_file": str(qf)}}))
        return str(p)
    small = {"k": 5, "context_tokens": 10_000}
    big = {"k": 20, "context_tokens": 40_000}
    files = [result("a", small, 6), result("b", small, 8), result("c", big, 9),
             result("d", {"k": 20, "full_context": True}, 10)]
    assert lme_curve.main(["--ref", "k=20 40k tok"] + files) == 0
    out = capsys.readouterr().out
    assert "k=5 10k tok       2   10  70.0" in out and "-20.0" in out
    assert "full context" in out and "MS 100" in out


def test_evidence_position_runs_are_rows_of_their_own():
    from bench.lme_curve import label
    assert label({"k": 20, "context_tokens": 10_000}) == "k=20 10k tok"
    assert label({"k": 20, "context_tokens": 10_000, "evidence_position": "first"}) == \
        "k=20 10k tok, evidence first"
    assert label({"full_context": True, "evidence_position": "date"}) == "full context"
