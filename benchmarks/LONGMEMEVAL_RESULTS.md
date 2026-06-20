# Feather on LongMemEval — retrieval recall

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) is a
long-term-memory benchmark: 500 questions, each over a haystack of ~48
user/assistant sessions (avg, `longmemeval_s`), with gold `answer_session_ids`.

`benchmarks/longmemeval.py` measures **retrieval recall@k** — does Feather
surface a gold evidence session in the top-k for each question. Each session is
ingested as one record; the question is the query.

## Results — BM25 (`keyword_search`), full 500 questions

No embedder, no API key, **11 s end-to-end** on a laptop.

| metric      | recall |
|-------------|--------|
| recall@1    | 0.874  |
| recall@3    | 0.942  |
| recall@5    | 0.974  |
| recall@10   | 0.986  |

By question type (recall@10):

| type                       |   n | recall@10 |
|----------------------------|-----|-----------|
| knowledge-update           |  78 | 1.000 |
| single-session-assistant   |  56 | 1.000 |
| single-session-user        |  70 | 1.000 |
| multi-session              | 133 | 0.985 |
| temporal-reasoning         | 133 | 0.985 |
| single-session-preference  |  30 | 0.900 |

## How to reproduce

```bash
# dataset (277 MB)
curl -sL -o /tmp/lme_s.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# BM25 baseline (no key)
python benchmarks/longmemeval.py --data /tmp/lme_s.json --mode keyword

# dense / hybrid with a real embedder
GOOGLE_API_KEY=... python benchmarks/longmemeval.py --data /tmp/lme_s.json \
    --mode hybrid --embed-provider gemini
```

## Honest scope

This is **retrieval recall** (did the evidence session make the top-k), not
end-to-end **QA accuracy** (an LLM answering from retrieved memory, judged
correct). Vendor headlines such as "92% on LongMemEval" usually refer to QA
accuracy — a different, harder metric — so these numbers are **not** directly
comparable to those. What they show is that Feather's retrieval foundation is
strong out of the box (97.4% @5 with BM25 alone), which is the prerequisite for
good QA. Dense (`--mode dense`) and hybrid RRF (`--mode hybrid`) are expected to
lift recall@1 and the `single-session-preference` tail; a QA-accuracy harness
(retrieve → LLM answer → judge) is the natural next layer.
