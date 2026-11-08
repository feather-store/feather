#!/usr/bin/env bash
# The reportable LongMemEval run: all 500 questions, official judge, none
# dropped, on a committed tree, once per frozen configuration.
#
#   scripts/lme_final.sh [flags chosen on dev, e.g. --reader verified]
#
# Measured 2026-10-30: the plain run (single reader + evidence packing)
# scored 0.912; --reader routed with --count-map retrieved scored 0.794.
#
# Runs the frozen configuration REPEATS times (default 3) and prints the
# headline (all 500) and the held-out 400 (the test split), each compared
# with BASELINE (a space-separated list of runs) by bench.compare: the mean
# over runs with a 95% CI, McNemar on the flips, and the rerun noise. Report
# the mean and its CI, the result JSONs' `command`, the per-question JSONLs
# and the reader model alongside the number.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
BASELINE=${BASELINE:-bench/results/longmemeval__s__20261030_131554__questions.jsonl}
REPEATS=${REPEATS:-3}

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (official judge)}"
[[ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]] ||
  { echo "set GEMINI_API_KEY or GOOGLE_API_KEY" >&2; exit 2; }

# Same rule as bench/runner.py's `dirty`: untracked code counts, the
# bench's own output does not.
if git status --porcelain |
     grep -vE '^.. bench/(results|reports)/' | grep -q .; then
  echo "working tree is dirty: commit first, or the result's commit does" \
       "not reproduce it" >&2
  git status --short | grep -vE '^.. bench/(results|reports)/' >&2
  exit 1
fi

runs=()
for rep in $(seq 1 "$REPEATS"); do
  echo "==> run $rep of $REPEATS"
  marker=$(mktemp)
  "$PY" -m bench run longmemeval --dataset s --k 20 --embedder gemini \
    --judge llm --judge-provider gemini-text --judge-model gemini-2.5-flash \
    --official-judge-provider openrouter \
    --official-judge-model openai/gpt-4o-2024-08-06 \
    --workers 8 --limit 0 --split all "$@"
  new=$(find bench/results -name '*__questions.jsonl' -size +0 -newer "$marker" |
        sort | tail -1)
  rm -f "$marker"
  [[ -n $new ]] || { echo "no questions file written" >&2; exit 1; }
  # A short or partly failed run must not be reported as the headline.
  "$PY" -m bench.lme_eval --check "$new" 500 ||
    { echo "$new is not a complete run: fix and rerun" >&2; exit 1; }
  runs+=("$new")
done

# shellcheck disable=SC2086  # BASELINE is a list of files
{
  echo; echo "== headline: all 500, $REPEATS runs =="
  "$PY" -m bench.compare lme --a $BASELINE --b "${runs[@]}" --split all
  echo; echo "== held out: test 400 =="
  "$PY" -m bench.compare lme --a $BASELINE --b "${runs[@]}" --split test
}
