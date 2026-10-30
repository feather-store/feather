#!/usr/bin/env bash
# LongMemEval dev ablations: each step runs the 100 frozen dev questions
# (bench/datasets/longmemeval_s_dev.json) and is compared question by
# question with BASELINE on those same questions. Tune here, never on test.
#
#   scripts/lme_dev.sh                  # every step
#   scripts/lme_dev.sh current countmap # chosen steps
#
# Steps add ONE change to the current code, so a gain can be attributed;
# combine the winners, then freeze and run all 500 once (scripts/lme_final.sh).
# Each dev run is ~4 min at 8 workers. Needs GEMINI_API_KEY (or
# GOOGLE_API_KEY) and OPENROUTER_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
BASELINE=${BASELINE:-bench/results/longmemeval__s__20261030_131554__questions.jsonl}
STRONG=${STRONG:-gemini-2.5-pro}       # the stronger reader to try
CHEAP=${CHEAP:-gemini-2.5-flash}       # map calls and fact extraction
TOKENS=${TOKENS:-80000}                # the larger budget to try
LOG=bench/results/dev_runs.tsv

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (official judge)}"
[[ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]] ||
  { echo "set GEMINI_API_KEY or GOOGLE_API_KEY" >&2; exit 2; }

BASE=(longmemeval --dataset s --k 20 --embedder gemini
      --judge llm --judge-provider gemini-text --judge-model gemini-2.5-flash
      --official-judge-provider openrouter
      --official-judge-model openai/gpt-4o-2024-08-06
      --workers 8 --limit 0 --split dev)

step_flags() {
  case $1 in
    current)  ;;                                       # today's code, no flags
    countmap) echo --count-map retrieved ;;
    strong)   echo --answerer-provider gemini-text --answerer-model "$STRONG" ;;
    facts)    echo --facts session --system-provider gemini-text \
                   --system-model "$CHEAP" ;;
    votes)    echo --count-votes 3 ;;
    verified) echo --reader verified ;;
    tokens)   echo --context-tokens "$TOKENS" ;;
    *) echo "unknown step $1 (current countmap strong facts votes tokens verified)" >&2
       return 1 ;;
  esac
}

steps=("$@")
[[ ${#steps[@]} -gt 0 ]] || steps=(current countmap strong facts votes tokens verified)
[[ -f $LOG ]] || printf "step\tquestions_file\tflags\n" > "$LOG"

for step in "${steps[@]}"; do
  flags=$(step_flags "$step")
  echo "==> dev: $step ${flags}"
  marker=$(mktemp)
  # shellcheck disable=SC2086  # flags are word lists by design
  "$PY" -m bench run "${BASE[@]}" $flags
  new=$(find bench/results -name '*__questions.jsonl' -newer "$marker" |
        sort | tail -1)
  rm -f "$marker"
  [[ -n $new ]] || { echo "no questions file from step $step" >&2; exit 1; }
  "$PY" -m bench.lme_eval --check "$new" 100 ||
    { echo "step $step: $new is not a complete dev run" >&2; exit 1; }
  printf "%s\t%s\t%s\n" "$step" "$new" "$flags" >> "$LOG"
  "$PY" -m bench.lme_eval "$BASELINE" "$new" official dev |
    tee "bench/results/dev_${step}_vs_baseline.txt"
done
echo "runs logged in $LOG"
