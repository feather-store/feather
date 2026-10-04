#!/bin/bash
# Sequential full benchmark run — each step logs to ~/fbench/out/<name>.log with wall time.
cd ~/fbench && . venv/bin/activate && cd feather
OUT=~/fbench/out; mkdir -p $OUT
C=~/.cache/feather
step() { local name=$1; shift; echo "=== START $name $(date +%T)" >> $OUT/_progress.log
  local t0=$(date +%s.%N); ( "$@" ) > $OUT/$name.log 2>&1; local rc=$?
  echo "=== END $name rc=$rc secs=$(echo "$(date +%s.%N)-$t0" | bc)" >> $OUT/_progress.log; }

# 1. core micro-suite
step suite_128_fsync  python benchmarks/full_suite.py --n 100000 --dim 128 --out $OUT/suite_128_fsync.json
step suite_128_nosync env FEATHER_WAL_SYNC=0 python benchmarks/full_suite.py --n 100000 --dim 128 --out $OUT/suite_128_nosync.json
step suite_768_fsync  python benchmarks/full_suite.py --n 50000 --dim 768 --links 20000 --queries 500 --out $OUT/suite_768_fsync.json
# 2. root feature tests (include perf numbers)
for t in test_auto_compact test_batch_ingest test_int8_ram test_parallel_load test_persist_graph test_prefiltered_search test_quantization test_secondary_index; do
  step root_$t python $t.py; done
# 3. legacy benchmark scripts (write .feather into cwd -> run in tmp)
mkdir -p /tmp/legacy && cp benchmarks/phase3_benchmark.py benchmarks/stress_test.py /tmp/legacy/
step legacy_phase3 bash -c "cd /tmp/legacy && PYTHONPATH=$PWD python phase3_benchmark.py"
step legacy_stress bash -c "cd /tmp/legacy && PYTHONPATH=$PWD python stress_test.py"
# 4. bench harness
step bench_ann_10k_128  python -m bench run vector_ann --dataset synthetic --n 10000 --dim 128 --queries 200
step bench_ann_50k_768  python -m bench run vector_ann --dataset synthetic --n 50000 --dim 768 --queries 200
step bench_siftsmall    python -m bench run vector_ann_real --dataset siftsmall --n 0 --queries 0 --ef-sweep 10,50,100,200
step bench_sift500k     env FEATHER_WAL_SYNC=0 python -m bench run vector_ann_real --dataset sift1m --n 500000 --queries 1000 --ef-sweep 10,50,100,200
step bench_lme_oracle   env FEATHER_WAL_SYNC=0 python -m bench run longmemeval --dataset oracle --limit 0
# 5. LongMemEval BM25 retrieval (real numbers, no API key)
step lme_bm25_s       python benchmarks/longmemeval.py --data $C/longmemeval/longmemeval_s_cleaned.json --mode keyword
step lme_bm25_oracle  python benchmarks/longmemeval.py --data $C/longmemeval/longmemeval_oracle.json --mode keyword
step lme_qa_dryrun    python benchmarks/longmemeval_qa.py --data $C/longmemeval/longmemeval_s_cleaned.json --mode keyword --limit 3 --dry-run
# 6. HTTP API
step api_20k python benchmarks/api_bench.py --n 20000 --dim 128 --out $OUT/api_20k.json
step bench_report python -m bench report
echo "=== ALL DONE $(date +%T)" >> $OUT/_progress.log
