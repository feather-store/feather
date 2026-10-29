#!/bin/bash
# Every benchmark, branch then master each time (so drift hits both alike).
SP=$SCRATCH
PY=$REPO/.venv/bin/python
BR=$REPO
MA=$SP/master
R=$SP/res
both() {   # name, command (run with cwd + PYTHONPATH = the tree)
  for t in branch master; do
    tree=$BR; [ $t = master ] && tree=$MA
    echo "[$(date +%H:%M:%S)] $1 / $t" >> $R/progress.log
    mkdir -p $R/$t
    (cd $tree && PYTHONPATH=$tree eval "$2") > $R/$t/$1.log 2>&1
    echo "   exit $?" >> $R/progress.log
  done
}
branch_only() {
  echo "[$(date +%H:%M:%S)] $1 / branch" >> $R/progress.log
  mkdir -p $R/branch
  (cd $BR && PYTHONPATH=$BR eval "$2") > $R/branch/$1.log 2>&1
  echo "   exit $?" >> $R/progress.log
}
: > $R/progress.log
both longmemeval "$PY benchmarks/longmemeval.py --data $SP/data/lme_s.json --mode keyword"
both torn "$PY $R/torn.py \$PWD 8"
both newest "$PY $R/newest_bench.py"
both full_suite "$PY benchmarks/full_suite.py --out $R/\$(basename \$PWD)_full_suite.json"
both search_simd_768 "$PY benchmarks/search_simd_bench.py --n 100000 --dim 768 --out $R/\$(basename \$PWD)_simd768.json"
both search_simd_768_int8 "$PY benchmarks/search_simd_bench.py --n 100000 --dim 768 --int8 --out $R/\$(basename \$PWD)_simd768_int8.json"
both concurrency "$PY benchmarks/concurrency_bench.py orchestrate --n 100000 --dim 128 --seconds 5 --out $R/\$(basename \$PWD)_concurrency"
both api_bench "$PY benchmarks/api_bench.py --n 20000 --dim 128 --port 8765 --out $R/\$(basename \$PWD)_api.json"
both api_concurrency "$PY benchmarks/api_concurrency_bench.py --n 20000 --dim 128 --seconds 8 --out $R/\$(basename \$PWD)_api_conc.json"
both stress_test "rm -f stress_test.feather*; $PY benchmarks/stress_test.py; rm -f stress_test.feather*"
both phase3 "$PY benchmarks/phase3_benchmark.py"
both readers_memory "f=\$(mktemp -d)/r.feather; $PY -c \"import feather_db,numpy as np; db=feather_db.DB.open('\$f',dim=384); db.add_batch(list(range(100000)), np.random.default_rng(0).random((100000,384),dtype=np.float32)); db.close()\"; $PY benchmarks/readers_memory.py \$f --readers 10 --dim 384"
branch_only save_cost "$PY benchmarks/save_cost.py"
branch_only int8_refine "$PY benchmarks/int8_refine.py"
branch_only namespace_search "$PY benchmarks/namespace_search.py"
branch_only idle_namespaces "$PY benchmarks/idle_namespaces.py"
echo "[$(date +%H:%M:%S)] ALL DONE" >> $R/progress.log
