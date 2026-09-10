#!/usr/bin/env bash
# Feather pre-release verification. Run from the repo root:
#   ./scripts/verify.sh          # everything
#   ./scripts/verify.sh quick    # skip the slow legs
#
# Exits non-zero if anything that should pass, fails. Each check prints why it
# exists, so a failure tells you what broke rather than just that something did.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
QUICK=${1:-full}
pass=0; fail=0; skip=0
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; pass=$((pass+1)); }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$1"; fail=$((fail+1)); }
skp()  { printf "  \033[33m-\033[0m %s\n" "$1"; skip=$((skip+1)); }
hdr()  { printf "\n\033[1m%s\033[0m\n" "$1"; }

hdr "1. Build"
if [ ! -x "$PY" ]; then bad "no interpreter at $PY (set PY=...)"; else
  rm -rf build
  if $PY setup.py build_ext --inplace >/tmp/_build.log 2>&1; then ok "C++ extension builds clean"
  else bad "build failed — see /tmp/_build.log"; tail -5 /tmp/_build.log; fi
fi

hdr "2. Test suite"
if $PY -m pytest tests -q --deselect tests/test_engine.py::TestProviderInterface::test_provider_str >/tmp/_pt.log 2>&1; then
  ok "$(grep -oE '[0-9]+ passed[^,]*' /tmp/_pt.log | tail -1)"
else bad "pytest failed"; tail -15 /tmp/_pt.log; fi

hdr "3. Format & feature scripts (v3-v9 compatibility, quantization, graph persistence)"
for t in test_secondary_index test_prefiltered_search test_auto_compact \
         test_quantization test_batch_ingest test_persist_graph; do
  if [ -f "$t.py" ]; then
    $PY "$t.py" >/dev/null 2>&1 && ok "$t" || bad "$t"
  else skp "$t (missing)"; fi
done

hdr "4. Data-integrity regressions (the four that could lose data)"
for t in tests/test_wal_recovery.py tests/test_wal_durability.py \
         tests/test_open_failure_safety.py tests/test_api_records.py; do
  n=$($PY -m pytest "$t" -q 2>/dev/null | grep -oE '[0-9]+ passed' | head -1)
  [ -n "$n" ] && ok "$(basename $t): $n" || bad "$(basename $t) failed"
done

hdr "5. Vendored Rust engine is in sync"
./scripts/sync-cpp.sh >/dev/null 2>&1
if git diff --quiet feather-cli/cpp/ 2>/dev/null; then ok "feather-cli/cpp matches include/"
else bad "vendored C++ has drifted — commit the sync"; git diff --stat feather-cli/cpp/ | tail -3; fi

hdr "6. Version consistency"
v_py=$(grep -oE '^version = "[^"]+"' pyproject.toml | head -1 | cut -d'"' -f2)
v_st=$(grep -oE 'version="[^"]+"' setup.py | head -1 | cut -d'"' -f2)
v_in=$(grep -oE '__version__ = "[^"]+"' feather_db/__init__.py | cut -d'"' -f2)
v_cg=$(grep -oE '^version = "[^"]+"' feather-cli/Cargo.toml | head -1 | cut -d'"' -f2)
if [ "$v_py" = "$v_st" ] && [ "$v_py" = "$v_in" ] && [ "$v_py" = "$v_cg" ]; then
  ok "all four declare $v_py"
else bad "version drift: pyproject=$v_py setup=$v_st __init__=$v_in Cargo=$v_cg"; fi

hdr "7. Release prerequisites"
command -v gh >/dev/null 2>&1 && {
  gh secret list 2>/dev/null | grep -q CARGO_REGISTRY_TOKEN \
    && ok "CARGO_REGISTRY_TOKEN is set" \
    || bad "CARGO_REGISTRY_TOKEN missing — crates.io publish will fail"
  gh api repos/feather-store/feather/environments --jq '.environments[].name' 2>/dev/null | grep -q pypi \
    && ok "pypi environment exists" \
    || bad "pypi environment missing"
  # Match an actual runner target, not the comments explaining why it was removed.
  if grep -hE "^[[:space:]]*(-[[:space:]]*)?(os:|runs-on:)" .github/workflows/*.yml 2>/dev/null \
       | grep -q "macos-13"; then
    bad "a workflow still targets macos-13 (retired 2025-12-04, queues forever)"
  else ok "no retired runner labels in any job target"; fi
} || skp "gh not installed — skipping release checks"

hdr "8. Live API behaviour (the two integration reports)"
if [ "$QUICK" != "quick" ]; then
  $PY - 2>/dev/null <<'PY' && ok "API contract checks passed" || bad "API contract checks failed"
import os, sys, shutil, logging, numpy as np
logging.disable(logging.INFO)
R=os.getcwd(); A=os.path.join(R,"feather-api")
os.environ.update(FEATHER_DATA_DIR="/tmp/_vfy", FEATHER_DB_DIM="32", FEATHER_DEV_MODE="1")
os.environ.pop("FEATHER_API_KEY",None)
shutil.rmtree("/tmp/_vfy",ignore_errors=True); os.makedirs("/tmp/_vfy")
sys.path[:0]=[R,A]
for m in [k for k in list(sys.modules) if k=="app" or k.startswith("app.")]: del sys.modules[m]
from fastapi.testclient import TestClient
from app.main import app
c=TestClient(app); c.__enter__(); NS="v"; D=32
def vec(s): 
    r=np.random.default_rng(s); v=r.random(D).astype(np.float32); return (v/np.linalg.norm(v)).tolist()
checks=[]
for i in range(20): c.post(f"/v1/{NS}/vectors", json={"id":i,"vector":vec(i),"metadata":{"content":f"d{i}"}})
for i in range(12): c.delete(f"/v1/{NS}/records/{i}")
rows=c.get(f"/v1/{NS}/records", params={"limit":5}).json()["results"]
checks.append(("tombstones do not eat the page", len(rows)==5))
checks.append(("import that stores nothing is not 200",
    c.post(f"/v1/{NS}/import", json={"items":[{"id":99,"vector":[0.1]*4}]}).status_code==400))
checks.append(("unknown metadata keys rejected",
    c.post(f"/v1/{NS}/vectors", json={"id":50,"vector":vec(50),"metadata":{"content":"x","bogus":"y"}}).status_code==422))
checks.append(("zero vector rejected",
    c.post(f"/v1/{NS}/search", json={"vector":[0.0]*D,"k":3}).status_code==400))
g=c.get(f"/v1/{NS}/records/15", params={"include_vector":"true"}).json()
checks.append(("stored vector readable", len(g.get("vector") or [])==D))
checks.append(("single-get has id+metadata", "id" in g and "metadata" in g))
r=c.post(f"/v1/{NS}/search", json={"vector":vec(15),"k":2,"raw_score":True,"track":False}).json()
checks.append(("raw cosine exposed", r["results"][0].get("cosine") is not None))
before=c.get(f"/v1/{NS}/records/15").json()["recall_count"]
c.post(f"/v1/{NS}/search", json={"vector":vec(15),"k":2,"track":False})
checks.append(("track=false does not mutate", c.get(f"/v1/{NS}/records/15").json()["recall_count"]==before))
c.__exit__(None,None,None)
bad=[n for n,okk in checks if not okk]
for n,okk in checks: print(("     ok  " if okk else "     FAIL ")+n)
sys.exit(1 if bad else 0)
PY
else skp "API contract checks (quick mode)"; fi

printf "\n\033[1m%d passed, %d failed, %d skipped\033[0m\n" "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
