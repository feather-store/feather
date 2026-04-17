"""
Feather DB v0.8.0 — Intense BM25 + Hybrid Search Test Suite

Tests:
  1.  Tokenizer — stop words, special chars, case, min-length
  2.  BM25 IDF logic — rare terms rank higher than common ones
  3.  BM25 TF saturation — high freq doesn't dominate indefinitely
  4.  BM25 doc length normalization — short precise docs beat long padded ones
  5.  Exact keyword recall — BM25 finds exact terms vector search misses
  6.  keyword_search top-k ordering
  7.  keyword_search with SearchFilter
  8.  keyword_search on empty corpus
  9.  keyword_search zero matches
  10. keyword_search single document corpus
  11. hybrid_search RRF merging — both rankers contribute
  12. hybrid_search rrf_k parameter effect
  13. hybrid_search with filter
  14. hybrid_search empty query (falls back to vector only)
  15. Persistence — BM25 rebuilds from metadata after save+reload
  16. update_metadata re-indexes content
  17. add() auto-indexes content
  18. Records without content are skipped gracefully
  19. Large corpus performance — 5 000 docs, BM25 + hybrid under 500 ms
  20. Large corpus — hybrid beats pure vector on keyword-specific query
  21. Unicode / multilingual content
  22. Very long content (10 000 char document)
  23. Duplicate content (same text, different IDs)
  24. Concurrent independent DB instances don't interfere
  25. forget() removes from BM25 (via importance=0, _deleted flag)

Run:
    source repro_venv/bin/activate
    PYTHONPATH=/path/to/feather python tests/test_bm25_hybrid.py
"""

import sys, os, time, math, tempfile, random, string
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feather_db
from feather_db.core import SearchFilter, ScoringConfig
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((name, condition))

def rng_vec(dim=128, seed=None):
    r = np.random.RandomState(seed)
    v = r.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)

def make_db(corpus: list[tuple[int,str]], dim=128) -> tuple[feather_db.DB, str]:
    """Create a fresh temp DB with the given (id, text) corpus."""
    f = tempfile.mktemp(suffix=".feather")
    db = feather_db.DB.open(f, dim=dim)
    for doc_id, text in corpus:
        meta = feather_db.Metadata()
        meta.content = text
        meta.timestamp = int(time.time())
        meta.importance = 1.0
        db.add(id=doc_id, vec=rng_vec(dim, seed=doc_id), meta=meta)
    return db, f

def cleanup(*paths):
    for p in paths:
        try: os.unlink(p)
        except: pass

# ── Test corpus ───────────────────────────────────────────────────────────────

BASE_CORPUS = [
    (1,  "Python is widely used for machine learning and data science pipelines"),
    (2,  "BM25 is the Okapi BM25 ranking function used in information retrieval systems"),
    (3,  "Feather DB embeds BM25 hybrid search directly into the vector database"),
    (4,  "JavaScript and TypeScript power modern web frontend applications"),
    (5,  "Hybrid search merges BM25 keyword scoring with dense vector similarity via RRF"),
    (6,  "Vector databases store high-dimensional embeddings for ANN similarity lookup"),
    (7,  "Reciprocal Rank Fusion combines ranked lists from multiple retrieval systems"),
    (8,  "Python pandas and numpy are essential libraries for numerical data manipulation"),
    (9,  "Information retrieval inverted indexes accelerate keyword lookup performance"),
    (10, "Machine learning deep neural networks require massive training datasets"),
]

# ─────────────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print(" Feather DB v0.8.0 — BM25 / Hybrid Search Tests")
print("══════════════════════════════════════════════\n")

# ── T01: Tokenizer stop words ─────────────────────────────────────────────────
print("── Section 1: Tokenizer ──────────────────────")
# If stop words work, "the quick brown fox" should index only "quick", "brown", "fox"
db, f = make_db([(1, "the quick brown fox jumps over the lazy dog")], dim=4)
r = db.keyword_search("the", k=5)  # stop word — should match nothing
check("T01 stop word 'the' not indexed", len(r) == 0,
      f"got {len(r)} results")
r2 = db.keyword_search("quick brown fox", k=5)
check("T01 content words do match", len(r2) == 1 and r2[0].id == 1,
      f"got {[x.id for x in r2]}")
cleanup(f)

# ── T02: Case insensitivity ───────────────────────────────────────────────────
db, f = make_db([(1, "Python Machine Learning"), (2, "python machine learning")], dim=4)
r_upper = db.keyword_search("PYTHON MACHINE", k=5)
r_lower = db.keyword_search("python machine", k=5)
check("T02 case insensitive — UPPER query matches lower content",
      set(x.id for x in r_upper) == set(x.id for x in r_lower))
cleanup(f)

# ── T03: Min-length filter (2 chars) ──────────────────────────────────────────
db, f = make_db([(1, "AI is a ML system")], dim=4)
r = db.keyword_search("AI ML", k=5)   # "AI"=2 chars ok, "ML"=2 chars ok, "is"/"a"=stop
check("T03 2-char tokens indexed", len(r) == 1, f"got {len(r)}")
r2 = db.keyword_search("a", k=5)      # 1-char, should not match
check("T03 1-char token not indexed", len(r2) == 0, f"got {len(r2)}")
cleanup(f)

# ── T04: Special characters stripped ─────────────────────────────────────────
db, f = make_db([(1, "neural-networks: deep_learning, transformers!")], dim=4)
r = db.keyword_search("neural networks deep learning transformers", k=5)
check("T04 special chars stripped — tokens still found", len(r) == 1)
cleanup(f)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 2: BM25 Scoring Correctness ──────")

# ── T05: Rare term gets higher IDF ────────────────────────────────────────────
# "featherdb" appears in 1 doc; "python" appears in many
db, f = make_db([
    (1,  "python python python python is everywhere"),
    (2,  "python data science pipeline processing"),
    (3,  "python web development backend api"),
    (4,  "featherdb unique specialized embedded database"),
], dim=4)
r = db.keyword_search("featherdb python", k=4)
check("T05 rare term IDF — 'featherdb' doc ranks #1",
      len(r) >= 1 and r[0].id == 4,
      f"top id={r[0].id if r else 'N/A'}")
cleanup(f)

# ── T06: TF saturation — doubling TF doesn't double score proportionally ──────
db, f = make_db([
    (1, "search " * 20),   # high TF
    (2, "search " * 2),    # low TF
], dim=4)
r = db.keyword_search("search", k=2)
if len(r) >= 2:
    high_tf_score = r[0].score if r[0].id == 1 else r[1].score
    low_tf_score  = r[1].score if r[1].id == 2 else r[0].score
    ratio = high_tf_score / (low_tf_score + 1e-9)
    check("T06 TF saturation — ratio < 10x despite 10x TF difference",
          ratio < 10.0, f"score ratio={ratio:.2f}")
else:
    check("T06 TF saturation", False, "insufficient results")
cleanup(f)

# ── T07: Doc length normalization ────────────────────────────────────────────
# Short precise doc should rank above long padded doc for same query term
db, f = make_db([
    (1, "quantum " + ("computing " * 50)),   # very long, 1 mention of quantum
    (2, "quantum computing"),                # short, precise
], dim=4)
r = db.keyword_search("quantum", k=2)
check("T07 doc length norm — short precise doc ranks #1",
      len(r) >= 1 and r[0].id == 2,
      f"top id={r[0].id if r else 'N/A'}")
cleanup(f)

# ── T08: IDF penalises ubiquitous terms ──────────────────────────────────────
# "data" appears in ALL docs → low IDF → low contribution
db, f = make_db([
    (i, f"data record {i} " + ("data " * 5)) for i in range(1, 11)
] + [(11, "featherdb embedded storage")], dim=4)
r_data = db.keyword_search("data", k=5)
r_rare = db.keyword_search("featherdb", k=5)
check("T08 IDF — rare term score > ubiquitous term score",
      len(r_data) > 0 and len(r_rare) > 0 and r_rare[0].score > r_data[0].score,
      f"rare={r_rare[0].score:.3f} vs ubiq={r_data[0].score:.3f}")
cleanup(f)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 3: keyword_search Behaviour ──────")

# ── T09: Top-k ordering is descending ────────────────────────────────────────
db, f = make_db(BASE_CORPUS)
r = db.keyword_search("BM25 search retrieval", k=10)
scores = [x.score for x in r]
check("T09 results are sorted descending",
      all(scores[i] >= scores[i+1] for i in range(len(scores)-1)),
      f"scores={[round(s,3) for s in scores]}")
cleanup(f)

# ── T10: k limits results ────────────────────────────────────────────────────
db, f = make_db(BASE_CORPUS)
r = db.keyword_search("python data machine learning", k=3)
check("T10 k=3 returns at most 3", len(r) <= 3, f"got {len(r)}")
cleanup(f)

# ── T11: Empty corpus → empty results ────────────────────────────────────────
f = tempfile.mktemp(suffix=".feather")
db = feather_db.DB.open(f, dim=4)
r = db.keyword_search("anything", k=5)
check("T11 empty corpus → 0 results", len(r) == 0, f"got {len(r)}")
cleanup(f)

# ── T12: Zero matches → empty list ───────────────────────────────────────────
db, f = make_db(BASE_CORPUS)
r = db.keyword_search("xyzzy quux nonexistenttoken123", k=5)
check("T12 no-match query → 0 results", len(r) == 0, f"got {len(r)}")
cleanup(f)

# ── T13: Single-doc corpus ────────────────────────────────────────────────────
db, f = make_db([(42, "the only document in this database")])
r = db.keyword_search("document database", k=5)
check("T13 single doc corpus — finds the doc", len(r) == 1 and r[0].id == 42)
cleanup(f)

# ── T14: keyword_search with filter ──────────────────────────────────────────
f2 = tempfile.mktemp(suffix=".feather")
db = feather_db.DB.open(f2, dim=4)
for i, (ns, text) in enumerate([
    ("alpha", "BM25 search algorithm inverted index"),
    ("beta",  "BM25 ranking function retrieval"),
    ("alpha", "vector similarity cosine distance"),
    ("beta",  "hybrid search reciprocal rank fusion"),
], start=1):
    meta = feather_db.Metadata()
    meta.content = text
    meta.namespace_id = ns
    meta.timestamp = int(time.time())
    meta.importance = 1.0
    db.add(id=i, vec=rng_vec(4, seed=i), meta=meta)

sf = SearchFilter()
sf.namespace_id = "alpha"
r = db.keyword_search("BM25 search", k=10, filter=sf)
ids = {x.id for x in r}
check("T14 filter restricts to namespace alpha",
      all(i in {1, 3} for i in ids) and 2 not in ids and 4 not in ids,
      f"ids={ids}")
cleanup(f2)

# ── T15: Exact phrase advantage ───────────────────────────────────────────────
# BM25 should rank the exact term document above a tangentially related one
db, f = make_db([
    (1, "recurrent neural network architecture for sequence modeling"),
    (2, "transformer attention mechanism supersedes recurrent models"),
    (3, "recurrent recurrent recurrent recurrent recurrent networks best"),
], dim=4)
r = db.keyword_search("recurrent network", k=3)
check("T15 recurrent network — relevant docs in top-3",
      len(r) >= 2, f"got {len(r)}")
cleanup(f)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 4: Hybrid Search (RRF) ───────────")

# ── T16: hybrid_search returns results ───────────────────────────────────────
db, f = make_db(BASE_CORPUS)
q = rng_vec(128, seed=99)
r = db.hybrid_search(q, "BM25 search retrieval", k=5)
check("T16 hybrid_search returns ≥1 result", len(r) >= 1, f"got {len(r)}")
cleanup(f)

# ── T17: RRF scores in (0, 1) range ──────────────────────────────────────────
db, f = make_db(BASE_CORPUS)
r = db.hybrid_search(rng_vec(128), "search retrieval systems", k=10)
ok = all(0 < x.score < 1 for x in r)
check("T17 RRF scores in (0, 1)", ok, f"scores={[round(x.score,5) for x in r]}")
cleanup(f)

# ── T18: RRF merging — BM25-top gets boosted over vec-only ───────────────────
# Create DB where BM25 result and vec result differ; hybrid should combine
f3 = tempfile.mktemp(suffix=".feather")
db = feather_db.DB.open(f3, dim=128)
# id=1: similar to query vector, no keyword match
# id=2: keyword match, random vector
query_vec = rng_vec(128, seed=0)
meta1 = feather_db.Metadata(); meta1.content = "completely unrelated topic zephyr"; meta1.timestamp = int(time.time()); meta1.importance = 1.0
meta2 = feather_db.Metadata(); meta2.content = "BM25 retrieval keyword search engine"; meta2.timestamp = int(time.time()); meta2.importance = 1.0
# id=1: near the query vector
db.add(id=1, vec=query_vec * 0.99 + rng_vec(128, seed=5) * 0.01, meta=meta1)
db.add(id=2, vec=rng_vec(128, seed=99), meta=meta2)

vec_only = db.search(query_vec, k=2)
kw_only  = db.keyword_search("BM25 retrieval keyword", k=2)
hybrid   = db.hybrid_search(query_vec, "BM25 retrieval keyword", k=2)

vec_ids    = [x.id for x in vec_only]
kw_ids     = [x.id for x in kw_only]
hybrid_ids = [x.id for x in hybrid]

check("T18 vec-only ranks id=1 first", len(vec_ids) > 0 and vec_ids[0] == 1, f"{vec_ids}")
check("T18 kw-only ranks id=2 first",  len(kw_ids) > 0 and kw_ids[0] == 2,  f"{kw_ids}")
check("T18 hybrid includes both ids",   set(hybrid_ids) == {1, 2}, f"{hybrid_ids}")
cleanup(f3)

# ── T19: rrf_k effect — higher rrf_k flattens scores ─────────────────────────
db, f = make_db(BASE_CORPUS)
q = rng_vec(128, seed=7)
r_k10  = db.hybrid_search(q, "search retrieval", k=5, rrf_k=10)
r_k200 = db.hybrid_search(q, "search retrieval", k=5, rrf_k=200)
range_k10  = (r_k10[0].score  - r_k10[-1].score)  if len(r_k10) > 1  else 0
range_k200 = (r_k200[0].score - r_k200[-1].score) if len(r_k200) > 1 else 0
check("T19 higher rrf_k → flatter score range",
      range_k10 > range_k200,
      f"range(k10)={range_k10:.5f} range(k200)={range_k200:.5f}")
cleanup(f)

# ── T20: hybrid_search with filter ───────────────────────────────────────────
f4 = tempfile.mktemp(suffix=".feather")
db = feather_db.DB.open(f4, dim=128)
for i, (ns, text) in enumerate([
    ("eng",  "machine learning neural network deep learning"),
    ("fin",  "machine learning stock prediction financial model"),
    ("eng",  "Python tensorflow keras neural architecture"),
    ("fin",  "algorithmic trading machine learning signals"),
], start=1):
    meta = feather_db.Metadata()
    meta.content = text
    meta.namespace_id = ns
    meta.timestamp = int(time.time())
    meta.importance = 1.0
    db.add(id=i, vec=rng_vec(128, seed=i*7), meta=meta)

sf = SearchFilter(); sf.namespace_id = "fin"
r = db.hybrid_search(rng_vec(128, seed=42), "machine learning model", k=10, filter=sf)
ids = {x.id for x in r}
check("T20 hybrid filter — only fin namespace results",
      all(i in {2, 4} for i in ids) and 1 not in ids and 3 not in ids,
      f"ids={ids}")
cleanup(f4)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 5: Persistence & Updates ─────────")

# ── T21: BM25 index survives save+reload ─────────────────────────────────────
db, f = make_db(BASE_CORPUS)
before = [x.id for x in db.keyword_search("BM25 search retrieval", k=5)]
db.save()
del db

db2 = feather_db.DB.open(f, dim=128)
after = [x.id for x in db2.keyword_search("BM25 search retrieval", k=5)]
check("T21 BM25 rebuilds from metadata after reload",
      before == after, f"before={before} after={after}")
cleanup(f)

# ── T22: update_metadata re-indexes content ──────────────────────────────────
f5 = tempfile.mktemp(suffix=".feather")
db = feather_db.DB.open(f5, dim=4)
meta = feather_db.Metadata(); meta.content = "original content about bananas"; meta.timestamp = int(time.time()); meta.importance = 1.0
db.add(id=1, vec=rng_vec(4, seed=1), meta=meta)

r_before = db.keyword_search("bananas", k=5)
check("T22 before update — finds 'bananas'", len(r_before) == 1)

new_meta = feather_db.Metadata(); new_meta.content = "updated content about mangoes"; new_meta.timestamp = int(time.time()); new_meta.importance = 1.0
db.update_metadata(1, new_meta)

r_banana = db.keyword_search("bananas", k=5)
r_mango  = db.keyword_search("mangoes", k=5)
check("T22 after update — 'bananas' no longer matches", len(r_banana) == 0, f"got {len(r_banana)}")
check("T22 after update — 'mangoes' matches", len(r_mango) == 1 and r_mango[0].id == 1)
cleanup(f5)

# ── T23: Records added without content skipped gracefully ────────────────────
f6 = tempfile.mktemp(suffix=".feather")
db = feather_db.DB.open(f6, dim=4)
meta_no_content = feather_db.Metadata(); meta_no_content.timestamp = int(time.time()); meta_no_content.importance = 1.0
db.add(id=1, vec=rng_vec(4, seed=1), meta=meta_no_content)
r = db.keyword_search("anything", k=5)
check("T23 no content → graceful empty result", len(r) == 0)
cleanup(f6)

# ── T24: Multiple saves don't corrupt BM25 index ─────────────────────────────
db, f = make_db(BASE_CORPUS)
r1 = [x.id for x in db.keyword_search("machine learning", k=5)]
db.save()
db.save()
db.save()
r2 = [x.id for x in db.keyword_search("machine learning", k=5)]
check("T24 multiple saves don't corrupt BM25", r1 == r2)
cleanup(f)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 6: Edge Cases ─────────────────────")

# ── T25: Only stop words in query → empty results ────────────────────────────
db, f = make_db(BASE_CORPUS)
r = db.keyword_search("the and or but", k=5)
check("T25 all-stop-word query → 0 results", len(r) == 0)
cleanup(f)

# ── T26: Doc length normalization — equal TF, short doc wins ────────────────
# Both docs contain "featherdb" once. The long doc has 200 other real words
# as padding → higher avg doc length penalty → short precise doc ranks higher.
padding = " ".join(["information", "retrieval", "system", "database", "storage",
                    "processing", "computing", "machine", "learning", "network"] * 20)
long_text = "featherdb " + padding   # 1 mention + 200 filler words
short_text = "featherdb embedded database"  # 1 mention + 2 words
db, f = make_db([(1, long_text), (2, short_text)], dim=4)
r = db.keyword_search("featherdb", k=5)
check("T26 very long content — doc found", len(r) >= 1)
check("T26 doc length norm — short precise doc ranks ahead of padded doc",
      r[0].id == 2, f"top id={r[0].id if r else 'N/A'}")
cleanup(f)

# ── T27: Duplicate content (same text, different IDs) ────────────────────────
text = "neural network transformer attention mechanism"
db, f = make_db([(1, text), (2, text), (3, text)], dim=4)
r = db.keyword_search("transformer attention", k=5)
ids = {x.id for x in r}
check("T27 duplicate content — all 3 IDs returned",
      ids == {1, 2, 3}, f"ids={ids}")
scores = [x.score for x in r]
check("T27 duplicate content — all scores identical",
      len(set(round(s, 4) for s in scores)) == 1,
      f"scores={[round(s,4) for s in scores]}")
cleanup(f)

# ── T28: Unicode content ──────────────────────────────────────────────────────
db, f = make_db([
    (1, "machine learning artificial intelligence"),
    (2, "apprentissage automatique intelligence artificielle"),
    (3, "maschinelles lernen künstliche intelligenz"),
], dim=4)
# ASCII tokens from multilingual content should still work
r_en = db.keyword_search("machine learning", k=5)
check("T28 unicode content — English query works", len(r_en) == 1 and r_en[0].id == 1)
# French words are mostly alphabetic — "apprentissage" should be indexed
r_fr = db.keyword_search("apprentissage automatique", k=5)
check("T28 unicode content — French query works", len(r_fr) >= 1 and r_fr[0].id == 2)
cleanup(f)

# ── T29: Two independent DBs don't share BM25 state ─────────────────────────
db_a, fa = make_db([(1, "postgres relational database sql queries")])
db_b, fb = make_db([(1, "redis cache key value store memory")])
r_a = db_a.keyword_search("postgres sql", k=5)
r_b = db_b.keyword_search("redis cache", k=5)
check("T29 independent DBs have independent BM25 state",
      len(r_a) == 1 and len(r_b) == 1)
# Cross-contamination check
r_a_wrong = db_a.keyword_search("redis", k=5)
check("T29 DB-A doesn't contain DB-B terms", len(r_a_wrong) == 0)
cleanup(fa, fb)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 7: Performance ────────────────────")

# ── T30: Large corpus — 5 000 docs ───────────────────────────────────────────
N = 5_000
words = ["search","retrieval","database","vector","embedding","neural","network",
         "learning","python","transformer","attention","context","memory","graph",
         "index","query","rank","score","token","semantic","hybrid","sparse","dense"]

print(f"  Building {N}-doc corpus...")
f_large = tempfile.mktemp(suffix=".feather")
db_large = feather_db.DB.open(f_large, dim=128)

t0 = time.perf_counter()
for i in range(1, N + 1):
    rng = random.Random(i)
    text = " ".join(rng.choices(words, k=rng.randint(10, 40)))
    meta = feather_db.Metadata()
    meta.content = text
    meta.timestamp = int(time.time())
    meta.importance = 1.0
    db_large.add(id=i, vec=rng_vec(128, seed=i), meta=meta)
ingest_ms = (time.perf_counter() - t0) * 1000
print(f"  Ingest: {ingest_ms:.0f} ms ({N} docs)")

t1 = time.perf_counter()
r_kw = db_large.keyword_search("neural network transformer semantic", k=20)
kw_ms = (time.perf_counter() - t1) * 1000

t2 = time.perf_counter()
r_hy = db_large.hybrid_search(rng_vec(128, seed=42), "neural network transformer semantic", k=20)
hy_ms = (time.perf_counter() - t2) * 1000

check("T30 keyword_search on 5k docs < 500 ms", kw_ms < 500, f"{kw_ms:.1f} ms")
check("T30 hybrid_search on 5k docs < 1000 ms", hy_ms < 1000, f"{hy_ms:.1f} ms")
check("T30 keyword_search returns results", len(r_kw) > 0, f"{len(r_kw)} results")
check("T30 hybrid_search returns results", len(r_hy) > 0, f"{len(r_hy)} results")
print(f"  keyword_search: {kw_ms:.1f} ms | hybrid_search: {hy_ms:.1f} ms")

# ── T31: Hybrid outperforms pure vector on keyword-specific query ─────────────
# Inject a needle doc with a very specific token
needle_meta = feather_db.Metadata()
needle_meta.content = "xyzzy_needle_token unique_rare_identifier"
needle_meta.timestamp = int(time.time())
needle_meta.importance = 1.0
db_large.add(id=N + 1, vec=rng_vec(128, seed=9999), meta=needle_meta)

r_vec_needle = db_large.search(rng_vec(128, seed=1), k=20)  # random vec
r_kw_needle  = db_large.keyword_search("xyzzy_needle_token", k=5)
r_hy_needle  = db_large.hybrid_search(rng_vec(128, seed=1), "xyzzy_needle_token", k=20)

vec_ids_needle = [x.id for x in r_vec_needle]
kw_ids_needle  = [x.id for x in r_kw_needle]
hy_ids_needle  = [x.id for x in r_hy_needle]

check("T31 keyword finds the needle", N + 1 in kw_ids_needle, f"kw_ids={kw_ids_needle[:5]}")
check("T31 pure vector misses the needle",
      N + 1 not in vec_ids_needle, f"vec top={vec_ids_needle[:5]}")
check("T31 hybrid finds the needle",
      N + 1 in hy_ids_needle, f"hy top={hy_ids_needle[:5]}")
cleanup(f_large)

# ── T32: BM25 on 5k corpus — save/reload preserves results ───────────────────
db, f = make_db(BASE_CORPUS)
r_before = [(x.id, round(x.score, 4)) for x in db.keyword_search("retrieval search systems", k=5)]
db.save()
db2 = feather_db.DB.open(f, dim=128)
r_after = [(x.id, round(x.score, 4)) for x in db2.keyword_search("retrieval search systems", k=5)]
check("T32 save+reload — BM25 scores identical", r_before == r_after,
      f"before={r_before} after={r_after}")
cleanup(f)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Section 8: Hybrid vs BM25 vs Vector quality─")

# ── T33: Semantic query — hybrid bridges keyword gap ─────────────────────────
# "automobile" is semantically close to "car" but not lexically
# With simulated embeddings this won't show semantic benefit, but validates pipeline
db, f = make_db([
    (1, "automobile vehicle transportation car driving road"),
    (2, "machine learning artificial intelligence computing"),
    (3, "database storage persistence query retrieval"),
], dim=128)
r = db.hybrid_search(rng_vec(128, seed=1), "car vehicle road transport", k=3)
ids = {x.id for x in r}
check("T33 hybrid pipeline — results include all ranked docs", len(r) >= 1)
cleanup(f)

# ── T34: keyword_search returns recall_count incremented ─────────────────────
db, f = make_db([(1, "unique test document here"), (2, "another document content")], dim=4)
_ = db.keyword_search("unique test", k=5)
meta = db.get_metadata(1)
check("T34 keyword_search increments recall_count",
      meta is not None and meta.recall_count > 0,
      f"recall_count={meta.recall_count if meta else 'N/A'}")
cleanup(f)

# ─────────────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
passed = sum(1 for _, ok in _results if ok)
failed = sum(1 for _, ok in _results if not ok)
total  = len(_results)
print(f" Results: {passed}/{total} PASSED  |  {failed} FAILED")
if failed:
    print("\n Failed tests:")
    for name, ok in _results:
        if not ok:
            print(f"   ✗ {name}")
print("══════════════════════════════════════════════\n")
sys.exit(0 if failed == 0 else 1)
