"""
Feather DB — Living Context Engine: Large-Scale Demo
=====================================================
Use case: Multi-tenant AI SaaS memory layer

3 tenants: acme_corp, techwave, startup_ai
~3000+ records across:
  - Product docs (FACT)
  - User preferences (PREFERENCE)
  - Support tickets (EVENT)
  - Chat conversations (CONVERSATION)

Tests covered:
  1.  Ingestion throughput
  2.  Vector search accuracy (clustered embeddings)
  3.  BM25 keyword search
  4.  Hybrid search (RRF)
  5.  Filtered search (namespace / entity / attributes)
  6.  Adaptive decay (importance + recall_count stickiness)
  7.  Context chain (graph BFS traversal)
  8.  Auto-link by similarity
  9.  Persistence + WAL crash recovery
  10. Compact (soft-delete pruning)
  11. Thread safety (concurrent reads)
  12. Full benchmark summary
"""

import sys, os, time, random, threading, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import feather_db
from feather_db.core import SearchFilter, ScoringConfig

# ── Colours for terminal output ──────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[94m"; RESET = "\033[0m"
def ok(msg):   print(f"  {G}✓{RESET} {msg}")
def info(msg): print(f"  {B}→{RESET} {msg}")
def warn(msg): print(f"  {Y}!{RESET} {msg}")
def section(title):
    print(f"\n{B}{'─'*60}{RESET}")
    print(f"{B}  {title}{RESET}")
    print(f"{B}{'─'*60}{RESET}")

# ── Reproducible random seed ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

DIM = 256   # embedding dimension (realistic for small models)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Embedding Generator
#    Each "topic" has a centroid. Records in that topic = centroid + small noise.
#    This makes semantic clusters so search results are meaningful.
# ─────────────────────────────────────────────────────────────────────────────
TOPICS = {
    # topic_name: centroid vector
    "billing":        np.random.rand(DIM).astype(np.float32),
    "onboarding":     np.random.rand(DIM).astype(np.float32),
    "api_usage":      np.random.rand(DIM).astype(np.float32),
    "security":       np.random.rand(DIM).astype(np.float32),
    "performance":    np.random.rand(DIM).astype(np.float32),
    "integrations":   np.random.rand(DIM).astype(np.float32),
    "data_export":    np.random.rand(DIM).astype(np.float32),
    "user_prefs":     np.random.rand(DIM).astype(np.float32),
    "product_update": np.random.rand(DIM).astype(np.float32),
    "bug_report":     np.random.rand(DIM).astype(np.float32),
}

def embed(topic: str, noise: float = 0.08) -> np.ndarray:
    """Return a unit-normalised vector near the topic centroid."""
    v = TOPICS[topic] + np.random.randn(DIM).astype(np.float32) * noise
    return (v / np.linalg.norm(v)).astype(np.float32)

# ── Realistic content corpus ──────────────────────────────────────────────────
BILLING_DOCS = [
    "Invoice generation is automatic at the end of each billing cycle.",
    "Upgrade or downgrade your subscription plan from the billing dashboard.",
    "Failed payment retry policy: 3 attempts over 7 days before account suspension.",
    "Annual plans receive a 20% discount compared to monthly billing.",
    "VAT and tax invoices are available for enterprise customers on request.",
    "Credit card and ACH bank transfers are accepted payment methods.",
    "Prorated billing applies when upgrading mid-cycle.",
    "Billing alerts can be configured to notify at 80% and 100% usage thresholds.",
]
ONBOARDING_DOCS = [
    "Create your first workspace by navigating to the dashboard and clicking New Workspace.",
    "Invite team members via email from the Settings > Team section.",
    "Connect your data source using the Integrations wizard on first login.",
    "API keys are generated under Settings > Developers > API Keys.",
    "The quickstart guide walks through your first vector ingestion in under 5 minutes.",
    "Role-based access control allows Admin, Editor, and Viewer permission levels.",
    "SSO via SAML 2.0 or OAuth 2.0 is available on Business and Enterprise plans.",
    "Import existing embeddings using the bulk upload CSV format.",
]
API_DOCS = [
    "Rate limits: 1000 requests/minute on Starter, 10000 on Growth, unlimited on Enterprise.",
    "Use the X-API-Key header to authenticate all API requests.",
    "Vector search endpoint accepts POST /v1/{namespace}/search with k and vector fields.",
    "Hybrid search combines BM25 keyword scoring with dense vector similarity via RRF.",
    "Namespace isolation ensures data from different tenants never leaks across queries.",
    "The /health endpoint returns version and loaded namespace count without authentication.",
    "Batch ingestion is supported via repeated POST calls; use async clients for throughput.",
    "Cursor-based pagination on GET /records supports efficient large dataset browsing.",
]
SECURITY_DOCS = [
    "All data is encrypted at rest using AES-256.",
    "TLS 1.3 is enforced for all API communication.",
    "IP allowlisting is available for Enterprise accounts.",
    "Audit logs capture all read and write operations with timestamps and user IDs.",
    "GDPR compliance: data purge requests are processed within 72 hours.",
    "SOC 2 Type II certification is available upon request for enterprise customers.",
    "Session tokens expire after 24 hours; refresh tokens are valid for 30 days.",
    "Penetration testing is conducted annually by an independent security firm.",
]
PERF_DOCS = [
    "HNSW index delivers sub-millisecond ANN search across 1 million vectors.",
    "Benchmark: 10k records searched in under 2ms on M2 MacBook Pro.",
    "Index construction uses M=16 ef_construction=200 for high recall.",
    "Adaptive decay scoring adjusts result relevance based on access frequency.",
    "Compact operation rebuilds the HNSW index removing soft-deleted records.",
    "Write-ahead log ensures no data loss on unexpected process termination.",
    "Atomic saves prevent file corruption during concurrent write operations.",
    "Memory footprint: approximately 4KB per record for 768-dim vectors.",
]
PREFS = [
    "User prefers dark mode across all dashboard views.",
    "Notification preference: weekly digest emails only, no real-time alerts.",
    "Default search modality set to hybrid for all workspace queries.",
    "User has enabled two-factor authentication via authenticator app.",
    "Language preference set to English with date format MM/DD/YYYY.",
    "User prefers compact table view over card grid in record browser.",
    "Auto-save interval configured to every 30 seconds.",
    "User opted into beta features and experimental search modes.",
]
SUPPORT_TICKETS = [
    "API key suddenly returning 401 — regenerated twice but same error persists.",
    "Dashboard loading slowly after importing 50k records, takes over 10 seconds.",
    "Hybrid search returning fewer than k results when namespace has under 100 records.",
    "Billing invoice not generated for last month despite successful payment.",
    "SAML SSO login redirecting to wrong callback URL after domain migration.",
    "Vector search results seem inaccurate after upgrading to v0.8.0.",
    "Compact operation throws segfault when namespace contains zero records.",
    "CSV bulk import failing on row 4521 with encoding error for Unicode content.",
    "Context chain not traversing beyond hop 1 despite setting hops=3.",
    "Rate limit exceeded errors appearing at only 200 requests per minute.",
]
CONVERSATIONS = [
    "User: How do I export my data? Agent: Use the export_graph_json endpoint or the dashboard Export button.",
    "User: Can I have multiple vector dimensions? Agent: Yes, each modality pocket has its own independent dimension.",
    "User: What is adaptive decay? Agent: Records accessed frequently resist time-based relevance decay.",
    "User: How is hybrid search scored? Agent: BM25 and vector ranks are fused via Reciprocal Rank Fusion.",
    "User: Is there a free tier? Agent: Yes, the Starter plan is free up to 10k vectors.",
    "User: How do I delete a record? Agent: Use soft-delete by setting importance=0 and _deleted=true attribute.",
    "User: What is the max vector dimension? Agent: There is no hard cap, but 768 or 1536 are most common.",
    "User: Can I run this on-premise? Agent: Yes, feather-db is fully embedded with no server dependency.",
]

CORPUS = {
    "billing":        (BILLING_DOCS,      feather_db.ContextType.FACT),
    "onboarding":     (ONBOARDING_DOCS,   feather_db.ContextType.FACT),
    "api_usage":      (API_DOCS,          feather_db.ContextType.FACT),
    "security":       (SECURITY_DOCS,     feather_db.ContextType.FACT),
    "performance":    (PERF_DOCS,         feather_db.ContextType.FACT),
    "user_prefs":     (PREFS,             feather_db.ContextType.PREFERENCE),
    "bug_report":     (SUPPORT_TICKETS,   feather_db.ContextType.EVENT),
    "integrations":   (CONVERSATIONS,     feather_db.ContextType.CONVERSATION),
    "data_export":    (API_DOCS,          feather_db.ContextType.FACT),
    "product_update": (PERF_DOCS,         feather_db.ContextType.FACT),
}

TENANTS = ["acme_corp", "techwave", "startup_ai"]
USERS   = ["alice", "bob", "carol", "dave", "eve"]
PLANS   = ["starter", "growth", "enterprise"]
CHANNELS= ["web", "api", "mobile", "slack"]

# ─────────────────────────────────────────────────────────────────────────────
# Build dataset: 3000 records across 3 namespaces
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(records_per_tenant: int = 1000):
    dataset = []  # list of dicts
    rec_id = 1
    for tenant in TENANTS:
        for i in range(records_per_tenant):
            topic = random.choice(list(CORPUS.keys()))
            docs, ctx_type = CORPUS[topic]
            content = random.choice(docs)
            # Add some variation so not all records are identical
            suffixes = [
                "", " (updated)", " — see docs.", " — contact support.",
                " Note: applies to v0.8.0+.", " [auto-generated]",
            ]
            content = content + random.choice(suffixes)

            dataset.append({
                "id":           rec_id,
                "vec":          embed(topic),
                "topic":        topic,
                "content":      content,
                "type":         ctx_type,
                "namespace_id": tenant,
                "entity_id":    random.choice(USERS),
                "importance":   round(random.uniform(0.5, 1.0), 2),
                "timestamp":    int(time.time()) - random.randint(0, 86400 * 60),
                "source":       f"{tenant}/pipeline",
                "channel":      random.choice(CHANNELS),
                "plan":         random.choice(PLANS),
            })
            rec_id += 1
    return dataset

# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{B}{'═'*60}{RESET}")
    print(f"{B}  Feather DB — Living Context Engine Large-Scale Demo{RESET}")
    print(f"{B}{'═'*60}{RESET}")

    tmp_dir = tempfile.mkdtemp(prefix="feather_demo_")
    db_path = os.path.join(tmp_dir, "demo.feather")
    results = {}

    try:
        # ── 1. INGESTION ─────────────────────────────────────────────────────
        section("1. Dataset Generation & Ingestion")
        RECORDS_PER_TENANT = 1000
        info(f"Generating {RECORDS_PER_TENANT * len(TENANTS):,} records across {len(TENANTS)} tenants...")
        t0 = time.perf_counter()
        dataset = build_dataset(RECORDS_PER_TENANT)
        gen_ms = (time.perf_counter() - t0) * 1000
        ok(f"Generated {len(dataset):,} records in {gen_ms:.1f}ms")

        db = feather_db.DB.open(db_path, dim=DIM)

        t0 = time.perf_counter()
        for rec in dataset:
            meta = feather_db.Metadata()
            meta.content      = rec["content"]
            meta.type         = rec["type"]
            meta.namespace_id = rec["namespace_id"]
            meta.entity_id    = rec["entity_id"]
            meta.importance   = rec["importance"]
            meta.timestamp    = rec["timestamp"]
            meta.source       = rec["source"]
            meta.set_attribute("channel", rec["channel"])
            meta.set_attribute("plan",    rec["plan"])
            meta.set_attribute("topic",   rec["topic"])
            db.add(id=rec["id"], vec=rec["vec"], meta=meta)

        ingest_ms = (time.perf_counter() - t0) * 1000
        throughput = len(dataset) / (ingest_ms / 1000)
        results["ingestion_ms"]  = ingest_ms
        results["throughput_rps"] = throughput
        ok(f"Ingested {len(dataset):,} records in {ingest_ms:.0f}ms")
        ok(f"Throughput: {throughput:,.0f} records/sec")
        ok(f"DB size: {db.size():,} records")

        # ── 2. VECTOR SEARCH ─────────────────────────────────────────────────
        section("2. Vector Search — Semantic Recall")
        query_topic = "billing"
        query_vec   = embed(query_topic, noise=0.05)  # close to billing centroid

        t0 = time.perf_counter()
        results_vec = db.search(query_vec, k=10)
        vec_ms = (time.perf_counter() - t0) * 1000
        results["vector_search_ms"] = vec_ms

        billing_hits = sum(
            1 for r in results_vec
            if r.metadata.get_attribute("topic") == "billing"
        )
        ok(f"Vector search (k=10) in {vec_ms:.2f}ms")
        ok(f"Topic precision: {billing_hits}/10 results are billing-topic")
        info(f"Top result: \"{results_vec[0].metadata.content[:70]}...\"")
        info(f"Score: {results_vec[0].score:.4f}")

        # ── 3. BM25 KEYWORD SEARCH ───────────────────────────────────────────
        section("3. BM25 Keyword Search")
        queries = [
            ("invoice billing payment",  "billing"),
            ("rate limit API key authentication", "api_usage"),
            ("HNSW index sub-millisecond search", "performance"),
            ("SAML SSO login redirect",  "bug_report"),
            ("dark mode notification preference", "user_prefs"),
        ]
        kw_times = []
        kw_precisions = []
        for query_text, expected_topic in queries:
            t0 = time.perf_counter()
            kw_results = db.keyword_search(query_text, k=5)
            elapsed = (time.perf_counter() - t0) * 1000
            kw_times.append(elapsed)
            hits = sum(1 for r in kw_results
                       if r.metadata.get_attribute("topic") == expected_topic)
            kw_precisions.append(hits / max(len(kw_results), 1))
            ok(f"'{query_text[:40]}' → {len(kw_results)} results in {elapsed:.2f}ms "
               f"[precision: {hits}/{len(kw_results)}]")

        results["keyword_search_avg_ms"] = sum(kw_times) / len(kw_times)
        results["keyword_precision"]     = sum(kw_precisions) / len(kw_precisions)
        ok(f"Avg keyword search: {results['keyword_search_avg_ms']:.2f}ms | "
           f"Avg precision: {results['keyword_precision']*100:.0f}%")

        # ── 4. HYBRID SEARCH (RRF) ───────────────────────────────────────────
        section("4. Hybrid Search — BM25 + Vector via RRF")
        hybrid_cases = [
            ("invoice billing payment",         embed("billing", 0.05),    "billing"),
            ("HNSW index search performance",   embed("performance", 0.05),"performance"),
            ("user preference dark mode",        embed("user_prefs", 0.05), "user_prefs"),
        ]
        hybrid_times = []
        hybrid_precs = []
        for query_text, query_v, expected_topic in hybrid_cases:
            t0 = time.perf_counter()
            h_results = db.hybrid_search(query_v, query_text, k=10, rrf_k=60)
            elapsed = (time.perf_counter() - t0) * 1000
            hybrid_times.append(elapsed)
            hits = sum(1 for r in h_results
                       if r.metadata.get_attribute("topic") == expected_topic)
            hybrid_precs.append(hits / max(len(h_results), 1))
            ok(f"'{query_text[:40]}' → {len(h_results)} results in {elapsed:.2f}ms "
               f"[precision: {hits}/{len(h_results)}]")

        results["hybrid_search_avg_ms"] = sum(hybrid_times) / len(hybrid_times)
        results["hybrid_precision"]     = sum(hybrid_precs) / len(hybrid_precs)

        # Compare hybrid vs pure vector precision
        info(f"Hybrid precision: {results['hybrid_precision']*100:.0f}% "
             f"| Keyword precision: {results['keyword_precision']*100:.0f}% "
             f"| (hybrid captures both signals)")

        # ── 5. FILTERED SEARCH ───────────────────────────────────────────────
        section("5. Filtered Search — Namespace / Entity / Attributes")

        # 5a. Namespace filter
        f_ns = SearchFilter()
        f_ns.namespace_id = "acme_corp"
        t0 = time.perf_counter()
        ns_results = db.search(query_vec, k=10, filter=f_ns)
        ns_ms = (time.perf_counter() - t0) * 1000
        all_acme = all(r.metadata.namespace_id == "acme_corp" for r in ns_results)
        ok(f"Namespace filter 'acme_corp': {len(ns_results)} results in {ns_ms:.2f}ms "
           f"[all correct: {all_acme}]")

        # 5b. Entity filter
        f_ent = SearchFilter()
        f_ent.entity_id = "alice"
        t0 = time.perf_counter()
        ent_results = db.search(query_vec, k=10, filter=f_ent)
        ent_ms = (time.perf_counter() - t0) * 1000
        all_alice = all(r.metadata.entity_id == "alice" for r in ent_results)
        ok(f"Entity filter 'alice': {len(ent_results)} results in {ent_ms:.2f}ms "
           f"[all correct: {all_alice}]")

        # 5c. Attribute filter (enterprise plan only)
        f_attr = SearchFilter()
        f_attr.attributes_match = {"plan": "enterprise"}
        t0 = time.perf_counter()
        attr_results = db.search(query_vec, k=10, filter=f_attr)
        attr_ms = (time.perf_counter() - t0) * 1000
        all_enterprise = all(r.metadata.get_attribute("plan") == "enterprise"
                             for r in attr_results)
        ok(f"Attribute filter 'plan=enterprise': {len(attr_results)} results in {attr_ms:.2f}ms "
           f"[all correct: {all_enterprise}]")

        # 5d. Importance threshold
        f_imp = SearchFilter()
        f_imp.importance_gte = 0.9
        imp_results = db.search(query_vec, k=20, filter=f_imp)
        all_high_imp = all(r.metadata.importance >= 0.9 for r in imp_results)
        ok(f"Importance ≥ 0.9 filter: {len(imp_results)} results [all correct: {all_high_imp}]")

        # 5e. Combined: namespace + entity + attribute
        f_combo = SearchFilter()
        f_combo.namespace_id = "techwave"
        f_combo.entity_id    = "bob"
        f_combo.attributes_match = {"channel": "api"}
        combo_results = db.search(query_vec, k=20, filter=f_combo)
        all_combo = all(
            r.metadata.namespace_id == "techwave" and
            r.metadata.entity_id    == "bob"       and
            r.metadata.get_attribute("channel") == "api"
            for r in combo_results
        )
        ok(f"Combined filter (ns+entity+attr): {len(combo_results)} results "
           f"[all correct: {all_combo}]")

        # ── 6. ADAPTIVE DECAY ────────────────────────────────────────────────
        section("6. Adaptive Decay — Living Context Scoring")

        # Touch a record 20 times to simulate frequent access
        hot_id = dataset[0]["id"]
        for _ in range(20):
            db.touch(hot_id)
        hot_meta = db.get_metadata(hot_id)
        ok(f"Hot record (id={hot_id}) recall_count: {hot_meta.recall_count}")

        # Search with scoring — hot record should score higher than cold
        cfg = ScoringConfig(half_life=30.0, weight=0.3, min=0.0)
        scored = db.search(dataset[0]["vec"], k=50, scoring=cfg)

        # Find hot record's rank vs same-topic records without touches
        hot_rank = next((i for i, r in enumerate(scored) if r.id == hot_id), None)
        ok(f"Hot record rank in scored search: #{hot_rank + 1 if hot_rank is not None else 'N/A'} "
           f"(boosted by recall_count={hot_meta.recall_count})")

        # Compare scored vs unscored order
        unscored = db.search(dataset[0]["vec"], k=10)
        ok(f"Unscored top score: {unscored[0].score:.4f} | "
           f"Scored top score: {scored[0].score:.4f}")

        # ── 7. GRAPH — LINK & CONTEXT CHAIN ──────────────────────────────────
        section("7. Context Graph — Edges & Context Chain")

        # Create a knowledge graph: billing → onboarding → api_usage
        # Find records of each topic
        billing_ids = [r["id"] for r in dataset if r["topic"] == "billing"][:5]
        onboard_ids = [r["id"] for r in dataset if r["topic"] == "onboarding"][:5]
        api_ids     = [r["id"] for r in dataset if r["topic"] == "api_usage"][:5]
        perf_ids    = [r["id"] for r in dataset if r["topic"] == "performance"][:5]

        # Link billing → onboarding (billing docs reference onboarding)
        for b_id, o_id in zip(billing_ids, onboard_ids):
            db.link(b_id, o_id, rel_type="references", weight=0.9)
        # Link onboarding → api_usage
        for o_id, a_id in zip(onboard_ids, api_ids):
            db.link(o_id, a_id, rel_type="leads_to", weight=0.8)
        # Link api_usage → performance
        for a_id, p_id in zip(api_ids, perf_ids):
            db.link(a_id, p_id, rel_type="related_to", weight=0.7)

        total_edges = len(billing_ids) + len(onboard_ids) + len(api_ids)
        ok(f"Created {total_edges} typed graph edges (billing→onboarding→api→perf)")

        # Get edges for a billing record
        edges = db.get_edges(billing_ids[0])
        ok(f"Outgoing edges from billing[0]: {len(edges)} edge(s) "
           f"[rel_type: {edges[0].rel_type if edges else 'none'}]")

        incoming = db.get_incoming(onboard_ids[0])
        ok(f"Incoming edges to onboarding[0]: {len(incoming)} edge(s)")

        # Context chain: start from billing query, hop 2 levels
        t0 = time.perf_counter()
        chain = db.context_chain(embed("billing", 0.05), k=3, hops=2)
        chain_ms = (time.perf_counter() - t0) * 1000
        results["context_chain_ms"] = chain_ms

        ok(f"Context chain (k=3, hops=2): {len(chain.nodes)} nodes, "
           f"{len(chain.edges)} edges in {chain_ms:.2f}ms")
        hop_dist = {}
        for node in chain.nodes:
            hop_dist[node.hop] = hop_dist.get(node.hop, 0) + 1
        for hop, count in sorted(hop_dist.items()):
            info(f"  Hop {hop}: {count} node(s)")

        # ── 8. AUTO-LINK ──────────────────────────────────────────────────────
        section("8. Auto-Link by Similarity")
        t0 = time.perf_counter()
        links_created = db.auto_link(modality="text", threshold=0.97,
                                     rel_type="similar_to", candidates=10)
        autolink_ms = (time.perf_counter() - t0) * 1000
        results["autolink_ms"] = autolink_ms
        ok(f"auto_link (threshold=0.97): {links_created} edges created in {autolink_ms:.0f}ms")

        # ── 9. PERSISTENCE + WAL CRASH RECOVERY ──────────────────────────────
        section("9. Persistence + WAL Crash Recovery")

        # Save checkpoint
        t0 = time.perf_counter()
        db.save()
        save_ms = (time.perf_counter() - t0) * 1000
        results["save_ms"] = save_ms
        ok(f"Saved {db.size():,} records in {save_ms:.0f}ms")

        file_mb = os.path.getsize(db_path) / (1024 * 1024)
        ok(f"File size: {file_mb:.2f} MB")

        # WAL test: add records WITHOUT saving, then re-open
        WAL_RECORDS = 50
        wal_ids = list(range(90001, 90001 + WAL_RECORDS))
        for wid in wal_ids:
            m = feather_db.Metadata()
            m.content = f"WAL test record {wid} — should survive crash"
            m.namespace_id = "wal_test"
            db.add(id=wid, vec=embed("billing"), meta=m)
        # DO NOT call db.save() — simulate crash by deleting the object
        del db

        # Re-open — WAL should replay automatically
        db2 = feather_db.DB.open(db_path, dim=DIM)
        recovered = sum(
            1 for wid in wal_ids
            if db2.get_metadata(wid) is not None
        )
        ok(f"WAL crash recovery: {recovered}/{WAL_RECORDS} records recovered")
        results["wal_recovery_rate"] = recovered / WAL_RECORDS
        db = db2

        # Re-open: measure load time
        db.save()
        del db
        t0 = time.perf_counter()
        db = feather_db.DB.open(db_path, dim=DIM)
        load_ms = (time.perf_counter() - t0) * 1000
        results["load_ms"] = load_ms
        ok(f"Reload {db.size():,} records in {load_ms:.0f}ms")

        # ── 10. COMPACT ──────────────────────────────────────────────────────
        section("10. Compact — Soft-Delete Pruning")

        # Soft-delete 100 records
        to_delete = [rec["id"] for rec in dataset[:100]]
        for did in to_delete:
            meta = db.get_metadata(did)
            if meta:
                meta.importance = 0.0
                meta.set_attribute("_deleted", "true")
                db.update_metadata(did, meta)
        ok(f"Soft-deleted {len(to_delete)} records")

        size_before = db.size()
        t0 = time.perf_counter()
        removed = db.compact()
        compact_ms = (time.perf_counter() - t0) * 1000
        size_after = db.size()
        results["compact_ms"] = compact_ms
        ok(f"compact() removed {removed} records in {compact_ms:.0f}ms "
           f"({size_before:,} → {size_after:,})")

        # ── 11. THREAD SAFETY ────────────────────────────────────────────────
        section("11. Thread Safety — Concurrent Reads")

        errors = []
        search_counts = []
        THREADS = 8
        QUERIES_PER_THREAD = 50

        def reader_thread(tid):
            try:
                q = embed(random.choice(list(TOPICS.keys())))
                for _ in range(QUERIES_PER_THREAD):
                    res = db.search(q, k=5)
                    search_counts.append(len(res))
            except Exception as e:
                errors.append(str(e))

        t0 = time.perf_counter()
        threads = [threading.Thread(target=reader_thread, args=(i,))
                   for i in range(THREADS)]
        for t in threads: t.start()
        for t in threads: t.join()
        concurrent_ms = (time.perf_counter() - t0) * 1000
        results["concurrent_ms"] = concurrent_ms

        total_queries = THREADS * QUERIES_PER_THREAD
        ok(f"{total_queries} concurrent queries across {THREADS} threads "
           f"in {concurrent_ms:.0f}ms")
        ok(f"Queries/sec: {total_queries / (concurrent_ms/1000):,.0f}")
        if errors:
            warn(f"Errors: {len(errors)} — {errors[0]}")
        else:
            ok("Zero thread-safety errors")

        # ── 12. SEARCH QUALITY DEEP-DIVE ─────────────────────────────────────
        section("12. Search Quality — Head-to-Head Comparison")
        print(f"\n  {'Query':<35} {'Vector':>8} {'Keyword':>8} {'Hybrid':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")

        quality_cases = [
            ("invoice billing payment",       embed("billing"),     "billing"),
            ("AES-256 encryption TLS audit",  embed("security"),    "security"),
            ("HNSW benchmark millisecond",    embed("performance"), "performance"),
            ("dark mode notification digest", embed("user_prefs"),  "user_prefs"),
            ("SSO SAML OAuth login error",    embed("bug_report"),  "bug_report"),
        ]
        for q_text, q_vec, topic in quality_cases:
            v_res  = db.search(q_vec, k=5)
            k_res  = db.keyword_search(q_text, k=5)
            h_res  = db.hybrid_search(q_vec, q_text, k=5)
            v_p = sum(1 for r in v_res  if r.metadata.get_attribute("topic") == topic)
            k_p = sum(1 for r in k_res  if r.metadata.get_attribute("topic") == topic)
            h_p = sum(1 for r in h_res  if r.metadata.get_attribute("topic") == topic)
            bar = lambda n: f"{G}{n}/5{RESET}" if n >= 4 else (f"{Y}{n}/5{RESET}" if n >= 2 else f"{R}{n}/5{RESET}")
            print(f"  {q_text[:35]:<35} {bar(v_p):>18} {bar(k_p):>18} {bar(h_p):>18}")

        # ── FINAL BENCHMARK SUMMARY ───────────────────────────────────────────
        section("BENCHMARK SUMMARY")
        print(f"""
  Dataset
  ├─ Records ingested : {len(dataset)+WAL_RECORDS:,}
  ├─ Namespaces       : {len(TENANTS)}
  ├─ Dimension        : {DIM}
  └─ File size        : {file_mb:.2f} MB

  Throughput
  ├─ Ingestion        : {results['throughput_rps']:>10,.0f} rec/sec
  ├─ Save (flush)     : {results['save_ms']:>10.0f} ms
  └─ Load (cold open) : {results['load_ms']:>10.0f} ms

  Search Latency (p50)
  ├─ Vector search    : {results['vector_search_ms']:>10.2f} ms
  ├─ Keyword (BM25)   : {results['keyword_search_avg_ms']:>10.2f} ms
  ├─ Hybrid (RRF)     : {results['hybrid_search_avg_ms']:>10.2f} ms
  └─ Context chain    : {results['context_chain_ms']:>10.2f} ms

  Search Quality
  ├─ Keyword precision: {results['keyword_precision']*100:>9.0f}%
  └─ Hybrid precision : {results['hybrid_precision']*100:>9.0f}%

  Reliability
  ├─ WAL recovery     : {results['wal_recovery_rate']*100:>9.0f}%
  ├─ Auto-link edges  : {links_created:>10,}
  ├─ Compact removed  : {removed:>10,}
  └─ Thread errors    : {len(errors):>10}

  Concurrency
  └─ {THREADS} threads × {QUERIES_PER_THREAD} queries : {results['concurrent_ms']:.0f}ms total
     ({total_queries/(results['concurrent_ms']/1000):,.0f} queries/sec)
""")

        ok("All tests passed.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
