"""
marketing_ads_dataset.py — Feather DB v0.5.0
Large synthetic Meta + Google Ads performance dataset

Records (~350 total):
  - 3 brands: Nike, Adidas, Puma
  - 2 platforms: Meta, Google
  - Per brand: 6 campaigns × 3-4 ad sets × 2-3 ads
  - Performance snapshots, insights, strategy notes

Embedding design (128-dim):
  Dims 0-15   → platform encoding (meta vs google)
  Dims 16-31  → objective encoding (awareness/consideration/conversion/retargeting)
  Dims 32-47  → performance tier (low/mid/high ROAS clusters)
  Dims 48-79  → audience type encoding
  Dims 80-127 → content noise + brand signal

Records in the same campaign cluster together.
High-ROAS ads cluster together across brands (performance similarity).
"""

import sys, time, json, os
import numpy as np
sys.path.insert(0, '.')

import feather_db
from feather_db import (DB, Metadata, ContextType, FilterBuilder,
                        MarketingProfile, RelType, visualize, export_graph,
                        ScoringConfig)

print("=" * 68)
print(f"  Feather DB v{feather_db.__version__} — Meta + Google Ads Dataset")
print("=" * 68)

DB_PATH   = "/tmp/ads_performance.feather"
DIM       = 128
rng       = np.random.default_rng(2025)

# ─────────────────────────────────────────────────────────────────
# Embedding helpers — structured so similar records cluster
# ─────────────────────────────────────────────────────────────────
PLATFORM_VECS  = { "meta": np.array([1,0]*8, dtype=np.float32), "google": np.array([0,1]*8, dtype=np.float32) }
OBJECTIVE_VECS = {
    "awareness":      np.array([1,0,0,0]*4, dtype=np.float32),
    "consideration":  np.array([0,1,0,0]*4, dtype=np.float32),
    "conversion":     np.array([0,0,1,0]*4, dtype=np.float32),
    "retargeting":    np.array([0,0,0,1]*4, dtype=np.float32),
    "search":         np.array([1,1,0,0]*4, dtype=np.float32),
    "shopping":       np.array([0,0,1,1]*4, dtype=np.float32),
    "display":        np.array([1,0,1,0]*4, dtype=np.float32),
    "performance_max":np.array([1,1,1,1]*4, dtype=np.float32),
}
AUDIENCE_VECS  = {
    "lookalike":    np.array([1,0,0,0]*8, dtype=np.float32),
    "interest":     np.array([0,1,0,0]*8, dtype=np.float32),
    "retargeting":  np.array([0,0,1,0]*8, dtype=np.float32),
    "broad":        np.array([0,0,0,1]*8, dtype=np.float32),
    "keyword":      np.array([1,1,0,0]*8, dtype=np.float32),
    "custom_intent":np.array([0,1,1,0]*8, dtype=np.float32),
}
BRAND_SEED = { "nike": 1, "adidas": 2, "puma": 3 }

def make_vec(platform, objective, roas_tier, audience, brand, noise=0.25):
    """Build a structured 128-dim embedding. Similar configs → close vectors."""
    pv  = PLATFORM_VECS.get(platform,  np.zeros(16, dtype=np.float32))
    ov  = OBJECTIVE_VECS.get(objective, np.zeros(16, dtype=np.float32))
    # ROAS tier: 0=low, 1=mid, 2=high — clusters by performance level
    rv  = np.array([roas_tier/2.0]*16, dtype=np.float32)
    av  = AUDIENCE_VECS.get(audience, np.zeros(32, dtype=np.float32))
    brand_rng = np.random.default_rng(BRAND_SEED[brand])
    bv  = brand_rng.random(48).astype(np.float32)
    base = np.concatenate([pv, ov, rv, av, bv])
    noise_vec = rng.random(DIM).astype(np.float32) * noise
    v = base + noise_vec
    return (v / np.linalg.norm(v)).astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# Dataset definition
# ─────────────────────────────────────────────────────────────────
BRANDS = ["nike", "adidas", "puma"]

META_CAMPAIGNS = [
    {"name": "Summer Awareness",      "obj": "awareness",     "budget": 25000, "platform": "meta"},
    {"name": "Product Consideration", "obj": "consideration", "budget": 18000, "platform": "meta"},
    {"name": "Conversion Push",       "obj": "conversion",    "budget": 35000, "platform": "meta"},
    {"name": "Retargeting Warmup",    "obj": "retargeting",   "budget": 12000, "platform": "meta"},
]
GOOGLE_CAMPAIGNS = [
    {"name": "Brand Search",          "obj": "search",          "budget": 20000, "platform": "google"},
    {"name": "Shopping Catalog",      "obj": "shopping",        "budget": 30000, "platform": "google"},
    {"name": "Display Reach",         "obj": "display",         "budget": 10000, "platform": "google"},
    {"name": "Performance Max",       "obj": "performance_max", "budget": 40000, "platform": "google"},
]

META_AD_SETS = [
    {"name": "Lookalike 1%",    "audience": "lookalike",    "age": "18-34"},
    {"name": "Interest Sports", "audience": "interest",     "age": "18-44"},
    {"name": "Retargeting 30d", "audience": "retargeting",  "age": "25-54"},
]
GOOGLE_AD_SETS = [
    {"name": "Branded Keywords",  "audience": "keyword",       "match": "exact"},
    {"name": "Competitor Terms",  "audience": "keyword",       "match": "broad"},
    {"name": "Custom Intent",     "audience": "custom_intent", "match": "n/a"},
]

AD_TEMPLATES = {
    "meta": [
        {"format": "reel",         "copy": "Just Move — 15s athlete sprint"},
        {"format": "carousel",     "copy": "5-product showcase — new collection"},
        {"format": "static_image", "copy": "Minimal hero shot — clean CTA"},
    ],
    "google": [
        {"format": "RSA",  "copy": "Responsive search — 10 headlines"},
        {"format": "DSA",  "copy": "Dynamic search — auto from landing page"},
        {"format": "video","copy": "YouTube 6s bumper — brand recall"},
    ],
}

INSIGHT_TEMPLATES = [
    "Short-form video outperforms static by {delta}% on CTR",
    "Lookalike 1pct audience showing {roas}x ROAS - scale budget",
    "Retargeting segment CPL down {pct}% vs last period",
    "Competitor bidding on brand terms - CPC up {delta}%",
    "Performance Max cannibilizing Shopping - overlap detected",
    "Weekend ROAS +{delta}% vs weekday - shift budget allocation",
    "Mobile-first creatives drive {pct}% of conversions",
]

# ─────────────────────────────────────────────────────────────────
# Build the DB
# ─────────────────────────────────────────────────────────────────
db = DB.open(DB_PATH, dim=DIM)
record_id   = 1
id_registry = {}   # label → id (for linking)
all_ids     = {"campaigns": {}, "adsets": {}, "ads": {}, "insights": {}, "perf": {}}

def ts_offset(days_ago=0):
    return int(time.time()) - days_ago * 86400

def add(rid, label, vec, meta):
    db.add(id=rid, vec=vec, meta=meta)
    id_registry[label] = rid

print("\n[1] Generating campaigns, ad sets, ads, performance & insights...")
t0 = time.time()

for brand in BRANDS:
    all_campaigns = META_CAMPAIGNS + GOOGLE_CAMPAIGNS
    for camp in all_campaigns:
        # ── Campaign record ──────────────────────────────────────
        roas_true = round(rng.uniform(1.8, 5.5), 2)
        roas_tier = 0 if roas_true < 2.5 else (1 if roas_true < 4.0 else 2)
        cid = record_id; record_id += 1
        camp_label = f"{brand}__{camp['platform']}__{camp['name'].replace(' ','_')}"
        mp = MarketingProfile().set_brand(brand).set_user("media_team")
        m  = mp.to_metadata()
        m.type = ContextType.EVENT; m.source = camp["platform"]
        m.timestamp = ts_offset(rng.integers(30, 90))
        m.importance = 0.85 + roas_tier * 0.05
        m.content = f"[{brand.upper()}] {camp['platform'].capitalize()} Campaign: {camp['name']}"
        m.set_attribute("record_type",  "campaign")
        m.set_attribute("platform",     camp["platform"])
        m.set_attribute("objective",    camp["obj"])
        m.set_attribute("budget",       str(camp["budget"]))
        m.set_attribute("roas",         str(roas_true))
        m.set_attribute("status",       "active")
        vec = make_vec(camp["platform"], camp["obj"], roas_tier, "broad", brand)
        add(cid, camp_label, vec, m)
        all_ids["campaigns"][camp_label] = cid

        ad_sets_def = META_AD_SETS if camp["platform"] == "meta" else GOOGLE_AD_SETS
        ad_templates = AD_TEMPLATES[camp["platform"]]

        for adset in ad_sets_def:
            # ── Ad Set record ─────────────────────────────────────
            as_roas  = round(roas_true + rng.uniform(-0.5, 0.8), 2)
            as_ctr   = round(rng.uniform(0.015, 0.092), 4)
            as_cpm   = round(rng.uniform(6.5, 28.0), 2)
            as_tier  = 0 if as_roas < 2.5 else (1 if as_roas < 4.0 else 2)
            asid = record_id; record_id += 1
            adset_label = f"{camp_label}__{adset['name'].replace(' ','_')}"
            mp2 = MarketingProfile().set_brand(brand).set_user("media_team")
            m2  = mp2.to_metadata()
            m2.type = ContextType.EVENT; m2.source = camp["platform"]
            m2.timestamp = ts_offset(rng.integers(1, 30))
            m2.importance = 0.7 + as_tier * 0.1
            m2.content = f"[{brand.upper()}] Ad Set: {adset['name']} — {camp['name']}"
            m2.set_attribute("record_type",  "ad_set")
            m2.set_attribute("platform",     camp["platform"])
            m2.set_attribute("objective",    camp["obj"])
            m2.set_attribute("audience",     adset["audience"])
            m2.set_attribute("ctr",          str(as_ctr))
            m2.set_attribute("cpm",          str(as_cpm))
            m2.set_attribute("roas",         str(as_roas))
            vec2 = make_vec(camp["platform"], camp["obj"], as_tier, adset["audience"], brand, noise=0.2)
            add(asid, adset_label, vec2, m2)
            all_ids["adsets"][adset_label] = asid

            # Ad Set is PART_OF Campaign
            db.link(asid, cid, RelType.PART_OF, weight=1.0)

            for ad_tmpl in ad_templates[:2]:   # 2 ads per ad set
                # ── Ad record ────────────────────────────────────
                ad_ctr  = round(as_ctr + rng.uniform(-0.01, 0.02), 4)
                ad_roas = round(as_roas + rng.uniform(-0.4, 0.6), 2)
                ad_cpc  = round(rng.uniform(0.4, 3.5), 2)
                ad_spend= round(rng.uniform(500, 8000), 0)
                ad_conv = int(ad_spend * ad_roas / rng.uniform(25, 120))
                ad_tier = 0 if ad_roas < 2.5 else (1 if ad_roas < 4.0 else 2)
                adid = record_id; record_id += 1
                ad_label = f"{adset_label}__{ad_tmpl['format']}"
                mp3 = MarketingProfile().set_brand(brand).set_user("creative_team")
                m3  = mp3.to_metadata()
                m3.type = ContextType.EVENT; m3.source = camp["platform"]
                m3.timestamp = ts_offset(rng.integers(1, 15))
                m3.importance = 0.6 + ad_tier * 0.15
                m3.content = f"[{brand.upper()}] Ad: {ad_tmpl['copy'][:50]}"
                m3.set_attribute("record_type",  "ad")
                m3.set_attribute("platform",     camp["platform"])
                m3.set_attribute("objective",    camp["obj"])
                m3.set_attribute("format",       ad_tmpl["format"])
                m3.set_attribute("audience",     adset["audience"])
                m3.set_attribute("ctr",          str(max(0.001, ad_ctr)))
                m3.set_attribute("cpc",          str(ad_cpc))
                m3.set_attribute("roas",         str(max(0.1, ad_roas)))
                m3.set_attribute("spend",        str(ad_spend))
                m3.set_attribute("conversions",  str(ad_conv))
                vec3 = make_vec(camp["platform"], camp["obj"], ad_tier, adset["audience"], brand, noise=0.15)
                add(adid, ad_label, vec3, m3)
                all_ids["ads"][ad_label] = adid

                # Ad is PART_OF Ad Set
                db.link(adid, asid, RelType.PART_OF, weight=1.0)

                # High-ROAS ads get a performance insight (CAUSED_BY relationship)
                if ad_roas >= 3.8:
                    iid = record_id; record_id += 1
                    tmpl = rng.choice(INSIGHT_TEMPLATES)
                    insight_text = tmpl.format(
                        roas=round(ad_roas, 1),
                        ctr=round(ad_ctr*100,1),
                        delta=int(rng.uniform(15,55)),
                        pct=int(rng.uniform(8,40))
                    )
                    mp4 = MarketingProfile().set_brand(brand).set_user("analytics_team")
                    m4  = mp4.to_metadata()
                    m4.type = ContextType.FACT; m4.source = "analytics"
                    m4.timestamp = ts_offset(rng.integers(0, 7))
                    m4.importance = 0.9
                    m4.content = f"[{brand.upper()}] Insight: {insight_text}"
                    m4.set_attribute("record_type",  "insight")
                    m4.set_attribute("platform",     camp["platform"])
                    m4.set_attribute("roas_trigger", str(round(ad_roas,2)))
                    vec4 = make_vec(camp["platform"], camp["obj"], 2, adset["audience"], brand, noise=0.3)
                    add(iid, f"insight_{iid}", vec4, m4)
                    all_ids["insights"][f"insight_{iid}"] = iid
                    # Insight DERIVED_FROM the high-ROAS ad
                    db.link(iid, adid, RelType.DERIVED_FROM, weight=0.95)
                    # Ad CAUSED_BY this insight cycle (feedback loop)
                    db.link(adid, iid, RelType.CAUSED_BY, weight=0.7)

# Cross-brand competitive links: Puma contradicts Nike on overlapping audiences
nike_camp_ids  = [v for k,v in all_ids["campaigns"].items() if "nike"  in k and "conversion" in k]
puma_camp_ids  = [v for k,v in all_ids["campaigns"].items() if "puma"  in k and "conversion" in k]
adidas_camp_ids= [v for k,v in all_ids["campaigns"].items() if "adidas" in k and "awareness" in k]
nike_aware_ids = [v for k,v in all_ids["campaigns"].items() if "nike"  in k and "awareness" in k]

for nc, pc in zip(nike_camp_ids[:2], puma_camp_ids[:2]):
    db.link(nc, pc, RelType.CONTRADICTS, weight=0.65)
for nc, ac in zip(nike_aware_ids[:2], adidas_camp_ids[:2]):
    db.link(nc, ac, RelType.REFERENCES, weight=0.5)

t_ingest = time.time() - t0
total_records = db.size()
print(f"   {total_records} records inserted in {t_ingest:.2f}s")
print(f"   Breakdown: {len(all_ids['campaigns'])} campaigns, {len(all_ids['adsets'])} ad sets, "
      f"{len(all_ids['ads'])} ads, {len(all_ids['insights'])} insights")


# ─────────────────────────────────────────────────────────────────
# [2] Auto-link by similarity
# ─────────────────────────────────────────────────────────────────
print("\n[2] Auto-linking by vector similarity (threshold=0.82)...")
t0 = time.time()
n_auto = db.auto_link(modality="text", threshold=0.82, rel_type=RelType.RELATED_TO, candidates=12)
print(f"   {n_auto} similarity edges created in {(time.time()-t0)*1000:.0f}ms")


# ─────────────────────────────────────────────────────────────────
# [3] Queries
# ─────────────────────────────────────────────────────────────────
print("\n[3] Running queries...")

# Q1: All Nike campaigns on Meta
f_nike_meta = FilterBuilder().namespace("nike").attribute("platform","meta").attribute("record_type","campaign").build()
r1 = db.search(make_vec("meta","conversion",2,"lookalike","nike"), k=20, filter=f_nike_meta)
print(f"\n  Q1 — Nike Meta campaigns: {len(r1)} found")
for r in r1[:4]:
    print(f"       id:{r.id:<4} roas={r.metadata.get_attribute('roas','?'):<6} obj={r.metadata.get_attribute('objective','?')}")

# Q2: High-ROAS ads across all brands (ROAS stored as string, filter by importance as proxy)
high_roas_filter = FilterBuilder().attribute("record_type","ad").build()
r2 = db.search(make_vec("meta","conversion",2,"lookalike","nike"), k=50, filter=high_roas_filter)
high_roas = [r for r in r2 if float(r.metadata.get_attribute("roas","0")) >= 4.0]
print(f"\n  Q2 — High-ROAS ads (ROAS ≥ 4.0): {len(high_roas)} found")
for r in sorted(high_roas, key=lambda x: float(x.metadata.get_attribute("roas","0")), reverse=True)[:5]:
    print(f"       id:{r.id:<4} brand={r.metadata.namespace_id:<8} roas={r.metadata.get_attribute('roas','?'):<6} fmt={r.metadata.get_attribute('format','?')}")

# Q3: Google Shopping ads specifically
f_shopping = FilterBuilder().attribute("platform","google").attribute("objective","shopping").attribute("record_type","ad").build()
r3 = db.search(make_vec("google","shopping",1,"keyword","adidas"), k=30, filter=f_shopping)
print(f"\n  Q3 — Google Shopping ads: {len(r3)} found")
for r in r3[:4]:
    print(f"       id:{r.id:<4} brand={r.metadata.namespace_id:<8} ctr={r.metadata.get_attribute('ctr','?'):<8} spend={r.metadata.get_attribute('spend','?')}")

# Q4: Retargeting audience ad sets across platforms
f_retarget = FilterBuilder().attribute("audience","retargeting").attribute("record_type","ad_set").build()
r4 = db.search(make_vec("meta","retargeting",1,"retargeting","nike"), k=20, filter=f_retarget)
print(f"\n  Q4 — Retargeting ad sets: {len(r4)} found")
for r in r4[:4]:
    print(f"       id:{r.id:<4} brand={r.metadata.namespace_id:<8} plat={r.metadata.get_attribute('platform','?'):<8} cpm={r.metadata.get_attribute('cpm','?')}")

# Q5: Insights derived from high performance
f_insight = FilterBuilder().attribute("record_type","insight").build()
r5 = db.search(make_vec("meta","conversion",2,"lookalike","nike"), k=20, filter=f_insight)
print(f"\n  Q5 — Performance insights: {len(r5)} found")
for r in r5[:4]:
    print(f"       id:{r.id:<4} brand={r.metadata.namespace_id:<8} content='{r.metadata.content[12:55]}'")

# Q6: Adidas-only search with scoring (adaptive decay)
cfg = ScoringConfig(half_life=14.0, weight=0.4, min=0.0)
f_adidas = FilterBuilder().namespace("adidas").build()
r6 = db.search(make_vec("meta","consideration",1,"interest","adidas"), k=10, filter=f_adidas, scoring=cfg)
print(f"\n  Q6 — Adidas scored search (adaptive decay): {len(r6)} found")
for r in r6[:4]:
    print(f"       id:{r.id:<4} score={r.score:.4f} type={r.metadata.get_attribute('record_type','?'):<10} obj={r.metadata.get_attribute('objective','?')}")


# ─────────────────────────────────────────────────────────────────
# [4] Context chain — from a high-ROAS Nike reel
# ─────────────────────────────────────────────────────────────────
print("\n[4] Context chain — high-ROAS campaign neighborhood...")
chain_query = make_vec("meta","conversion",2,"lookalike","nike")
chain = db.context_chain(chain_query, k=5, hops=2, modality="text")
print(f"   Subgraph: {len(chain.nodes)} nodes, {len(chain.edges)} edges")
print("   Top 8 nodes by score:")
for n in chain.nodes[:8]:
    hop_tag = "seed" if n.hop == 0 else f"hop {n.hop}"
    rtype = n.metadata.get_attribute("record_type", "?")
    roas  = n.metadata.get_attribute("roas", "?")
    print(f"     [{hop_tag}] id:{n.id:<4} score={n.score:.3f} type={rtype:<10} roas={roas:<6} brand={n.metadata.namespace_id}")

print("   Edge types in chain:")
from collections import Counter
rel_counts = Counter(e.rel_type for e in chain.edges)
for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
    print(f"     {rel:<20} {count:3d} edges")


# ─────────────────────────────────────────────────────────────────
# [5] Edge inspection — campaign hierarchy
# ─────────────────────────────────────────────────────────────────
print("\n[5] Campaign hierarchy walk...")
# Pick first Nike Meta conversion campaign
sample_camp_label = next(k for k in all_ids["campaigns"] if "nike" in k and "meta" in k and "Conversion" in k)
camp_id = all_ids["campaigns"][sample_camp_label]
camp_meta = db.get_metadata(camp_id)
inc = db.get_incoming(camp_id)
print(f"   Campaign: '{camp_meta.content}'")
print(f"   Incoming edges (ad sets reporting into this campaign): {len(inc)}")
for ie in inc[:5]:
    child_meta = db.get_metadata(ie.source_id)
    if child_meta:
        print(f"     ← id:{ie.source_id:<4} [{ie.rel_type}] '{child_meta.content[:55]}'")

# Walk down: pick one ad set, list its ads
if inc:
    adset_id   = inc[0].source_id
    adset_meta = db.get_metadata(adset_id)
    adset_inc  = db.get_incoming(adset_id)
    print(f"\n   Ad Set: '{adset_meta.content[:60]}'")
    print(f"   Ads under this ad set: {len(adset_inc)}")
    for ie in adset_inc[:4]:
        ad_meta = db.get_metadata(ie.source_id)
        if ad_meta:
            roas = ad_meta.get_attribute("roas","?")
            ctr  = ad_meta.get_attribute("ctr","?")
            fmt  = ad_meta.get_attribute("format","?")
            print(f"     ← id:{ie.source_id:<4} [{fmt:<12}] roas={roas:<6} ctr={ctr}")


# ─────────────────────────────────────────────────────────────────
# [6] Cross-brand competitive graph
# ─────────────────────────────────────────────────────────────────
print("\n[6] Cross-brand competitive signal check...")
for cid in nike_camp_ids[:2]:
    out = db.get_edges(cid)
    contradicts = [e for e in out if e.rel_type == RelType.CONTRADICTS]
    meta = db.get_metadata(cid)
    if contradicts:
        for e in contradicts:
            rival_meta = db.get_metadata(e.target_id)
            print(f"   Nike '{meta.content[7:40]}' CONTRADICTS Puma '{rival_meta.content[7:40]}'  w={e.weight}")


# ─────────────────────────────────────────────────────────────────
# [7] Graph export + stats
# ─────────────────────────────────────────────────────────────────
print("\n[7] Graph export stats...")
g_all    = export_graph(db)
g_nike   = export_graph(db, namespace_filter="nike")
g_adidas = export_graph(db, namespace_filter="adidas")
g_puma   = export_graph(db, namespace_filter="puma")

print(f"   Full graph:    {len(g_all['nodes']):4d} nodes  {len(g_all['edges']):5d} edges")
print(f"   Nike only:     {len(g_nike['nodes']):4d} nodes  {len(g_nike['edges']):5d} edges")
print(f"   Adidas only:   {len(g_adidas['nodes']):4d} nodes  {len(g_adidas['edges']):5d} edges")
print(f"   Puma only:     {len(g_puma['nodes']):4d} nodes  {len(g_puma['edges']):5d} edges")

# Per rel_type breakdown
from collections import Counter
rel_dist = Counter(e["rel_type"] for e in g_all["edges"])
print(f"\n   Edge type distribution:")
for rel, cnt in sorted(rel_dist.items(), key=lambda x: -x[1]):
    bar = "█" * min(cnt//3, 30)
    print(f"     {rel:<20} {cnt:4d}  {bar}")


# ─────────────────────────────────────────────────────────────────
# [8] Visualizations — one per brand + full graph
# ─────────────────────────────────────────────────────────────────
print("\n[8] Generating visualizations...")
paths = {}
for brand in ["nike", "adidas", "puma", ""]:
    label = brand if brand else "full"
    ns_filter = brand
    out_path = f"/tmp/ads_graph_{label}.html"
    p = visualize(db, output_path=out_path, namespace_filter=ns_filter)
    paths[label] = p

print("\n   Open these in your browser:")
for label, path in paths.items():
    g = g_nike if label=="nike" else g_adidas if label=="adidas" else g_puma if label=="puma" else g_all
    print(f"   [{label:<8}]  {len(g['nodes'])} nodes · {len(g['edges'])} edges  →  file://{path}")


# ─────────────────────────────────────────────────────────────────
# [9] Save + reload verification
# ─────────────────────────────────────────────────────────────────
print("\n[9] Persistence check...")
db.save()
db2 = DB.open(DB_PATH, dim=DIM)
assert db2.size() == total_records, f"Size mismatch: {db2.size()} vs {total_records}"

# Verify edges survived reload
test_id = list(all_ids["adsets"].values())[0]
edges_orig   = db.get_edges(test_id)
edges_reload = db2.get_edges(test_id)
assert len(edges_reload) == len(edges_orig), "Edge count mismatch after reload"

# Verify reverse index rebuilt
incoming_reload = db2.get_incoming(camp_id)
assert len(incoming_reload) > 0, "Reverse index missing after reload"

# Spot check attribute
m_check = db2.get_metadata(test_id)
assert m_check.get_attribute("record_type") in ["ad_set","campaign","ad","insight"]

print(f"   ✓ {total_records} records reloaded correctly")
print(f"   ✓ Edges persisted: {len(edges_reload)} edges on sample node")
print(f"   ✓ Reverse index rebuilt: {len(incoming_reload)} incoming on campaign")
print(f"   ✓ Attributes intact: record_type={m_check.get_attribute('record_type')}")


# ─────────────────────────────────────────────────────────────────
# [10] Performance summary
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("  PERFORMANCE SUMMARY")
print("=" * 68)
print(f"  Total records:         {total_records}")
print(f"  Campaigns:             {len(all_ids['campaigns'])}")
print(f"  Ad Sets:               {len(all_ids['adsets'])}")
print(f"  Ads:                   {len(all_ids['ads'])}")
print(f"  Insights:              {len(all_ids['insights'])}")
print(f"  Manual edges:          {sum(len(v) for v in [nike_camp_ids, puma_camp_ids])}")
print(f"  Auto-link edges:       {n_auto}")
print(f"  Total graph edges:     {len(g_all['edges'])}")
print(f"  Ingest time:           {t_ingest:.2f}s")

import os
db_size = os.path.getsize(DB_PATH) / 1024
print(f"  DB file size:          {db_size:.1f} KB")

# Search latency benchmark
t_search = time.time()
for _ in range(200):
    q_bench = rng.random(DIM).astype(np.float32)
    db.search(q_bench, k=10)
t_search = (time.time() - t_search) / 200 * 1000
print(f"  Search latency (k=10): {t_search:.3f} ms/query")

t_chain = time.time()
for _ in range(50):
    q_bench = rng.random(DIM).astype(np.float32)
    db.context_chain(q_bench, k=5, hops=2)
t_chain = (time.time() - t_chain) / 50 * 1000
print(f"  context_chain latency: {t_chain:.3f} ms/query (k=5, hops=2)")
print("=" * 68)
print(f"\n  Visualizer links:")
for label, path in paths.items():
    print(f"    file://{path}")
print()
