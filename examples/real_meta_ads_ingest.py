"""
real_meta_ads_ingest.py — Feather DB v0.5.0
Ingest real Meta Ads performance data from real_data/meta_performance_data.csv

Dataset: 352 ad records across 59 campaigns, 114 ad sets
Embeddings: built from ad naming conventions + performance tiers + creative type
Graph edges:
  - part_of: ad → ad_set → campaign
  - derived_from: ad insights encoded in extracted_entities
  - caused_by: high/low ROAS linked to performance signals
  - references: similar ads (auto_link)
"""

import sys, json, time, math, hashlib, re
import numpy as np

sys.path.insert(0, '.')
import feather_db
from feather_db import DB, Metadata, ContextType, FilterBuilder, RelType, visualize, export_graph

# ─── Parse CSV ───────────────────────────────────────────────────────────────
def parse_csv(path):
    with open(path, 'rb') as f:
        raw = f.read()
    lines = raw.split(b'\n')
    header_cols, data_start = [], 0
    for i, line in enumerate(lines):
        if b'\t' in line:
            data_start = i
            break
        col = line.decode('utf-8', errors='replace').strip()
        if col:
            header_cols.append(col)
    all_cols = ['row_num'] + header_cols
    rows = []
    for line in lines[data_start:]:
        decoded = line.decode('utf-8', errors='replace').strip()
        if not decoded:
            continue
        parts = decoded.split('\t')
        if len(parts) == len(all_cols):
            rows.append(dict(zip(all_cols, parts)))
    return rows

NULL_SYM = 'ᴺᵁᴸᴸ'

def safe_float(v, default=0.0):
    if v is None or v == NULL_SYM or v == '':
        return default
    try:
        return float(v)
    except:
        return default

def safe_int(v, default=0):
    try:
        return int(float(v))
    except:
        return default

# ─── Embedding Builder ────────────────────────────────────────────────────────
# 128-dim structured embedding from real fields
# Dims:  0-15   product line (CC/FD/MF/SB/Bonds/RD)
#        16-31  objective (Purchase/Installs/Registration/Traffic/Awareness)
#        32-47  geo signal (Metro/South/TN/KL/ROI/India)
#        48-63  creative type (image/video/static/carousel/reel/DSA)
#        64-79  roas tier (0/low/mid/high/very_high)
#        80-95  spend tier
#        96-111 ctr tier
#        112-127 campaign_id hash noise (keeps same-campaign ads close)

DIM = 128

PRODUCT_MAP = {'CC': 0, 'FD': 1, 'MF': 2, 'SB': 3, 'Bonds': 4, 'Bond': 4,
               'RD': 5, 'AppPromotion': 6, 'RET': 7, 'Ret': 7, 'SM': 8}
OBJ_MAP = {'Purchase': 0, 'Installs': 1, 'Registration': 2,
           'Traffic': 3, 'Clicks': 3, 'Awareness': 4}
GEO_MAP = {'Metro': 0, 'South': 1, 'TamilNadu': 2, 'KL': 3,
           'ROI': 4, 'India': 5, 'INT': 6}
CREATIVE_MAP = {'Static': 0, 'Video': 1, 'Carousel': 2, 'Reel': 3, 'DSA': 4}

def one_hot_16(val, mapping):
    vec = np.zeros(16, dtype=np.float32)
    idx = mapping.get(val, 15)  # 15 = unknown
    vec[idx % 16] = 1.0
    return vec

def roas_tier_vec(roas):
    """Smooth encoding: 0=zero, 1=low(<1), 2=mid(1-5), 3=high(5-20), 4=very_high(>20)"""
    vec = np.zeros(16, dtype=np.float32)
    if roas <= 0:        vec[0] = 1.0
    elif roas < 1:       vec[1] = 1.0; vec[2] = roas
    elif roas < 5:       vec[2] = 1.0; vec[3] = roas / 5.0
    elif roas < 20:      vec[3] = 1.0; vec[4] = roas / 20.0
    else:                vec[4] = 1.0; vec[5] = min(roas / 100.0, 1.0)
    return vec

def spend_tier_vec(spend):
    vec = np.zeros(16, dtype=np.float32)
    if spend <= 0:           vec[0] = 1.0
    elif spend < 1000:       vec[1] = 1.0; vec[2] = spend / 1000.0
    elif spend < 10000:      vec[2] = 1.0; vec[3] = spend / 10000.0
    elif spend < 100000:     vec[3] = 1.0; vec[4] = spend / 100000.0
    elif spend < 1000000:    vec[4] = 1.0; vec[5] = spend / 1000000.0
    else:                    vec[5] = 1.0
    return vec

def ctr_tier_vec(ctr):
    vec = np.zeros(16, dtype=np.float32)
    if ctr <= 0:       vec[0] = 1.0
    elif ctr < 0.5:    vec[1] = 1.0; vec[2] = ctr / 0.5
    elif ctr < 1.5:    vec[2] = 1.0; vec[3] = ctr / 1.5
    elif ctr < 3.0:    vec[3] = 1.0; vec[4] = ctr / 3.0
    else:              vec[4] = 1.0
    return vec

def campaign_noise(campaign_id):
    """Deterministic noise seeded by campaign_id — keeps same-campaign ads nearby."""
    h = int(hashlib.md5(campaign_id.encode()).hexdigest(), 16)
    rng = np.random.default_rng(h % (2**32))
    return rng.random(16).astype(np.float32) * 0.15

def extract_field(name, mapping):
    """Extract first matching key from a naming-convention string."""
    for key in mapping:
        if key in name:
            return key
    return 'unknown'

def make_embedding(row):
    product   = extract_field(row['campaign_name'], PRODUCT_MAP)
    objective = extract_field(row['campaign_name'], OBJ_MAP)
    geo       = extract_field(row['adset_name'], GEO_MAP)
    creative  = extract_field(row['ad_name'], CREATIVE_MAP)
    roas      = safe_float(row['roas'])
    spend     = safe_float(row['total_spend'])
    ctr       = safe_float(row['ctr'])
    cid       = row['campaign_id']

    vec = np.concatenate([
        one_hot_16(product, PRODUCT_MAP),    # 0-15
        one_hot_16(objective, OBJ_MAP),      # 16-31
        one_hot_16(geo, GEO_MAP),            # 32-47
        one_hot_16(creative, CREATIVE_MAP),  # 48-63
        roas_tier_vec(roas),                 # 64-79
        spend_tier_vec(spend),               # 80-95
        ctr_tier_vec(ctr),                   # 96-111
        campaign_noise(cid),                 # 112-127
    ])
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


# ─── Parse creative type from URL ────────────────────────────────────────────
def creative_type(url):
    if 'videos' in url:
        return 'video'
    if 'images' in url:
        return 'image'
    return 'other'


# ─── Parse product line from campaign name ────────────────────────────────────
def product_line(campaign_name):
    # Format: HM_NA_SM_<product>_<obj>_<geo>_<date>
    parts = campaign_name.split('_')
    for i, p in enumerate(parts):
        if p in PRODUCT_MAP:
            return p
    return 'other'


# ─── Main Ingest ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  Feather DB v0.5.0 — Real Meta Ads Ingestion")
print("=" * 65)

CSV_PATH = "real_data/meta_performance_data.csv"
DB_PATH  = "/tmp/real_meta_ads.feather"

rows = parse_csv(CSV_PATH)
print(f"\n[1] Loaded {len(rows)} records from {CSV_PATH}")

db = DB.open(DB_PATH, dim=DIM)

# ─── Build ID maps ────────────────────────────────────────────────────────────
# Campaigns and ad sets get their own nodes
# IDs: ads = 1..N, adsets = N+1..N+M, campaigns = N+M+1..N+M+K

unique_campaigns = {}   # campaign_id -> sequential ID
unique_adsets    = {}   # adset_id    -> sequential ID

for r in rows:
    cid = r['campaign_id']
    aid = r['adset_id']
    if cid not in unique_campaigns:
        unique_campaigns[cid] = len(unique_campaigns) + 1
    if aid not in unique_adsets:
        unique_adsets[aid] = len(unique_adsets) + 1

N_ADS      = len(rows)
N_ADSETS   = len(unique_adsets)
N_CAMPAIGNS= len(unique_campaigns)
ADSET_BASE = N_ADS + 1
CAMP_BASE  = N_ADS + N_ADSETS + 1

print(f"   Ads: {N_ADS}   Ad Sets: {N_ADSETS}   Campaigns: {N_CAMPAIGNS}")


# ─── Insert Campaign nodes ────────────────────────────────────────────────────
print(f"\n[2] Inserting {N_CAMPAIGNS} campaign nodes...")
for cid, seq in unique_campaigns.items():
    feather_id = CAMP_BASE + seq - 1
    cname = None
    for r in rows:
        if r['campaign_id'] == cid:
            cname = r['campaign_name']
            break

    m = Metadata()
    m.timestamp  = int(time.time())
    m.type       = ContextType.FACT
    m.source     = "meta_ads_api"
    m.content    = cname or cid
    m.importance = 0.9
    m.namespace_id = "hawky_meta"
    m.entity_id    = cid
    m.set_attribute("record_type", "campaign")
    m.set_attribute("campaign_id", cid)
    m.set_attribute("product_line", product_line(cname or ''))

    # Campaign vector: avg of its ads' embeddings
    camp_rows = [r for r in rows if r['campaign_id'] == cid]
    vecs = [make_embedding(r) for r in camp_rows]
    avg_vec = np.mean(vecs, axis=0).astype(np.float32)
    avg_vec /= (np.linalg.norm(avg_vec) or 1.0)
    db.add(id=feather_id, vec=avg_vec, meta=m)


# ─── Insert Ad Set nodes ──────────────────────────────────────────────────────
print(f"[3] Inserting {N_ADSETS} ad set nodes...")
for asid, seq in unique_adsets.items():
    feather_id = ADSET_BASE + seq - 1
    aname = None
    cid   = None
    for r in rows:
        if r['adset_id'] == asid:
            aname = r['adset_name']
            cid   = r['campaign_id']
            break

    m = Metadata()
    m.timestamp  = int(time.time())
    m.type       = ContextType.FACT
    m.source     = "meta_ads_api"
    m.content    = aname or asid
    m.importance = 0.75
    m.namespace_id = "hawky_meta"
    m.entity_id    = asid
    m.set_attribute("record_type", "ad_set")
    m.set_attribute("adset_id", asid)
    if cid:
        m.set_attribute("campaign_id", cid)

    adset_rows = [r for r in rows if r['adset_id'] == asid]
    vecs = [make_embedding(r) for r in adset_rows]
    avg_vec = np.mean(vecs, axis=0).astype(np.float32)
    avg_vec /= (np.linalg.norm(avg_vec) or 1.0)
    db.add(id=feather_id, vec=avg_vec, meta=m)


# ─── Insert Ad nodes ──────────────────────────────────────────────────────────
print(f"[4] Inserting {N_ADS} ad nodes...")
t0 = time.time()
adset_to_adids = {}   # feather adset_id -> list of ad feather_ids
camp_to_adids  = {}

for i, r in enumerate(rows):
    feather_id = i + 1
    roas  = safe_float(r['roas'])
    spend = safe_float(r['total_spend'])
    ctr   = safe_float(r['ctr'])
    cpm   = safe_float(r['cpm'])
    cpc   = safe_float(r['cpc'])
    impressions = safe_int(r['total_impressions'])
    clicks      = safe_int(r['total_clicks'])
    installs    = safe_int(r['total_installs'])
    purchase_val= safe_float(r['total_purchase_value'])
    hook_rate   = safe_float(r['hook_rate'])
    hold_rate   = safe_float(r['hold_rate'])

    # Importance scaled by spend (log-normalised, max ~6M)
    imp = min(0.5 + 0.5 * math.log1p(spend) / math.log1p(6_131_954), 1.0)

    # Parse ad_created_time
    try:
        import datetime
        dt = datetime.datetime.strptime(r['ad_created_time'], '%Y-%m-%d %H:%M:%S')
        ts = int(dt.timestamp())
    except:
        ts = int(time.time())

    # Classify context type
    if roas > 5:
        ctype = ContextType.FACT       # strong performer
    elif installs > 0 or purchase_val > 0:
        ctype = ContextType.EVENT      # conversion event
    else:
        ctype = ContextType.PREFERENCE  # awareness / engagement

    m = Metadata()
    m.timestamp  = ts
    m.type       = ctype
    m.source     = "meta_ads_api"
    m.content    = r['ad_name']
    m.importance = imp
    m.namespace_id = "hawky_meta"
    m.entity_id    = r['ad_id']        # Meta's ad ID
    m.set_attribute("record_type",   "ad")
    m.set_attribute("ad_name",       r['ad_name'])
    m.set_attribute("adset_id",      r['adset_id'])
    m.set_attribute("campaign_id",   r['campaign_id'])
    m.set_attribute("creative_type", creative_type(r['url']))
    m.set_attribute("product_line",  product_line(r['campaign_name']))
    m.set_attribute("roas",          str(round(roas, 4)))
    m.set_attribute("spend",         str(round(spend, 2)))
    m.set_attribute("ctr",           str(round(ctr, 4)))
    m.set_attribute("cpm",           str(round(cpm, 2)))
    m.set_attribute("cpc",           str(round(cpc, 2)))
    m.set_attribute("impressions",   str(impressions))
    m.set_attribute("clicks",        str(clicks))
    m.set_attribute("installs",      str(installs))
    m.set_attribute("hook_rate",     str(round(hook_rate, 4)))
    m.set_attribute("hold_rate",     str(round(hold_rate, 4)))
    m.set_attribute("url",           r['url'][:200])

    vec = make_embedding(r)
    db.add(id=feather_id, vec=vec, meta=m)

    # Track for edge creation
    as_fid = ADSET_BASE + unique_adsets[r['adset_id']] - 1
    c_fid  = CAMP_BASE  + unique_campaigns[r['campaign_id']] - 1
    adset_to_adids.setdefault(as_fid, []).append(feather_id)
    camp_to_adids.setdefault(c_fid, []).append(feather_id)

print(f"   Inserted in {time.time()-t0:.2f}s")


# ─── Build hierarchy edges ────────────────────────────────────────────────────
print("\n[5] Building hierarchy edges (part_of)...")

# Ad → Ad Set
for r in rows:
    ad_fid    = int(r['row_num'])
    adset_fid = ADSET_BASE + unique_adsets[r['adset_id']] - 1
    db.link(ad_fid, adset_fid, RelType.PART_OF, weight=1.0)

# Ad Set → Campaign
for asid, seq in unique_adsets.items():
    adset_fid = ADSET_BASE + seq - 1
    cid = None
    for r in rows:
        if r['adset_id'] == asid:
            cid = r['campaign_id']
            break
    if cid:
        camp_fid = CAMP_BASE + unique_campaigns[cid] - 1
        db.link(adset_fid, camp_fid, RelType.PART_OF, weight=1.0)

print(f"   {N_ADS} ad→adset + {N_ADSETS} adset→campaign edges")


# ─── Performance signal edges ─────────────────────────────────────────────────
print("\n[6] Building performance signal edges...")

# Find top performers (ROAS > 5) and link them to their campaigns with caused_by
caused_edges = 0
supports_edges = 0
for r in rows:
    roas = safe_float(r['roas'])
    ad_fid   = int(r['row_num'])
    camp_fid = CAMP_BASE + unique_campaigns[r['campaign_id']] - 1

    if roas > 5:
        # top performer causes campaign success signal
        db.link(camp_fid, ad_fid, RelType.CAUSED_BY, weight=min(roas/50.0, 1.0))
        caused_edges += 1
    elif roas > 1:
        # decent performer supports campaign
        db.link(camp_fid, ad_fid, RelType.SUPPORTS, weight=roas/5.0)
        supports_edges += 1

print(f"   {caused_edges} caused_by (ROAS>5)   {supports_edges} supports (ROAS 1-5)")


# ─── Cross-product line edges (references) ────────────────────────────────────
print("\n[7] Cross-product reference edges (same objective, different product)...")
# Group by objective, link top ads across product lines
objective_groups = {}
for r in rows:
    obj = extract_field(r['campaign_name'], OBJ_MAP)
    roas = safe_float(r['roas'])
    if roas > 3:
        objective_groups.setdefault(obj, []).append((int(r['row_num']), roas, product_line(r['campaign_name'])))

ref_edges = 0
for obj, ads in objective_groups.items():
    # Sort by ROAS desc, link top 3 across different product lines
    ads.sort(key=lambda x: -x[1])
    seen_products = set()
    top = []
    for fid, roas, pl in ads:
        if pl not in seen_products:
            top.append((fid, roas, pl))
            seen_products.add(pl)
        if len(top) >= 4:
            break
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            db.link(top[i][0], top[j][0], RelType.REFERENCES, weight=0.6)
            ref_edges += 1

print(f"   {ref_edges} cross-product reference edges")


# ─── Auto-link by vector similarity ──────────────────────────────────────────
print("\n[8] Auto-linking by vector similarity (threshold=0.88)...")
t0 = time.time()
n_auto = db.auto_link(modality="text", threshold=0.88, rel_type=RelType.RELATED_TO, candidates=8)
print(f"   {n_auto} similarity edges created in {time.time()-t0:.0f}ms")


# ─── Queries ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  QUERIES")
print("=" * 65)

# Q1: Top performing ads by ROAS
print("\nQ1 — Top 10 ads by ROAS:")
top_roas = sorted(
    [(int(r['row_num']), safe_float(r['roas']), r['ad_name'], r['campaign_name'],
      creative_type(r['url']), safe_float(r['total_spend']))
     for r in rows if safe_float(r['roas']) > 0],
    key=lambda x: -x[1]
)[:10]
for fid, roas, name, camp, ctype, spend in top_roas:
    print(f"  id:{fid:<4} roas={roas:>8.2f}  spend={spend:>10,.0f}  type={ctype:<5}  {name[:45]}")

# Q2: Filter video ads with ROAS > 2
print("\nQ2 — Video ads with ROAS > 2:")
f = FilterBuilder().namespace("hawky_meta").attribute("creative_type", "video").attribute("record_type", "ad")
q_vec = make_embedding(rows[0])
results = db.search(q_vec, k=50, filter=f.build())
video_high = [(r.id, safe_float(r.metadata.get_attribute("roas","0")), r.metadata.content)
              for r in results if safe_float(r.metadata.get_attribute("roas","0")) > 2]
video_high.sort(key=lambda x: -x[1])
for fid, roas, name in video_high[:8]:
    print(f"  id:{fid:<4} roas={roas:.2f}  {name[:55]}")

# Q3: Product line breakdown by avg ROAS
print("\nQ3 — Avg ROAS by product line:")
pl_roas = {}
for r in rows:
    pl = product_line(r['campaign_name'])
    roas = safe_float(r['roas'])
    if roas > 0:
        pl_roas.setdefault(pl, []).append(roas)
for pl, vals in sorted(pl_roas.items(), key=lambda x: -sum(x[1])/len(x[1])):
    avg = sum(vals)/len(vals)
    median = sorted(vals)[len(vals)//2]
    print(f"  {pl:<14}  n={len(vals):<4}  avg_roas={avg:>8.2f}  median={median:>8.2f}")

# Q4: Context chain from a high-ROAS ad
print("\nQ4 — Context chain from top ROAS ad (2 hops):")
top_ad_id = top_roas[0][0]
top_vec = make_embedding(rows[top_ad_id - 1])
chain = db.context_chain(top_vec, k=5, hops=2, modality="text")
print(f"   Subgraph: {len(chain.nodes)} nodes, {len(chain.edges)} edges")
print("   Top nodes:")
for n in chain.nodes[:8]:
    rtype = n.metadata.get_attribute("record_type", "?")
    roas_a = n.metadata.get_attribute("roas", "?")
    hop_tag = "(seed)" if n.hop == 0 else f"(hop {n.hop})"
    print(f"     id:{n.id:<5} score={n.score:.3f}  {rtype:<10} roas={roas_a:<8} {hop_tag}  {n.metadata.content[:40]}")

# Q5: High-spend low-ROAS (inefficient ads to review)
print("\nQ5 — High spend, low ROAS (potential waste):")
waste = sorted(
    [(int(r['row_num']), safe_float(r['roas']), safe_float(r['total_spend']), r['ad_name'])
     for r in rows if safe_float(r['total_spend']) > 10000 and safe_float(r['roas']) < 1],
    key=lambda x: -x[2]
)[:8]
for fid, roas, spend, name in waste:
    print(f"  id:{fid:<4} spend={spend:>10,.0f}  roas={roas:.4f}  {name[:50]}")

# Q6: Campaign spend + ROAS roll-up
print("\nQ6 — Campaign roll-up (spend, ROAS, # ads):")
camp_stats = {}
for r in rows:
    cname = r['campaign_name']
    roas  = safe_float(r['roas'])
    spend = safe_float(r['total_spend'])
    camp_stats.setdefault(cname, {'spend': 0, 'roas_vals': [], 'n': 0})
    camp_stats[cname]['spend'] += spend
    camp_stats[cname]['n'] += 1
    if roas > 0:
        camp_stats[cname]['roas_vals'].append(roas)

for cname, s in sorted(camp_stats.items(), key=lambda x: -x[1]['spend'])[:10]:
    avg_r = sum(s['roas_vals'])/len(s['roas_vals']) if s['roas_vals'] else 0
    print(f"  spend={s['spend']:>10,.0f}  avg_roas={avg_r:>7.2f}  n={s['n']:<3}  {cname[:60]}")


# ─── Graph export & visualization ────────────────────────────────────────────
print("\n" + "=" * 65)
print("  GRAPH EXPORT + VISUALIZATION")
print("=" * 65)

g = export_graph(db, namespace_filter="hawky_meta")
print(f"\n  Full graph: {len(g['nodes'])} nodes  {len(g['edges'])} edges")

# Count edge types
from collections import Counter
edge_types = Counter(e['rel_type'] for e in g['edges'])
print("  Edge type distribution:")
for etype, cnt in edge_types.most_common():
    bar = '█' * min(cnt // 10, 40)
    print(f"    {etype:<20} {cnt:>5}  {bar}")

print("\n  Generating visualizations...")
html_full = visualize(db, output_path="/tmp/real_meta_full.html",
                      namespace_filter="hawky_meta",
                      title="Real Meta Ads — Full Graph")

# CC product only
f_cc = FilterBuilder().namespace("hawky_meta").attribute("product_line", "CC")
cc_ads = [int(r['row_num']) for r in rows if product_line(r['campaign_name']) == 'CC']
print(f"  CC product ads: {len(cc_ads)}")

# MF product only
mf_ads = [int(r['row_num']) for r in rows if product_line(r['campaign_name']) == 'MF']
print(f"  MF product ads: {len(mf_ads)}")

print()
print("  Visualizer links:")
print(f"    file:///tmp/real_meta_full.html   ({len(g['nodes'])} nodes, {len(g['edges'])} edges)")


# ─── Persistence check ───────────────────────────────────────────────────────
print("\n[9] Persistence check...")
db.save()
db2 = DB.open(DB_PATH, dim=DIM)
assert db2.size() == db.size(), "Record count mismatch after reload"
sample = db2.get_metadata(1)
assert sample is not None
assert sample.get_attribute("record_type") == "ad"
edges = db2.get_edges(1)
assert any(e.rel_type == RelType.PART_OF for e in edges), "Missing part_of edge after reload"
print(f"  Records: {db2.size()}  Edges on ad#1: {len(edges)}  Attributes: OK")
print("  All assertions passed")

print()
print("=" * 65)
print("  DONE")
print(f"  {db.size()} nodes  |  Real Meta Ads fully ingested + graphed")
print("  Open: file:///tmp/real_meta_full.html")
print("=" * 65)
