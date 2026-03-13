#!/usr/bin/env python3
"""
Real-Data Time-Series Intelligence Simulation — Feb 1–2, 2026
==============================================================
Uses actual Stable Money Meta ad data (meta_performance_data.csv) to simulate
hourly performance snapshots across a 48-hour window, layered with synthesized
creative intelligence, competitor events, and social media trends.

Story:
  Feb 1, 2026 = India Union Budget Day (announcement at 11 AM IST)
    → FD & Bond spike (favorable small-savings provisions in Budget)
    → MF dips (LTCG tax not eased, STT raised on equity F&O)
    → CC moderate (interest rate cap discussion)
    → Competitor Bajaj Finance + Groww react to the FD/MF delta
  Feb 2, 2026 = Post-Budget day
    → Sustained FD/Bond performance, MF recovery begins
    → #Budget2026, #FDvsGold trending on social media
    → Valentine's Week creatives start launching (CTR lift for lifestyle ads)

Node Layers:
  1000-1009  Product nodes (FD, CC, Bond, MF)
  2001-2060  Real ad nodes (top 15 per product from CSV)
  3001-6000  Hourly performance snapshots (ad × 48 hours)
  7001-7030  Creative intelligence (from extracted_entities)
  8001-8020  Competitor intelligence
  9001-9015  Social media trend nodes
  9101-9110  Industry event nodes (Budget, RBI, Valentine's)
"""

import sys, os, json, math, random, time
from datetime import datetime, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feather_db
import numpy as np

random.seed(42)
np.random.seed(42)

DB_PATH  = "/tmp/real_ts_demo.feather"
HTML_OUT = "/tmp/real_ts_demo.html"
CSV_PATH = os.path.join(os.path.dirname(__file__), "../real_data/meta_performance_data.csv")
DIM      = 64

# ─── Parse CSV ─────────────────────────────────────────────────────────────────

def parse_csv(path):
    with open(path) as f:
        lines = f.readlines()
    headers, data_rows = [], []
    for line in lines:
        s = line.strip()
        if s and s[0].isdigit() and "\t" in s:
            data_rows.append(s.split("\t"))
        else:
            headers.append(s)
    return [dict(zip(headers, r[1:])) for r in data_rows]

def classify_product(row):
    name = (row.get("ad_name", "") + " " + row.get("campaign_name", "")).upper()
    if "_CC_" in name or name.startswith("SM_CC") or "CREDITCARD" in name: return "CC"
    if "_FD_" in name or "_FD " in name or "SM_FD" in name[:20]:           return "FD"
    if "BOND" in name or "SB_" in name[:5]:                                return "Bond"
    if "_MF_" in name or "SM_MF" in name[:20]:                             return "MF"
    return "Other"

def safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except:
        return default

# ─── Embedding ─────────────────────────────────────────────────────────────────

VOCAB = [
    "ctr","roas","cpm","cpc","impressions","clicks","spend","leads",
    "fd","fixeddeposit","bond","creditcard","mutualfund","rd",
    "budget","india","finance","invest","return","rate","interest","growth",
    "campaign","ad","creative","static","video","carousel","influencer",
    "morning","afternoon","evening","night","peak","hour","daily","weekly",
    "competitor","groww","bajaj","hdfc","paytm","zerodha","finflex","sbi",
    "social","trend","sentiment","viral","engagement","twitter","instagram",
    "spike","surge","decline","drop","recovery","stable","volatile",
    "valentine","february","budget2026","rbi","fiscal","tax","savings",
    "install","purchase","registration","lead","conversion","acquisition",
    "hook","hold","visual","dialogue","voiceover","calltoaction","cta",
    "roi","retention","metro","tier2","south","north","hindi","tamil","bengali",
]

def embed_text(text):
    text = text.lower()
    vec = np.zeros(DIM, dtype=np.float32)
    tokens = text.replace(",", " ").replace(".", " ").replace("_", " ").replace("/", " ").split()
    for tok in tokens:
        for i, kw in enumerate(VOCAB):
            if kw in tok or tok in kw:
                idx = (i * 7 + len(tok) * 3) % DIM
                vec[idx] += 1.0 + len(tok) * 0.05
    vec += np.random.rand(DIM).astype(np.float32) * 0.03
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec

def embed_perf(ctr, roas, cpm, impressions, hour, day, product, budget_impact=0.0):
    vec = np.zeros(DIM, dtype=np.float32)
    vec[0]  = min(ctr / 5.0, 1.0)
    vec[1]  = min(roas / 300.0, 1.0)
    vec[2]  = min(cpm / 500.0, 1.0)
    vec[3]  = min(impressions / 150000.0, 1.0)
    vec[4]  = hour / 23.0
    vec[5]  = float(day)
    vec[6]  = max(-1.0, min(1.0, budget_impact))
    prod_map = {"FD": 7, "CC": 8, "Bond": 9, "MF": 10}
    if product in prod_map:
        vec[prod_map[product]] = 1.0
    vec[11] = 1.0 if 9  <= hour <= 11 else 0.0   # morning peak
    vec[12] = 1.0 if 18 <= hour <= 21 else 0.0   # evening peak
    vec[13] = 1.0 if 12 <= hour <= 14 else 0.0   # lunch dip
    vec += np.random.rand(DIM).astype(np.float32) * 0.015
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec

# ─── Time patterns ─────────────────────────────────────────────────────────────

TOD_PATTERN = {
    0:0.28, 1:0.18, 2:0.13, 3:0.11, 4:0.12, 5:0.18,
    6:0.42, 7:0.72, 8:0.88, 9:1.00, 10:1.08, 11:1.18,
    12:0.88, 13:0.82, 14:0.90, 15:1.02, 16:1.08, 17:1.14,
    18:1.22, 19:1.32, 20:1.28, 21:1.12, 22:0.78, 23:0.48,
}

def budget_effect(product, day, hour):
    """Returns (ctr_mult, roas_mult, budget_impact_score [-1,1])"""
    if day == 0 and hour >= 11:
        h = hour - 11
        ramp = min(1.0, h / 4.0)
        if product == "FD":
            return 1.3 + ramp * 0.45, 1.5 + ramp * 0.6,  0.85
        elif product == "Bond":
            if h < 2: return 0.88, 0.82, 0.45
            return 1.18 + ramp * 0.2, 1.35 + ramp * 0.2, 0.70
        elif product == "CC":
            return 1.06, 1.08, 0.30
        elif product == "MF":
            drop = min(1.0, h / 5.0)
            return 0.82 - drop * 0.12, 0.78 - drop * 0.18, -0.55
    elif day == 1:
        if product == "FD":    return 1.22, 1.32, 0.65
        elif product == "Bond": return 1.14, 1.22, 0.52
        elif product == "MF":   return 1.02, 0.96, -0.12   # recovery
        elif product == "CC":   return 1.10, 1.06, 0.22
    return 1.0, 1.0, 0.0

# ─── Build DB ──────────────────────────────────────────────────────────────────

print("─" * 60)
print("  Stable Money Intelligence — Feb 1–2, 2026 Time Series")
print("─" * 60)

rows = parse_csv(CSV_PATH)
print(f"Loaded {len(rows)} real ads from CSV")

# Group by product, pick top N by spend
by_product = defaultdict(list)
for r in rows:
    p = classify_product(r)
    if p == "Other": continue
    by_product[p].append(r)

PRODUCTS = ["FD", "CC", "Bond", "MF"]
PRODUCT_COLORS = {"FD": "#4f8ef7", "CC": "#f7874f", "Bond": "#4ff7a0", "MF": "#f7e24f"}
TOP_N = 12   # top ads per product

selected_ads = {}   # product → [row, ...]
for prod in PRODUCTS:
    ads = sorted(by_product[prod], key=lambda r: safe_float(r.get("total_spend", 0)), reverse=True)
    selected_ads[prod] = ads[:TOP_N]

total_ads = sum(len(v) for v in selected_ads.values())
print(f"Selected {total_ads} top ads ({TOP_N}/product)")

# ─── Open DB ───────────────────────────────────────────────────────────────────
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

db = feather_db.DB.open(DB_PATH, dim=DIM)

# ─── Product nodes ─────────────────────────────────────────────────────────────
PRODUCT_IDS = {"FD": 1001, "CC": 1002, "Bond": 1003, "MF": 1004}
PRODUCT_DESCS = {
    "FD":   "Fixed Deposit — guaranteed returns, FDIC-style insurance",
    "CC":   "Credit Card — airport lounge, cashback, spend-linked rewards",
    "Bond": "Bonds — corporate & government bonds, monthly payout",
    "MF":   "Mutual Fund — SIP-based equity & gold fund investment",
}

for prod, pid in PRODUCT_IDS.items():
    meta = feather_db.Metadata()
    meta.timestamp  = int(datetime(2026,2,1,0,0,tzinfo=timezone.utc).timestamp())
    meta.importance = 1.0
    meta.type       = feather_db.ContextType.FACT
    meta.content    = PRODUCT_DESCS[prod]
    meta.namespace_id = "stable_money"
    meta.entity_id    = prod.lower()
    meta.set_attribute("entity_type", "product")
    meta.set_attribute("product", prod)
    db.add(id=pid, vec=embed_text(PRODUCT_DESCS[prod]), meta=meta)

print(f"Added {len(PRODUCT_IDS)} product nodes")

# ─── Real Ad nodes ─────────────────────────────────────────────────────────────
ad_id_map = {}  # row["ad_id"] → feather_id
feather_ad_ids = {}  # feather_id → {prod, row}
ad_counter = 2001

for prod in PRODUCTS:
    for row in selected_ads[prod]:
        raw_id = row.get("ad_id", "")
        feather_id = ad_counter
        ad_counter += 1
        ad_id_map[raw_id] = feather_id
        feather_ad_ids[feather_id] = {"prod": prod, "row": row}

        meta = feather_db.Metadata()
        meta.timestamp    = int(datetime(2026,1,15,tzinfo=timezone.utc).timestamp())
        meta.importance   = min(1.0, safe_float(row.get("total_spend",0)) / 6000000.0)
        if meta.importance < 0.1: meta.importance = 0.3
        meta.type         = feather_db.ContextType.FACT
        meta.source       = "meta_ads_api"
        meta.content      = row.get("ad_name","")
        meta.namespace_id = "stable_money"
        meta.entity_id    = prod.lower()
        meta.set_attribute("entity_type",    "ad")
        meta.set_attribute("product",        prod)
        meta.set_attribute("ad_id",          row.get("ad_id",""))
        meta.set_attribute("adset_name",     row.get("adset_name","")[:80])
        meta.set_attribute("campaign_name",  row.get("campaign_name","")[:80])
        meta.set_attribute("campaign_id",    row.get("campaign_id",""))
        meta.set_attribute("total_spend",    row.get("total_spend","0"))
        meta.set_attribute("ctr",            row.get("ctr","0"))
        meta.set_attribute("roas",           row.get("roas","0"))
        meta.set_attribute("cpm",            row.get("cpm","0"))

        text = f"{row.get('ad_name','')} {row.get('campaign_name','')} {prod}"
        db.add(id=feather_id, vec=embed_text(text), meta=meta)

        # Link to product node
        db.link(from_id=feather_id, to_id=PRODUCT_IDS[prod], rel_type="part_of", weight=0.9)

print(f"Added {ad_counter - 2001} real ad nodes")

# ─── Hourly performance snapshots (48 hours × 48 ads) ─────────────────────────
FEB1_TS = int(datetime(2026,2,1,0,0,tzinfo=timezone.utc).timestamp())
HOUR_SEC = 3600
snap_counter = 3001
snaps_by_ad_hour = {}  # (feather_ad_id, day, hour) → snap_feather_id

# Store hourly series data for visualization
timeseries_data = defaultdict(list)   # product → [{hour_idx, hour_label, ctr, roas}]

print("Simulating 48 hours of performance data...")
for ad_fid, ad_info in feather_ad_ids.items():
    prod = ad_info["prod"]
    row  = ad_info["row"]
    base_ctr  = max(0.05, safe_float(row.get("ctr", 0.3)))
    base_roas = max(0.0,  safe_float(row.get("roas", 1.0)))
    base_cpm  = max(1.0,  safe_float(row.get("cpm", 50.0)))
    base_impr = max(100,  safe_float(row.get("total_impressions", 5000)) / 30)

    for day in range(2):
        for hour in range(24):
            hour_idx = day * 24 + hour
            ts = FEB1_TS + hour_idx * HOUR_SEC

            tod     = TOD_PATTERN[hour]
            be_ctr, be_roas, bi_score = budget_effect(prod, day, hour)
            noise_c = 1.0 + random.gauss(0, 0.04)
            noise_r = 1.0 + random.gauss(0, 0.06)

            sim_ctr  = base_ctr  * tod * be_ctr  * noise_c
            sim_roas = base_roas * tod * be_roas  * noise_r
            sim_cpm  = base_cpm  * (0.8 + random.uniform(0, 0.4))
            sim_impr = base_impr * tod * (1.0 + 0.3 * random.random())
            sim_ctr  = max(0.0, sim_ctr)
            sim_roas = max(0.0, sim_roas)

            snap_id = snap_counter
            snap_counter += 1
            snaps_by_ad_hour[(ad_fid, day, hour)] = snap_id

            meta = feather_db.Metadata()
            meta.timestamp    = ts
            meta.importance   = 0.4 + abs(bi_score) * 0.3
            meta.type         = feather_db.ContextType.EVENT
            meta.source       = "meta_ads_hourly"
            meta.content      = (
                f"{prod} hourly snapshot — "
                f"{'Feb 1' if day==0 else 'Feb 2'} {hour:02d}:00 | "
                f"CTR={sim_ctr:.3f}% ROAS={sim_roas:.2f}"
            )
            meta.namespace_id = "stable_money"
            meta.entity_id    = prod.lower()
            meta.set_attribute("entity_type",   "perf_snapshot")
            meta.set_attribute("product",       prod)
            meta.set_attribute("ad_id",         str(ad_fid))
            meta.set_attribute("day",           str(day))
            meta.set_attribute("hour",          str(hour))
            meta.set_attribute("hour_idx",      str(hour_idx))
            meta.set_attribute("ctr",           f"{sim_ctr:.4f}")
            meta.set_attribute("roas",          f"{sim_roas:.3f}")
            meta.set_attribute("cpm",           f"{sim_cpm:.2f}")
            meta.set_attribute("impressions",   f"{sim_impr:.0f}")
            meta.set_attribute("budget_impact", f"{bi_score:.3f}")

            vec = embed_perf(sim_ctr, sim_roas, sim_cpm, sim_impr, hour, day, prod, bi_score)
            db.add(id=snap_id, vec=vec, meta=meta)
            db.link(from_id=snap_id, to_id=ad_fid, rel_type="derived_from", weight=0.85)

            # Collect series for product-level chart
            timeseries_data[prod].append({
                "hour_idx": hour_idx,
                "ctr": sim_ctr,
                "roas": sim_roas,
            })

print(f"Added {snap_counter - 3001} hourly performance snapshots")

# ─── Creative Intelligence (from extracted_entities) ───────────────────────────
print("Synthesizing creative intelligence nodes...")

creative_intel_nodes = []
ci_counter = 7001

# Pick up to 2 ads per product with valid extracted_entities
for prod in PRODUCTS:
    seen = 0
    for row in selected_ads[prod]:
        raw = row.get("extracted_entities", "")
        if not raw or raw in ("ᴺᵁᴸᴸ", "NULL", ""):
            continue
        try:
            entities = json.loads(raw)
        except:
            continue

        # Build rich creative summary
        cta          = entities.get("Call To Action", entities.get("Call To Action Text", ""))
        emotional    = entities.get("Emotional Appeal", "")
        hook         = entities.get("Hook Element", "")
        key_selling  = entities.get("Key Selling Points", "")
        language     = entities.get("Language", "")
        usp          = entities.get("Unique Selling Points", "")
        return_rate  = entities.get("Return Rate Mentioned", "")
        visual_style = entities.get("Key Visual Style", entities.get("Video Style", ""))
        presenter    = entities.get("Presenter Type", "")
        dialogue     = entities.get("Dialogues", "")[:200] if entities.get("Dialogues") else ""

        summary = (
            f"Creative for {prod} | Lang: {language} | "
            f"CTA: {cta} | Hook: {hook} | "
            f"Emotional: {emotional} | USP: {usp} | "
            f"Return: {return_rate} | Style: {visual_style} | Presenter: {presenter}"
        )

        node_id = ci_counter
        ci_counter += 1

        meta = feather_db.Metadata()
        meta.timestamp    = int(datetime(2026,2,1,tzinfo=timezone.utc).timestamp())
        meta.importance   = 0.8
        meta.type         = feather_db.ContextType.FACT
        meta.source       = "creative_intelligence"
        meta.content      = summary
        meta.namespace_id = "stable_money"
        meta.entity_id    = prod.lower()
        meta.set_attribute("entity_type",    "creative_intel")
        meta.set_attribute("product",        prod)
        meta.set_attribute("cta",            cta[:60])
        meta.set_attribute("emotional_appeal", emotional[:60])
        meta.set_attribute("hook",           hook[:60])
        meta.set_attribute("language",       language[:40])
        meta.set_attribute("visual_style",   visual_style[:60])
        meta.set_attribute("return_rate",    return_rate[:40])
        meta.set_attribute("presenter",      presenter[:40])
        meta.set_attribute("ad_id",          row.get("ad_id",""))

        db.add(id=node_id, vec=embed_text(summary + " " + dialogue), meta=meta)

        # Link to product and the real ad node
        ad_fid = ad_id_map.get(row.get("ad_id",""))
        if ad_fid:
            db.link(from_id=node_id, to_id=ad_fid, rel_type="references", weight=0.9)
        db.link(from_id=node_id, to_id=PRODUCT_IDS[prod], rel_type="part_of", weight=0.7)

        creative_intel_nodes.append(node_id)
        seen += 1
        if seen >= 3:
            break

print(f"Added {len(creative_intel_nodes)} creative intelligence nodes")

# ─── Competitor Intelligence ────────────────────────────────────────────────────
print("Adding competitor intelligence nodes...")

COMPETITORS = [
    {
        "id": 8001, "name": "Bajaj Finance FD Push",
        "product": "FD",
        "content": "Bajaj Finance launches 8.85% FD campaign post-Budget on Feb 1. Heavy Instagram/Meta spend. Targeting Stable Money FD segment with higher rate messaging. CTR 2.1x industry avg.",
        "date": "Feb 1 14:00", "day": 0, "severity": "high",
        "links": [PRODUCT_IDS["FD"]],
    },
    {
        "id": 8002, "name": "Groww MF Budget Campaign",
        "product": "MF",
        "content": "Groww activates Budget Day MF campaign: 'Invest before LTCG changes land'. Emergency creative push. 40% budget uplift. Targets 25-35 salaried Metro cohort.",
        "date": "Feb 1 12:30", "day": 0, "severity": "medium",
        "links": [PRODUCT_IDS["MF"]],
    },
    {
        "id": 8003, "name": "HDFC Bank CC Offer",
        "product": "CC",
        "content": "HDFC Bank CC team activates Valentine's Week lounge access offer — 2 free airport lounge visits with first spend ₹5K. Directly competitive to Stable Money CC Airport Lounge creative.",
        "date": "Feb 2 09:00", "day": 1, "severity": "medium",
        "links": [PRODUCT_IDS["CC"]],
    },
    {
        "id": 8004, "name": "Zerodha Bonds Retargeting",
        "product": "Bond",
        "content": "Zerodha Coin launches bond retargeting: 'Post-budget 9.1% corporate bond available'. Competitive to Stable Money Bond narrative. High hold-rate creative (Telugu voiceover).",
        "date": "Feb 1 16:00", "day": 0, "severity": "medium",
        "links": [PRODUCT_IDS["Bond"]],
    },
    {
        "id": 8005, "name": "Paytm Money FD + MF Bundle",
        "product": "FD",
        "content": "Paytm Money launches FD+MF combo offer post Budget. 'Save tax + earn more — FD for stability, MF for growth'. Budget-specific creative. Feb 2 heavy push.",
        "date": "Feb 2 11:00", "day": 1, "severity": "high",
        "links": [PRODUCT_IDS["FD"], PRODUCT_IDS["MF"]],
    },
    {
        "id": 8006, "name": "SBI FD Rate Match",
        "product": "FD",
        "content": "SBI Digital revises FD landing page to highlight 7.1% rate (Budget announcement of senior citizen FD benefit). Lower than Stable Money but brand trust drives volume.",
        "date": "Feb 1 15:30", "day": 0, "severity": "low",
        "links": [PRODUCT_IDS["FD"]],
    },
    {
        "id": 8007, "name": "Angel One MF Budget Blast",
        "product": "MF",
        "content": "Angel One MF budget special: ELSS push before LTCG clarity. ₹500 SIP messaging. Instagram Stories & Reels blitz targeting 18-28 age group. CTR spike observed.",
        "date": "Feb 1 13:00", "day": 0, "severity": "medium",
        "links": [PRODUCT_IDS["MF"]],
    },
    {
        "id": 8008, "name": "Navi Finance FD Creative",
        "product": "FD",
        "content": "Navi Finance reactive FD creative: 'Budget 2026 loves savers — 8.5% FD'. Same-day creative turnaround. Lower production quality but CTR competitive (0.48%).",
        "date": "Feb 1 17:00", "day": 0, "severity": "low",
        "links": [PRODUCT_IDS["FD"]],
    },
]

for c in COMPETITORS:
    ts = FEB1_TS + (c["day"] * 24 * HOUR_SEC)
    meta = feather_db.Metadata()
    meta.timestamp    = ts
    meta.importance   = {"high": 0.9, "medium": 0.7, "low": 0.5}[c["severity"]]
    meta.type         = feather_db.ContextType.EVENT
    meta.source       = "competitor_intel"
    meta.content      = c["content"]
    meta.namespace_id = "stable_money"
    meta.entity_id    = "competitor"
    meta.set_attribute("entity_type", "competitor_intel")
    meta.set_attribute("competitor",  c["name"][:60])
    meta.set_attribute("product",     c["product"])
    meta.set_attribute("date",        c["date"])
    meta.set_attribute("severity",    c["severity"])

    db.add(id=c["id"], vec=embed_text(c["content"] + " " + c["product"]), meta=meta)
    for link_id in c.get("links", []):
        db.link(from_id=c["id"], to_id=link_id, rel_type="contradicts", weight=0.8)

print(f"Added {len(COMPETITORS)} competitor intel nodes")

# ─── Social Media Trend Nodes ──────────────────────────────────────────────────
print("Adding social media trend nodes...")

SOCIAL_TRENDS = [
    {
        "id": 9001, "tag": "#Budget2026",
        "content": "#Budget2026 trending #1 in India on Feb 1 from 11:05 AM. 4.2M tweets in 6 hours. Dominant sentiment: 'FD is now the king investment'. FD-related search volume +340%.",
        "product": "FD", "day": 0, "hour": 11, "sentiment": "positive",
    },
    {
        "id": 9002, "tag": "#FDvsMF",
        "content": "#FDvsMF trending on Twitter Feb 1 afternoon. Key narrative: Budget makes FD more attractive than MF (LTCG angle). Influencers: @NithinKamath tweets supporting MF long-term.",
        "product": "FD", "day": 0, "hour": 14, "sentiment": "mixed",
    },
    {
        "id": 9003, "tag": "#BondInvesting",
        "content": "#BondInvesting +180% search volume post-Budget. 'Monthly payout bonds' queries surge. YouTube: 'Best bonds to buy after Budget 2026' — 2.1M views in 24h.",
        "product": "Bond", "day": 0, "hour": 13, "sentiment": "positive",
    },
    {
        "id": 9004, "tag": "#MFSahi Debate",
        "content": "AMFI's #MFSahiHai campaign counters negative MF sentiment. LTCG confusion creates 'hold or sell?' queries. MF search volume dips 22% Feb 1 afternoon before recovering.",
        "product": "MF", "day": 0, "hour": 15, "sentiment": "negative",
    },
    {
        "id": 9005, "tag": "#ValentinesWeek",
        "content": "#ValentinesWeek starts Feb 2. Financial brands pivot: 'Gift your partner financial security' angle. CC lounge + FD gift ideas trending. Lifestyle × Finance content CTR lift +15%.",
        "product": "CC", "day": 1, "hour": 7, "sentiment": "positive",
    },
    {
        "id": 9006, "tag": "#BudgetForSavers",
        "content": "#BudgetForSavers organic campaign: senior citizen FD benefit highlighted. Facebook groups: 'Which FD to open after Budget?'. Organic reach benefits paid FD ads (+8% CTR halo).",
        "product": "FD", "day": 0, "hour": 16, "sentiment": "positive",
    },
    {
        "id": 9007, "tag": "Instagram Reels Finance",
        "content": "Finance Reels engagement spike Feb 2. Format: 15s 'Budget Winner/Loser' carousel. FD creatives with rate callout performing 2.3x vs. pre-budget Reels. Hook rate +40%.",
        "product": "FD", "day": 1, "hour": 10, "sentiment": "positive",
    },
    {
        "id": 9008, "tag": "South India FD Interest",
        "content": "Tamil & Telugu FD search queries +210% post-Budget. Vernacular FD creatives (Tamil voiceover) see CTR 0.78% vs. 0.31% English average. High opportunity signal.",
        "product": "FD", "day": 1, "hour": 9, "sentiment": "positive",
    },
    {
        "id": 9009, "tag": "Mutual Fund Outflow Alert",
        "content": "SEBI data: Feb 1 equity MF redemptions spike 18% (preliminary). Debt fund inflows up 12%. Consumer pivot toward FD + bonds. Sentiment: risk-off mode post-budget.",
        "product": "MF", "day": 0, "hour": 18, "sentiment": "negative",
    },
    {
        "id": 9010, "tag": "CC Airport Lounge Viral",
        "content": "Airport lounge access viral moment: influencer 'Broke to First Class' Reel hits 8M views Feb 2. CC lounge benefit search +190%. Stable Money CC lounge creative gets organic tailwind.",
        "product": "CC", "day": 1, "hour": 14, "sentiment": "positive",
    },
]

for s in SOCIAL_TRENDS:
    ts = FEB1_TS + s["day"] * 24 * HOUR_SEC + s["hour"] * HOUR_SEC
    meta = feather_db.Metadata()
    meta.timestamp    = ts
    meta.importance   = 0.75
    meta.type         = feather_db.ContextType.EVENT
    meta.source       = "social_intelligence"
    meta.content      = s["content"]
    meta.namespace_id = "stable_money"
    meta.entity_id    = "social"
    meta.set_attribute("entity_type", "social_trend")
    meta.set_attribute("tag",         s["tag"])
    meta.set_attribute("product",     s["product"])
    meta.set_attribute("day",         str(s["day"]))
    meta.set_attribute("hour",        str(s["hour"]))
    meta.set_attribute("sentiment",   s["sentiment"])

    db.add(id=s["id"], vec=embed_text(s["content"] + " " + s["tag"]), meta=meta)
    db.link(from_id=s["id"], to_id=PRODUCT_IDS[s["product"]], rel_type="references", weight=0.75)

print(f"Added {len(SOCIAL_TRENDS)} social media trend nodes")

# ─── Industry Event Nodes ──────────────────────────────────────────────────────
EVENTS = [
    {
        "id": 9101,
        "content": "Union Budget 2026 announced Feb 1 11:00 AM. Key provisions: (1) FD interest up to ₹1.5L tax-free for seniors. (2) LTCG on equity unchanged at 12.5%. (3) STT on F&O up 25%. (4) Corporate bond withholding tax reduced.",
        "label": "Budget 2026 Announcement",
        "day": 0, "hour": 11,
        "links": [PRODUCT_IDS["FD"], PRODUCT_IDS["Bond"], PRODUCT_IDS["MF"], PRODUCT_IDS["CC"]],
    },
    {
        "id": 9102,
        "content": "RBI MPC minutes released Feb 2: repo rate held at 6.25%. Forward guidance: cautiously accommodative. Bond market positive. FD rates expected to stay elevated 6–9 months.",
        "label": "RBI Policy Signal",
        "day": 1, "hour": 10,
        "links": [PRODUCT_IDS["FD"], PRODUCT_IDS["Bond"]],
    },
    {
        "id": 9103,
        "content": "Valentine's Week 2026 starts Feb 7. Creative strategy signal: 'Gift financial security' angle. CC wallet + FD joint-holder offer opportunity. Pre-flight creative rotation advised from Feb 3.",
        "label": "Valentine's Week Signal",
        "day": 1, "hour": 8,
        "links": [PRODUCT_IDS["CC"], PRODUCT_IDS["FD"]],
    },
]

for e in EVENTS:
    ts = FEB1_TS + e["day"] * 24 * HOUR_SEC + e["hour"] * HOUR_SEC
    meta = feather_db.Metadata()
    meta.timestamp    = ts
    meta.importance   = 1.0
    meta.type         = feather_db.ContextType.EVENT
    meta.source       = "industry_event"
    meta.content      = e["content"]
    meta.namespace_id = "stable_money"
    meta.entity_id    = "event"
    meta.set_attribute("entity_type", "industry_event")
    meta.set_attribute("label",       e["label"])
    meta.set_attribute("day",         str(e["day"]))
    meta.set_attribute("hour",        str(e["hour"]))

    db.add(id=e["id"], vec=embed_text(e["content"]), meta=meta)
    for lid in e["links"]:
        db.link(from_id=e["id"], to_id=lid, rel_type="caused_by", weight=1.0)

# Cross-link: competitor actions caused by industry event
db.link(from_id=8001, to_id=9101, rel_type="caused_by", weight=0.9)
db.link(from_id=8002, to_id=9101, rel_type="caused_by", weight=0.9)
db.link(from_id=8006, to_id=9101, rel_type="caused_by", weight=0.8)
db.link(from_id=8007, to_id=9101, rel_type="caused_by", weight=0.8)
db.link(from_id=8003, to_id=9103, rel_type="caused_by", weight=0.7)
db.link(from_id=9005, to_id=9103, rel_type="supports",  weight=0.8)
db.link(from_id=9001, to_id=9101, rel_type="derived_from", weight=0.95)
db.link(from_id=9002, to_id=9101, rel_type="derived_from", weight=0.85)
db.link(from_id=9004, to_id=9101, rel_type="derived_from", weight=0.85)
db.link(from_id=9009, to_id=9101, rel_type="caused_by", weight=0.8)
db.link(from_id=9003, to_id=9102, rel_type="supports",  weight=0.7)

print(f"Added {len(EVENTS)} industry event nodes")

db.save()
print(f"\nDB saved → {DB_PATH}")

total_nodes = (len(PRODUCT_IDS) + total_ads + (snap_counter - 3001) +
               len(creative_intel_nodes) + len(COMPETITORS) + len(SOCIAL_TRENDS) + len(EVENTS))
print(f"Total nodes: {total_nodes}")

# ─── Sample queries ─────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("  INTELLIGENCE QUERIES")
print("─"*60)

def search_vec(text, k=5):
    return db.search(embed_text(text), k=k)

print("\n[Q1] Budget Day FD Performance — context_chain from FD budget spike hour")
q_vec = embed_perf(ctr=1.8, roas=2.5, cpm=80, impressions=12000,
                   hour=14, day=0, product="FD", budget_impact=0.85)
chain = db.context_chain(q_vec, k=3, hops=2, modality="text")
for node in sorted(chain.nodes, key=lambda n: n.hop)[:8]:
    m = node.metadata
    etype = m.get_attribute("entity_type")
    if etype in ("industry_event","competitor_intel","social_trend","creative_intel"):
        print(f"  hop={node.hop} [{etype}] {m.content[:90]}")

print("\n[Q2] Competitor response to FD Budget spike")
results = search_vec("budget FD competitor bajaj groww rate campaign", k=8)
for r in results:
    m = r.metadata
    if m.get_attribute("entity_type") in ("competitor_intel","industry_event"):
        print(f"  score={r.score:.3f} [{m.get_attribute('competitor') or m.get_attribute('label')}]")
        print(f"    {m.content[:90]}")

print("\n[Q3] Social trend signal for MF — negative sentiment")
results = search_vec("mutual fund LTCG sentiment outflow negative", k=8)
for r in results:
    m = r.metadata
    if m.get_attribute("entity_type") == "social_trend":
        print(f"  [{m.get_attribute('tag')}] sentiment={m.get_attribute('sentiment')} {m.content[:80]}")

print("\n[Q4] Creative intelligence for FD — best hook + emotional appeal")
results = search_vec("FD creative hook emotional appeal visual style influencer", k=10)
for r in results[:5]:
    m = r.metadata
    if m.get_attribute("entity_type") == "creative_intel":
        print(f"  CTR_base={m.get_attribute('ctr','?')} Hook: {m.get_attribute('hook','N/A')[:60]}")
        print(f"    Emotional: {m.get_attribute('emotional_appeal','N/A')[:60]}")

# ─── Compute product-level hourly aggregates for chart ─────────────────────────
# Aggregate CTR + ROAS per product per hour (average over all ads in product)
hourly_agg = {}  # product → {hour_idx → {ctr_avg, roas_avg, n}}
for prod in PRODUCTS:
    hourly_agg[prod] = defaultdict(lambda: {"ctr_sum": 0.0, "roas_sum": 0.0, "n": 0})

for (ad_fid, day, hour), snap_id in snaps_by_ad_hour.items():
    prod = feather_ad_ids[ad_fid]["prod"]
    meta = db.get_metadata(snap_id)
    hour_idx = day * 24 + hour
    hourly_agg[prod][hour_idx]["ctr_sum"]  += safe_float(meta.get_attribute("ctr"))
    hourly_agg[prod][hour_idx]["roas_sum"] += safe_float(meta.get_attribute("roas"))
    hourly_agg[prod][hour_idx]["n"]        += 1

chart_series = {}
for prod in PRODUCTS:
    series = []
    for hidx in range(48):
        d = hourly_agg[prod][hidx]
        n = d["n"] if d["n"] > 0 else 1
        series.append({
            "hour_idx": hidx,
            "ctr":  round(d["ctr_sum"]  / n, 4),
            "roas": round(d["roas_sum"] / n, 3),
        })
    chart_series[prod] = series

# ─── Collect intel nodes for force graph ───────────────────────────────────────
intel_node_ids = (
    list(PRODUCT_IDS.values()) +
    list(feather_ad_ids.keys())[:20] +   # top 20 ads for graph
    creative_intel_nodes +
    [c["id"] for c in COMPETITORS] +
    [s["id"] for s in SOCIAL_TRENDS] +
    [e["id"] for e in EVENTS]
)

graph_nodes = []
graph_edges = []
exported = set()

type_config = {
    "product":        {"color": "#4f8ef7", "r": 22, "label": "Product"},
    "ad":             {"color": "#a78bfa", "r": 14, "label": "Ad"},
    "creative_intel": {"color": "#34d399", "r": 16, "label": "Creative Intel"},
    "competitor_intel":{"color":"#f87171",  "r": 18, "label": "Competitor"},
    "social_trend":   {"color": "#fbbf24", "r": 14, "label": "Social Trend"},
    "industry_event": {"color": "#f472b6", "r": 22, "label": "Industry Event"},
}

for nid in intel_node_ids:
    meta = db.get_metadata(nid)
    etype = meta.get_attribute("entity_type") or "ad"
    cfg = type_config.get(etype, {"color": "#94a3b8", "r": 8, "label": etype})
    label = (meta.get_attribute("tag") or
             meta.get_attribute("competitor") or
             meta.get_attribute("label") or
             meta.get_attribute("product") or
             meta.content[:30])
    graph_nodes.append({
        "id":    nid,
        "label": label[:40],
        "color": cfg["color"],
        "r":     cfg["r"],
        "type":  etype,
        "content": meta.content[:120],
        "product": meta.get_attribute("product",""),
        "day":     meta.get_attribute("day",""),
        "hour":    meta.get_attribute("hour",""),
        "severity": meta.get_attribute("severity",""),
        "sentiment": meta.get_attribute("sentiment",""),
        "ad_id":   meta.get_attribute("ad_id",""),
        "ctr":     meta.get_attribute("ctr",""),
    })
    exported.add(nid)

# Edges between intel nodes
for nid in exported:
    meta = db.get_metadata(nid)
    for edge in meta.edges:
        if edge.target_id in exported:
            graph_edges.append({
                "source": nid,
                "target": edge.target_id,
                "rel":    edge.rel_type,
                "weight": edge.weight,
            })

# ─── Build HTML ────────────────────────────────────────────────────────────────
print("\nBuilding visualization HTML...")

# Prepare chart data JSON
import json as _json
chart_json   = _json.dumps(chart_series)
nodes_json   = _json.dumps(graph_nodes)
edges_json   = _json.dumps(graph_edges)
colors_json  = _json.dumps(PRODUCT_COLORS)

# Budget event marker hours
BUDGET_MARKERS = [
    {"hour_idx": 11, "label": "Budget Announced", "color": "#f472b6"},
    {"hour_idx": 34, "label": "RBI Signal", "color": "#60a5fa"},
    {"hour_idx": 32, "label": "Valentine's Signal", "color": "#f87171"},
]
markers_json = _json.dumps(BUDGET_MARKERS)

D3_PATH = os.path.join(os.path.dirname(__file__), "../feather_db/d3.min.js")
with open(D3_PATH) as f:
    d3_js = f.read()

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stable Money — Intel Dashboard Feb 1-2, 2026</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0f1117; color:#e2e8f0; font-family:'Inter',system-ui,sans-serif; }}
header {{
  padding:16px 24px; background:#1a1d2e; border-bottom:1px solid #2d3148;
  display:flex; align-items:center; justify-content:space-between;
}}
header h1 {{ font-size:18px; font-weight:700; color:#e2e8f0; }}
header .sub {{ font-size:12px; color:#94a3b8; margin-top:2px; }}
.badge {{ font-size:10px; padding:2px 8px; border-radius:9999px; font-weight:600; }}
.badge-blue {{ background:#1e3a5f; color:#60a5fa; }}
.badge-green {{ background:#14402a; color:#34d399; }}
.tab-bar {{
  display:flex; gap:4px; padding:12px 24px; background:#151821;
  border-bottom:1px solid #2d3148;
}}
.tab {{
  padding:6px 18px; border-radius:6px; cursor:pointer; font-size:13px;
  color:#94a3b8; border:1px solid transparent; transition:all 0.2s;
}}
.tab.active {{ background:#1e3a5f; color:#60a5fa; border-color:#2563eb; }}
.tab:hover:not(.active) {{ background:#1a1d2e; color:#e2e8f0; }}
.panel {{ display:none; padding:24px; }}
.panel.active {{ display:block; }}
/* ── Timeline ── */
#chart-container {{
  background:#151821; border-radius:12px; border:1px solid #2d3148;
  padding:20px; position:relative;
}}
#chart-container h2 {{ font-size:15px; font-weight:600; margin-bottom:16px; color:#e2e8f0; }}
#timeline-svg {{ width:100%; display:block; }}
.legend {{ display:flex; gap:20px; margin-top:12px; flex-wrap:wrap; }}
.legend-item {{ display:flex; align-items:center; gap:6px; font-size:12px; color:#94a3b8; }}
.legend-dot {{ width:12px; height:12px; border-radius:50%; }}
.event-cards {{
  display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; margin-top:20px;
}}
.event-card {{
  background:#151821; border:1px solid #2d3148; border-radius:10px; padding:14px;
}}
.event-card .ec-header {{ font-size:12px; font-weight:700; margin-bottom:6px; }}
.event-card .ec-body {{ font-size:11px; color:#94a3b8; line-height:1.5; }}
/* ── Intel Graph ── */
#graph-container {{
  background:#151821; border-radius:12px; border:1px solid #2d3148; overflow:hidden;
}}
#graph-svg {{ width:100%; height:620px; display:block; cursor:grab; }}
#graph-svg:active {{ cursor:grabbing; }}
.tooltip {{
  position:fixed; background:#1a1d2e; border:1px solid #4f8ef7;
  border-radius:8px; padding:10px 14px; font-size:12px; color:#e2e8f0;
  max-width:300px; pointer-events:none; display:none; z-index:9999;
  box-shadow:0 8px 32px rgba(0,0,0,0.5);
}}
.tooltip strong {{ color:#60a5fa; }}
.filter-bar {{ display:flex; gap:8px; padding:12px; background:#1a1d2e; flex-wrap:wrap; }}
.filter-btn {{
  padding:4px 14px; border-radius:9999px; cursor:pointer; font-size:11px;
  border:1px solid #2d3148; color:#94a3b8; background:transparent; transition:all 0.2s;
}}
.filter-btn.active {{ border-color:#4f8ef7; color:#4f8ef7; background:#0d1f3c; }}
/* ── Stats row ── */
.stats-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px; }}
.stat-card {{
  background:#151821; border:1px solid #2d3148; border-radius:10px; padding:14px;
}}
.stat-card .sv {{ font-size:24px; font-weight:700; color:#4f8ef7; }}
.stat-card .sl {{ font-size:11px; color:#94a3b8; margin-top:2px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Stable Money — Intelligence Dashboard</h1>
    <div class="sub">Feb 1–2, 2026 · Budget Day Time Series · {total_nodes} nodes · Real Meta Ads Data</div>
  </div>
  <div style="display:flex;gap:8px">
    <span class="badge badge-blue">Budget Day</span>
    <span class="badge badge-green">Live Intel</span>
  </div>
</header>

<div class="tab-bar">
  <div class="tab active" onclick="switchTab('timeline',this)">📈 Timeline</div>
  <div class="tab" onclick="switchTab('graph',this)">🕸 Intel Graph</div>
</div>

<div class="panel active" id="panel-timeline">
  <div class="stats-row">
    <div class="stat-card">
      <div class="sv">{total_ads}</div>
      <div class="sl">Real Ads Loaded</div>
    </div>
    <div class="stat-card">
      <div class="sv">{snap_counter - 3001}</div>
      <div class="sl">Hourly Snapshots</div>
    </div>
    <div class="stat-card">
      <div class="sv">{len(COMPETITORS)}</div>
      <div class="sl">Competitor Events</div>
    </div>
    <div class="stat-card">
      <div class="sv">{len(SOCIAL_TRENDS)}</div>
      <div class="sl">Social Trends</div>
    </div>
  </div>

  <div id="chart-container">
    <h2>Hourly CTR by Product — Feb 1 (00:00) → Feb 2 (23:00)</h2>
    <svg id="timeline-svg" height="320"></svg>
    <div class="legend" id="legend"></div>
  </div>

  <div class="event-cards" style="margin-top:20px">
    <div class="event-card">
      <div class="ec-header" style="color:#f472b6">⚡ Budget 2026 Announcement — Feb 1, 11:00 AM</div>
      <div class="ec-body">FD interest up to ₹1.5L tax-free for seniors. LTCG on equity unchanged (12.5%). STT on F&O raised 25%. Corporate bond withholding tax reduced. FD CTR surges +45% within 3 hours.</div>
    </div>
    <div class="event-card">
      <div class="ec-header" style="color:#f87171">🏦 Competitor Reactions</div>
      <div class="ec-body">Bajaj Finance 8.85% FD campaign live by 2 PM. Groww activates MF Budget blast. HDFC CC Valentine's offer. Zerodha Bonds retargeting. 8 competitor moves logged across products.</div>
    </div>
    <div class="event-card">
      <div class="ec-header" style="color:#fbbf24">📱 Social Media Signal</div>
      <div class="ec-body">#Budget2026 hits 4.2M tweets by 5 PM. #FDvsMF trending. South India vernacular FD queries +210%. Valentine's Week creative opportunity emerging. MF sentiment turns cautiously negative.</div>
    </div>
    <div class="event-card">
      <div class="ec-header" style="color:#60a5fa">🏛 RBI Signal — Feb 2, 10:00 AM</div>
      <div class="ec-body">Repo rate held at 6.25%. Accommodative forward guidance. Bond market positive — FD rates expected elevated for 6-9 months. FD narrative strengthened heading into Feb 2.</div>
    </div>
  </div>
</div>

<div class="panel" id="panel-graph">
  <div id="graph-container">
    <div class="filter-bar">
      <button class="filter-btn active" onclick="filterGraph('all',this)">All</button>
      <button class="filter-btn" onclick="filterGraph('product',this)">Products</button>
      <button class="filter-btn" onclick="filterGraph('industry_event',this)">Events</button>
      <button class="filter-btn" onclick="filterGraph('competitor_intel',this)">Competitors</button>
      <button class="filter-btn" onclick="filterGraph('social_trend',this)">Social</button>
      <button class="filter-btn" onclick="filterGraph('creative_intel',this)">Creative Intel</button>
      <button class="filter-btn" onclick="filterGraph('fd',this)">FD</button>
      <button class="filter-btn" onclick="filterGraph('cc',this)">CC</button>
      <button class="filter-btn" onclick="filterGraph('bond',this)">Bond</button>
      <button class="filter-btn" onclick="filterGraph('mf',this)">MF</button>
    </div>
    <svg id="graph-svg"></svg>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
{d3_js}
</script>
<script>
// ─── Data ───────────────────────────────────────────────────────────────────
const chartSeries  = {chart_json};
const allNodes     = {nodes_json};
const allEdges     = {edges_json};
const productColors = {colors_json};
const MARKERS       = {markers_json};
const PRODUCTS = ["FD","CC","Bond","MF"];

// ─── Tab switching ───────────────────────────────────────────────────────────
function switchTab(name, el) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
  if (name === 'graph') renderGraph('all');
}}

// ─── Timeline chart ──────────────────────────────────────────────────────────
(function buildTimeline() {{
  const svg = d3.select('#timeline-svg');
  const W = document.getElementById('timeline-svg').parentElement.clientWidth - 40;
  const H = 300;
  const margin = {{top:20, right:20, bottom:50, left:52}};
  const iW = W - margin.left - margin.right;
  const iH = H - margin.top - margin.bottom;

  svg.attr('width', W).attr('height', H);
  const g = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

  // Hour labels
  const hourLabels = [];
  for (let i=0;i<48;i++) {{
    const d = i < 24 ? 'Feb 1' : 'Feb 2';
    const h = i % 24;
    hourLabels.push(`${{d}} ${{String(h).padStart(2,'0')}}:00`);
  }}

  const x = d3.scaleLinear().domain([0,47]).range([0, iW]);
  const allCtrs = PRODUCTS.flatMap(p => chartSeries[p].map(d => d.ctr));
  const y = d3.scaleLinear().domain([0, d3.max(allCtrs)*1.15]).range([iH, 0]);

  // Grid
  g.append('g').attr('class','grid')
    .call(d3.axisLeft(y).tickSize(-iW).tickFormat(''))
    .selectAll('line').attr('stroke','#2d3148').attr('stroke-dasharray','3,3');
  g.select('.grid .domain').remove();

  // Axes
  g.append('g').attr('transform',`translate(0,${{iH}})`)
    .call(d3.axisBottom(x)
      .tickValues(d3.range(0,48,6))
      .tickFormat(i => hourLabels[i])
    )
    .selectAll('text').attr('fill','#64748b').attr('font-size','10px')
      .attr('transform','rotate(-25)').attr('text-anchor','end');

  g.append('g').call(d3.axisLeft(y).ticks(6).tickFormat(d => d.toFixed(2)+'%'))
    .selectAll('text').attr('fill','#64748b').attr('font-size','10px');

  g.selectAll('.domain').attr('stroke','#2d3148');

  // Budget markers
  MARKERS.forEach(m => {{
    g.append('line')
      .attr('x1', x(m.hour_idx)).attr('x2', x(m.hour_idx))
      .attr('y1', 0).attr('y2', iH)
      .attr('stroke', m.color).attr('stroke-dasharray','5,3').attr('stroke-width',1.5);
    g.append('text')
      .attr('x', x(m.hour_idx)+4).attr('y', 12)
      .attr('fill', m.color).attr('font-size','10px').text(m.label);
  }});

  // Lines
  const line = d3.line().x(d => x(d.hour_idx)).y(d => y(d.ctr)).curve(d3.curveCatmullRom.alpha(0.5));
  PRODUCTS.forEach(prod => {{
    const color = productColors[prod];
    g.append('path')
      .datum(chartSeries[prod])
      .attr('fill','none').attr('stroke',color).attr('stroke-width',2.5)
      .attr('d', line).attr('opacity',0.9);
  }});

  // Legend
  const legend = document.getElementById('legend');
  PRODUCTS.forEach(prod => {{
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-dot" style="background:${{productColors[prod]}}"></div><span>${{prod}}</span>`;
    legend.appendChild(item);
  }});
}})();

// ─── Intel Force Graph ───────────────────────────────────────────────────────
let simulation, svg, link, node, label, currentFilter = 'all';

function filterGraph(filter, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentFilter = filter;
  renderGraph(filter);
}}

function renderGraph(filter) {{
  const svgEl = document.getElementById('graph-svg');
  svgEl.innerHTML = '';
  if (simulation) simulation.stop();

  const W = svgEl.parentElement.clientWidth;
  const H = 620;

  // Filter nodes
  let visNodes = allNodes;
  if (filter === 'all') {{
    visNodes = allNodes;
  }} else if (['fd','cc','bond','mf'].includes(filter)) {{
    const prodMap = {{fd:'FD',cc:'CC',bond:'Bond',mf:'MF'}};
    const prod = prodMap[filter];
    visNodes = allNodes.filter(n =>
      n.product === prod ||
      n.type === 'industry_event' ||
      (n.type === 'social_trend' && n.product === prod)
    );
  }} else {{
    visNodes = allNodes.filter(n => n.type === filter || n.type === 'product');
  }}

  const visIds = new Set(visNodes.map(n => n.id));
  const visEdges = allEdges.filter(e => visIds.has(e.source) && visIds.has(e.target));

  const nodesData = visNodes.map(n => ({{...n}}));
  const edgesData = visEdges.map(e => ({{...e}}));

  svg = d3.select(svgEl).attr('width', W).attr('height', H);
  const g = svg.append('g');

  // Zoom
  svg.call(d3.zoom().scaleExtent([0.2,4]).on('zoom', ev => g.attr('transform', ev.transform)));

  // Defs
  const defs = svg.append('defs');
  defs.append('marker').attr('id','arr').attr('viewBox','0 -4 8 8')
    .attr('refX',12).attr('markerWidth',6).attr('markerHeight',6).attr('orient','auto')
    .append('path').attr('d','M0,-4L8,0L0,4').attr('fill','#4f8ef750');

  const relColors = {{
    caused_by:'#f472b6', supports:'#34d399', part_of:'#60a5fa',
    derived_from:'#fbbf24', references:'#a78bfa', contradicts:'#f87171', related_to:'#94a3b8',
  }};

  link = g.append('g').selectAll('line').data(edgesData).enter().append('line')
    .attr('stroke', d => relColors[d.rel] || '#4f8ef730')
    .attr('stroke-width', d => 1 + d.weight)
    .attr('stroke-opacity', 0.6)
    .attr('marker-end','url(#arr)');

  node = g.append('g').selectAll('circle').data(nodesData).enter().append('circle')
    .attr('r', d => d.r)
    .attr('fill', d => d.color)
    .attr('fill-opacity', 0.85)
    .attr('stroke', '#0f1117').attr('stroke-width',1.5)
    .on('mouseover', showTooltip)
    .on('mousemove', moveTooltip)
    .on('mouseout',  hideTooltip)
    .call(d3.drag()
      .on('start', (ev,d) => {{ if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
      .on('drag',  (ev,d) => {{ d.fx=ev.x; d.fy=ev.y; }})
      .on('end',   (ev,d) => {{ if (!ev.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }})
    );

  label = g.append('g').selectAll('text').data(nodesData.filter(n => n.r >= 12)).enter().append('text')
    .text(d => d.label.length > 22 ? d.label.slice(0,22)+'…' : d.label)
    .attr('text-anchor','middle').attr('dy',d => d.r + 12)
    .attr('fill','#cbd5e1').attr('font-size','10px').attr('pointer-events','none');

  // ID map for simulation
  const idToIdx = new Map(nodesData.map((n,i) => [n.id, i]));
  edgesData.forEach(e => {{
    e.source = idToIdx.has(e.source) ? nodesData[idToIdx.get(e.source)] : e.source;
    e.target = idToIdx.has(e.target) ? nodesData[idToIdx.get(e.target)] : e.target;
  }});

  simulation = d3.forceSimulation(nodesData)
    .force('link', d3.forceLink(edgesData).distance(d => 80 + (d.source.r||8) + (d.target.r||8)).strength(0.4))
    .force('charge', d3.forceManyBody().strength(d => d.r >= 16 ? -300 : -120))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collide', d3.forceCollide().radius(d => d.r + 8))
    .alphaDecay(0.025)
    .on('tick', () => {{
      link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
          .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
      node.attr('cx',d=>d.x).attr('cy',d=>d.y);
      label.attr('x',d=>d.x).attr('y',d=>d.y);
    }});
}}

function showTooltip(ev, d) {{
  const tt = document.getElementById('tooltip');
  let html = `<strong>${{d.label}}</strong><br><span style="color:#94a3b8;font-size:10px">${{d.type}}</span><br><br>`;
  html += d.content;
  if (d.product)   html += `<br><span style="color:#4f8ef7">Product: ${{d.product}}</span>`;
  if (d.severity)  html += `<br>Severity: <b style="color:${{d.severity==='high'?'#f87171':d.severity==='medium'?'#fbbf24':'#34d399'}}">${{d.severity}}</b>`;
  if (d.sentiment) html += `<br>Sentiment: <b>${{d.sentiment}}</b>`;
  if (d.ctr)       html += `<br>CTR: ${{d.ctr}}%`;
  tt.innerHTML = html;
  tt.style.display = 'block';
  tt.style.left = (ev.clientX + 14) + 'px';
  tt.style.top  = (ev.clientY - 10) + 'px';
}}
function moveTooltip(ev) {{
  const tt = document.getElementById('tooltip');
  tt.style.left = (ev.clientX + 14) + 'px';
  tt.style.top  = (ev.clientY - 10) + 'px';
}}
function hideTooltip() {{
  document.getElementById('tooltip').style.display = 'none';
}}
</script>
</body>
</html>"""

with open(HTML_OUT, "w") as f:
    f.write(html)

print(f"Saved → {HTML_OUT}")
print("\n─" * 60)
print(f"  Summary")
print("─" * 60)
print(f"  Product nodes:          {len(PRODUCT_IDS)}")
print(f"  Real ad nodes:          {total_ads}")
print(f"  Hourly perf snapshots:  {snap_counter - 3001}")
print(f"  Creative intel nodes:   {len(creative_intel_nodes)}")
print(f"  Competitor intel nodes: {len(COMPETITORS)}")
print(f"  Social trend nodes:     {len(SOCIAL_TRENDS)}")
print(f"  Industry event nodes:   {len(EVENTS)}")
print(f"  Total:                  {total_nodes}")
print(f"\n  DB:   {DB_PATH}")
print(f"  HTML: {HTML_OUT}")
