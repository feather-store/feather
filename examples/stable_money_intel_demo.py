"""
Stable Money — Intelligence Layer Demo
=======================================
Simulates a performance marketing intelligence graph for a fintech brand
with three products: Fixed Deposit (FD), Credit Card, Bond.

What this demonstrates:
  - Human knowledge / strategic intelligence ingested as nodes
  - Competitor events attributed via typed edges
  - Daily performance data connected to creatives → campaigns → products
  - A "shock" event (competitor cuts price) propagating through the graph
  - context_chain revealing WHY performance dropped
  - Any-direction querying across all layers

Run:
    python setup.py build_ext --inplace
    python3 examples/stable_money_intel_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feather_db
import numpy as np
import time, math, json, random

DB_PATH = "/tmp/stable_money.feather"
DIM = 128

# ─── Vocabulary (financial + marketing + strategic) ──────────────────────────

VOCAB = [
    # Financial
    "fixed deposit","fd","interest rate","maturity","minimum deposit",
    "1000 inr","500 inr","rupee","bond","yield","credit card","lounge",
    "airport","premium","reward","cashback","annual fee","approval","cagr",
    "return","investment","savings","lock-in","tenure","renewal","nri",
    # Marketing
    "ctr","roas","impressions","clicks","spend","conversion","cpc","cpm",
    "reach","frequency","engagement","instagram","google","youtube","meta",
    "performance","creative","campaign","audience","targeting","awareness",
    "consideration","retargeting","video","image","carousel",
    # Strategic
    "acquisition","funnel","entry product","cross-sell","upsell",
    "positioning","competitor","differentiation","value proposition",
    "hook","intent","strategy","counter offer","pricing","threshold",
    "response","launch","reduce","increase","feature","benefit",
    # Entities
    "stable money","finflex","q1","q2","millennial","salaried",
    "business owner","india","drop","decline","growth","spike","anomaly",
]
VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}


def embed_text(text, dim=DIM):
    """TF-IDF style sparse embedding over domain vocabulary."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for w in words:
        # exact match
        if w in VOCAB_INDEX:
            idx = VOCAB_INDEX[w] % dim
            vec[idx] += 1.0
        # substring match (e.g. "fd" in "fixed-deposit")
        for vocab_w, vi in VOCAB_INDEX.items():
            if vocab_w in text.lower():
                vec[vi % dim] += 0.5
    # add small structured hash dims (48-127)
    for i, ch in enumerate(text[:40]):
        vec[48 + (ord(ch) * 7 + i * 13) % 80] += 0.1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def make_perf_vec(ctr, roas, spend, impressions, day, product_id):
    """128-dim structured performance embedding."""
    vec = np.zeros(DIM, dtype=np.float32)
    vec[0]  = min(ctr / 0.05, 1.0)          # CTR tier
    vec[1]  = min(roas / 5.0, 1.0)           # ROAS tier
    vec[2]  = min(spend / 50000, 1.0)        # spend tier
    vec[3]  = min(impressions / 500000, 1.0) # reach tier
    vec[4]  = (day % 7) / 7.0               # day of week
    vec[5]  = (day % 30) / 30.0             # day of month
    # product signal
    vec[6 + product_id] = 1.0
    # add noise
    noise = np.random.RandomState(day * 31 + product_id).rand(DIM).astype(np.float32) * 0.05
    vec += noise
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ─── Node ID registry ─────────────────────────────────────────────────────────

class IDs:
    # Products
    FD           = 1
    CREDIT_CARD  = 2
    BOND         = 3

    # Campaigns (10-29)
    FD_AWARENESS   = 10
    FD_PERF        = 11
    FD_RETARGET    = 12
    CC_ACQUISITION = 13
    CC_REWARDS     = 14
    BOND_YIELD     = 15

    # Creatives (100-129)
    FD_LOUNGE_VIDEO   = 100   # "Open FD, get airport lounge access"
    FD_1000_BANNER    = 101   # "Start with just ₹1000"
    FD_INTEREST_REEL  = 102   # "8.5% interest — India's best FD"
    CC_LOUNGE_VIDEO   = 103   # "Unlimited lounge access with our CC"
    CC_CASHBACK_POST  = 104   # "5% cashback on all spends"
    BOND_YIELD_POST   = 105   # "11% yield — invest in bonds"

    # Human intelligence / strategy (200-229)
    STRATEGY_FD_CC_FUNNEL   = 200  # FD is entry intent to get CC
    STRATEGY_LOUNGE_HOOK    = 201  # airport lounge is the acquisition hook
    STRATEGY_1000_INR_POS   = 202  # 1000 INR is deliberate premium positioning
    STRATEGY_CC_UPSELL      = 203  # CC customers → bond investors
    STRATEGY_COUNTER_OFFER  = 204  # Response to FinFlex: FD + CC combo
    INTEL_AUDIENCE_INSIGHT  = 205  # Millennials: lounge > interest rate
    INTEL_CHANNEL_INSIGHT   = 206  # Instagram outperforms Google for FD
    INTEL_CREATIVE_INSIGHT  = 207  # Video > image for FD acquisition

    # Competitor events (300-319)
    COMP_FINFLEX_500_INR    = 300  # FinFlex cuts minimum FD to ₹500
    COMP_FINFLEX_CAMPAIGN   = 301  # FinFlex launches "₹500 se shuru" campaign
    COMP_PAYTM_FD_LAUNCH    = 302  # Paytm relaunches FD product
    COMP_HDFC_LOUNGE_AD     = 303  # HDFC runs lounge ad on same audiences

    # Performance snapshots: 1000 + (day * 10) + creative_offset
    PERF_BASE = 1000


def perf_id(day, creative_offset):
    return IDs.PERF_BASE + day * 10 + creative_offset


# ─── Build DB ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Stable Money — Intelligence Layer Demo")
print("=" * 60)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

db = feather_db.DB.open(DB_PATH, dim=DIM)
NOW = int(time.time())
DAY = 86400

# ── 1. Products ───────────────────────────────────────────────────────────────
print("\n[1/6] Ingesting products...")

products = [
    (IDs.FD, "Fixed Deposit",
     "Stable Money Fixed Deposit: 8.5% interest rate, minimum ₹1000, tenures 3-60 months, "
     "premature withdrawal allowed, auto-renewal option, India's safest investment"),
    (IDs.CREDIT_CARD, "Credit Card",
     "Stable Money Credit Card: unlimited airport lounge access, 5% cashback on spends, "
     "zero annual fee first year, credit limit up to 5L, instant approval for FD holders"),
    (IDs.BOND, "Bond",
     "Stable Money Bond: 11% yield, minimum ₹10000, 24-month lock-in, "
     "rated AA+, quarterly interest payout, ideal for salaried investors"),
]

for pid, name, desc in products:
    meta = feather_db.Metadata()
    meta.timestamp = NOW - 90 * DAY
    meta.importance = 1.0
    meta.type = feather_db.ContextType.FACT
    meta.source = "product_catalog"
    meta.content = desc
    meta.namespace_id = "stable_money"
    meta.entity_id = "product"
    meta.set_attribute("product_name", name)
    meta.set_attribute("product_code", ["fd","credit_card","bond"][pid-1])
    db.add(id=pid, vec=embed_text(desc), meta=meta)

# ── 2. Campaigns ──────────────────────────────────────────────────────────────
print("[2/6] Ingesting campaigns...")

campaigns = [
    (IDs.FD_AWARENESS, "FD Awareness Q1",
     "FD brand awareness campaign targeting millennials on Instagram and YouTube. "
     "Positioning FD as entry product leading to credit card acquisition. "
     "Lounge access is primary hook. Budget 5L/month.",
     "fd", "instagram,youtube"),
    (IDs.FD_PERF, "FD Performance",
     "FD performance campaign on Google and Meta targeting salaried professionals. "
     "Focus on 8.5% interest rate and 1000 INR minimum deposit. Direct response.",
     "fd", "google,meta"),
    (IDs.FD_RETARGET, "FD Retargeting",
     "Retargeting campaign for FD website visitors and app users. "
     "Counter competitor messaging with superior interest rate.",
     "fd", "meta,google"),
    (IDs.CC_ACQUISITION, "CC Acquisition",
     "Credit card acquisition campaign. Primary audience: existing FD holders. "
     "Unlimited lounge access as main benefit. Targeting premium millennials.",
     "credit_card", "instagram,meta"),
    (IDs.CC_REWARDS, "CC Rewards Awareness",
     "Cashback and rewards awareness for credit card. Targeting urban spenders.",
     "credit_card", "instagram,youtube"),
    (IDs.BOND_YIELD, "Bond Yield Campaign",
     "Bond investment campaign targeting business owners and NRI investors. "
     "11% yield as primary value proposition.",
     "bond", "google,linkedin"),
]

for cid, name, desc, product, channels in campaigns:
    meta = feather_db.Metadata()
    meta.timestamp = NOW - 85 * DAY
    meta.importance = 0.9
    meta.type = feather_db.ContextType.FACT
    meta.source = "campaign_manager"
    meta.content = desc
    meta.namespace_id = "stable_money"
    meta.entity_id = "campaign"
    meta.set_attribute("campaign_name", name)
    meta.set_attribute("product", product)
    meta.set_attribute("channels", channels)
    db.add(id=cid, vec=embed_text(desc), meta=meta)

# ── 3. Creatives ──────────────────────────────────────────────────────────────
print("[3/6] Ingesting creatives (text + visual modality)...")

creatives = [
    (IDs.FD_LOUNGE_VIDEO, "FD Lounge Video",
     "Open a Fixed Deposit with Stable Money and get unlimited airport lounge access. "
     "Minimum ₹1000. Start your premium banking journey today. Apply now.",
     "fd", "video", IDs.FD_AWARENESS),
    (IDs.FD_1000_BANNER, "FD 1000 INR Banner",
     "Start your investment journey with just ₹1000. Stable Money FD at 8.5% interest. "
     "Safe, secure, and smart. Open FD in 3 minutes.",
     "fd", "image", IDs.FD_PERF),
    (IDs.FD_INTEREST_REEL, "FD Interest Rate Reel",
     "India's best Fixed Deposit rate: 8.5% per annum. Stable Money FD beats inflation. "
     "Lock in your rate today. Minimum ₹1000. Instant activation.",
     "fd", "video", IDs.FD_PERF),
    (IDs.CC_LOUNGE_VIDEO, "CC Lounge Video",
     "Unlimited airport lounge access with Stable Money Credit Card. "
     "Exclusive for FD holders. Zero annual fee. Apply in 2 minutes.",
     "credit_card", "video", IDs.CC_ACQUISITION),
    (IDs.CC_CASHBACK_POST, "CC Cashback Post",
     "5% cashback on all online spends. Stable Money Credit Card rewards you every day. "
     "No cap, no conditions. Apply now.",
     "credit_card", "image", IDs.CC_REWARDS),
    (IDs.BOND_YIELD_POST, "Bond Yield Post",
     "Earn 11% yield with Stable Money Bonds. AA+ rated. Quarterly payouts. "
     "Minimum ₹10000. Ideal for NRI and business investors.",
     "bond", "image", IDs.BOND_YIELD),
]

for cid, name, desc, product, fmt, campaign_id in creatives:
    meta = feather_db.Metadata()
    meta.timestamp = NOW - 80 * DAY
    meta.importance = 0.85
    meta.type = feather_db.ContextType.FACT
    meta.source = "creative_studio"
    meta.content = desc
    meta.namespace_id = "stable_money"
    meta.entity_id = "creative"
    meta.set_attribute("creative_name", name)
    meta.set_attribute("product", product)
    meta.set_attribute("format", fmt)
    meta.set_attribute("campaign_id", str(campaign_id))
    db.add(id=cid, vec=embed_text(desc), meta=meta, modality="text")

    # Visual modality — simulate visual feature vector from creative content
    vis_seed = np.random.RandomState(cid).rand(DIM).astype(np.float32)
    vis_seed[0] = 0.8 if fmt == "video" else 0.3   # brightness proxy
    vis_seed[1] = 0.6 if "lounge" in desc.lower() else 0.4
    vis_seed = vis_seed / np.linalg.norm(vis_seed)
    db.add(id=cid, vec=vis_seed, meta=meta, modality="visual")

# ── 4. Human Intelligence / Strategic Knowledge ───────────────────────────────
print("[4/6] Ingesting human intelligence nodes...")

intelligence = [
    (IDs.STRATEGY_FD_CC_FUNNEL, feather_db.ContextType.PREFERENCE, 1.0,
     "Strategic intent: Fixed Deposit is the primary entry product to acquire credit card customers. "
     "FD opens the relationship. CC is the upsell. The funnel is: FD acquisition → CC offer → Bond investment. "
     "All FD creatives must subtly hint at the credit card benefit.",
     "strategy"),

    (IDs.STRATEGY_LOUNGE_HOOK, feather_db.ContextType.PREFERENCE, 0.95,
     "Insight: For target audience (millennials, 25-35, salaried, urban), "
     "airport lounge access is a stronger acquisition hook than interest rate. "
     "Test results show lounge-featured creatives have 2.3x higher CTR than rate-focused creatives. "
     "Prioritize lounge messaging across all FD and CC campaigns.",
     "strategy"),

    (IDs.STRATEGY_1000_INR_POS, feather_db.ContextType.PREFERENCE, 0.95,
     "Positioning decision: Stable Money FD minimum deposit is ₹1000. "
     "This is a deliberate premium positioning signal — not a barrier. "
     "We are NOT competing on lowest minimum deposit. We compete on trust, safety, and product bundle. "
     "Do not reduce minimum deposit below ₹1000 even if competitors go lower.",
     "strategy"),

    (IDs.STRATEGY_CC_UPSELL, feather_db.ContextType.PREFERENCE, 0.9,
     "Cross-sell strategy: CC holders with 6+ months tenure are the highest-propensity bond investors. "
     "Bond campaign should target this cohort specifically. "
     "Messaging: from premium banking to wealth creation.",
     "strategy"),

    (IDs.STRATEGY_COUNTER_OFFER, feather_db.ContextType.PREFERENCE, 0.98,
     "Counter-strategy response to FinFlex ₹500 FD launch: "
     "Do not reduce our minimum deposit. Instead, run a combo offer: "
     "Open FD ₹1000 → instant CC approval → 3 months free lounge access. "
     "Position ₹1000 as premium, not expensive. Campaign: FD + CC bundle launch.",
     "strategy"),

    (IDs.INTEL_AUDIENCE_INSIGHT, feather_db.ContextType.FACT, 0.85,
     "Audience intelligence: Primary FD audience — millennials aged 25-35, salaried, metro cities. "
     "Key motivations: safety (68%), returns (54%), lounge access (71% among CC-eligible). "
     "Secondary audience: NRI investors seeking INR-denominated returns.",
     "intelligence"),

    (IDs.INTEL_CHANNEL_INSIGHT, feather_db.ContextType.FACT, 0.8,
     "Channel performance intelligence: Instagram Reels outperform Google Search for FD acquisition "
     "by 40% on cost-per-lead. YouTube performs best for bond investment campaigns. "
     "Google Search is best for retargeting FD drop-offs. Meta is efficient for CC lookalike audiences.",
     "intelligence"),

    (IDs.INTEL_CREATIVE_INSIGHT, feather_db.ContextType.FACT, 0.8,
     "Creative intelligence: Video creatives outperform static images for FD acquisition by 2.1x CTR. "
     "Best performing hook: 'Open FD, get airport lounge' — 3.8% CTR. "
     "Worst performing: pure interest rate messaging — 0.9% CTR. "
     "Recommended format: 15s vertical video with lounge visual in first 3 seconds.",
     "intelligence"),
]

for nid, ctx_type, importance, content, entity in intelligence:
    meta = feather_db.Metadata()
    meta.timestamp = NOW - 60 * DAY
    meta.importance = importance
    meta.type = ctx_type
    meta.source = "human_intelligence"
    meta.content = content
    meta.namespace_id = "stable_money"
    meta.entity_id = entity
    db.add(id=nid, vec=embed_text(content), meta=meta)

# ── 5. Competitor Events ──────────────────────────────────────────────────────
print("[5/6] Ingesting competitor intelligence events...")

DAY_21 = NOW - 9 * DAY  # "21 days ago" in simulation = the shock event

competitor_events = [
    (IDs.COMP_FINFLEX_500_INR, DAY_21, 0.99,
     "COMPETITOR EVENT: FinFlex Bank announced reduction of minimum FD deposit to ₹500 INR. "
     "Previous minimum was ₹2000. This undercuts Stable Money's ₹1000 minimum. "
     "FinFlex is positioning as 'most accessible FD in India'. "
     "Risk: price-sensitive segment may shift. Observed same day as FD CTR drop.",
     "competitor_signal", "finflex"),

    (IDs.COMP_FINFLEX_CAMPAIGN, DAY_21 + 3600, 0.9,
     "COMPETITOR EVENT: FinFlex launched campaign '₹500 se shuru' (start with ₹500) on Instagram and YouTube. "
     "Heavy spend observed. Estimated ₹50L campaign budget. Targeting same millennial audience. "
     "Creative format: video, lounge access featured. Direct competitive attack.",
     "competitor_signal", "finflex"),

    (IDs.COMP_PAYTM_FD_LAUNCH, DAY_21 - 5 * DAY, 0.75,
     "COMPETITOR EVENT: Paytm Money relaunched Fixed Deposit product with 8.75% interest rate. "
     "Slightly above Stable Money's 8.5%. No minimum deposit change. "
     "Targeting existing Paytm users. Impact: moderate pressure on interest rate positioning.",
     "competitor_signal", "paytm"),

    (IDs.COMP_HDFC_LOUNGE_AD, DAY_21 + 2 * DAY, 0.7,
     "COMPETITOR EVENT: HDFC Bank running aggressive airport lounge ad campaign on Instagram. "
     "Targeting same lounge-interested audience. May be reducing effectiveness of our lounge hook. "
     "Audience fatigue risk for lounge messaging.",
     "competitor_signal", "hdfc"),
]

for nid, ts, importance, content, entity, competitor in competitor_events:
    meta = feather_db.Metadata()
    meta.timestamp = ts
    meta.importance = importance
    meta.type = feather_db.ContextType.EVENT
    meta.source = "market_intelligence"
    meta.content = content
    meta.namespace_id = "stable_money"
    meta.entity_id = entity
    meta.set_attribute("competitor", competitor)
    meta.set_attribute("severity", "critical" if importance > 0.9 else "moderate")
    db.add(id=nid, vec=embed_text(content), meta=meta)

# ── 6. Daily Performance Snapshots (30 days) ─────────────────────────────────
print("[6/6] Ingesting 30 days of performance data...")

# Performance baseline per creative
baselines = {
    IDs.FD_LOUNGE_VIDEO:  {"ctr": 0.038, "roas": 3.2, "spend": 45000, "impressions": 420000},
    IDs.FD_1000_BANNER:   {"ctr": 0.018, "roas": 2.1, "spend": 22000, "impressions": 280000},
    IDs.FD_INTEREST_REEL: {"ctr": 0.022, "roas": 2.4, "spend": 30000, "impressions": 310000},
    IDs.CC_LOUNGE_VIDEO:  {"ctr": 0.041, "roas": 4.1, "spend": 55000, "impressions": 500000},
    IDs.CC_CASHBACK_POST: {"ctr": 0.025, "roas": 3.0, "spend": 35000, "impressions": 350000},
    IDs.BOND_YIELD_POST:  {"ctr": 0.019, "roas": 5.2, "spend": 40000, "impressions": 200000},
}

creative_offsets = {
    IDs.FD_LOUNGE_VIDEO: 0,
    IDs.FD_1000_BANNER: 1,
    IDs.FD_INTEREST_REEL: 2,
    IDs.CC_LOUNGE_VIDEO: 3,
    IDs.CC_CASHBACK_POST: 4,
    IDs.BOND_YIELD_POST: 5,
}

product_of = {
    IDs.FD_LOUNGE_VIDEO: "fd", IDs.FD_1000_BANNER: "fd",
    IDs.FD_INTEREST_REEL: "fd", IDs.CC_LOUNGE_VIDEO: "credit_card",
    IDs.CC_CASHBACK_POST: "credit_card", IDs.BOND_YIELD_POST: "bond",
}

SHOCK_DAY = 21  # competitor event fires on day 21
rng = np.random.RandomState(42)

perf_nodes_created = 0
for day in range(1, 31):
    ts = NOW - (30 - day) * DAY
    for creative_id, base in baselines.items():
        offset = creative_offsets[creative_id]
        product = product_of[creative_id]

        # Apply shock: FD creatives drop 20% CTR, 0.4 ROAS after day 21
        shock = 0.0
        recovering = False
        if day >= SHOCK_DAY and product == "fd":
            days_since_shock = day - SHOCK_DAY
            if days_since_shock <= 5:
                shock = -0.22 * (1 - days_since_shock * 0.05)  # drop
            else:
                shock = -0.22 + (days_since_shock - 5) * 0.04  # recovery
                recovering = True

        noise_ctr  = rng.normal(0, 0.002)
        noise_roas = rng.normal(0, 0.1)

        ctr  = max(0.005, base["ctr"]  * (1 + shock) + noise_ctr)
        roas = max(0.5,   base["roas"] * (1 + shock * 0.8) + noise_roas)
        spend = base["spend"] * (1 + rng.normal(0, 0.05))
        impr  = base["impressions"] * (1 + rng.normal(0, 0.08))

        node_id = perf_id(day, offset)
        meta = feather_db.Metadata()
        meta.timestamp = ts
        meta.importance = min(1.0, roas / 5.0)
        meta.type = feather_db.ContextType.EVENT
        meta.source = "ads_platform"
        meta.content = (
            f"Daily performance snapshot: {product} creative day {day}. "
            f"CTR={ctr:.3f} ROAS={roas:.2f} spend=₹{spend:.0f} "
            f"impressions={impr:.0f}"
            + (" [POST-SHOCK RECOVERY]" if recovering else "")
            + (" [COMPETITOR IMPACT]" if shock < -0.1 else "")
        )
        meta.namespace_id = "stable_money"
        meta.entity_id = "perf_snapshot"
        meta.set_attribute("product", product)
        meta.set_attribute("creative_id", str(creative_id))
        meta.set_attribute("day", str(day))
        meta.set_attribute("ctr", f"{ctr:.4f}")
        meta.set_attribute("roas", f"{roas:.2f}")
        meta.set_attribute("shock", "true" if shock < -0.1 else "false")

        product_idx = ["fd","credit_card","bond"].index(product)
        db.add(id=node_id, vec=make_perf_vec(ctr, roas, spend, impr, day, product_idx), meta=meta)
        perf_nodes_created += 1

print(f"    {perf_nodes_created} performance snapshot nodes created.")

# ── 7. Edges ──────────────────────────────────────────────────────────────────
print("\n[edges] Building knowledge graph edges...")

# Product funnel
db.link(IDs.FD, IDs.CREDIT_CARD, rel_type="supports", weight=0.95)
db.link(IDs.CREDIT_CARD, IDs.BOND, rel_type="supports", weight=0.75)

# Campaign → Product
for cid in [IDs.FD_AWARENESS, IDs.FD_PERF, IDs.FD_RETARGET]:
    db.link(cid, IDs.FD, rel_type="part_of", weight=1.0)
for cid in [IDs.CC_ACQUISITION, IDs.CC_REWARDS]:
    db.link(cid, IDs.CREDIT_CARD, rel_type="part_of", weight=1.0)
db.link(IDs.BOND_YIELD, IDs.BOND, rel_type="part_of", weight=1.0)

# Creative → Campaign
for cid, _, _, _, _, camp in creatives:
    db.link(cid, camp, rel_type="part_of", weight=1.0)

# Creative → Strategy
db.link(IDs.FD_LOUNGE_VIDEO, IDs.STRATEGY_LOUNGE_HOOK,    rel_type="supports", weight=0.95)
db.link(IDs.FD_LOUNGE_VIDEO, IDs.STRATEGY_FD_CC_FUNNEL,   rel_type="supports", weight=0.9)
db.link(IDs.FD_1000_BANNER,  IDs.STRATEGY_1000_INR_POS,   rel_type="supports", weight=0.95)
db.link(IDs.CC_LOUNGE_VIDEO, IDs.STRATEGY_LOUNGE_HOOK,    rel_type="supports", weight=0.9)
db.link(IDs.CC_LOUNGE_VIDEO, IDs.STRATEGY_FD_CC_FUNNEL,   rel_type="supports", weight=0.85)

# Strategy → Strategy
db.link(IDs.STRATEGY_FD_CC_FUNNEL, IDs.STRATEGY_LOUNGE_HOOK,   rel_type="derived_from", weight=0.8)
db.link(IDs.STRATEGY_CC_UPSELL,    IDs.STRATEGY_FD_CC_FUNNEL,  rel_type="derived_from", weight=0.85)
db.link(IDs.STRATEGY_COUNTER_OFFER,IDs.STRATEGY_1000_INR_POS,  rel_type="derived_from", weight=0.9)

# Competitor → Strategy (contradicts)
db.link(IDs.COMP_FINFLEX_500_INR,  IDs.STRATEGY_1000_INR_POS,  rel_type="contradicts",  weight=0.98)
db.link(IDs.COMP_FINFLEX_CAMPAIGN, IDs.STRATEGY_LOUNGE_HOOK,   rel_type="contradicts",  weight=0.85)
db.link(IDs.COMP_HDFC_LOUNGE_AD,   IDs.STRATEGY_LOUNGE_HOOK,   rel_type="contradicts",  weight=0.7)

# Counter strategy → Competitor (references)
db.link(IDs.STRATEGY_COUNTER_OFFER, IDs.COMP_FINFLEX_500_INR,  rel_type="references",   weight=0.95)

# Performance drops → Competitor (caused_by)
for day in range(SHOCK_DAY, SHOCK_DAY + 6):
    for offset in range(3):  # FD creatives only
        pid = perf_id(day, offset)
        db.link(pid, IDs.COMP_FINFLEX_500_INR,  rel_type="caused_by", weight=0.85)
        db.link(pid, IDs.COMP_FINFLEX_CAMPAIGN, rel_type="caused_by", weight=0.75)

# Intelligence → Campaigns
db.link(IDs.INTEL_CHANNEL_INSIGHT, IDs.FD_AWARENESS,   rel_type="supports", weight=0.8)
db.link(IDs.INTEL_CREATIVE_INSIGHT,IDs.FD_AWARENESS,   rel_type="supports", weight=0.85)
db.link(IDs.INTEL_AUDIENCE_INSIGHT,IDs.CC_ACQUISITION, rel_type="supports", weight=0.8)

db.save()
print(f"    Graph built. Total nodes: ~{len(db.get_all_ids('text'))} (text modality)")

# ─── QUERIES ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  QUERIES")
print("=" * 60)

from feather_db import FilterBuilder, ScoringConfig

# ── Q1: Performance anomaly — FD CTR drop ─────────────────────────────────────
print("\n[Q1] What happened to FD performance around day 21?")
print("-" * 50)

# Search for shocked performance nodes
shock_query = embed_text("fd performance drop competitor impact ctr decline")
f = FilterBuilder().namespace("stable_money").entity("perf_snapshot").attribute("shock","true").build()
results = db.search(shock_query, k=6, filter=f)

print(f"Found {len(results)} impacted FD performance nodes:")
for r in results[:3]:
    m = r.metadata
    print(f"  Node {r.id} | day={m.get_attribute('day')} product={m.get_attribute('product')} "
          f"CTR={m.get_attribute('ctr')} ROAS={m.get_attribute('roas')}")

# ── Q2: context_chain — trace the WHY ────────────────────────────────────────
print("\n[Q2] context_chain: Starting from FD performance drop → trace causes...")
print("-" * 50)

drop_node = perf_id(SHOCK_DAY, 0)  # day 21, FD lounge video
chain = db.context_chain(make_perf_vec(0.025, 2.1, 40000, 380000, SHOCK_DAY, 0),
                          k=3, hops=2, modality="text")

print(f"Chain returned {len(chain.nodes)} nodes, {len(chain.edges)} edges")
print("Nodes by hop distance:")
for node in sorted(chain.nodes, key=lambda n: n.hop):
    m = node.metadata
    if m:
        label = m.content[:80].replace("\n", " ")
        print(f"  hop={node.hop} id={node.id} score={node.score:.3f}")
        print(f"    └─ {label}...")

# ── Q3: Competitor intelligence sweep ────────────────────────────────────────
print("\n[Q3] What competitor signals are active and what do they contradict?")
print("-" * 50)

f = FilterBuilder().namespace("stable_money").entity("competitor_signal").build()
comp_results = db.search(embed_text("competitor finflex 500 inr launch campaign"), k=5, filter=f)

for r in comp_results:
    m = r.metadata
    print(f"  [{m.get_attribute('competitor').upper()}] severity={m.get_attribute('severity')}")
    print(f"    {m.content[:100]}...")
    edges = db.get_edges(r.id)
    for e in edges:
        target = db.get_metadata(e.target_id)
        if target:
            print(f"    --{e.rel_type}(w={e.weight:.2f})--> {target.content[:60]}...")

# ── Q4: Which creatives align with FD→CC strategy? ───────────────────────────
print("\n[Q4] Which creatives implement the FD→CC acquisition strategy?")
print("-" * 50)

strategy_meta = db.get_metadata(IDs.STRATEGY_FD_CC_FUNNEL)
strategy_vec = db.get_vector(IDs.STRATEGY_FD_CC_FUNNEL, modality="text")
f = FilterBuilder().namespace("stable_money").entity("creative").build()
results = db.search(strategy_vec, k=6, filter=f)

for r in results:
    m = r.metadata
    print(f"  {m.get_attribute('creative_name')} | format={m.get_attribute('format')} "
          f"product={m.get_attribute('product')} score={r.score:.3f}")

# ── Q5: Time-decayed performance — what's performing NOW? ────────────────────
print("\n[Q5] Current FD performance (time-decay weighted, 7-day half-life):")
print("-" * 50)

cfg = ScoringConfig(half_life=7.0, weight=0.65, min=0.0)
f = FilterBuilder().namespace("stable_money").entity("perf_snapshot").attribute("product","fd").build()
perf_query = embed_text("fd performance ctr roas spend")
results = db.search(perf_query, k=8, filter=f, scoring=cfg)

print("  Most relevant recent FD snapshots (recency-weighted):")
for r in results[:5]:
    m = r.metadata
    print(f"  day={m.get_attribute('day'):>3} CTR={m.get_attribute('ctr')} "
          f"ROAS={m.get_attribute('roas')} score={r.score:.3f} "
          f"{'⚠ IMPACTED' if m.get_attribute('shock')=='true' else ''}")

# ── Q6: Human intelligence relevant to current FD situation ─────────────────
print("\n[Q6] What strategic intelligence is most relevant to the FD situation?")
print("-" * 50)

intel_query = embed_text("fd minimum deposit 1000 inr competitor strategy response counter")
f = FilterBuilder().namespace("stable_money").entity("strategy").build()
results = db.search(intel_query, k=5, filter=f)

for r in results:
    m = r.metadata
    print(f"  [score={r.score:.3f}] {m.content[:120]}...")

# ── Q7: Reverse — what attributes to the FinFlex event? ─────────────────────
print("\n[Q7] Reverse query: everything attributed TO the FinFlex ₹500 event")
print("-" * 50)

incoming = db.get_incoming(IDs.COMP_FINFLEX_500_INR)
print(f"  {len(incoming)} nodes reference this competitor event:")
from collections import Counter
rel_counts = Counter(e.rel_type for e in incoming)
for rel, count in rel_counts.most_common():
    print(f"    {rel}: {count} nodes")

print("\n  Sample attributed nodes:")
for e in incoming[:4]:
    m = db.get_metadata(e.source_id)
    if m:
        print(f"  [{e.rel_type}] {m.content[:80]}...")

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Products:              3")
print(f"  Campaigns:             6")
print(f"  Creatives:             6 (text + visual modality)")
print(f"  Human intelligence:    8 nodes (strategy + intel)")
print(f"  Competitor events:     4 nodes")
print(f"  Performance snapshots: {perf_nodes_created}")
print(f"  Total (text modality): {len(db.get_all_ids('text'))}")
print(f"  Shock day: day {SHOCK_DAY} — FinFlex ₹500 FD launch")
print(f"\n  context_chain reveals: perf drop → caused_by → competitor event")
print(f"                         → contradicts → 1000 INR strategy")
print(f"                         → derived_from → counter offer strategy")
print(f"\n  DB saved: {DB_PATH}")
print("=" * 60)
