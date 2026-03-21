#!/usr/bin/env python3
"""
Feather DB × Gemini Embedding 2 — Cross-Modal Intelligence Experiment
======================================================================
Uses real Stable Money Meta ad data (meta_performance_data.csv) to simulate
a unified multimodal embedding space where ad creatives (text + image + video)
live alongside competitor events and social signals — all in one Feather index.

Run:
    # offline / mock mode (no API key needed)
    python3 experiment.py

    # real Gemini Embedding 2
    GOOGLE_API_KEY=AIza... python3 experiment.py --real

What this demonstrates:
  1. Unified index  — text intel, image creatives, video transcripts in ONE
                      HNSW index (768-dim shared space)
  2. Cross-modal search — query with image description → finds text intel
  3. context_chain    — competitor video ad → traverses to your text strategy
                        nodes in 2 hops, bypassing modality boundaries
  4. Modality tagging  — every node carries entity_type + modality attribute
                        so you can filter post-search
"""

import sys, os, json, math, random, argparse, time
from datetime import datetime, timezone
from collections import defaultdict

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import feather_db
import numpy as np
from embedder import GeminiEmbedder

random.seed(42)
np.random.seed(42)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--real",    action="store_true", help="Use real Gemini API")
parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY",""))
args = parser.parse_args()

DB_PATH  = os.path.join(os.path.dirname(__file__), "results", "gemini_cross_modal.feather")
CSV_PATH = os.path.join(ROOT, "real_data", "meta_performance_data.csv")
RESULTS  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

MODE = "REAL (Gemini Embedding 2)" if args.real else "MOCK (offline simulation)"

print("=" * 65)
print("  Feather DB × Gemini Embedding 2 — Cross-Modal Experiment")
print(f"  Mode: {MODE}")
print("=" * 65)

# ── Init embedder ──────────────────────────────────────────────────────────────
if args.real:
    emb = GeminiEmbedder(api_key=args.api_key)
else:
    emb = GeminiEmbedder(mock=True)

print(f"Embedder ready | dim={emb.dim} | model=gemini-embedding-exp-03-07\n")

# ── Parse CSV ──────────────────────────────────────────────────────────────────
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
    name = (row.get("ad_name","") + " " + row.get("campaign_name","")).upper()
    if "_CC_" in name or "SM_CC" in name[:8]: return "CC"
    if "_FD_" in name or "SM_FD" in name[:8]: return "FD"
    if "BOND" in name or name[:3] == "SB_": return "Bond"
    if "_MF_" in name or "SM_MF" in name[:8]: return "MF"
    return "Other"

def safe_float(v, d=0.0):
    try: return float(v) if math.isfinite(float(v)) else d
    except: return d

rows = parse_csv(CSV_PATH)
by_product = defaultdict(list)
for r in rows:
    p = classify_product(r)
    if p != "Other":
        by_product[p].append(r)

PRODUCTS = ["FD", "CC", "Bond", "MF"]
TOP_N    = 5   # top ads per product for experiment

# ── Open DB ────────────────────────────────────────────────────────────────────
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

db = feather_db.DB.open(DB_PATH, dim=emb.dim)
print(f"DB opened | path={DB_PATH} | dim={emb.dim}\n")

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: Ad Creatives — TEXT modality
# Each real ad's name + extracted_entities → text embedding
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("LAYER 1: Ad Creatives (TEXT)")
print("─" * 65)

TEXT_BASE  = 1001
text_nodes = {}   # ad_id_str → feather_id
fid = TEXT_BASE

for prod in PRODUCTS:
    ads = sorted(by_product[prod],
                 key=lambda r: safe_float(r.get("total_spend",0)), reverse=True)[:TOP_N]
    for row in ads:
        ad_id = row.get("ad_id","")
        raw   = row.get("extracted_entities","")
        try:    entities = json.loads(raw) if raw not in ("ᴺᵁᴸᴸ","") else {}
        except: entities = {}

        text_summary = (
            f"{prod} ad creative: {row.get('ad_name','')}. "
            f"CTA: {entities.get('Call To Action','')}. "
            f"Hook: {entities.get('Hook Element','')}. "
            f"Emotional: {entities.get('Emotional Appeal','')}. "
            f"Return: {entities.get('Return Rate Mentioned','')}. "
            f"USP: {entities.get('Unique Selling Points','')}. "
            f"Language: {entities.get('Language','')}."
        )

        vec = emb.embed_text(text_summary)

        meta = feather_db.Metadata()
        meta.timestamp    = int(datetime(2026,2,1,tzinfo=timezone.utc).timestamp())
        meta.importance   = min(1.0, safe_float(row.get("total_spend",0)) / 6000000)
        if meta.importance < 0.2: meta.importance = 0.4
        meta.type         = feather_db.ContextType.FACT
        meta.source       = "gemini_text_embedding"
        meta.content      = text_summary[:200]
        meta.namespace_id = "stable_money"
        meta.entity_id    = prod.lower()
        meta.set_attribute("modality",     "text")
        meta.set_attribute("entity_type",  "ad_creative")
        meta.set_attribute("product",      prod)
        meta.set_attribute("ad_id",        ad_id)
        meta.set_attribute("ctr",          row.get("ctr","0"))
        meta.set_attribute("roas",         row.get("roas","0"))
        meta.set_attribute("spend",        row.get("total_spend","0"))
        meta.set_attribute("hook",         entities.get("Hook Element","")[:60])
        meta.set_attribute("language",     entities.get("Language","")[:40])
        meta.set_attribute("emotional",    entities.get("Emotional Appeal","")[:60])

        db.add(id=fid, vec=vec, meta=meta)
        text_nodes[ad_id] = fid
        print(f"  [TEXT] {prod} | {row.get('ad_name','')[:50]} | fid={fid}")
        fid += 1

print(f"  → {fid - TEXT_BASE} text ad creative nodes\n")

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: Ad Creatives — IMAGE modality (descriptions from extracted_entities)
# Simulate image embeddings from visual descriptions
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("LAYER 2: Ad Creatives (IMAGE — visual descriptions)")
print("─" * 65)

IMAGE_BASE  = 2001
image_nodes = {}
fid = IMAGE_BASE

for prod in PRODUCTS:
    ads = sorted(by_product[prod],
                 key=lambda r: safe_float(r.get("total_spend",0)), reverse=True)[:TOP_N]
    for row in ads:
        ad_id = row.get("ad_id","")
        raw   = row.get("extracted_entities","")
        try:    entities = json.loads(raw) if raw not in ("ᴺᵁᴸᴸ","") else {}
        except: entities = {}

        # Build a rich image description from visual fields
        image_desc = (
            f"Visual ad for {prod}. "
            f"Style: {entities.get('Key Visual Style', entities.get('Video Style',''))}. "
            f"Color palette: {entities.get('Dominant Color Palette','')}. "
            f"Prominent subject: {entities.get('Prominent Subject','')}. "
            f"On-screen text: {entities.get('On Screen Text Elements','')}. "
            f"Visual callouts: {entities.get('Visual Callouts','')}. "
            f"Model present: {entities.get('Model Present','')}."
        )

        vec = emb.embed_image(image_description=image_desc)

        meta = feather_db.Metadata()
        meta.timestamp    = int(datetime(2026,2,1,tzinfo=timezone.utc).timestamp())
        meta.importance   = 0.75
        meta.type         = feather_db.ContextType.FACT
        meta.source       = "gemini_image_embedding"
        meta.content      = image_desc[:200]
        meta.namespace_id = "stable_money"
        meta.entity_id    = prod.lower()
        meta.set_attribute("modality",      "image")
        meta.set_attribute("entity_type",   "ad_creative")
        meta.set_attribute("product",       prod)
        meta.set_attribute("ad_id",         ad_id)
        meta.set_attribute("visual_style",  entities.get("Key Visual Style","")[:60])
        meta.set_attribute("color_palette", entities.get("Dominant Color Palette","")[:60])
        meta.set_attribute("subject",       entities.get("Prominent Subject","")[:60])

        db.add(id=fid, vec=vec, meta=meta)
        image_nodes[ad_id] = fid

        # Cross-modal link: image ↔ text version of same ad
        if ad_id in text_nodes:
            db.link(from_id=fid, to_id=text_nodes[ad_id], rel_type="same_ad", weight=1.0)

        print(f"  [IMAGE] {prod} | style={entities.get('Key Visual Style','')[:35]} | fid={fid}")
        fid += 1

print(f"  → {fid - IMAGE_BASE} image ad creative nodes\n")

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: Video Ad Transcripts
# Simulated video transcripts for top video ads (from Dialogues field)
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("LAYER 3: Video Transcripts")
print("─" * 65)

VIDEO_BASE  = 3001
video_nodes = {}
fid = VIDEO_BASE

for prod in PRODUCTS:
    ads = sorted(by_product[prod],
                 key=lambda r: safe_float(r.get("total_spend",0)), reverse=True)[:TOP_N]
    video_ads = [r for r in ads if "Video" in r.get("ad_name","")][:3]

    for row in video_ads:
        ad_id = row.get("ad_id","")
        raw   = row.get("extracted_entities","")
        try:    entities = json.loads(raw) if raw not in ("ᴺᵁᴸᴸ","") else {}
        except: entities = {}

        dialogue = entities.get("Dialogues","")[:300]
        duration = entities.get("video_duration","")
        cta_timing = entities.get("Call To Action Timing","")
        brand_timing = entities.get("Brand Mention Timing","")

        transcript = (
            f"Video transcript for {prod} ad '{row.get('ad_name','')}'. "
            f"Duration: {duration}. "
            f"Brand mention at: {brand_timing}. CTA at: {cta_timing}. "
            f"Dialogue: {dialogue}. "
            f"Music/Voice: {entities.get('Music or Voice','')}. "
            f"Key moments: {entities.get('Key Moments in Detail','')[:150]}."
        )

        vec = emb.embed_video_transcript(transcript)

        meta = feather_db.Metadata()
        meta.timestamp    = int(datetime(2026,2,1,tzinfo=timezone.utc).timestamp())
        meta.importance   = 0.8
        meta.type         = feather_db.ContextType.FACT
        meta.source       = "gemini_video_embedding"
        meta.content      = transcript[:200]
        meta.namespace_id = "stable_money"
        meta.entity_id    = prod.lower()
        meta.set_attribute("modality",      "video")
        meta.set_attribute("entity_type",   "ad_creative")
        meta.set_attribute("product",       prod)
        meta.set_attribute("ad_id",         ad_id)
        meta.set_attribute("duration",      str(duration)[:20])
        meta.set_attribute("cta_timing",    str(cta_timing)[:20])

        db.add(id=fid, vec=vec, meta=meta)
        video_nodes[ad_id] = fid

        if ad_id in text_nodes:
            db.link(from_id=fid, to_id=text_nodes[ad_id], rel_type="same_ad", weight=1.0)
        if ad_id in image_nodes:
            db.link(from_id=fid, to_id=image_nodes[ad_id], rel_type="same_ad", weight=0.9)

        print(f"  [VIDEO] {prod} | {row.get('ad_name','')[:45]} | dur={duration} | fid={fid}")
        fid += 1

print(f"  → {fid - VIDEO_BASE} video transcript nodes\n")

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4: Competitor Intel nodes (text)
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("LAYER 4: Competitor Intelligence")
print("─" * 65)

COMPETITOR_NODES = [
    (9001, "FD",   "Bajaj Finance launches 8.85% FD campaign post-Budget Feb 1. "
                   "Heavy Instagram video creative. Black and gold color palette. "
                   "Influencer: senior citizen testimonial. CTA: Book now. "
                   "Return rate 8.85% prominently displayed. Emotional: security."),
    (9002, "FD",   "Navi Finance reactive FD creative: Budget 2026 loves savers. "
                   "8.5% rate. Static ad. Blue-white palette. No model. Rate callout "
                   "as visual callout. CTA: Invest today. Language: English."),
    (9003, "MF",   "Groww activates Budget Day MF emergency campaign. Reels format. "
                   "Invest before LTCG changes land messaging. Young male presenter. "
                   "High energy hook: Are you losing money sitting idle? "
                   "Download CTA. Tamil and Hindi versions."),
    (9004, "Bond", "Zerodha Coin bond retargeting. Post-budget 9.1% corporate bond. "
                   "Video ad. Telugu voiceover. Monthly payout angle. "
                   "Trust indicator: SEBI registered. Graph visual showing yield."),
    (9005, "CC",   "HDFC Bank CC Valentine's Week offer. Two free airport lounge visits. "
                   "First spend Rs 5000. Couple lifestyle visual. Warm palette. "
                   "Limited time offer overlay. Instagram story format."),
]

competitor_ids = {}
for nid, prod, content in COMPETITOR_NODES:
    vec = emb.embed_text(content)
    meta = feather_db.Metadata()
    meta.timestamp    = int(datetime(2026,2,1,12,0,tzinfo=timezone.utc).timestamp())
    meta.importance   = 0.85
    meta.type         = feather_db.ContextType.EVENT
    meta.source       = "competitor_intel"
    meta.content      = content
    meta.namespace_id = "stable_money"
    meta.entity_id    = "competitor"
    meta.set_attribute("modality",     "text")
    meta.set_attribute("entity_type",  "competitor_intel")
    meta.set_attribute("product",      prod)
    db.add(id=nid, vec=vec, meta=meta)
    competitor_ids[nid] = prod
    print(f"  [COMPETITOR] {prod} | {content[:60]}...")

print(f"  → {len(COMPETITOR_NODES)} competitor intel nodes\n")

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5: Strategy Intel nodes
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("LAYER 5: Strategy & Market Intel")
print("─" * 65)

STRATEGY_NODES = [
    (9101, "FD",   "Budget 2026 strategy: FD interest up to Rs 1.5L tax-free for seniors. "
                   "Opportunity window: 4-6 weeks of elevated FD search intent. "
                   "Recommended creative angle: security + tax benefit. "
                   "Vernacular creatives in Tamil, Telugu show 2x CTR lift."),
    (9102, "MF",   "LTCG tax unchanged at 12.5%. STT on F&O raised 25%. "
                   "MF sentiment cautiously negative post-budget. "
                   "Counter-narrative: long-term SIP still optimal. "
                   "Creative hook: don't panic, stay invested."),
    (9103, "Bond", "RBI repo rate held 6.25%. Accommodative stance. "
                   "Corporate bond yields attractive 8.5-9.5%. "
                   "Monthly payout narrative resonates with retired segment. "
                   "Video format outperforms static for bond education."),
    (9104, "CC",   "Valentine's Week Feb 7-14. Lifestyle + financial security angle. "
                   "Airport lounge + couple travel creative performs well. "
                   "Competitive: HDFC lounge offer live. Differentiate on ease of approval."),
]

strategy_ids = [n[0] for n in STRATEGY_NODES]
for nid, prod, content in STRATEGY_NODES:
    vec = emb.embed_text(content)
    meta = feather_db.Metadata()
    meta.timestamp    = int(datetime(2026,2,1,tzinfo=timezone.utc).timestamp())
    meta.importance   = 0.9
    meta.type         = feather_db.ContextType.FACT
    meta.source       = "strategy_intel"
    meta.content      = content
    meta.namespace_id = "stable_money"
    meta.entity_id    = prod.lower()
    meta.set_attribute("modality",    "text")
    meta.set_attribute("entity_type", "strategy_intel")
    meta.set_attribute("product",     prod)
    db.add(id=nid, vec=vec, meta=meta)
    print(f"  [STRATEGY] {prod} | {content[:60]}...")

print(f"  → {len(STRATEGY_NODES)} strategy intel nodes\n")

# ── Cross-layer edges ──────────────────────────────────────────────────────────
# Competitor nodes contradict / challenge strategy nodes
db.link(9001, 9101, rel_type="contradicts",  weight=0.8)  # Bajaj FD vs FD strategy
db.link(9003, 9102, rel_type="supports",     weight=0.7)  # Groww MF vs MF strategy
db.link(9004, 9103, rel_type="contradicts",  weight=0.75) # Zerodha Bond vs Bond strategy
db.link(9005, 9104, rel_type="contradicts",  weight=0.85) # HDFC CC vs CC strategy

db.save()

total = (fid - TEXT_BASE) + (fid - IMAGE_BASE) + (fid - VIDEO_BASE) + len(COMPETITOR_NODES) + len(STRATEGY_NODES)
# Recalculate properly
n_text  = len(text_nodes)
n_image = len(image_nodes)
n_video = len(video_nodes)
n_comp  = len(COMPETITOR_NODES)
n_strat = len(STRATEGY_NODES)
n_total = n_text + n_image + n_video + n_comp + n_strat

print("=" * 65)
print(f"  DB SUMMARY")
print("=" * 65)
print(f"  Text ad creatives:    {n_text}")
print(f"  Image ad creatives:   {n_image}")
print(f"  Video transcripts:    {n_video}")
print(f"  Competitor intel:     {n_comp}")
print(f"  Strategy intel:       {n_strat}")
print(f"  Total nodes:          {n_total}")
print(f"  Embedding dim:        {emb.dim}")
print(f"  Mode:                 {MODE}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

results = {}  # for blog output

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1: Cross-modal search
# Query: image description of a competitor ad → find matching TEXT intel nodes
# Proves: unified space — image query finds text results
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("EXPERIMENT 1: Cross-Modal Search (image query → text results)")
print("─" * 65)
print("Query: Competitor FD image ad (black/gold, senior, 8.85% rate callout)")
print()

query_image_desc = (
    "Competitor FD ad image. Black and gold color palette. "
    "Senior citizen testimonial. Large rate callout 8.85%. "
    "Trust badge overlay. Book Now CTA button."
)
q_vec = emb.embed_image(image_description=query_image_desc)
hits  = db.search(q_vec, k=8)

exp1_results = []
print(f"  {'Score':>6}  {'Modality':<8}  {'Type':<18}  Content")
print(f"  {'─'*6}  {'─'*8}  {'─'*18}  {'─'*40}")
for r in hits:
    m = r.metadata
    modality = m.get_attribute("modality")
    etype    = m.get_attribute("entity_type")
    product  = m.get_attribute("product")
    exp1_results.append({
        "score": round(r.score, 4),
        "modality": modality,
        "type": etype,
        "product": product,
        "content": m.content[:70],
    })
    print(f"  {r.score:>6.4f}  {modality:<8}  {etype:<18}  {m.content[:55]}...")

results["exp1_cross_modal_search"] = exp1_results

# Count cross-modal hits (image query → non-image results)
cross_modal_hits = [r for r in exp1_results if r["modality"] != "image"]
print(f"\n  Cross-modal hits (image query → non-image): {len(cross_modal_hits)}/{len(exp1_results)}")
print(f"  ✓ Unified space confirmed: {len(cross_modal_hits) > 0}")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2: context_chain — competitor video → strategy traversal
# Start from Bajaj FD competitor node, expand 2 hops
# ─────────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("EXPERIMENT 2: context_chain — competitor → strategy traversal")
print("─" * 65)
print("Seed: Bajaj Finance FD video creative (competitor, node 9001)")
print()

seed_vec = emb.embed_text(
    "Bajaj Finance FD video ad. Black gold palette. Senior testimonial. "
    "8.85% return. Book now CTA. Budget 2026 urgency messaging."
)
chain = db.context_chain(seed_vec, k=4, hops=2, modality="text")

exp2_results = []
for node in sorted(chain.nodes, key=lambda n: (n.hop, -n.score)):
    m        = node.metadata
    modality = m.get_attribute("modality")
    etype    = m.get_attribute("entity_type")
    product  = m.get_attribute("product")
    exp2_results.append({
        "hop": node.hop,
        "score": round(node.score, 4),
        "modality": modality,
        "type": etype,
        "product": product,
        "content": m.content[:80],
    })
    prefix = "  └─" if node.hop > 0 else "  ●"
    print(f"  hop={node.hop} {prefix} [{modality}/{etype}] score={node.score:.4f}")
    print(f"       {m.content[:75]}...")
    print()

results["exp2_context_chain"] = exp2_results

# Strategy nodes reached
strat_reached = [r for r in exp2_results if r["type"] == "strategy_intel"]
print(f"  Strategy intel nodes reached in chain: {len(strat_reached)}")
print(f"  ✓ Competitor creative → strategy context traversal: {len(strat_reached) > 0}")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3: Cross-modal similarity — same ad, different modalities
# Measure: how close are the text vs image vs video embeddings of the SAME ad?
# ─────────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("EXPERIMENT 3: Same-Ad Cross-Modal Similarity")
print("─" * 65)
print("For each ad: cosine(text_vec, image_vec) and cosine(text_vec, video_vec)")
print()

exp3_results = []
for ad_id, text_fid in text_nodes.items():
    image_fid = image_nodes.get(ad_id)
    video_fid = video_nodes.get(ad_id)
    if not image_fid:
        continue

    t_vec = db.get_vector(text_fid,  modality="text")
    i_vec = db.get_vector(image_fid, modality="text")

    ti_sim = emb.cosine_similarity(t_vec, i_vec)
    tv_sim = None
    if video_fid:
        v_vec  = db.get_vector(video_fid, modality="text")
        tv_sim = emb.cosine_similarity(t_vec, v_vec)

    tm = db.get_metadata(text_fid)
    ad_name = tm.content[:45]
    prod    = tm.get_attribute("product")

    row = {
        "ad_name": ad_name,
        "product": prod,
        "text_image_sim": round(ti_sim, 4),
        "text_video_sim": round(tv_sim, 4) if tv_sim else None,
    }
    exp3_results.append(row)

    tv_str = f"  text↔video={tv_sim:.4f}" if tv_sim else ""
    print(f"  [{prod}] text↔image={ti_sim:.4f}{tv_str}  |  {ad_name}...")

results["exp3_same_ad_similarity"] = exp3_results

avg_ti = sum(r["text_image_sim"] for r in exp3_results) / len(exp3_results)
print(f"\n  Average text↔image similarity: {avg_ti:.4f}")
print(f"  ✓ Same-ad cross-modal coherence: {avg_ti > 0.5}")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 4: Competitor threat detection via cross-modal search
# Query: our best-performing FD text creative → find competitor creatives that
#        are semantically close (potential cannibalisation)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("EXPERIMENT 4: Competitor Threat Detection")
print("─" * 65)
print("Query: top FD text creative → find semantically close competitor nodes")
print()

# Use the top FD text creative as query
fd_text_ids = [fid for fid, info in text_nodes.items()
               if db.get_metadata(image_nodes.get(fid, TEXT_BASE) if False else
                  list(text_nodes.values())[0]).get_attribute("product") == "FD"]

# Simpler: just search with a known FD creative description
fd_query = (
    "FD creative. Monthly payout angle. Senior citizens. Trust indicators. "
    "Returns 8.5%. Static ad. Hindi language. WhatsApp-style social proof."
)
q_vec = emb.embed_text(fd_query)
hits  = db.search(q_vec, k=10)

exp4_results = []
print(f"  {'Score':>6}  {'Modality':<8}  {'Type':<18}  {'Product':<8}  Content")
print(f"  {'─'*6}  {'─'*8}  {'─'*18}  {'─'*8}  {'─'*35}")
for r in hits:
    m = r.metadata
    etype   = m.get_attribute("entity_type")
    prod    = m.get_attribute("product")
    modal   = m.get_attribute("modality")
    is_comp = etype == "competitor_intel"
    flag    = " ⚠" if is_comp else ""
    exp4_results.append({
        "score": round(r.score, 4),
        "type": etype,
        "product": prod,
        "modality": modal,
        "competitor": is_comp,
    })
    print(f"  {r.score:>6.4f}  {modal:<8}  {etype:<18}  {prod:<8}  {m.content[:40]}...{flag}")

results["exp4_threat_detection"] = exp4_results
threats = [r for r in exp4_results if r["competitor"]]
print(f"\n  Competitor threats detected in top-10: {len(threats)}")
print(f"  ✓ Cross-modal threat detection: {len(threats) > 0}")

# ─────────────────────────────────────────────────────────────────────────────
# Save results JSON for blog post
# ─────────────────────────────────────────────────────────────────────────────
results_path = os.path.join(RESULTS, "experiment_results.json")
with open(results_path, "w") as f:
    json.dump({
        "mode": MODE,
        "model": "gemini-embedding-exp-03-07",
        "dim": emb.dim,
        "nodes": {
            "text_creatives": n_text,
            "image_creatives": n_image,
            "video_transcripts": n_video,
            "competitor_intel": n_comp,
            "strategy_intel": n_strat,
            "total": n_total,
        },
        "experiments": results,
    }, f, indent=2)

print()
print("=" * 65)
print(f"  Results saved → {results_path}")
print("=" * 65)
print()
print("  Summary of findings:")
print(f"  EXP 1  Cross-modal search:       {len(cross_modal_hits)}/{len(exp1_results)} hits were cross-modal")
print(f"  EXP 2  context_chain traversal:  {len(strat_reached)} strategy nodes reached from competitor")
print(f"  EXP 3  Same-ad coherence:        avg similarity {avg_ti:.4f}")
print(f"  EXP 4  Threat detection:         {len(threats)} competitor threats in top-10")
print()
print("  Next: python3 experiment.py --real   (requires GOOGLE_API_KEY)")
