"""
marketing_living_context.py — Feather DB v0.4.0 demo

Demonstrates:
  - MarketingProfile to create branded, user-tagged context assets
  - namespace_id (brand) + entity_id (user) filtering
  - attribute-based filtering (channel, campaign)
  - update_importance() feeding real performance signals back in
  - Multi-brand isolation in a single .feather file
  - FilterBuilder fluent API: .namespace().entity().attribute()
"""

import time
import numpy as np
import feather_db
from feather_db import DB, Metadata, ContextType, FilterBuilder, MarketingProfile

DB_PATH = "/tmp/marketing_living_context_v4.feather"
DIM = 128

rng = np.random.default_rng(42)

def rand_vec():
    return rng.random(DIM).astype(np.float32)


# ─────────────────────────────────────────────
# 1. Open DB
# ─────────────────────────────────────────────
db = DB.open(DB_PATH, dim=DIM)
print(f"[feather_db v{feather_db.__version__}] DB opened at {DB_PATH}")


# ─────────────────────────────────────────────
# 2. Ingest — two brands, two users each
# ─────────────────────────────────────────────
records = [
    # brand="nike", user="user_001"
    dict(id=1, brand="nike",    user="user_001", channel="instagram", campaign="Q1_summer",  ctr=0.045, roas=3.2, importance=0.9),
    dict(id=2, brand="nike",    user="user_001", channel="email",     campaign="Q1_summer",  ctr=0.021, roas=2.1, importance=0.7),
    dict(id=3, brand="nike",    user="user_002", channel="instagram", campaign="Q1_summer",  ctr=0.038, roas=2.8, importance=0.85),
    dict(id=4, brand="nike",    user="user_002", channel="tiktok",    campaign="brand_aware", ctr=0.067, roas=4.1, importance=0.95),
    # brand="adidas", user="user_010"
    dict(id=5, brand="adidas",  user="user_010", channel="instagram", campaign="spring_drop", ctr=0.052, roas=3.5, importance=0.88),
    dict(id=6, brand="adidas",  user="user_010", channel="google",    campaign="spring_drop", ctr=0.018, roas=1.9, importance=0.6),
    dict(id=7, brand="adidas",  user="user_011", channel="tiktok",    campaign="collab_2026", ctr=0.071, roas=4.8, importance=0.92),
]

for r in records:
    mp = (MarketingProfile()
          .set_brand(r["brand"])
          .set_user(r["user"])
          .set_channel(r["channel"])
          .set_campaign(r["campaign"])
          .set_ctr(r["ctr"])
          .set_roas(r["roas"]))
    meta = mp.to_metadata()
    meta.timestamp = int(time.time())
    meta.importance = r["importance"]
    meta.type = ContextType.EVENT
    meta.content = f"{r['brand']} / {r['user']} / {r['channel']} — ctr={r['ctr']}"
    db.add(id=r["id"], vec=rand_vec(), meta=meta)

print(f"\nInserted {len(records)} records across 2 brands.")


# ─────────────────────────────────────────────
# 3. Search — all modalities, no filter
# ─────────────────────────────────────────────
query = rand_vec()
results = db.search(query, k=5)
print(f"\n[Unfiltered top-5 results]")
for res in results:
    m = res.metadata
    print(f"  id={res.id:2d}  score={res.score:.4f}  ns={m.namespace_id}  entity={m.entity_id}  ch={m.attributes.get('channel','?')}")


# ─────────────────────────────────────────────
# 4. Filter by brand (namespace_id)
# ─────────────────────────────────────────────
f_nike = FilterBuilder().namespace("nike").build()
results_nike = db.search(query, k=10, filter=f_nike)
print(f"\n[Nike-only results]  ({len(results_nike)} found)")
for res in results_nike:
    print(f"  id={res.id}  entity={res.metadata.entity_id}  ch={res.metadata.attributes.get('channel','?')}")


# ─────────────────────────────────────────────
# 5. Filter by brand + user (namespace + entity)
# ─────────────────────────────────────────────
f_user = FilterBuilder().namespace("nike").entity("user_001").build()
results_user = db.search(query, k=10, filter=f_user)
print(f"\n[Nike / user_001 results]  ({len(results_user)} found)")
for res in results_user:
    print(f"  id={res.id}  ch={res.metadata.attributes.get('channel','?')}  ctr={res.metadata.attributes.get('ctr','?')}")


# ─────────────────────────────────────────────
# 6. Filter by attribute (channel = instagram)
# ─────────────────────────────────────────────
f_ig = FilterBuilder().attribute("channel", "instagram").build()
results_ig = db.search(query, k=10, filter=f_ig)
print(f"\n[Instagram-only results (all brands)]  ({len(results_ig)} found)")
for res in results_ig:
    print(f"  id={res.id}  ns={res.metadata.namespace_id}  entity={res.metadata.entity_id}")


# ─────────────────────────────────────────────
# 7. Filter: adidas + tiktok
# ─────────────────────────────────────────────
f_combo = FilterBuilder().namespace("adidas").attribute("channel", "tiktok").build()
results_combo = db.search(query, k=10, filter=f_combo)
print(f"\n[Adidas + TikTok]  ({len(results_combo)} found)")
for res in results_combo:
    print(f"  id={res.id}  roas={res.metadata.attributes.get('roas','?')}  importance={res.metadata.importance:.2f}")


# ─────────────────────────────────────────────
# 8. Signal feedback — update_importance()
#    Imagine a real-time performance feed bumps high-ROAS records
# ─────────────────────────────────────────────
high_roas_ids = [r["id"] for r in records if r["roas"] >= 4.0]
for rid in high_roas_ids:
    db.update_importance(rid, 1.0)
    print(f"  Boosted importance to 1.0 for id={rid}")

print(f"\n[Post-boost importance check]")
for rid in high_roas_ids:
    meta = db.get_metadata(rid)
    if meta:
        print(f"  id={rid}  importance={meta.importance:.2f}  roas={meta.attributes.get('roas','?')}")


# ─────────────────────────────────────────────
# 9. update_metadata() — full record overwrite
#    Re-tag record 2 to a new campaign
# ─────────────────────────────────────────────
meta2 = db.get_metadata(2)
if meta2:
    mp2 = MarketingProfile(meta2)
    mp2.set_campaign("Q2_reactivation").set_ctr(0.031).set_roas(2.6)
    db.update_metadata(2, mp2.to_metadata())
    updated = db.get_metadata(2)
    print(f"\n[Record 2 after update_metadata]")
    print(f"  campaign={updated.attributes.get('campaign_id','?')}  ctr={updated.attributes.get('ctr','?')}  roas={updated.attributes.get('roas','?')}")


# ─────────────────────────────────────────────
# 10. Save & verify backward compat
# ─────────────────────────────────────────────
db.save()
print(f"\nDB saved. Reloading to verify persistence...")

db2 = DB.open(DB_PATH, dim=DIM)
meta_check = db2.get_metadata(4)
if meta_check:
    mp_check = MarketingProfile(meta_check)
    print(f"  id=4  brand={mp_check.brand_id}  user={mp_check.user_id}  channel={mp_check.channel}  roas={mp_check.roas}")
    assert mp_check.brand_id == "nike",   "namespace_id mismatch"
    assert mp_check.user_id == "user_002", "entity_id mismatch"
    assert mp_check.channel == "tiktok",   "channel attribute mismatch"
    print("  Persistence assertions passed.")
else:
    print("  WARNING: metadata for id=4 not found after reload")

print("\nDone.")
