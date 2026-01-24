import feather_db
from feather_db import DB, ContextType, Metadata, ScoringConfig, FilterBuilder
import numpy as np
import time

def embed(text, dim=768):
    # Dummy embedding function: use hash of text to seed a random vector
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    np.random.seed(int.from_bytes(h[:4], "little"))
    return np.random.random(dim).astype(np.float32)

def main():
    print("🚀 Feather DB - Phase 2: Context Engine Example")
    print("==============================================")

    # 1. Open database
    dim = 768
    db = DB.open("context.feather", dim=dim)

    # 2. Add context with full metadata
    # Facts
    meta1 = Metadata()
    meta1.content = "Ashwath is building Feather.ai"
    meta1.type = ContextType.FACT
    meta1.source = "onboarding"
    meta1.importance = 0.9
    meta1.tags_json = '["founder", "company"]'
    meta1.timestamp = int(time.time())
    db.add(id=1, vec=embed(meta1.content), meta=meta1)

    # Preferences
    meta2 = Metadata()
    meta2.content = "User prefers concise responses"
    meta2.type = ContextType.PREFERENCE
    meta2.source = "chat:abc123"
    meta2.importance = 0.9
    meta2.tags_json = '["communication", "style"]'
    meta2.timestamp = int(time.time())
    db.add(id=2, vec=embed(meta2.content), meta=meta2)

    # Events (simulating an older event)
    meta3 = Metadata()
    meta3.content = "Had a meeting with WPP about Q1 campaigns"
    meta3.type = ContextType.EVENT
    meta3.source = "calendar:sync"
    meta3.importance = 0.9
    meta3.timestamp = 1736438400  # Jan 9, 2024 (approx)
    meta3.tags_json = '["meeting", "wpp", "campaigns"]'
    db.add(id=3, vec=embed(meta3.content), meta=meta3)

    # 3. Smart Retrieval (Time + Similarity + Importance)
    query_text = "What about WPP?"
    print(f"\n🔍 Searching for: '{query_text}'")
    print("Note: With dummy embeddings, 'Feather.ai' wins because it is newer (100% recency) vs. WPP (~1 year old).")
    
    # Construction of filter
    fb = FilterBuilder()
    filter_both = fb.types([ContextType.FACT, ContextType.EVENT]).build()
    
    # Scoring configuration (30% weight to recency)
    scoring = ScoringConfig(half_life=30, weight=0.3)
    
    query_vec = embed(query_text)
    results = db.search(query_vec, k=5, filter=filter_both, scoring=scoring)

    for r in results:
        print(f"[{r.metadata.type}] {r.metadata.content} (score: {r.score:.3f})")

    # 4. Filtered search (Domain logic: Only show WPP)
    print("\n🎯 Domain Filter: Searching only within 'calendar:sync' source...")
    filter_wpp = FilterBuilder().source("calendar:sync").build()
    results_wpp = db.search(query_vec, k=5, filter=filter_wpp)
    
    for r in results_wpp:
        print(f"[{r.metadata.type}] {r.metadata.content} (score: {r.score:.3f})")

    # 5. Similarity Only (No time weighting)
    print("\n⚖️ Similarity Only: Searching without temporal weight...")
    results_sim = db.search(query_vec, k=5, filter=filter_both, scoring=ScoringConfig(weight=0.0))
    for r in results_sim:
        print(f"[{r.metadata.type}] {r.metadata.content} (score: {r.score:.3f})")

    # 5. Persistence test
    print("\n💾 Saving and reloading...")
    db.save()
    
    db_reloaded = DB.open("context.feather", dim=dim)
    meta = db_reloaded.get_metadata(1)
    if meta:
        print(f"✅ Successfully reloaded fact: '{meta.content}'")
    else:
        print("❌ Failed to reload metadata")

    # Cleanup
    import os
    # os.remove("context.feather")

if __name__ == "__main__":
    main()
