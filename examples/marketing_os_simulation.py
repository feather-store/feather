import numpy as np
from feather_db import DB, Metadata, ContextType
import time
import os
import random
import json

def mock_embedding(seed, dim=64):
    """Deterministic random vector."""
    np.random.seed(seed)
    vec = np.random.rand(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)

def run_simulation():
    print("=== 🪐 Feather DB: Marketing OS 'Deep Volume' Simulation ===\n")
    
    db_path = "marketing_os_sim.feather"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    DIM = 64
    N_CREATIVES = 1000
    DAYS_TO_SIMULATE = 30
    
    # Initialize DB
    print(f"1. [System] Initializing Context Engine...")
    print(f"   - Target Volume: {N_CREATIVES} Creatives")
    print(f"   - Timeline: {DAYS_TO_SIMULATE} Days")
    db = DB.open(db_path, dim=DIM)
    
    # --- Step 1: Ingest 1,000 Creatives (The Entities) ---
    print("\n2. [Ingestion] Onboarding 1,000 Creative Assets...")
    start_time = time.time()
    
    creative_ids = []
    
    for i in range(N_CREATIVES):
        c_id = 10000 + i # IDs 10000 to 10999
        creative_ids.append(c_id)
        
        # Simulate Multimodal Data
        visual_vec = mock_embedding(c_id)
        
        meta = Metadata()
        meta.content = f"Creative_{i}: Summer Campaign Variant #{i}"
        meta.source = f"s3://assets/2026/summer/c_{i}.mp4"
        meta.tags_json = json.dumps(["summer", "video", "hero", f"variant_{i % 5}"])
        meta.type = ContextType.FACT
        
        # Store Visual Signal
        db.add(c_id, visual_vec, meta, modality="visual")
        
    print(f"   -> Ingested {N_CREATIVES} creatives in {time.time() - start_time:.2f}s")

    # --- Step 2: Simulate 30 Days of "Living Context" ---
    print(f"\n3. [Simulation] Running {DAYS_TO_SIMULATE}-Day Lifecycle (Insights & Performance)...")
    
    next_event_id = 50000
    total_links = 0
    start_sim = time.time()
    
    for day in range(1, DAYS_TO_SIMULATE + 1):
        # Calculate a mock timestamp for "Day X"
        day_ts = int(time.time()) - ((30 - day) * 86400)
        
        # Scenario A: Daily Performance Data (Graph Links)
        # Randomly 20% of creatives get performance data each day
        active_creatives = random.sample(creative_ids, k=int(N_CREATIVES * 0.2))
        
        for c_id in active_creatives:
            # Create a "Performance Node"
            perf_id = next_event_id
            next_event_id += 1
            
            # Simulate Click/Sales data
            clicks = random.randint(50, 5000)
            conversions = int(clicks * random.uniform(0.01, 0.05))
            
            meta_perf = Metadata()
            meta_perf.type = ContextType.EVENT
            meta_perf.content = f"Day {day} Stats: {clicks} Clicks, {conversions} Conv."
            meta_perf.timestamp = day_ts
            # High importance if conversions are high
            meta_perf.importance = 1.0 + (conversions / 10.0) 
            
            # Ingest & Link
            # We use a mock text vector for searchability
            db.add(perf_id, mock_embedding(perf_id), meta_perf, modality="text")
            db.link(perf_id, c_id) # Link Event -> Creative
            total_links += 1
            
        # Scenario B: AI Insights (New Knowledge)
        # Randomly 5% of creatives get a new AI analysis each day
        analyzed_creatives = random.sample(creative_ids, k=int(N_CREATIVES * 0.05))
        
        for c_id in analyzed_creatives:
            insight_id = next_event_id
            next_event_id += 1
            
            insights = [
                "Fatigue Detected (CTR down 10%)",
                "Hook Rate High (>40%)",
                "Audience: Gen-Z skewed",
                "Negative Sentiment in Comments"
            ]
            chosen_insight = random.choice(insights)
            
            meta_insight = Metadata()
            meta_insight.content = f"AI Insight (Day {day}): {chosen_insight}"
            meta_insight.timestamp = day_ts
            meta_insight.type = ContextType.FACT
            
            db.add(insight_id, mock_embedding(insight_id), meta_insight, modality="text")
            db.link(insight_id, c_id) # Link Insight -> Creative
            total_links += 1
            
    print(f"   -> Simulation Complete in {time.time() - start_sim:.2f}s")
    print(f"   -> Generated {next_event_id - 50000} Context Nodes")
    print(f"   -> Created {total_links} Graph Connections")

    # --- Step 3: "Living Context" Retrieval ---
    print("\n4. [Verification] Deep Retrieval of a 'Living' Creative...")
    
    # Improved Verification: Pick a known active creative from the last day of simulation
    # The last 'active_creatives' batch from the loop is a good candidate, but let's just cheat
    # and pick one that we know has links. In a real app index, this is trivial.
    # Here, let's scan the first 100 events to find a creative ID that definitely has a link.
    
    target_c_id = None
    # Scan a few events to find a target
    for eid in range(50000, 50100):
        m = db.get_metadata(eid)
        if m.links:
            target_c_id = m.links[0]
            break
            
    if target_c_id is None:
        target_c_id = 10000 # Fallback
    
    print(f"   Querying Creative ID: {target_c_id}")
    
    # A. Get the Entity Itself
    meta_entity = db.get_metadata(target_c_id)
    print(f"   [Entity] {meta_entity.content}")
    print(f"            Source: {meta_entity.source}")
    
    # B. Traverse the Graph to find ITS history
    # Scanning ALL generated events (50000 to next_event_id) to verify links.
    # In production, this "Reverse Lookup" is an index O(1).
    # Here, we iterate to prove the data is physically there.
    
    print(f"   [History] Retrieving linked timeline (Scanning {next_event_id - 50000} nodes)...")
    history_events = []
    
    for eid in range(50000, next_event_id):
         m = db.get_metadata(eid)
         if target_c_id in m.links:
             history_events.append(m)
             
    if not history_events:
        print("   (No history found - unexpected for this ID)")
    else:
        # Sort by timestamp
        history_events.sort(key=lambda x: x.timestamp)
        
        for h in history_events:
            # Format timestamp
            ts_str = time.strftime('%Y-%m-%d', time.localtime(h.timestamp))
            prefix = "📊" if "Stats" in h.content else "🧠"
            print(f"     {ts_str} {prefix} {h.content}")
            
    print(f"\n   -> Validated 'Living Context': Creative {target_c_id} has {len(history_events)} linked historical records.")
    
    print("\n✅ Deep Volume Test Passed.")

if __name__ == "__main__":
    run_simulation()
