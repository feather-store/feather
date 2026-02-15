import numpy as np
from feather_db import DB, Metadata, ContextType
import time
import os
import json

def mock_embedding(seed, dim=4):
    """Creates a deterministic random vector for demo purposes."""
    np.random.seed(seed)
    vec = np.random.rand(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)

def run_advanced_demo():
    print("=== 🪶 Feather DB: Advanced Knowledge Graph Demo ===\n")
    print("Scenario: Correlating 'Visual Insights' to 'Sales Data' via MongoDB Attribution\n")
    
    db_path = "knowledge_graph.feather"
    if os.path.exists(db_path):
        os.remove(db_path)
    db = DB.open(db_path, dim=4)

    # --- 1. The "Source of Truth" (Simulated MongoDB) ---
    # In a real app, Feather IDs map to these MongoDB ObjectIDs
    mongo_products = {
        "prod_001": {"name": "Smart Coffee Maker Pro", "category": "Home"},
        "prod_002": {"name": "Robot Vacuum X1", "category": "Home"}
    }
    print(f"1. [System] Synced with MongoDB Products: {[k for k in mongo_products.keys()]}")

    # --- 2. Ingesting Product Entity (The Anchor) ---
    # Feather ID 1 <-> MongoDB 'prod_001'
    prod_vec = mock_embedding(1) 
    meta_prod = Metadata()
    meta_prod.content = json.dumps(mongo_products["prod_001"])
    meta_prod.source = "mongo://products/prod_001" # <--- ATTRIBUTION
    meta_prod.type = ContextType.FACT
    
    db.add(1, prod_vec, meta_prod, modality="text")
    print(f"   -> Created Knowledge Node 1 for 'Smart Coffee Maker Pro' (Linked to mongo:prod_001)")

    # --- 3. Ingesting "Rich Media & Insights" (The Context) ---
    # A Video Review where a user complains about heating
    # Feather ID 2
    video_vec = mock_embedding(2)
    
    meta_video = Metadata()
    meta_video.source = "s3://reviews/video_review_99.mp4"
    meta_video.content = "AI Insight: User appears frustrated. Audio mentions 'lukewarm water'."
    meta_video.tags_json = '["negative_sentiment", "heating_issue", "video"]'
    
    # Store Visual Signal (Frame) AND Text Signal (Transcript/Insight)
    db.add(2, video_vec, meta_video, modality="visual")
    db.add(2, mock_embedding(3), meta_video, modality="text")
    
    # LINK Video (2) -> Product (1)
    db.link(2, 1)
    print(f"   -> Ingested Video Review (Node 2) with AI Insights. Linked to Product (Node 1).")

    # --- 4. Ingesting Business Event (The Signal) ---
    # Sales dipped specifically for this product
    # Feather ID 3
    event_vec = mock_embedding(4)
    
    meta_event = Metadata()
    meta_event.type = ContextType.EVENT
    meta_event.content = "ALERT: Sales dropped 15% in Q1 2026"
    meta_event.importance = 10.0 # Critical
    
    # LINK Event (3) -> Product (1)
    db.add(3, event_vec, meta_event, modality="text")
    db.link(3, 1)
    print(f"   -> Ingested Sales Alert (Node 3). Linked to Product (Node 1).")

    # --- 5. The "Powerful Context" Query ---
    print("\n--- ANALYST QUERY: 'Why are we losing sales?' ---")
    
    # 1. Search for Negative Sentiment / Issues (using vector search on insights)
    # We query for "issues" or "frustrated" (simulated vector)
    query_vec = mock_embedding(3) # Matches the "Video Insight" vector
    results = db.search(query_vec, k=1, modality="text")
    
    for res in results:
        print(f"\n🔍 FOUND INSIGHT (Node {res.id}): '{res.metadata.content}'")
        print(f"   Source: {res.metadata.source}")
        
        # 2. Graph Traversal (The "Knowledge" step)
        # Who is this insight about?
        links = res.metadata.links
        print(f"   🔗 Graph Links: {links}")
        
        if 1 in links: # Linked to Product
            # Fetch Product Context
            prod_meta = db.get_metadata(1)
            print(f"      └── RELATED PRODUCT: {prod_meta.content}")
            print(f"          └── MongoDB ID: {prod_meta.source}")
            
            # 3. Reverse Lookup (Simulated)
            # In a full graph walk, we'd check what ELSE is linked to Product 1
            # (Simulating finding the Sales Alert which is also linked to Node 1)
            print(f"      └── ⚠️ CROSS-REFERENCE: System detects 'Sales Alert' (Node 3) also linked to this Product.")
            print(f"          >>> CONCLUSION: Quality Issue in Video (Node 2) likely caused Sales Drop (Node 3).")

    # --- 6. Time-Series Knowledge Evolution (The User's latest request) ---
    print("\n--- SCENARIO: Daily Insights & Evolving Conclusions ---")
    
    # Creative Entity (The Constant)
    # Creative Entity (The Constant)
    creative_id = 500
    meta_creative = Metadata()
    meta_creative.content = "Ad Creative: 'Hero Video v1'"
    db.add(creative_id, mock_embedding(500), meta_creative, modality="visual")
    print(f"1. Created Creative Asset (Node {creative_id})")

    # Day 1 Insight (linked to Creative)
    day1_id = 501
    meta_d1 = Metadata()
    meta_d1.content = "Day 1 Insight: High CTR, low conversion."
    meta_d1.timestamp = int(time.time()) - 172800 # 2 days ago
    db.add(day1_id, mock_embedding(501), meta_d1, modality="text")
    db.link(day1_id, creative_id)
    print(f"   -> Added Day 1 Insight (Node {day1_id}). Linked to Creative.")

    # Day 2 Insight (linked to Creative)
    day2_id = 502
    meta_d2 = Metadata()
    meta_d2.content = "Day 2 Insight: Audience dropoff at 0:03s."
    meta_d2.timestamp = int(time.time()) - 86400 # 1 day ago
    db.add(day2_id, mock_embedding(502), meta_d2, modality="text")
    db.link(day2_id, creative_id)
    print(f"   -> Added Day 2 Insight (Node {day2_id}). Linked to Creative.")

    # Day 3 Insight (Today - The Conclusion)
    day3_id = 503
    meta_d3 = Metadata()
    meta_d3.content = "Day 3 Conclusion: Creative fatigue detected. Recommend refreshing intro."
    meta_d3.timestamp = int(time.time()) # Now
    meta_d3.importance = 2.0 # Higher importance
    db.add(day3_id, mock_embedding(503), meta_d3, modality="text")
    db.link(day3_id, creative_id)
    print(f"   -> Added Day 3 Conclusion (Node {day3_id}). Linked to Creative.")

    # Query: Get the latest insights for this Creative
    print(f"\n   [Query] Retrieving evolution of Creative {creative_id}...")
    
    # In a real app, we might search by vector or just traverse "backwards" from the Creative
    # Here, let's simulate a search for "insights" relevant to this creative
    # We simulate this by searching near the concept of "performance insight"
    # and filtering for things linked to our Creative ID (in a real graph traversal)
    
    # For this demo, let's inspect the graph neighbors of ID 500 manually to show the architecture
    # (Since we don't have a full Graph Query Language yet, we do it in app logic)
    
    # NOTE: In Phase 3.2, we will add `get_links(id)` to the Python API. 
    # For now, we verified links exist via the search results in previous steps.
    # Let's search for the "Conclusion" text vector
    
    results = db.search(mock_embedding(503), k=1, modality="text")
    for res in results:
        print(f"   Latest Insight: '{res.metadata.content}'")
        if creative_id in res.metadata.links:
             print(f"   (Correctly attributed to Creative {creative_id})")

if __name__ == "__main__":
    run_advanced_demo()
