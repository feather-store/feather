import numpy as np
from feather_db import DB, Metadata, ContextType, FilterBuilder
import os
import time

def mock_embedding(seed, dim=4):
    """Creates a deterministic random vector for demo purposes."""
    np.random.seed(seed)
    vec = np.random.rand(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)

def run_marketing_demo():
    print("=== 🪶 Feather DB: Marketing Multimodal Demo (Week 1) ===\n")
    
    db_path = "marketing_demo.feather"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Initialize DB (using small dim for demo)
    db = DB.open(db_path, dim=4)
    print("1. [System] Initialized 'Living Context Engine'")

    # --- Scenario: Ingesting Campaigns (Multimodal Pockets) ---
    print("\n2. [Ingestion] Processing Marketing Assets...")
    
    # --- Scenario: Ingesting a Video Ad (Multimodal) ---
    print("\n2. [Ingestion] Processing Video Asset: 'video_ad_summer_drop.mp4'...")
    
    # Simulate processing a 10-second video 
    # We extract a "Key Frame" at 00:05 which shows the product clearly
    video_asset_id = 100
    frame_vec = mock_embedding(100) # Simulating CLIP embedding of Frame 5s
    
    meta_video = Metadata()
    meta_video.content = "Asset: video_ad_summer_drop.mp4 (Frame 00:05)"
    meta_video.source = "s3://marketing-assets/summer-2026/video_ad_summer_drop.mp4"
    meta_video.tags_json = '["summer", "video", "product_reveal"]'
    meta_video.type = ContextType.FACT # It's a fact that this content exists
    
    # Store Visual Signal for the Video
    db.add(video_asset_id, frame_vec, meta_video, modality="visual")
    print(f"   -> Ingested Video Frame (ID {video_asset_id}) from '{meta_video.source}'")

    # --- Scenario: Ingesting an Image Ad ---
    print("\n   [Ingestion] Processing Image Asset: 'insta_story_v1.jpg'...")
    image_asset_id = 200
    image_vec = mock_embedding(200)
    
    meta_image = Metadata()
    meta_image.content = "Asset: insta_story_v1.jpg"
    meta_image.source = "s3://marketing-assets/summer-2026/insta_story_v1.jpg"
    meta_image.tags_json = '["instagram", "story", "static"]'
    
    db.add(image_asset_id, image_vec, meta_image, modality="visual")
    print(f"   -> Ingested Image Ad (ID {image_asset_id})")

    # --- Scenario: Attribution (Graph Linking) ---
    print("\n3. [Attribution] Linking Real-World Performance...")
    
    # Conversion Event: User clicked the VIDEO ad and bought item
    conversion_id = 900
    conv_vec = mock_embedding(900)
    
    meta_conv = Metadata()
    meta_conv.type = ContextType.EVENT
    meta_conv.content = "CONVERSION: Order #8821 ($150.00)"
    meta_conv.importance = 1.0 # Standard importance
    meta_conv.timestamp = int(time.time())
    
    db.add(conversion_id, conv_vec, meta_conv, modality="text")
    
    # LINK the Video Asset (100) to the Conversion (900)
    # This creates the "Attribution Edge"
    db.link(from_id=video_asset_id, to_id=conversion_id)
    print(f"   -> 🔗 LINKED: Video Asset ({video_asset_id}) ==> Conversion Event ({conversion_id})")
    
    # --- Scenario: Multimodal Retrieval ---
    print("\n4. [Retrieval] Marketer Query: 'Show me high-converting summer video moments'")
    
    # Search for visuals similar to our video keyframe
    results = db.search(frame_vec, k=5, modality="visual")
    
    print(f"   > Searching 'visual' index...")
    for res in results:
        print(f"\n   [Result ID {res.id}] match_score: {res.score:.4f}")
        print(f"     Media: {res.metadata.source}")
        print(f"     Context: {res.metadata.content}")
        
        # Check Attribution Links
        if conversion_id in res.metadata.links:
            print(f"     💰 ATTRIBUTION: This asset is linked to Order #8821 ($150.00)!")
        else:
            print(f"     (No direct conversion linked)")

    # --- Scenario: Text Query (Cross-Modal) ---
    print("\n5. [Retrieval] Marketer Query: 'Find assets about winter cozy vibes'")
    # We search the 'text' index but can retrieve the ID which has visual pockets too
    text_query_vec = mock_embedding(201) # Simulating "Winter" text embedding
    
    # Let's add a text signal to the Image Asset (200) so we can find it by text too
    meta_image_text = Metadata()
    meta_image_text.content = "Instagram Story Caption: Cozy winter vibes"
    db.add(200, text_query_vec, meta_image_text, modality="text")
    print("   -> Added Text Signal to Image Asset (200)")

    results_text = db.search(text_query_vec, k=1, modality="text")
    
    for res in results_text:
        print(f"   [Result ID {res.id}] Content: {res.metadata.content}")
        print(f"     > Retrieved via Text Signal")

    # --- Scenario: Adaptive Decay (Sticky Memory) ---
    print("\n6. [Intelligence] Testing Adaptive Decay (Context Stickiness)...")
    # We will 'touch' the Video Asset (100) multiple times to simulate high engagement
    # This should make it decay SLOWER than a non-recalled item if time passes.
    
    print(f"   > Simulating 50 recalls for Video Asset (100)...")
    for _ in range(50):
        db.touch(100) # This updates recall_count
        
    meta_100 = db.get_metadata(100)
    print(f"   > Video Asset (100) Recall Count: {meta_100.recall_count}")
    
    # In a real test, we would fast-forward time.
    # Here we verify that the system is *tracking* the context interaction.
    if meta_100.recall_count >= 50:
         print("   ✓ 'Living Memory' is active: asset is becoming stickier.")
    else:
         print("   ❌ Memory update failed.")

    # --- Verification ---
    print("\n7. [Verification] Checking Persistence...")
    db.save()
    del db
    
    db2 = DB.open(db_path)
    meta_reloaded = db2.get_metadata(100)
    if 900 in meta_reloaded.links:
         print("   ✓ Graph Links successfully persisted.")
    else:
         print("   ❌ Persistence Failed.")

if __name__ == "__main__":
    run_marketing_demo()

