import numpy as np
from feather_db import DB, Metadata, ContextType
import os
import time

def test_phase3():
    db_path = "test_phase3.feather"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Open DB with default dim 3 (for simple testing)
    db = DB.open(db_path, dim=3)
    
    # 1. Test Multimodal Pockets
    # Add a visual signal for Product A (ID 1)
    visual_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    meta1 = Metadata()
    meta1.content = "Product A Hero Image"
    db.add(1, visual_vec, meta1, modality="visual")
    
    # Add a textual signal for Product A (ID 1)
    text_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    db.add(1, text_vec, meta1, modality="text")
    
    # 2. Test Record Linking (Graph)
    # Add a conversion event (ID 2)
    conv_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    meta2 = Metadata()
    meta2.type = ContextType.EVENT
    meta2.content = "Conversion of Product A"
    db.add(2, conv_vec, meta2, modality="text")
    
    # Link Product A to Conversion
    db.link(from_id=1, to_id=2)
    print("✓ Linked Product A (1) -> Conversion (2)")
    
    # 3. Test Retrieval and Salience (Touch)
    # Search in text modality
    print("\nSearching text modality for query [0, 1, 0]...")
    results = db.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), k=1, modality="text")
    for r in results:
        print(f"Found: {r.id}, Score: {r.score:.4f}, Content: {r.metadata.content}")
        # touch() is called internally by search() in my current C++ implementation
        
    # Verify metadata updates
    meta = db.get_metadata(1)
    print(f"\nMetadata for Product A (1):")
    print(f" - Recall Count: {meta.recall_count}")
    print(f" - Links: {meta.links}")
    
    assert 2 in meta.links, "Link missing!"
    assert meta.recall_count > 0, "Recall count not updated!"
    
    # 4. Test Persistence
    db.save()
    del db
    
    print("\nReloading DB...")
    db2 = DB.open(db_path)
    meta_reloaded = db2.get_metadata(1)
    print(f"Reloaded Metadata for Product A (1):")
    print(f" - Recall Count: {meta_reloaded.recall_count}")
    print(f" - Links: {meta_reloaded.links}")
    
    assert 2 in meta_reloaded.links, "Link missing after reload!"
    assert meta_reloaded.recall_count > 0, "Recall count missing after reload!"
    
    print("\n✓ Phase 3 Verification Successful!")

if __name__ == "__main__":
    test_phase3()
