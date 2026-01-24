import time
import numpy as np
import os
from feather_db import DB, Metadata, ContextType, ScoringConfig, FilterBuilder

# Configuration
NUM_RECORDS = 10000
DIM = 128  # Realistic dimension size
DB_PATH = "scale_test.feather"

def generate_random_vector(dim):
    vec = np.random.rand(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)

def run_scale_test():
    print(f"🚀 Starting Scale Test with {NUM_RECORDS} records (Dim={DIM})...")
    
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    db = DB.open(DB_PATH, DIM)
    
    # -------------------------------------------------------------------------
    # 1. Ingestion Test
    # -------------------------------------------------------------------------
    print("\n📦 Phase 1: Ingestion")
    start_time = time.time()
    
    for i in range(NUM_RECORDS):
        meta = Metadata()
        meta.timestamp = int(time.time()) - np.random.randint(0, 86400 * 30) # Random time last 30 days
        meta.importance = np.random.random()
        meta.type = ContextType(np.random.randint(0, 4)) # Random type
        meta.source = "benchmark"
        meta.content = f"Record {i} content payload"
        
        # Add tags effectively
        tags = []
        if i % 2 == 0: tags.append("even")
        if i % 10 == 0: tags.append("milestone")
        meta.tags_json = str(tags).replace("'", '"')
        
        vec = generate_random_vector(DIM)
        db.add(i, vec, meta)
        
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{NUM_RECORDS} records...", end="\r")
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✅ Ingestion Complete: {duration:.2f}s ({NUM_RECORDS/duration:.0f} records/sec)")
    
    # -------------------------------------------------------------------------
    # 2. Query Latency Test
    # -------------------------------------------------------------------------
    print("\n⚡ Phase 2: Query Performance")
    
    # Warmup
    q_vec = generate_random_vector(DIM)
    db.search(q_vec, k=10)
    
    latencies = []
    
    # Test A: Unfiltered
    start_time = time.time()
    for _ in range(100):
        db.search(q_vec, k=10)
    avg_latency = (time.time() - start_time) / 100 * 1000
    print(f"   Simple Search (k=10): {avg_latency:.3f} ms")
    
    # Test B: Filtered (Type = FACT)
    fb = FilterBuilder()
    f_fact = fb.types(ContextType.FACT).build()
    start_time = time.time()
    for _ in range(100):
        db.search(q_vec, k=10, filter=f_fact)
    avg_latency = (time.time() - start_time) / 100 * 1000
    print(f"   Filtered Search (Type=FACT): {avg_latency:.3f} ms")
    
    # Test C: Complex Filter + Scoring
    f_complex = fb.types([ContextType.FACT, ContextType.EVENT]).min_importance(0.8).build()
    scoring = ScoringConfig(half_life=86400 * 7, weight=0.5)
    
    start_time = time.time()
    for _ in range(100):
        db.search(q_vec, k=10, filter=f_complex, scoring=scoring)
    avg_latency = (time.time() - start_time) / 100 * 1000
    print(f"   Complex Search (Filter+Score): {avg_latency:.3f} ms")
    
    # -------------------------------------------------------------------------
    # 3. Persistence Test
    # -------------------------------------------------------------------------
    print("\n💾 Phase 3: Persistence Size")
    db.save()
    file_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"   Database Size: {file_size_mb:.2f} MB")
    
    print("\n✅ Scale Test Passed.")

if __name__ == "__main__":
    run_scale_test()
