import feather_db
import numpy as np
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor

# Configuration
DIM = 768
NUM_VECTORS = 100_000
BATCH_SIZE = 10_000
DB_PATH = "stress_test.feather"

def generate_batch(size, dim):
    return np.random.rand(size, dim).astype(np.float32)

def print_memory():
    process = psutil.Process(os.getpid())
    print(f"[Mem] {process.memory_info().rss / 1024 / 1024:.2f} MB")

def run_stress_test():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print(f"=== 🚀 Feather DB Stress Test ({NUM_VECTORS} vectors, {DIM} dim) ===")
    
    # 1. Ingestion
    print("\n[Phase 1] Ingestion Speed Test")
    db = feather_db.DB.open(DB_PATH, dim=DIM)
    
    start_time = time.time()
    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        batch = generate_batch(BATCH_SIZE, DIM)
        for j in range(BATCH_SIZE):
            db.add(i + j, batch[j])
        print(f"   Saved {i + BATCH_SIZE}...")
        print_memory()
    
    duration = time.time() - start_time
    print(f"✅ Ingestion Complete: {duration:.2f}s ({NUM_VECTORS / duration:.0f} vectors/sec)")

    # 2. Latency Test (Single Query)
    print("\n[Phase 2] Search Latency (P99)")
    query = np.random.rand(DIM).astype(np.float32)
    latencies = []
    
    # Warmup
    for _ in range(100): db.search(query, k=10)

    # Measure
    for _ in range(1000):
        t0 = time.perf_counter()
        db.search(query, k=10)
        latencies.append((time.perf_counter() - t0) * 1000) # ms

    latencies.sort()
    p50 = latencies[500]
    p99 = latencies[990]
    print(f"✅ Search Latency: P50={p50:.3f}ms, P99={p99:.3f}ms")

    # 3. Graph/Link Stress
    print("\n[Phase 3] Graph Linking Stress")
    # Link every 10th item to the previous 10 items (dense local web)
    link_start = time.time()
    count = 0
    for i in range(10, 20000, 10): # First 20k
        for j in range(1, 6):
           db.link(i, i-j)
           count += 1
    
    link_duration = time.time() - link_start
    print(f"✅ Created {count} links in {link_duration:.3f}s ({count / link_duration:.0f} links/sec)")

    db.save()
    print("\nDone.")

if __name__ == "__main__":
    run_stress_test()
