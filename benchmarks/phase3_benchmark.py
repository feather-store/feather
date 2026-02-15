import numpy as np
from feather_db import DB, Metadata, ContextType
import time
import os
import random

def run_benchmark():
    db_path = "benchmark_phase3.feather"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    DIM = 128 # Standard-ish dimension for performance test
    N_ITEMS = 10_000
    N_LINKS = 20_000
    
    print(f"=== ⚡️ Feather DB: Phase 3 Performance Benchmark ===")
    print(f"Config: {N_ITEMS} Items (Multimodal), {N_LINKS} Graph Links, Dim={DIM}")
    
    # Initialize
    db = DB.open(db_path, dim=DIM)
    
    # --- 1. Ingestion Benchmark ---
    print("\n[1] Benchmarking Ingestion...")
    vectors = np.random.rand(N_ITEMS, DIM).astype(np.float32)
    start_time = time.time()
    
    for i in range(N_ITEMS):
        # Alternate modalities to test "Feather Pockets"
        modality = "visual" if i % 2 == 0 else "text"
        
        meta = Metadata()
        meta.content = f"Item {i} content..."
        meta.type = ContextType.FACT
        
        db.add(i, vectors[i], meta, modality=modality)
        
    ingest_time = time.time() - start_time
    print(f"   -> Ingested {N_ITEMS} multimodal records in {ingest_time:.4f}s")
    print(f"   -> Rate: {N_ITEMS / ingest_time:.0f} vectors/sec")
    
    # --- 2. Graph Linking Benchmark ---
    print("\n[2] Benchmarking Graph Construction...")
    start_time = time.time()
    
    # Create random links (simulating a dense-ish knowledge graph)
    for _ in range(N_LINKS):
        from_id = random.randint(0, N_ITEMS - 1)
        to_id = random.randint(0, N_ITEMS - 1)
        db.link(from_id, to_id)
        
    link_time = time.time() - start_time
    print(f"   -> Created {N_LINKS} links in {link_time:.4f}s")
    print(f"   -> Rate: {N_LINKS / link_time:.0f} links/sec")

    # --- 3. Retrieval + Traversal Benchmark ---
    print("\n[3] Benchmarking 'Search + Graph Walk'...")
    # This simulates: "Find relevant node (Vector Search) AND fetch its linked neighbors (Graph Walk)"
    query = np.random.rand(DIM).astype(np.float32)
    
    start_time = time.time()
    N_QUERIES = 100
    
    total_neighbors_fetched = 0
    
    for _ in range(N_QUERIES):
        # 1. Vector Search (Visual)
        results = db.search(query, k=5, modality="visual")
        
        # 2. Graph Walk (Simulated "Expansion")
        for res in results:
            neighbors = res.metadata.links
            total_neighbors_fetched += len(neighbors)
            # In a real app, we would db.get_metadata(n) here.
            # Getting metadata is a hash map lookup O(1).
            for n_id in neighbors:
                 _ = db.get_metadata(n_id)

    query_time = time.time() - start_time
    avg_latency = (query_time / N_QUERIES) * 1000 # ms
    
    print(f"   -> Ran {N_QUERIES} 'Search+Traverse' queries in {query_time:.4f}s")
    print(f"   -> Avg Latency: {avg_latency:.2f} ms")
    print(f"   -> Total Graph Nodes Visited: {total_neighbors_fetched}")
    
    # --- 4. Adaptive Decay Overhead ---
    print("\n[4] Benchmarking Scorer Overhead (Adaptive Decay)...")
    # Search WITHOUT decay enabled is just raw HNSW.
    # Our search() now ALWAYS runs Scorer logic in C++.
    # So the latency above INCLUDES the decay calculation.
    print(f"   -> (Note: The {avg_latency:.2f} ms latency INCLUDES the real-time Adaptive Decay calculation)")

if __name__ == "__main__":
    run_benchmark()
