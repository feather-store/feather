#!/usr/bin/env python3
"""
Feather DB - Batch Processing Example
This example shows how to efficiently process large numbers of vectors.
"""

import feather_db
import numpy as np
import time

def main():
    print("=" * 70)
    print("Feather DB - Batch Processing Example")
    print("=" * 70)
    
    # Configuration
    DIM = 512
    TOTAL_VECTORS = 10000
    BATCH_SIZE = 1000
    
    print(f"\nConfiguration:")
    print(f"  Dimensions: {DIM}")
    print(f"  Total vectors: {TOTAL_VECTORS:,}")
    print(f"  Batch size: {BATCH_SIZE:,}")
    
    # Create database
    print("\n1. Creating database...")
    db = feather_db.DB.open("batch_test.feather", dim=128)
    print("   ✓ Database created")
    
    # Add vectors in batches
    print(f"\n2. Adding {TOTAL_VECTORS:,} vectors in batches...")
    start_time = time.time()
    
    for batch_start in range(0, TOTAL_VECTORS, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, TOTAL_VECTORS)
        
        # Add vectors in this batch
        for i in range(batch_start, batch_end):
            vector = np.random.random(DIM).astype(np.float32)
            db.add(id=i, vec=vector)
        
        # Save periodically
        db.save()
        
        # Progress update
        progress = (batch_end / TOTAL_VECTORS) * 100
        elapsed = time.time() - start_time
        rate = batch_end / elapsed if elapsed > 0 else 0
        
        print(f"   Progress: {progress:5.1f}% ({batch_end:,}/{TOTAL_VECTORS:,}) "
              f"- {rate:.0f} vectors/sec")
    
    total_time = time.time() - start_time
    print(f"\n   ✓ Added {TOTAL_VECTORS:,} vectors in {total_time:.2f} seconds")
    print(f"   ✓ Average rate: {TOTAL_VECTORS/total_time:.0f} vectors/second")
    
    # Perform searches
    print("\n3. Testing search performance...")
    num_searches = 10
    search_times = []
    
    for i in range(num_searches):
        query = np.random.random(DIM).astype(np.float32)
        
        start = time.time()
        ids, distances = db.search(query, k=10)
        search_time = (time.time() - start) * 1000  # Convert to ms
        
        search_times.append(search_time)
    
    avg_search_time = np.mean(search_times)
    print(f"   ✓ Performed {num_searches} searches")
    print(f"   ✓ Average search time: {avg_search_time:.2f} ms")
    print(f"   ✓ Searches per second: {1000/avg_search_time:.0f}")
    
    # Final save
    print("\n4. Final save...")
    db.save()
    print("   ✓ Database saved")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Total vectors: {TOTAL_VECTORS:,}")
    print(f"  Dimension: {DIM}")
    print(f"  Add time: {total_time:.2f} seconds")
    print(f"  Add rate: {TOTAL_VECTORS/total_time:.0f} vectors/second")
    print(f"  Search time: {avg_search_time:.2f} ms")
    print(f"  Memory per vector: ~{DIM * 4 / 1024:.1f} KB")
    print(f"  Estimated total size: ~{TOTAL_VECTORS * DIM * 4 / (1024*1024):.1f} MB")
    print("=" * 70)

if __name__ == "__main__":
    main()
