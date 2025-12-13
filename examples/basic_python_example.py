#!/usr/bin/env python3
"""
Feather DB - Basic Python Example
This example shows how to create a database, add vectors, and search.
"""

import feather_db
import numpy as np

def main():
    print("=" * 60)
    print("Feather DB - Basic Example")
    print("=" * 60)
    
    # Step 1: Create or open a database
    print("\n1. Creating database...")
    db = feather_db.DB.open("example.feather", dim=128)
    print("   ✓ Database created with 128 dimensions")
    
    # Step 2: Add some vectors
    print("\n2. Adding vectors...")
    num_vectors = 10
    
    for i in range(num_vectors):
        # Create a random vector (in real use, these would be embeddings)
        vector = np.random.random(128).astype(np.float32)
        db.add(id=i, vec=vector)
    
    print(f"   ✓ Added {num_vectors} vectors")
    
    # Step 3: Save the database
    print("\n3. Saving database...")
    db.save()
    print("   ✓ Database saved to disk")
    
    # Step 4: Search for similar vectors
    print("\n4. Searching for similar vectors...")
    query = np.random.random(128).astype(np.float32)
    ids, distances = db.search(query, k=5)
    
    print(f"   ✓ Found {len(ids)} similar vectors:")
    for i, (id, dist) in enumerate(zip(ids, distances), 1):
        print(f"      {i}. ID: {id:2d}, Distance: {dist:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
