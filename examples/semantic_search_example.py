#!/usr/bin/env python3
"""
Feather DB - Semantic Search Example
This example demonstrates how to build a simple semantic search system.
Note: This uses random vectors as a placeholder. In production, use a real
embedding model like sentence-transformers.
"""

import feather_db
import numpy as np

# Simulated embedding function (replace with real model in production)
def get_embedding(text, dim=384):
    """
    In production, replace this with a real embedding model:
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text)
    """
    # For demo purposes, we create deterministic "embeddings" based on text length
    np.random.seed(hash(text) % (2**32))
    return np.random.random(dim).astype(np.float32)

def main():
    print("=" * 70)
    print("Feather DB - Semantic Search Example")
    print("=" * 70)
    
    # Our document collection
    documents = {
        0: "Python is a high-level programming language",
        1: "Machine learning is a subset of artificial intelligence",
        2: "The weather is sunny and warm today",
        3: "Vector databases enable semantic search capabilities",
        4: "Deep learning uses neural networks with multiple layers",
        5: "I love eating pizza and pasta",
        6: "Natural language processing analyzes human language",
        7: "The cat sat on the mat",
        8: "Embeddings represent text as numerical vectors",
        9: "Artificial intelligence is transforming technology"
    }
    
    # Step 1: Create database
    print("\n1. Creating database...")
    db = feather_db.DB.open("semantic_search.feather", dim=384)
    print("   ✓ Database created")
    
    # Step 2: Add documents
    print("\n2. Adding documents to database...")
    for doc_id, text in documents.items():
        embedding = get_embedding(text)
        db.add(id=doc_id, vec=embedding)
        print(f"   Added: [{doc_id}] {text[:50]}...")
    
    db.save()
    print("   ✓ All documents added and saved")
    
    # Step 3: Search with different queries
    queries = [
        "What is artificial intelligence?",
        "Tell me about programming languages",
        "How's the weather?",
        "Explain neural networks"
    ]
    
    print("\n3. Performing searches...")
    print("=" * 70)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        
        # Get query embedding and search
        query_embedding = get_embedding(query)
        ids, distances = db.search(query_embedding, k=3)
        
        print("Top 3 results:")
        for i, (doc_id, dist) in enumerate(zip(ids, distances), 1):
            similarity = 1 / (1 + dist)  # Convert distance to similarity score
            print(f"  {i}. [Score: {similarity:.3f}] {documents[doc_id]}")
    
    print("\n" + "=" * 70)
    print("Semantic search completed!")
    print("=" * 70)
    
    print("\n💡 Tip: For production use, replace get_embedding() with a real model:")
    print("   from sentence_transformers import SentenceTransformer")
    print("   model = SentenceTransformer('all-MiniLM-L6-v2')")
    print("   embedding = model.encode(text)")

if __name__ == "__main__":
    main()
