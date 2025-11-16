# Feather DB - Complete Usage Guide

## What is Feather DB?

Feather DB is a **fast, lightweight vector database** for storing and searching high-dimensional vectors using approximate nearest neighbor (ANN) search. Think of it as a specialized database for finding similar items based on their vector representations.

### Common Use Cases:
- **Semantic Search**: Find similar documents, sentences, or paragraphs
- **Image Similarity**: Find visually similar images
- **Recommendation Systems**: Find similar products, movies, or content
- **Embeddings Storage**: Store and search ML model embeddings
- **RAG Systems**: Retrieval-Augmented Generation for AI applications

---

## How It Works

### The Concept

1. **Vectors**: Your data (text, images, etc.) is converted to numerical vectors (arrays of numbers)
2. **Storage**: Vectors are stored in a database with unique IDs
3. **Search**: Given a query vector, find the most similar vectors in the database
4. **Distance**: Similarity is measured using L2 (Euclidean) distance - smaller = more similar

### Example Flow:
```
Text: "machine learning"
   ↓ (embedding model)
Vector: [0.1, 0.3, -0.2, ..., 0.5]  (768 dimensions)
   ↓ (store in Feather DB)
ID: 42
   ↓ (search with query)
Query: "artificial intelligence"
   ↓ (find similar)
Results: ID 42 (distance: 0.15) ← Very similar!
```

---

## Installation

### Prerequisites

```bash
# macOS
xcode-select --install  # C++ compiler

# Python 3.8+
python3 --version

# Rust (for CLI)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build from Source

```bash
# 1. Clone repository
git clone <your-repo-url>
cd feather

# 2. Build C++ Core
g++ -O3 -std=c++17 -fPIC -c src/feather_core.cpp -o feather_core.o
ar rcs libfeather.a feather_core.o

# 3. Build Python Bindings
pip install pybind11 numpy
python setup.py build_ext --inplace
pip install -e .

# 4. Build Rust CLI
cd feather-cli
cargo build --release
cd ..
```

---

## Usage Guide

## 1. Python API (Recommended for Most Users)

### Installation Check
```python
import feather_py
import numpy as np
print("Feather DB is ready!")
```

### Basic Usage

#### Step 1: Create a Database
```python
import feather_py
import numpy as np

# Create or open a database
# dim = dimension of your vectors (must be consistent)
db = feather_py.DB.open("my_vectors.feather", dim=768)
```

#### Step 2: Add Vectors
```python
# Create a vector (768 dimensions)
vector = np.random.random(768).astype(np.float32)

# Add with a unique ID
db.add(id=1, vec=vector)

# Add more vectors
for i in range(2, 100):
    vec = np.random.random(768).astype(np.float32)
    db.add(id=i, vec=vec)

# Save to disk
db.save()
```

#### Step 3: Search for Similar Vectors
```python
# Create a query vector
query = np.random.random(768).astype(np.float32)

# Search for 5 most similar vectors
ids, distances = db.search(query, k=5)

print(f"Found {len(ids)} similar vectors:")
for i, (id, dist) in enumerate(zip(ids, distances)):
    print(f"  {i+1}. ID: {id}, Distance: {dist:.4f}")
```

### Real-World Example: Semantic Search

```python
import feather_py
import numpy as np

# Assume you have a function that converts text to embeddings
# (e.g., using sentence-transformers, OpenAI, etc.)
def get_embedding(text):
    # This is a placeholder - use your actual embedding model
    # Example: model.encode(text)
    return np.random.random(384).astype(np.float32)

# Create database
db = feather_py.DB.open("documents.feather", dim=384)

# Your documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Vector databases enable semantic search",
    "Deep learning uses neural networks"
]

# Add documents to database
for i, doc in enumerate(documents):
    embedding = get_embedding(doc)
    db.add(id=i, vec=embedding)

db.save()

# Search for similar documents
query = "What is AI and machine learning?"
query_embedding = get_embedding(query)
ids, distances = db.search(query_embedding, k=3)

print("Most relevant documents:")
for id, dist in zip(ids, distances):
    print(f"  [{dist:.3f}] {documents[id]}")
```

### Batch Processing

```python
import feather_py
import numpy as np

db = feather_py.DB.open("large_dataset.feather", dim=512)

# Process in batches
batch_size = 1000
total_vectors = 100000

for batch_start in range(0, total_vectors, batch_size):
    print(f"Processing batch {batch_start // batch_size + 1}...")
    
    for i in range(batch_size):
        vector_id = batch_start + i
        vector = np.random.random(512).astype(np.float32)
        db.add(vector_id, vector)
    
    # Save periodically
    if batch_start % 10000 == 0:
        db.save()
        print(f"  Saved {batch_start + batch_size} vectors")

db.save()
print("All vectors added!")
```

### Working with Real Embeddings

```python
# Example with sentence-transformers
from sentence_transformers import SentenceTransformer
import feather_py

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

# Create database
db = feather_py.DB.open("sentences.feather", dim=384)

# Your data
sentences = [
    "I love programming in Python",
    "Machine learning is fascinating",
    "The weather is nice today",
    "Neural networks are powerful"
]

# Add to database
for i, sentence in enumerate(sentences):
    embedding = model.encode(sentence)
    db.add(i, embedding)

db.save()

# Search
query = "I enjoy coding"
query_embedding = model.encode(query)
ids, distances = db.search(query_embedding, k=2)

print("Similar sentences:")
for id, dist in zip(ids, distances):
    print(f"  [{1-dist:.3f}] {sentences[id]}")
```

---

## 2. C++ API (For Performance-Critical Applications)

### Basic Usage

```cpp
#include "include/feather.h"
#include <iostream>
#include <vector>

int main() {
    // Create or open database
    auto db = feather::DB::open("vectors.feather", 768);
    
    // Create a vector
    std::vector<float> vec(768, 0.1f);
    
    // Add vector with ID
    db->add(1, vec);
    
    // Add more vectors
    for (uint64_t i = 2; i <= 100; ++i) {
        std::vector<float> v(768);
        for (auto& val : v) {
            val = static_cast<float>(rand()) / RAND_MAX;
        }
        db->add(i, v);
    }
    
    // Search
    std::vector<float> query(768, 0.1f);
    auto results = db->search(query, 5);
    
    std::cout << "Top 5 results:\n";
    for (auto [id, distance] : results) {
        std::cout << "  ID: " << id 
                  << ", Distance: " << distance << "\n";
    }
    
    // Save
    db->save();
    
    return 0;
}
```

### Compile and Run

```bash
# Compile
g++ -O3 -std=c++17 -I./include my_app.cpp libfeather.a -o my_app

# Run
./my_app
```

### Advanced Example

```cpp
#include "include/feather.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main() {
    const size_t DIM = 512;
    const size_t NUM_VECTORS = 10000;
    
    // Create database
    auto db = feather::DB::open("benchmark.feather", DIM);
    
    // Random number generator
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Add vectors
    auto start = std::chrono::high_resolution_clock::now();
    
    for (uint64_t i = 0; i < NUM_VECTORS; ++i) {
        std::vector<float> vec(DIM);
        for (auto& v : vec) {
            v = dist(gen);
        }
        db->add(i, vec);
        
        if (i % 1000 == 0) {
            std::cout << "Added " << i << " vectors\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Added " << NUM_VECTORS << " vectors in " 
              << duration.count() << "ms\n";
    
    // Search benchmark
    std::vector<float> query(DIM);
    for (auto& v : query) {
        v = dist(gen);
    }
    
    start = std::chrono::high_resolution_clock::now();
    auto results = db->search(query, 10);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Search completed in " << duration.count() << "μs\n";
    
    // Save
    db->save();
    
    return 0;
}
```

---

## 3. Rust CLI (For Command-Line Operations)

### Basic Commands

#### Create a New Database
```bash
./feather-cli/target/release/feather-cli new my_db.feather --dim 768
```

#### Add Vectors from .npy Files

First, create .npy files with Python:
```python
import numpy as np

# Create and save vectors
vec1 = np.random.random(768).astype(np.float32)
np.save('vector1.npy', vec1)

vec2 = np.random.random(768).astype(np.float32)
np.save('vector2.npy', vec2)
```

Then add to database:
```bash
# Add vector with ID 1
./feather-cli/target/release/feather-cli add my_db.feather 1 -n vector1.npy

# Add vector with ID 2
./feather-cli/target/release/feather-cli add my_db.feather 2 -n vector2.npy
```

#### Search for Similar Vectors

```bash
# Create query vector
python3 -c "import numpy as np; np.save('query.npy', np.random.random(768).astype(np.float32))"

# Search
./feather-cli/target/release/feather-cli search my_db.feather -n query.npy --k 5
```

### Complete Workflow Example

```bash
#!/bin/bash

# 1. Create database
./feather-cli/target/release/feather-cli new products.feather --dim 512

# 2. Generate and add product embeddings
python3 << 'EOF'
import numpy as np

products = [
    "Laptop Computer",
    "Wireless Mouse",
    "Mechanical Keyboard",
    "USB-C Cable",
    "External Monitor"
]

# Simulate embeddings (use real model in production)
for i, product in enumerate(products):
    embedding = np.random.random(512).astype(np.float32)
    np.save(f'product_{i}.npy', embedding)
    print(f"Created embedding for: {product}")
EOF

# 3. Add to database
for i in {0..4}; do
    ./feather-cli/target/release/feather-cli add products.feather $i -n product_${i}.npy
    echo "Added product $i"
done

# 4. Search
python3 -c "import numpy as np; np.save('query.npy', np.random.random(512).astype(np.float32))"
./feather-cli/target/release/feather-cli search products.feather -n query.npy --k 3

# 5. Cleanup
rm product_*.npy query.npy
```

---

## Understanding the Results

### Distance Interpretation

```python
ids, distances = db.search(query, k=5)

# Distance values:
# 0.0 - 1.0   : Very similar (almost identical)
# 1.0 - 5.0   : Similar
# 5.0 - 20.0  : Somewhat related
# 20.0+       : Not very similar
```

### Converting Distance to Similarity Score

```python
def distance_to_similarity(distance):
    """Convert L2 distance to similarity score (0-1)"""
    return 1 / (1 + distance)

ids, distances = db.search(query, k=5)
for id, dist in zip(ids, distances):
    similarity = distance_to_similarity(dist)
    print(f"ID: {id}, Similarity: {similarity:.3f}")
```

---

## Best Practices

### 1. Choose the Right Dimension

```python
# Common embedding dimensions:
# - 384: sentence-transformers (MiniLM)
# - 512: Many custom models
# - 768: BERT, OpenAI ada-002
# - 1536: OpenAI text-embedding-3-small
# - 3072: OpenAI text-embedding-3-large

# Higher dimensions = more accurate but slower and more memory
db = feather_py.DB.open("db.feather", dim=768)
```

### 2. Normalize Vectors (Optional)

```python
import numpy as np

def normalize(vec):
    """Normalize vector to unit length"""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# Use normalized vectors for cosine similarity-like behavior
vector = get_embedding("some text")
normalized = normalize(vector)
db.add(id=1, vec=normalized)
```

### 3. Batch Operations

```python
# Good: Batch processing with periodic saves
for i in range(0, 10000, 100):
    for j in range(100):
        db.add(i + j, vectors[i + j])
    if i % 1000 == 0:
        db.save()

# Bad: Save after every add
for i in range(10000):
    db.add(i, vectors[i])
    db.save()  # Too slow!
```

### 4. Error Handling

```python
import feather_py
import numpy as np

try:
    db = feather_py.DB.open("db.feather", dim=768)
    
    vector = np.random.random(768).astype(np.float32)
    db.add(id=1, vec=vector)
    
    # Wrong dimension - will raise error
    wrong_vec = np.random.random(512).astype(np.float32)
    db.add(id=2, vec=wrong_vec)  # RuntimeError!
    
except RuntimeError as e:
    print(f"Error: {e}")
finally:
    db.save()
```

### 5. Memory Management

```python
# For large datasets, process in chunks
def process_large_dataset(vectors, db, chunk_size=1000):
    for i in range(0, len(vectors), chunk_size):
        chunk = vectors[i:i+chunk_size]
        for j, vec in enumerate(chunk):
            db.add(i + j, vec)
        db.save()
        print(f"Processed {i + len(chunk)} vectors")
```

---

## Performance Tips

### 1. Use Appropriate k Value

```python
# Small k (1-10): Fast, good for most use cases
results = db.search(query, k=5)

# Large k (50+): Slower, use only if needed
results = db.search(query, k=100)
```

### 2. File Storage

```python
# Use SSD for better performance
# Good: /path/to/ssd/vectors.feather
# Avoid: /path/to/hdd/vectors.feather
```

### 3. Dimension Trade-offs

```
Dimension | Memory/Vector | Search Speed | Accuracy
----------|---------------|--------------|----------
128       | 512 bytes     | Very Fast    | Good
384       | 1.5 KB        | Fast         | Better
768       | 3 KB          | Medium       | Best
1536      | 6 KB          | Slower       | Excellent
```

---

## Troubleshooting

### Common Issues

#### 1. Dimension Mismatch
```python
# Error: RuntimeError: Dimension mismatch
# Solution: Ensure all vectors have same dimension as database
db = feather_py.DB.open("db.feather", dim=768)
vec = np.random.random(768).astype(np.float32)  # Must be 768!
```

#### 2. File Not Found
```python
# Error: Cannot open file
# Solution: Check path and permissions
import os
os.path.exists("db.feather")  # Check if file exists
```

#### 3. Memory Issues
```python
# Error: Out of memory
# Solution: Process in smaller batches
# Reduce max_elements in HNSW (requires C++ code change)
```

#### 4. Slow Search
```python
# Issue: Search is slow
# Solutions:
# - Reduce dimension if possible
# - Reduce k value
# - Use SSD storage
# - Ensure database is properly indexed
```

---

## Complete Example: Building a Document Search System

```python
import feather_py
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentSearch:
    def __init__(self, db_path="documents.feather", model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.db = feather_py.DB.open(db_path, dim=self.dim)
        self.documents = {}
    
    def add_document(self, doc_id, text):
        """Add a document to the database"""
        embedding = self.model.encode(text)
        self.db.add(id=doc_id, vec=embedding)
        self.documents[doc_id] = text
        return doc_id
    
    def add_documents(self, documents):
        """Add multiple documents"""
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)
        self.db.save()
    
    def search(self, query, k=5):
        """Search for similar documents"""
        query_embedding = self.model.encode(query)
        ids, distances = self.db.search(query_embedding, k=k)
        
        results = []
        for id, dist in zip(ids, distances):
            results.append({
                'id': id,
                'text': self.documents.get(id, "Unknown"),
                'distance': float(dist),
                'similarity': 1 / (1 + dist)
            })
        return results
    
    def save(self):
        """Save database to disk"""
        self.db.save()

# Usage
search_engine = DocumentSearch()

# Add documents
docs = {
    1: "Python is a high-level programming language",
    2: "Machine learning enables computers to learn from data",
    3: "Vector databases store high-dimensional embeddings",
    4: "Natural language processing analyzes human language",
    5: "Deep learning uses neural networks with multiple layers"
}

search_engine.add_documents(docs)

# Search
results = search_engine.search("What is AI and ML?", k=3)

print("Search Results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. [Similarity: {result['similarity']:.3f}]")
    print(f"   {result['text']}")

search_engine.save()
```

---

## Next Steps

1. **Start Simple**: Begin with the Python API and small datasets
2. **Experiment**: Try different embedding models and dimensions
3. **Scale Up**: Move to larger datasets and optimize performance
4. **Production**: Use C++ API for performance-critical applications
5. **Automate**: Use Rust CLI for batch processing and scripts

## Resources

- **Test Files**: Check `p-test/` directory for examples
- **Documentation**: See `p-test/QUICK_REFERENCE.md` for quick tips
- **Architecture**: Read `p-test/architecture-diagram.md` for internals

---

**Happy Vector Searching! 🚀**
