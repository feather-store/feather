# Feather DB - How to Use (Complete Guide)

## 📚 Table of Contents

1. [What is Feather DB?](#what-is-feather-db)
2. [Quick Start](#quick-start)
3. [Three Ways to Use](#three-ways-to-use)
4. [Common Use Cases](#common-use-cases)
5. [Step-by-Step Tutorials](#step-by-step-tutorials)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## What is Feather DB?

**Feather DB is a vector database** - a specialized database for storing and searching high-dimensional vectors (arrays of numbers).

### Why Use It?

Modern AI applications convert data (text, images, audio) into **embeddings** - numerical representations that capture meaning. Feather DB lets you:

- **Store** millions of embeddings efficiently
- **Search** for similar items in milliseconds
- **Build** semantic search, recommendations, and RAG systems

### How It Works

```
Your Data → Embedding Model → Vector → Feather DB → Similar Items
```

**Example:**
```
"machine learning" → [0.1, 0.3, -0.2, ...] → Store with ID 42
                                              ↓
"artificial intelligence" → [0.1, 0.3, -0.1, ...] → Search
                                              ↓
                                         Find ID 42 (very similar!)
```

---

## Quick Start

### Installation

```bash
# 1. Build C++ core
g++ -O3 -std=c++17 -fPIC -c src/feather_core.cpp -o feather_core.o
ar rcs libfeather.a feather_core.o

# 2. Install Python bindings
pip install pybind11 numpy
pip install -e .

# 3. Build Rust CLI (optional)
cd feather-cli && cargo build --release && cd ..
```

### Your First Program (Python)

```python
import feather_py
import numpy as np

# 1. Create database
db = feather_py.DB.open("my_db.feather", dim=128)

# 2. Add vectors
for i in range(10):
    vector = np.random.random(128).astype(np.float32)
    db.add(id=i, vec=vector)

# 3. Search
query = np.random.random(128).astype(np.float32)
ids, distances = db.search(query, k=5)

print(f"Found {len(ids)} similar vectors!")
db.save()
```

**Run it:**
```bash
python3 my_first_program.py
```

---

## Three Ways to Use

### 1. Python API (Recommended)

**Best for:** Most applications, data science, prototyping

```python
import feather_py
import numpy as np

db = feather_py.DB.open("db.feather", dim=768)
db.add(id=1, vec=vector)
ids, distances = db.search(query, k=5)
db.save()
```

**Pros:**
- Easy to use
- Integrates with NumPy
- Perfect for ML workflows

### 2. C++ API

**Best for:** Performance-critical applications, embedded systems

```cpp
#include "include/feather.h"

auto db = feather::DB::open("db.feather", 768);
db->add(1, vector);
auto results = db->search(query, 5);
db->save();
```

**Pros:**
- Maximum performance
- Low memory overhead
- Direct control

### 3. Rust CLI

**Best for:** Command-line tools, scripts, batch processing

```bash
feather new db.feather --dim 768
feather add db.feather 1 -n vector.npy
feather search db.feather -n query.npy --k 5
```

**Pros:**
- No coding required
- Works with .npy files
- Great for automation

---

## Common Use Cases

### 1. Semantic Search

**Problem:** Find documents similar to a query

```python
from sentence_transformers import SentenceTransformer
import feather_py

# Setup
model = SentenceTransformer('all-MiniLM-L6-v2')
db = feather_py.DB.open("docs.feather", dim=384)

# Add documents
documents = ["doc1 text", "doc2 text", ...]
for i, doc in enumerate(documents):
    embedding = model.encode(doc)
    db.add(i, embedding)

# Search
query_emb = model.encode("your search query")
ids, _ = db.search(query_emb, k=5)

# Get results
results = [documents[id] for id in ids]
```

### 2. Image Similarity

**Problem:** Find visually similar images

```python
from torchvision import models, transforms
import feather_py
from PIL import Image

# Setup
model = models.resnet50(pretrained=True)
db = feather_py.DB.open("images.feather", dim=2048)

# Add images
for i, img_path in enumerate(image_paths):
    img = Image.open(img_path)
    features = extract_features(model, img)
    db.add(i, features)

# Find similar
query_features = extract_features(model, query_image)
similar_ids, _ = db.search(query_features, k=10)
```

### 3. Recommendation System

**Problem:** Recommend similar products/content

```python
import feather_py

db = feather_py.DB.open("products.feather", dim=256)

# Add products
for product in products:
    embedding = create_product_embedding(product)
    db.add(product.id, embedding)

# Recommend
user_preferences = get_user_embedding(user)
recommended_ids, _ = db.search(user_preferences, k=10)
```

### 4. RAG (Retrieval-Augmented Generation)

**Problem:** Find relevant context for LLM prompts

```python
import feather_py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
db = feather_py.DB.open("knowledge.feather", dim=384)

# Index knowledge base
for i, chunk in enumerate(knowledge_chunks):
    embedding = model.encode(chunk)
    db.add(i, embedding)

# Retrieve context for LLM
def get_context(question):
    q_emb = model.encode(question)
    ids, _ = db.search(q_emb, k=3)
    return [knowledge_chunks[id] for id in ids]

# Use with LLM
context = get_context("What is machine learning?")
prompt = f"Context: {context}\n\nQuestion: {question}"
```

---

## Step-by-Step Tutorials

### Tutorial 1: Build a Document Search Engine

**Goal:** Search through 1000 documents

**Step 1: Install dependencies**
```bash
pip install sentence-transformers feather-db
```

**Step 2: Prepare data**
```python
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    # ... 998 more documents
]
```

**Step 3: Create embeddings**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = [model.encode(doc) for doc in documents]
```

**Step 4: Build database**
```python
import feather_py

db = feather_py.DB.open("search.feather", dim=384)
for i, emb in enumerate(embeddings):
    db.add(i, emb)
db.save()
```

**Step 5: Search**
```python
query = "What is programming?"
query_emb = model.encode(query)
ids, distances = db.search(query_emb, k=5)

for id, dist in zip(ids, distances):
    print(f"[{1/(1+dist):.3f}] {documents[id]}")
```

**Complete code:** See `examples/semantic_search_example.py`

---

### Tutorial 2: Process Large Dataset

**Goal:** Add 100,000 vectors efficiently

**Step 1: Setup**
```python
import feather_py
import numpy as np

db = feather_py.DB.open("large.feather", dim=512)
```

**Step 2: Batch processing**
```python
BATCH_SIZE = 1000
TOTAL = 100000

for batch_start in range(0, TOTAL, BATCH_SIZE):
    # Process batch
    for i in range(BATCH_SIZE):
        vec_id = batch_start + i
        vector = get_vector(vec_id)  # Your data source
        db.add(vec_id, vector)
    
    # Save periodically
    db.save()
    print(f"Processed {batch_start + BATCH_SIZE}/{TOTAL}")
```

**Step 3: Verify**
```python
# Test search
query = np.random.random(512).astype(np.float32)
ids, distances = db.search(query, k=10)
print(f"Search works! Found {len(ids)} results")
```

**Complete code:** See `examples/batch_processing_example.py`

---

### Tutorial 3: Use Rust CLI

**Goal:** Manage vectors from command line

**Step 1: Create test data**
```python
import numpy as np

# Create vectors
for i in range(5):
    vec = np.random.random(128).astype(np.float32)
    np.save(f'vec{i}.npy', vec)
```

**Step 2: Create database**
```bash
./feather-cli/target/release/feather-cli new mydb.feather --dim 128
```

**Step 3: Add vectors**
```bash
for i in {0..4}; do
    ./feather-cli/target/release/feather-cli add mydb.feather $i -n vec${i}.npy
done
```

**Step 4: Search**
```bash
# Create query
python3 -c "import numpy as np; np.save('query.npy', np.random.random(128).astype(np.float32))"

# Search
./feather-cli/target/release/feather-cli search mydb.feather -n query.npy --k 3
```

---

## Best Practices

### 1. Choose Right Dimension

| Dimension | Use Case | Speed | Accuracy |
|-----------|----------|-------|----------|
| 128-256 | Simple tasks | Fast | Good |
| 384-512 | Most applications | Medium | Better |
| 768-1024 | High accuracy needed | Slower | Best |
| 1536+ | Maximum quality | Slowest | Excellent |

**Recommendation:** Start with 384 (sentence-transformers default)

### 2. Normalize Vectors

```python
def normalize(vec):
    """Normalize to unit length"""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# Use normalized vectors
normalized = normalize(embedding)
db.add(id=1, vec=normalized)
```

**Why?** Better similarity scores, especially for cosine similarity

### 3. Batch Operations

```python
# ✅ Good: Batch with periodic saves
for i in range(0, 10000, 1000):
    for j in range(1000):
        db.add(i+j, vectors[i+j])
    db.save()

# ❌ Bad: Save after every add
for i in range(10000):
    db.add(i, vectors[i])
    db.save()  # Too slow!
```

### 4. Error Handling

```python
try:
    db = feather_py.DB.open("db.feather", dim=768)
    db.add(id=1, vec=vector)
except RuntimeError as e:
    print(f"Error: {e}")
    # Handle dimension mismatch, file errors, etc.
finally:
    db.save()
```

### 5. Monitor Performance

```python
import time

# Measure add time
start = time.time()
db.add(id=1, vec=vector)
print(f"Add: {(time.time()-start)*1000:.2f}ms")

# Measure search time
start = time.time()
ids, dists = db.search(query, k=10)
print(f"Search: {(time.time()-start)*1000:.2f}ms")
```

---

## Troubleshooting

### Problem: "Dimension mismatch"

**Error:**
```
RuntimeError: Dimension mismatch
```

**Solution:**
```python
# Ensure vector dimension matches database
db = feather_py.DB.open("db.feather", dim=768)
vector = np.random.random(768).astype(np.float32)  # Must be 768!
db.add(id=1, vec=vector)
```

### Problem: "Module not found: feather_py"

**Error:**
```
ModuleNotFoundError: No module named 'feather_py'
```

**Solution:**
```bash
# Rebuild and install
python setup.py build_ext --inplace
pip install -e .
```

### Problem: Slow performance

**Symptoms:** Adding/searching takes too long

**Solutions:**
1. Reduce dimension (768 → 384)
2. Use batch processing
3. Use SSD storage
4. Reduce k value in search

### Problem: Out of memory

**Symptoms:** Crash when adding many vectors

**Solutions:**
1. Process in smaller batches
2. Save more frequently
3. Reduce dimension
4. Use C++ API for lower overhead

### Problem: Rust CLI not found

**Error:**
```
command not found: feather
```

**Solution:**
```bash
# Build CLI
cd feather-cli
cargo build --release
cd ..

# Use full path
./feather-cli/target/release/feather-cli --help
```

---

## Performance Guide

### Expected Performance

**M1 MacBook Pro, 10,000 vectors:**

| Operation | Dim 128 | Dim 384 | Dim 768 |
|-----------|---------|---------|---------|
| Add rate | 5,000/s | 3,000/s | 2,000/s |
| Search (k=10) | 0.5ms | 1.0ms | 1.5ms |
| Memory/vector | 512B | 1.5KB | 3KB |

### Optimization Tips

**1. Dimension**
```python
# Faster but less accurate
db = feather_py.DB.open("db.feather", dim=128)

# Slower but more accurate
db = feather_py.DB.open("db.feather", dim=768)
```

**2. Batch Size**
```python
# Too small: Overhead from frequent saves
BATCH_SIZE = 10  # ❌

# Too large: Memory issues
BATCH_SIZE = 100000  # ❌

# Just right
BATCH_SIZE = 1000  # ✅
```

**3. k Value**
```python
# Fast
ids, _ = db.search(query, k=5)  # ✅

# Slower
ids, _ = db.search(query, k=100)  # Use only if needed
```

---

## Quick Reference

### Python API

```python
import feather_py
import numpy as np

# Create/open
db = feather_py.DB.open("db.feather", dim=768)

# Add
vector = np.random.random(768).astype(np.float32)
db.add(id=1, vec=vector)

# Search
ids, distances = db.search(query, k=5)

# Save
db.save()

# Get dimension
dim = db.dim()
```

### Rust CLI

```bash
# Create
feather new db.feather --dim 768

# Add
feather add db.feather 1 -n vector.npy

# Search
feather search db.feather -n query.npy --k 5
```

### C++ API

```cpp
#include "include/feather.h"

// Create/open
auto db = feather::DB::open("db.feather", 768);

// Add
std::vector<float> vec(768, 0.1f);
db->add(1, vec);

// Search
auto results = db->search(query, 5);

// Save
db->save();
```

---

## Resources

### Documentation
- **Complete Guide**: `USAGE_GUIDE.md`
- **Quick Reference**: `p-test/QUICK_REFERENCE.md`
- **Architecture**: `p-test/architecture-diagram.md`
- **Test Results**: `p-test/TEST_RESULTS.md`

### Examples
- **Basic**: `examples/basic_python_example.py`
- **Semantic Search**: `examples/semantic_search_example.py`
- **Batch Processing**: `examples/batch_processing_example.py`
- **Demo Outputs**: `examples/DEMO_OUTPUT.md`

### Getting Help
1. Check examples directory
2. Read USAGE_GUIDE.md
3. Review test results in p-test/
4. Check architecture diagrams

---

## Next Steps

1. **Try examples** - Run the three Python examples
2. **Build something** - Start with semantic search
3. **Optimize** - Tune dimensions and batch sizes
4. **Scale up** - Process larger datasets
5. **Deploy** - Use in production applications

**Happy vector searching! 🚀**
