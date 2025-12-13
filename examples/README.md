# Feather DB Examples

This directory contains practical examples showing how to use Feather DB.

## Quick Start

### 1. Basic Example
Learn the fundamentals: create database, add vectors, search.

```bash
python3 examples/basic_python_example.py
```

**What it does:**
- Creates a database with 128 dimensions
- Adds 10 random vectors
- Searches for 5 most similar vectors
- Shows basic API usage

**Output:**
```
1. Creating database...
   ✓ Database created with 128 dimensions
2. Adding vectors...
   ✓ Added 10 vectors
3. Saving database...
   ✓ Database saved to disk
4. Searching for similar vectors...
   ✓ Found 5 similar vectors:
      1. ID:  3, Distance: 12.3456
      ...
```

---

### 2. Semantic Search Example
Build a simple document search system.

```bash
python3 examples/semantic_search_example.py
```

**What it does:**
- Creates a collection of 10 documents
- Converts documents to embeddings (simulated)
- Searches with natural language queries
- Returns most relevant documents

**Example queries:**
- "What is artificial intelligence?"
- "Tell me about programming languages"
- "How's the weather?"

**Note:** Uses simulated embeddings. For production, use a real model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text)
```

---

### 3. Batch Processing Example
Handle large datasets efficiently.

```bash
python3 examples/batch_processing_example.py
```

**What it does:**
- Adds 10,000 vectors in batches
- Measures insertion performance
- Tests search performance
- Shows progress and statistics

**Performance metrics:**
- Vectors per second
- Search latency
- Memory usage estimates

---

## Running the Examples

### Prerequisites

```bash
# Install Feather DB
pip install -e .

# Install NumPy (if not already installed)
pip install numpy
```

### Run All Examples

```bash
# Basic example
python3 examples/basic_python_example.py

# Semantic search
python3 examples/semantic_search_example.py

# Batch processing
python3 examples/batch_processing_example.py
```

---

## Example Use Cases

### 1. Document Search
```python
import feather_db
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
db = feather_db.DB.open("docs.feather", dim=384)

# Add documents
for i, doc in enumerate(documents):
    embedding = model.encode(doc)
    db.add(i, embedding)

# Search
query_emb = model.encode("your query")
ids, distances = db.search(query_emb, k=5)
```

### 2. Image Similarity
```python
import feather_db
from torchvision import models, transforms
from PIL import Image

# Load image model
model = models.resnet50(pretrained=True)
model.eval()

db = feather_db.DB.open("images.feather", dim=2048)

# Add images
for i, img_path in enumerate(image_paths):
    img = Image.open(img_path)
    embedding = extract_features(model, img)
    db.add(i, embedding)

# Find similar images
query_emb = extract_features(model, query_image)
ids, distances = db.search(query_emb, k=10)
```

### 3. Recommendation System
```python
import feather_db
import numpy as np

db = feather_db.DB.open("products.feather", dim=256)

# Add product embeddings
for product_id, features in products.items():
    embedding = create_product_embedding(features)
    db.add(product_id, embedding)

# Find similar products
product_emb = get_product_embedding(product_id)
similar_ids, _ = db.search(product_emb, k=5)
```

---

## Tips for Production Use

### 1. Use Real Embedding Models

**For Text:**
```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims
# or
model = SentenceTransformer('all-mpnet-base-v2')  # 768 dims
```

**For Images:**
```bash
pip install torch torchvision
```

```python
from torchvision import models
model = models.resnet50(pretrained=True)
```

### 2. Batch Processing

```python
# Good: Process in batches
for i in range(0, len(data), 1000):
    batch = data[i:i+1000]
    for item in batch:
        db.add(item.id, item.embedding)
    db.save()  # Save periodically

# Bad: Save after every add
for item in data:
    db.add(item.id, item.embedding)
    db.save()  # Too slow!
```

### 3. Error Handling

```python
try:
    db = feather_py.DB.open("db.feather", dim=768)
    db.add(id=1, vec=vector)
except RuntimeError as e:
    print(f"Error: {e}")
finally:
    db.save()
```

### 4. Normalize Vectors (Optional)

```python
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# Use for cosine similarity-like behavior
normalized = normalize(embedding)
db.add(id=1, vec=normalized)
```

---

## Performance Benchmarks

Typical performance on M1 MacBook Pro:

| Operation | Dimension | Performance |
|-----------|-----------|-------------|
| Add vectors | 128 | ~50,000/sec |
| Add vectors | 512 | ~30,000/sec |
| Add vectors | 768 | ~20,000/sec |
| Search (k=10) | 128 | ~0.5 ms |
| Search (k=10) | 512 | ~1.0 ms |
| Search (k=10) | 768 | ~1.5 ms |

*Results may vary based on hardware and dataset size*

---

## Next Steps

1. **Try the examples** - Run all three examples to understand the API
2. **Read the guide** - Check `USAGE_GUIDE.md` for comprehensive documentation
3. **Build your app** - Use these examples as templates for your use case
4. **Optimize** - Tune dimensions and batch sizes for your needs

---

## Need Help?

- **Documentation**: See `USAGE_GUIDE.md` in the root directory
- **Quick Reference**: Check `p-test/QUICK_REFERENCE.md`
- **Architecture**: Read `p-test/architecture-diagram.md`
- **Test Results**: See `p-test/TEST_RESULTS.md`

Happy coding! 🚀
