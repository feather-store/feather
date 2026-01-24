# Feather DB 🪶

**Fast, lightweight context-aware vector database**

*Part of [Hawky.ai](https://hawky.ai) - AI Native Digital Marketing OS*

[![PyPI](https://img.shields.io/pypi/v/feather-db)](https://pypi.org/project/feather-db/)
[![Crates.io](https://img.shields.io/crates/v/feather-db-cli)](https://crates.io/crates/feather-db-cli)
[![Website](https://img.shields.io/badge/website-getfeather.store-blue)](https://www.getfeather.store/)
[![GitHub](https://img.shields.io/github/stars/feather-store/feather?style=social)](https://github.com/feather-store/feather)

A fast, lightweight vector database built with C++ and HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search.

## Features

- 🚀 **High Performance**: Built with C++ and optimized HNSW algorithm
- 🧠 **Context Engine**: Structured metadata storage (Facts, Preferences, Events, Conversations)
- ⏳ **Temporal Retrieval**: Time-weighted scoring with exponential decay
- 🔍 **Filtered Search**: Domain-logic filtering (by type, source, tags) during HNSW search
- 🐍 **Python Integration**: Native Python bindings with `FilterBuilder` support
- 🦀 **Rust CLI**: Enhanced CLI for metadata and filtered operations
- 💾 **Persistent Storage**: Version 2 binary format with automatic metadata persistence

[![PyPI](https://img.shields.io/pypi/v/feather-db)](https://pypi.org/project/feather-db/)
[![Crates.io](https://img.shields.io/crates/v/feather-db-cli)](https://crates.io/crates/feather-db-cli)

## Quick Start

### Python Usage

```python
import feather_db
import numpy as np

# Open or create a database
db = feather_db.DB.open("my_vectors.feather", dim=768)

# Add vectors
vector = np.random.random(768).astype(np.float32)
db.add(id=1, vec=vector)

# Search for similar vectors
query = np.random.random(768).astype(np.float32)
ids, distances = db.search(query, k=5)

print(f"Found {len(ids)} similar vectors")
for i, (id, dist) in enumerate(zip(ids, distances)):
    print(f"  {i+1}. ID: {id}, Distance: {dist:.4f}")

# Save the database
db.save()

### Context Engine (Phase 2)

```python
from feather_db import DB, Metadata, ContextType, FilterBuilder

# Add with metadata
meta = Metadata()
meta.content = "User prefers dark mode"
meta.type = ContextType.PREFERENCE
meta.importance = 0.9
db.add(id=1, vec=embedding, meta=meta)

# Search with filters and temporal decay
fb = FilterBuilder()
filter = fb.types(ContextType.PREFERENCE).min_importance(0.5).build()

results = db.search(query, k=5, filter=filter, scoring=ScoringConfig(half_life=30))
```
```

### C++ Usage

```cpp
#include "include/feather.h"
#include <vector>

int main() {
    // Open database
    auto db = feather::DB::open("my_vectors.feather", 768);
    
    // Add a vector
    std::vector<float> vec(768, 0.1f);
    db->add(1, vec);
    
    // Search
    std::vector<float> query(768, 0.1f);
    auto results = db->search(query, 5);
    
    for (auto [id, distance] : results) {
        std::cout << "ID: " << id << ", Distance: " << distance << std::endl;
    }
    
    return 0;
}
```

### CLI Usage

```bash
# Create a new database
feather new my_db.feather --dim 768

# Add vectors from NumPy files
feather add my_db.feather 1 --npy vector1.npy
feather add my_db.feather 2 --npy vector2.npy

# Search for similar vectors
feather search my_db.feather --npy query.npy --k 10
```

### Rust CLI

The CLI is available as a native binary for fast database management.

```bash
# Add with metadata
feather add --npy vector.npy --content "Hello world" --source "cli" my_db 123

# Search with filters
feather search --npy query.npy --type-filter 0 --source-filter "cli" my_db
```

## Installation

### Python Package (Recommended)

```bash
pip install feather-db
```

### Build from Source

#### Prerequisites

- **C++17** compatible compiler
- **Python 3.8+** (for Python bindings)
- **Rust 1.70+** (for CLI tool)
- **pybind11** (for Python bindings)

#### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd feather
   ```

2. **Install Python Package**
   ```bash
   pip install .
   ```

3. **Build Rust CLI (Optional)**
   ```bash
   cd feather-cli
   cargo build --release
   ```

## Architecture

### Core Components

- **`feather::DB`**: Main C++ class providing vector database functionality
- **HNSW Index**: Hierarchical Navigable Small World algorithm for fast ANN search
- **Binary Format**: Custom storage format with magic number validation
- **Multi-language Bindings**: Python (pybind11) and Rust (FFI) interfaces

### File Format

Feather uses a custom binary format:
```
[4 bytes] Magic number: 0x46454154 ("FEAT")
[4 bytes] Version: 1
[4 bytes] Dimension
[Records] ID (8 bytes) + Vector data (dim * 4 bytes)
```

### Performance Characteristics

- **Index Type**: HNSW with L2 distance
- **Max Elements**: 1,000,000 (configurable)
- **Construction Parameters**: M=16, ef_construction=200
- **Memory Usage**: ~4 bytes per dimension per vector + index overhead

## API Reference

### Python API

#### `feather_db.DB`

- **`DB.open(path: str, dim: int = 768)`**: Open or create database
- **`add(id: int, vec: np.ndarray)`**: Add vector with ID
- **`search(query: np.ndarray, k: int = 5)`**: Search k nearest neighbors
- **`save()`**: Persist database to disk
- **`dim()`**: Get vector dimension

### C++ API

#### `feather::DB`

- **`static std::unique_ptr<DB> open(path, dim)`**: Factory method
- **`void add(uint64_t id, const std::vector<float>& vec)`**: Add vector
- **`auto search(const std::vector<float>& query, size_t k)`**: Search vectors
- **`void save()`**: Save to disk
- **`size_t dim() const`**: Get dimension

### CLI Commands

- **`feather new <path> --dim <dimension>`**: Create new database
- **`feather add <db> <id> --npy <file>`**: Add vector from .npy file
- **`feather search <db> --npy <query> --k <count>`**: Search similar vectors

## Examples

### Semantic Search with Embeddings

```python
import feather_db
import numpy as np

# Create database for sentence embeddings
db = feather_db.DB.open("sentences.feather", dim=384)

# Add document embeddings
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Vector databases enable semantic search capabilities"
]

for i, doc in enumerate(documents):
    # Assume get_embedding() returns a 384-dim vector
    embedding = get_embedding(doc)
    db.add(i, embedding)

# Search for similar documents
query_embedding = get_embedding("What is machine learning?")
ids, distances = db.search(query_embedding, k=2)

for id, dist in zip(ids, distances):
    print(f"Document: {documents[id]}")
    print(f"Similarity: {1 - dist:.3f}\n")
```

### Batch Processing

```python
import feather_db
import numpy as np

db = feather_db.DB.open("large_dataset.feather", dim=512)

# Batch add vectors
batch_size = 1000
for batch_start in range(0, 100000, batch_size):
    for i in range(batch_size):
        vector_id = batch_start + i
        vector = np.random.random(512).astype(np.float32)
        db.add(vector_id, vector)
    
    # Periodic save
    if batch_start % 10000 == 0:
        db.save()
        print(f"Processed {batch_start + batch_size} vectors")
```

## Performance Tips

1. **Batch Operations**: Add vectors in batches and save periodically
2. **Memory Management**: Consider vector dimension vs. memory usage trade-offs
3. **Search Parameters**: Adjust `k` parameter based on your precision/recall needs
4. **File I/O**: Use SSD storage for better performance with large databases

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Built on top of [hnswlib](https://github.com/nmslib/hnswlib)
- Uses [pybind11](https://github.com/pybind/pybind11) for Python bindings
- CLI built with [clap](https://github.com/clap-rs/clap) for Rust