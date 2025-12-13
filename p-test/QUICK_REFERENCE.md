# Feather Rust CLI - Quick Reference Card

## 🎯 TL;DR

**What is it?** Command-line tool to manage vector databases for similarity search.

**Current Status:** Code is ready, test data created, **Rust needs to be installed**.

**To test:** Install Rust → Run `./p-test/run_tests.sh`

---

## 📦 Installation

```bash
# Install Rust (one-time setup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Build the CLI
cd feather-cli
cargo build --release
cd ..
```

Binary location: `feather-cli/target/release/feather-cli`

---

## 🚀 Quick Start

```bash
# 1. Create database
./feather-cli/target/release/feather-cli new vectors.feather --dim 128

# 2. Add vectors (requires .npy files)
./feather-cli/target/release/feather-cli add vectors.feather 1 -n vec1.npy
./feather-cli/target/release/feather-cli add vectors.feather 2 -n vec2.npy

# 3. Search
./feather-cli/target/release/feather-cli search vectors.feather -n query.npy --k 5
```

---

## 📝 Commands

### `new` - Create Database
```bash
feather new <path> --dim <dimension>
```
**Example:** `feather new db.feather --dim 768`

**What it does:**
- Creates empty database file
- Initializes HNSW index
- Sets vector dimensions

---

### `add` - Add Vector
```bash
feather add <db_path> <id> -n <npy_file>
```
**Example:** `feather add db.feather 42 -n embedding.npy`

**What it does:**
- Opens existing database
- Reads vector from .npy file
- Adds to index with specified ID
- Saves to disk

**Requirements:**
- Database must exist (use `new` first)
- .npy file must match database dimensions
- ID must be unique

---

### `search` - Find Similar Vectors
```bash
feather search <db_path> -n <query_npy> --k <count>
```
**Example:** `feather search db.feather -n query.npy --k 10`

**What it does:**
- Opens database
- Reads query vector
- Finds k most similar vectors
- Returns IDs and distances

**Output format:**
```
ID: 5  dist: 0.1234
ID: 2  dist: 0.2345
ID: 8  dist: 0.3456
```

---

## 🧪 Testing

### Automated Test
```bash
./p-test/run_tests.sh
```

### Manual Test
```bash
# Create test data first
python3 p-test/test_rust_cli.py

# Then run commands
./feather-cli/target/release/feather-cli new p-test/test.feather --dim 128
./feather-cli/target/release/feather-cli add p-test/test.feather 1 -n p-test/test-data/vector1.npy
./feather-cli/target/release/feather-cli search p-test/test.feather -n p-test/test-data/query.npy --k 3
```

---

## 📊 File Formats

### .npy Files (NumPy Arrays)
```python
import numpy as np

# Create vector
vec = np.random.randn(128).astype(np.float32)

# Save
np.save('vector.npy', vec)

# Load
vec = np.load('vector.npy')
```

### .feather Files (Database)
```
Binary format:
[4 bytes] Magic: "FEAT" (0x46454154)
[4 bytes] Version: 1
[4 bytes] Dimension: e.g., 768
[Records] ID (8 bytes) + Vector (dim * 4 bytes)
```

---

## 🏗️ Architecture

```
User Command
    ↓
Rust CLI (parses args, handles I/O)
    ↓
FFI Bindings (safe Rust wrappers)
    ↓
C++ Core (extern "C" functions)
    ↓
C++ Database (HNSW index, file I/O)
```

---

## 🔧 Troubleshooting

| Error | Solution |
|-------|----------|
| `cargo: command not found` | Install Rust |
| `linking with cc failed` | Install C++ compiler: `xcode-select --install` |
| `cannot find -lfeather` | Build C++ core first |
| `Dimension mismatch` | Check .npy dimensions match database |
| `Failed to create DB` | Check file permissions |

---

## 📚 Documentation Files

- `TESTING_SUMMARY.md` - Complete testing guide
- `complete-rust-cli-guide.md` - Comprehensive documentation
- `rust-cli-analysis.md` - Architecture analysis
- `architecture-diagram.md` - Visual diagrams
- `QUICK_REFERENCE.md` - This file

---

## 🎓 Key Concepts

**HNSW:** Hierarchical Navigable Small World - fast approximate nearest neighbor search

**FFI:** Foreign Function Interface - allows Rust to call C++ code

**L2 Distance:** Euclidean distance - `sqrt(sum((a - b)^2))`

**.npy:** NumPy binary format for arrays

**Approximate Search:** Fast but not exact - good enough for most use cases

---

## 💡 Tips

1. **Batch operations:** Add multiple vectors before searching
2. **Dimension choice:** Higher = more accurate, but slower and more memory
3. **k parameter:** Start with k=10, adjust based on needs
4. **File size:** ~4 bytes per dimension per vector + overhead
5. **Performance:** Use SSD for large databases

---

## 🔗 Related Tools

**Python API:**
```python
import feather_db
db = feather_db.DB.open("db.feather", dim=768)
db.add(1, numpy_array)
ids, dists = db.search(query, k=5)
```

**C++ API:**
```cpp
auto db = feather::DB::open("db.feather", 768);
db->add(1, vector);
auto results = db->search(query, 5);
```

---

## ✅ Checklist

Before testing:
- [ ] Rust installed (`rustc --version`)
- [ ] C++ compiler available (`g++ --version`)
- [ ] Test data created (`ls p-test/test-data/`)
- [ ] CLI built (`ls feather-cli/target/release/feather-cli`)

Ready to test:
- [ ] Run `./p-test/run_tests.sh`
- [ ] Verify output matches expected results
- [ ] Check database file created

---

## 🚨 Current Status

✅ **Ready:**
- C++ core compiled
- Python bindings working
- Test data created
- Test scripts prepared
- Documentation complete

❌ **Blocked:**
- Rust not installed
- CLI not built
- Tests not run

**Next step:** Install Rust and run tests!

---

## 📞 Quick Help

**Install Rust:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Run tests:**
```bash
./p-test/run_tests.sh
```

**That's it!** 🎉
