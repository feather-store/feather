# Rust CLI Complete Analysis & Testing Guide

## Overview
The Feather Rust CLI provides a command-line interface to the Feather vector database, allowing users to create databases, add vectors, and search for similar vectors using .npy files.

## Architecture Understanding

### 1. **File Structure**
```
feather-cli/
├── Cargo.toml          # Rust package configuration
├── build.rs            # Build script (compiles C++ core)
├── main.rs             # CLI entry point (WRONG LOCATION - should be in src/)
└── src/
    ├── lib.rs          # FFI bindings to C++ core
    └── main.rs         # CLI entry point (CORRECT LOCATION)
```

### 2. **How It Works**

#### **Layer 1: C++ Core (FFI Interface)**
Located in `src/feather_core.cpp`, provides C-compatible functions:
- `feather_open()` - Opens/creates database
- `feather_add()` - Adds vector with ID
- `feather_search()` - Searches for k-nearest neighbors
- `feather_save()` - Persists database to disk
- `feather_close()` - Cleanup

#### **Layer 2: Rust FFI Bindings**
Located in `feather-cli/src/lib.rs`:
```rust
pub struct DB(*mut c_void);  // Opaque pointer to C++ DB object

impl DB {
    pub fn open(path: &Path, dim: usize) -> Option<Self>
    pub fn add(&self, id: u64, vec: &[f32])
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<u64>, Vec<f32>)
    pub fn save(&self)
}
```

#### **Layer 3: CLI Interface**
Located in `feather-cli/src/main.rs` (or `feather-cli/main.rs`):
- Uses `clap` for argument parsing
- Three commands: `new`, `add`, `search`
- Reads/writes .npy files using `ndarray-npy`

### 3. **Build Process**

The `build.rs` script:
1. Compiles `src/feather_core.cpp` using the `cc` crate
2. Links against the C++ standard library
3. Creates a static library that Rust can link to

**Current Issue**: There are TWO `main.rs` files:
- `feather-cli/main.rs` (wrong location, uses `ReadNpyExt` trait)
- `feather-cli/src/main.rs` (correct location, uses `read_npy()` function)

This causes confusion - Cargo will use `src/main.rs` by default.

## CLI Commands Explained

### Command 1: `new` - Create Database
```bash
feather new <path> --dim <dimension>
```

**What it does:**
1. Calls `DB::open(path, dim)` which creates a new database file
2. Initializes HNSW index with specified dimensions
3. Creates empty database file with header

**Example:**
```bash
feather new vectors.feather --dim 768
```

### Command 2: `add` - Add Vector
```bash
feather add <db_path> <id> -n <npy_file>
```

**What it does:**
1. Opens existing database (dim=0 means auto-detect from file)
2. Reads vector from .npy file (NumPy array format)
3. Adds vector with specified ID to database
4. Saves database to disk

**Example:**
```bash
feather add vectors.feather 1 -n vector1.npy
```

### Command 3: `search` - Find Similar Vectors
```bash
feather search <db_path> -n <query_npy> --k <num_results>
```

**What it does:**
1. Opens existing database
2. Reads query vector from .npy file
3. Performs k-nearest neighbor search using HNSW
4. Returns IDs and distances of k most similar vectors

**Example:**
```bash
feather search vectors.feather -n query.npy --k 5
```

## Code Flow Analysis

### Example: Adding a Vector

```
User runs: feather add db.feather 42 -n vec.npy

1. main.rs parses arguments with clap
   ↓
2. Matches Commands::Add { db, id, npy }
   ↓
3. DB::open(&db, 0) 
   → Calls feather_open() via FFI
   → C++ creates/loads DB object
   ↓
4. npy.read_npy() or read_npy(&file)
   → Reads NumPy array from file
   → Returns Array1<f32>
   ↓
5. db.add(id, arr.as_slice().unwrap())
   → Calls feather_add() via FFI
   → C++ adds vector to HNSW index
   ↓
6. db.save()
   → Calls feather_save() via FFI
   → C++ writes to disk
   ↓
7. Drop trait calls feather_close()
   → Cleanup C++ resources
```

## Current Status

### ✅ What's Working:
- C++ core is compiled (`libfeather.a` exists)
- Rust library is built (`libfeather_cli.rlib` exists)
- FFI bindings are properly defined
- Build system is configured correctly

### ❌ What's Not Working:
- **No CLI binary built** - only library exists
- **Duplicate main.rs files** - confusion about which is used
- **Different implementations** - two versions use different npy reading methods

## Testing Plan

To properly test the Rust CLI, we need to:

1. **Fix the file structure** - Ensure correct main.rs is used
2. **Build the binary** - Compile with `--bin` flag
3. **Create test data** - Generate .npy files for testing
4. **Run test sequence** - Test all three commands
5. **Verify results** - Check database files and search results

## Next Steps

Would you like me to:
1. Fix the file structure and build the CLI binary?
2. Create test .npy files and run a complete test sequence?
3. Debug any specific issues you're encountering?
