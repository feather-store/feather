# Complete Rust CLI Testing Guide for Feather DB

## 🔍 Current Status Analysis

### What I Found:
1. ✅ C++ core is compiled and working
2. ✅ Python bindings are built and functional
3. ❌ **Rust is NOT installed on your system**
4. ⚠️ Rust CLI library was built previously but binary doesn't exist
5. ⚠️ Two `main.rs` files exist (one in wrong location)

## 📋 Complete Understanding of Rust CLI Code

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Command Line                         │
│  $ feather new db.feather --dim 768                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Rust CLI (main.rs)                              │
│  - Parses arguments with clap                                │
│  - Handles three commands: new, add, search                  │
│  - Reads/writes .npy files                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Rust FFI Bindings (lib.rs)                         │
│  pub struct DB(*mut c_void)                                  │
│  - open()   → feather_open()                                 │
│  - add()    → feather_add()                                  │
│  - search() → feather_search()                               │
│  - save()   → feather_save()                                 │
└────────────────────┬────────────────────────────────────────┘
                     │ FFI (Foreign Function Interface)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         C++ Core (feather_core.cpp)                          │
│  extern "C" functions:                                       │
│  - feather_open()   → Creates DB object                      │
│  - feather_add()    → Adds vector to HNSW index              │
│  - feather_search() → Searches k-nearest neighbors           │
│  - feather_save()   → Persists to disk                       │
│  - feather_close()  → Cleanup                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              C++ Database (feather.h)                        │
│  class DB {                                                  │
│    - HNSW index for vector search                            │
│    - Binary file I/O                                         │
│    - L2 distance calculations                                │
│  }                                                            │
└─────────────────────────────────────────────────────────────┘
```

### Code Walkthrough

#### 1. **main.rs** - CLI Entry Point

```rust
// Defines CLI structure using clap
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// Three commands available
enum Commands {
    New { path, dim },      // Create new database
    Add { db, id, npy },    // Add vector from .npy file
    Search { db, npy, k },  // Search for similar vectors
}
```

**Command Flow:**
- `New`: Creates empty database with specified dimensions
- `Add`: Loads .npy file → adds vector with ID → saves
- `Search`: Loads .npy query → searches HNSW index → prints results

#### 2. **lib.rs** - FFI Bindings

```rust
// Opaque pointer to C++ DB object
pub struct DB(*mut c_void);

// Declares C functions from feather_core.cpp
extern "C" {
    fn feather_open(path: *const c_char, dim: usize) -> *mut c_void;
    fn feather_add(db: *mut c_void, id: u64, vec: *const f32, len: usize);
    fn feather_search(db: *mut c_void, query: *const f32, len: usize, 
                      k: usize, out_ids: *mut u64, out_dists: *mut f32);
    fn feather_save(db: *mut c_void);
    fn feather_close(db: *mut c_void);
}

// Safe Rust wrappers
impl DB {
    pub fn open(path: &Path, dim: usize) -> Option<Self> {
        // Converts Rust Path to C string
        // Calls C++ feather_open()
        // Returns None if failed
    }
    
    pub fn add(&self, id: u64, vec: &[f32]) {
        // Passes slice pointer to C++
        unsafe { feather_add(self.0, id, vec.as_ptr(), vec.len()) }
    }
    
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<u64>, Vec<f32>) {
        // Allocates output buffers
        // Calls C++ search
        // Returns results as Rust vectors
    }
}

// Automatic cleanup when DB goes out of scope
impl Drop for DB {
    fn drop(&mut self) { 
        unsafe { feather_close(self.0) } 
    }
}
```

#### 3. **build.rs** - Compilation Script

```rust
fn main() {
    cc::Build::new()
        .cpp(true)                          // Enable C++ mode
        .file("../src/feather_core.cpp")    // Compile C++ core
        .include("../include")              // Add header path
        .compile("feather");                // Output: libfeather.a
}
```

**What happens during build:**
1. Cargo runs `build.rs` before compiling Rust code
2. `cc` crate compiles C++ files
3. Creates static library `libfeather.a`
4. Rust code links against this library
5. Final binary includes both Rust and C++ code

### Data Flow Example: Adding a Vector

```
1. User Command:
   $ feather add vectors.feather 42 -n embedding.npy

2. Rust CLI (main.rs):
   - Parses: db="vectors.feather", id=42, npy="embedding.npy"
   - Matches Commands::Add branch

3. Open Database:
   DB::open("vectors.feather", 0)
   → lib.rs converts path to CString
   → Calls feather_open() via FFI
   → C++ opens file, loads existing vectors
   → Returns pointer to DB object

4. Read NumPy File:
   npy.read_npy()
   → ndarray-npy reads binary .npy format
   → Returns Array1<f32> with vector data
   → Example: [0.1, 0.2, 0.3, ..., 0.768]

5. Add Vector:
   db.add(42, arr.as_slice().unwrap())
   → lib.rs calls feather_add(ptr, 42, data_ptr, 768)
   → C++ receives: id=42, vec=[0.1, 0.2, ...], len=768
   → Adds to HNSW index with label 42

6. Save Database:
   db.save()
   → Calls feather_save() via FFI
   → C++ writes binary format:
      [MAGIC][VERSION][DIM][ID1][VEC1][ID2][VEC2]...

7. Cleanup:
   Drop trait automatically calls feather_close()
   → Frees C++ memory
```

## 🛠️ How to Test the Rust CLI

### Prerequisites

**You need to install Rust first:**

```bash
# Install Rust using rustup (official installer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow prompts, then reload shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Step 1: Fix File Structure

The `main.rs` should be in `src/` directory only:

```bash
# Check which main.rs is being used
ls -la feather-cli/main.rs
ls -la feather-cli/src/main.rs

# If both exist, remove the one in root
rm feather-cli/main.rs  # (if it exists)
```

### Step 2: Build the CLI

```bash
cd feather-cli
cargo build --release

# Binary will be at:
# target/release/feather-cli (or just 'feather' on some systems)
```

### Step 3: Create Test Data

You need Python with NumPy to create test .npy files:

```python
import numpy as np

# Create test vectors (128 dimensions)
vector1 = np.random.randn(128).astype(np.float32)
vector2 = np.random.randn(128).astype(np.float32)
query = np.random.randn(128).astype(np.float32)

# Save as .npy files
np.save('test_vec1.npy', vector1)
np.save('test_vec2.npy', vector2)
np.save('test_query.npy', query)
```

### Step 4: Run Test Sequence

```bash
# Test 1: Create new database
./target/release/feather-cli new test.feather --dim 128
# Expected: "Created: "test.feather""

# Test 2: Add first vector
./target/release/feather-cli add test.feather 1 -n test_vec1.npy
# Expected: "Added ID 1"

# Test 3: Add second vector
./target/release/feather-cli add test.feather 2 -n test_vec2.npy
# Expected: "Added ID 2"

# Test 4: Search for similar vectors
./target/release/feather-cli search test.feather -n test_query.npy --k 2
# Expected: 
# ID: 1  dist: 123.4567
# ID: 2  dist: 234.5678
```

### Step 5: Verify Results

```bash
# Check database file was created
ls -lh test.feather

# Check file format (should start with "FEAT" magic number)
xxd test.feather | head -5
```

## 🐛 Common Issues & Solutions

### Issue 1: "cargo: command not found"
**Solution:** Install Rust (see Prerequisites above)

### Issue 2: "linking with `cc` failed"
**Solution:** Install C++ compiler
```bash
# macOS
xcode-select --install

# Linux
sudo apt-get install build-essential
```

### Issue 3: "cannot find -lfeather"
**Solution:** Build C++ core first
```bash
g++ -O3 -std=c++17 -fPIC -c src/feather_core.cpp -o feather_core.o
ar rcs libfeather.a feather_core.o
```

### Issue 4: "Dimension mismatch"
**Solution:** Ensure .npy file dimensions match database dimensions

### Issue 5: "Failed to create DB"
**Solution:** Check file permissions and path exists

## 📊 Expected Behavior

### Successful Output Examples:

**Creating database:**
```
$ feather new vectors.feather --dim 768
Created: "vectors.feather"
```

**Adding vectors:**
```
$ feather add vectors.feather 1 -n vec1.npy
Added ID 1
```

**Searching:**
```
$ feather search vectors.feather -n query.npy --k 3
ID: 5  dist: 0.1234
ID: 2  dist: 0.2345
ID: 8  dist: 0.3456
```

## 🎯 Summary

The Rust CLI works by:
1. **Parsing commands** with clap
2. **Calling C++ functions** through FFI
3. **Reading/writing .npy files** for vector data
4. **Managing memory** safely with RAII patterns

**To test it, you need:**
- ✅ Rust installed
- ✅ C++ core compiled
- ✅ Test .npy files created
- ✅ Correct file structure

**Current blocker:** Rust is not installed on your system.

Would you like me to help you install Rust and run the complete test sequence?
