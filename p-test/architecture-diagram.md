# Feather Rust CLI - Visual Architecture

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                                                                       │
│  $ feather new db.feather --dim 768                                  │
│  $ feather add db.feather 1 -n vector.npy                            │
│  $ feather search db.feather -n query.npy --k 5                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RUST CLI LAYER (main.rs)                          │
│                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐         │
│  │   Command   │  │   Command   │  │      Command        │         │
│  │     NEW     │  │     ADD     │  │      SEARCH         │         │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘         │
│         │                │                     │                     │
│         │ Creates DB     │ Reads .npy         │ Reads .npy          │
│         │ with dim       │ Adds vector        │ Searches k-NN       │
│         │                │ Saves DB           │ Returns results     │
└─────────┼────────────────┼─────────────────────┼─────────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  RUST FFI BINDINGS (lib.rs)                          │
│                                                                       │
│  pub struct DB(*mut c_void);  // Opaque pointer to C++ object       │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ DB::open()   │  │  DB::add()   │  │ DB::search() │              │
│  │              │  │              │  │              │              │
│  │ Converts     │  │ Passes slice │  │ Allocates    │              │
│  │ Path → CStr  │  │ pointer      │  │ output bufs  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                  │                       │
│         │ unsafe FFI      │ unsafe FFI       │ unsafe FFI            │
│         ▼                 ▼                  ▼                       │
│  feather_open()    feather_add()      feather_search()              │
└─────────┼─────────────────┼──────────────────┼───────────────────────┘
          │                 │                  │
          │ C ABI           │ C ABI            │ C ABI
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              C++ FFI LAYER (feather_core.cpp)                        │
│                                                                       │
│  extern "C" {                                                        │
│    void* feather_open(const char* path, size_t dim)                 │
│    void  feather_add(void* db, uint64_t id, float* vec, size_t len) │
│    void  feather_search(void* db, float* q, size_t len, size_t k,   │
│                         uint64_t* ids, float* dists)                 │
│    void  feather_save(void* db)                                      │
│    void  feather_close(void* db)                                     │
│  }                                                                   │
│                                                                       │
│  Wraps C++ DB class with C-compatible interface                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                C++ DATABASE CORE (feather.h)                         │
│                                                                       │
│  class DB {                                                          │
│    private:                                                          │
│      unique_ptr<HierarchicalNSW<float>> index_;  // HNSW index      │
│      size_t dim_;                                 // Vector dims     │
│      string path_;                                // File path       │
│                                                                       │
│    public:                                                           │
│      static unique_ptr<DB> open(path, dim)       // Factory         │
│      void add(id, vector)                        // Add to index    │
│      auto search(query, k)                       // k-NN search     │
│      void save()                                 // Persist         │
│  }                                                                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HNSW ALGORITHM (hnswlib)                          │
│                                                                       │
│  HierarchicalNSW<float>:                                             │
│    - Hierarchical graph structure                                    │
│    - M = 16 (connections per node)                                   │
│    - ef_construction = 200 (build quality)                           │
│    - L2 distance metric (Euclidean)                                  │
│    - SIMD optimizations (AVX512/AVX/SSE)                             │
│                                                                       │
│  Operations:                                                         │
│    - addPoint(vector, label) → Insert into graph                     │
│    - searchKnn(query, k) → Find k nearest neighbors                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PERSISTENT STORAGE                                │
│                                                                       │
│  Binary File Format (.feather):                                      │
│  ┌──────────────────────────────────────────────────────┐           │
│  │ Header (12 bytes)                                     │           │
│  │  [4] Magic: 0x46454154 ("FEAT")                      │           │
│  │  [4] Version: 1                                       │           │
│  │  [4] Dimension: e.g., 768                             │           │
│  ├──────────────────────────────────────────────────────┤           │
│  │ Records (variable length)                             │           │
│  │  [8] ID: uint64_t                                     │           │
│  │  [dim*4] Vector: float32 array                        │           │
│  │  [8] ID: uint64_t                                     │           │
│  │  [dim*4] Vector: float32 array                        │           │
│  │  ...                                                  │           │
│  └──────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Adding a Vector

```
Step 1: User Command
┌────────────────────────────────────────────┐
│ $ feather add db.feather 42 -n vec.npy    │
└────────────────┬───────────────────────────┘
                 │
Step 2: Parse Arguments (Rust)
┌────────────────▼───────────────────────────┐
│ Commands::Add {                            │
│   db: "db.feather",                        │
│   id: 42,                                  │
│   npy: "vec.npy"                           │
│ }                                          │
└────────────────┬───────────────────────────┘
                 │
Step 3: Open Database (Rust → C++)
┌────────────────▼───────────────────────────┐
│ DB::open("db.feather", 0)                  │
│   → CString::new("db.feather")             │
│   → feather_open(c_str, 0)                 │
│   → C++ loads existing database            │
│   → Returns pointer to DB object           │
└────────────────┬───────────────────────────┘
                 │
Step 4: Read NumPy File (Rust)
┌────────────────▼───────────────────────────┐
│ npy.read_npy()                             │
│   → Opens "vec.npy"                        │
│   → Parses NumPy binary format             │
│   → Returns Array1<f32>                    │
│   → Example: [0.1, 0.2, 0.3, ..., 0.768]  │
└────────────────┬───────────────────────────┘
                 │
Step 5: Add Vector (Rust → C++)
┌────────────────▼───────────────────────────┐
│ db.add(42, arr.as_slice().unwrap())        │
│   → feather_add(ptr, 42, data_ptr, 768)   │
│   → C++ receives: id=42, vec=[...], len   │
│   → index_->addPoint(vec.data(), 42)       │
│   → HNSW inserts into graph structure     │
└────────────────┬───────────────────────────┘
                 │
Step 6: Save Database (Rust → C++)
┌────────────────▼───────────────────────────┐
│ db.save()                                  │
│   → feather_save(ptr)                      │
│   → C++ writes binary format:              │
│      [MAGIC][VERSION][DIM]                 │
│      [ID1][VEC1][ID2][VEC2]...             │
└────────────────┬───────────────────────────┘
                 │
Step 7: Cleanup (Automatic)
┌────────────────▼───────────────────────────┐
│ Drop trait triggered                       │
│   → feather_close(ptr)                     │
│   → C++ deletes DB object                  │
│   → Memory freed                           │
└────────────────────────────────────────────┘
```

## Memory Management

```
┌─────────────────────────────────────────────────────────────┐
│                    RUST SIDE                                 │
│                                                              │
│  Stack:                                                      │
│  ┌──────────────────────┐                                   │
│  │ DB(ptr) ────────────┼─────────┐                          │
│  └──────────────────────┘         │                          │
│                                    │                          │
│  Heap:                             │                          │
│  (Rust doesn't own C++ memory)     │                          │
└────────────────────────────────────┼──────────────────────────┘
                                     │ Opaque pointer
                                     │ (*mut c_void)
┌────────────────────────────────────┼──────────────────────────┐
│                    C++ SIDE        │                          │
│                                    │                          │
│  Heap:                             ▼                          │
│  ┌─────────────────────────────────────────┐                 │
│  │ unique_ptr<DB>                          │                 │
│  │  ├─ unique_ptr<HierarchicalNSW>         │                 │
│  │  │   ├─ data_level0_memory_ (vectors)   │                 │
│  │  │   ├─ linkLists_ (graph structure)    │                 │
│  │  │   └─ label_lookup_ (ID mapping)      │                 │
│  │  ├─ dim_ (768)                           │                 │
│  │  └─ path_ ("db.feather")                 │                 │
│  └─────────────────────────────────────────┘                 │
│                                                              │
│  When feather_close() called:                                │
│    → delete unique_ptr<DB>                                   │
│    → Cascading destructors free all memory                   │
└─────────────────────────────────────────────────────────────┘
```

## Build Process

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Cargo Build Starts                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Run build.rs                                        │
│                                                              │
│  cc::Build::new()                                            │
│    .cpp(true)                                                │
│    .file("../src/feather_core.cpp")                          │
│    .include("../include")                                    │
│    .compile("feather");                                      │
│                                                              │
│  Output: libfeather.a (static library)                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Compile Rust Code                                   │
│                                                              │
│  - Compiles lib.rs (FFI bindings)                            │
│  - Compiles main.rs (CLI interface)                          │
│  - Links against libfeather.a                                │
│  - Links against C++ standard library                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Link Everything                                     │
│                                                              │
│  Final Binary:                                               │
│  ┌──────────────────────────────────────┐                   │
│  │ feather-cli                          │                   │
│  │  ├─ Rust code (CLI + FFI)            │                   │
│  │  ├─ libfeather.a (C++ core)          │                   │
│  │  ├─ libc++ (C++ stdlib)              │                   │
│  │  └─ Dependencies (clap, ndarray...)  │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
│  Location: target/release/feather-cli                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. FFI (Foreign Function Interface)
- Allows Rust to call C/C++ functions
- Uses `extern "C"` for C ABI compatibility
- Requires `unsafe` blocks in Rust
- Pointers passed as `*mut c_void` (opaque)

### 2. HNSW (Hierarchical Navigable Small World)
- Graph-based approximate nearest neighbor search
- Faster than brute force for large datasets
- Trade-off: speed vs accuracy
- Parameters: M (connections), ef (search quality)

### 3. Memory Safety
- Rust: Ownership, borrowing, lifetimes
- C++: Smart pointers (unique_ptr)
- FFI: Manual coordination required
- Drop trait ensures cleanup

### 4. Binary Format
- Custom format for persistence
- Magic number for validation
- Version for compatibility
- Efficient storage and loading
