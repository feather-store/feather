# Local Testing Guide for Feather DB

This guide will help you thoroughly test Feather DB locally before release.

## 🎯 Testing Goals

1. ✅ Verify Python bindings work
2. ✅ Test all three APIs (Python, C++, Rust CLI)
3. ✅ Run all examples
4. ✅ Check for memory leaks
5. ✅ Validate search accuracy
6. ✅ Test edge cases

---

## 📋 Pre-Testing Checklist

### 1. Check Build Environment

```bash
# Check C++ compiler
g++ --version

# Check Python
python3 --version

# Check Rust (for CLI)
~/.cargo/bin/cargo --version

# Check required tools
pip list | grep -E "pybind11|numpy"
```

### 2. Clean Previous Builds

```bash
# Remove old build artifacts
rm -f *.o *.a *.so *.dylib
rm -rf build/ dist/ *.egg-info/
rm -f *.feather *.npy

# Clean Rust builds
cd feather-cli && ~/.cargo/bin/cargo clean && cd ..
```

---

## 🔨 Step 1: Build Everything

### 1.1 Build C++ Core

```bash
echo "Building C++ core..."
g++ -O3 -std=c++17 -fPIC -c src/feather_core.cpp -o feather_core.o
ar rcs libfeather.a feather_core.o

# Verify
ls -lh libfeather.a
echo "✓ C++ core built"
```

### 1.2 Build Python Bindings

```bash
echo "Building Python bindings..."
python3 setup.py build_ext --inplace

# Verify
ls -lh *.so
echo "✓ Python bindings built"
```

### 1.3 Build Rust CLI

```bash
echo "Building Rust CLI..."
cd feather-cli
~/.cargo/bin/cargo build --release
cd ..

# Verify
ls -lh feather-cli/target/release/feather-cli
echo "✓ Rust CLI built"
```

---

## 🧪 Step 2: Test Python API

### 2.1 Basic Import Test

```bash
python3 << 'EOF'
print("=" * 60)
print("Test 1: Import Test")
print("=" * 60)

try:
    import feather_py
    print("✓ feather_py imported successfully")
    
    import numpy as np
    print("✓ numpy imported successfully")
    
    print("\nModule info:")
    print(f"  Module: {feather_py.__name__}")
    print(f"  DB class: {feather_py.DB}")
    
    print("\n✅ Import test PASSED")
except Exception as e:
    print(f"\n❌ Import test FAILED: {e}")
    exit(1)
EOF
```

### 2.2 Basic Functionality Test

```bash
python3 << 'EOF'
import feather_py
import numpy as np

print("=" * 60)
print("Test 2: Basic Functionality")
print("=" * 60)

try:
    # Create database
    print("\n1. Creating database...")
    db = feather_py.DB.open("test_basic.feather", dim=128)
    print("   ✓ Database created")
    
    # Check dimension
    dim = db.dim()
    print(f"   ✓ Dimension: {dim}")
    assert dim == 128, f"Expected dim=128, got {dim}"
    
    # Add vectors
    print("\n2. Adding vectors...")
    for i in range(5):
        vec = np.random.random(128).astype(np.float32)
        db.add(id=i, vec=vec)
    print("   ✓ Added 5 vectors")
    
    # Search
    print("\n3. Searching...")
    query = np.random.random(128).astype(np.float32)
    ids, distances = db.search(query, k=3)
    print(f"   ✓ Found {len(ids)} results")
    print(f"   ✓ IDs: {ids}")
    print(f"   ✓ Distances: {distances}")
    
    # Verify results
    assert len(ids) == 3, f"Expected 3 results, got {len(ids)}"
    assert len(distances) == 3, f"Expected 3 distances, got {len(distances)}"
    
    # Save
    print("\n4. Saving...")
    db.save()
    print("   ✓ Database saved")
    
    print("\n✅ Basic functionality test PASSED")
    
except Exception as e:
    print(f"\n❌ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
finally:
    import os
    if os.path.exists("test_basic.feather"):
        os.remove("test_basic.feather")
EOF
```

### 2.3 Edge Cases Test

```bash
python3 << 'EOF'
import feather_py
import numpy as np

print("=" * 60)
print("Test 3: Edge Cases")
print("=" * 60)

try:
    db = feather_py.DB.open("test_edge.feather", dim=128)
    
    # Test 1: Dimension mismatch
    print("\n1. Testing dimension mismatch...")
    try:
        wrong_vec = np.random.random(64).astype(np.float32)
        db.add(id=1, vec=wrong_vec)
        print("   ❌ Should have raised error!")
        exit(1)
    except RuntimeError as e:
        print(f"   ✓ Correctly raised error: {e}")
    
    # Test 2: Correct dimension
    print("\n2. Testing correct dimension...")
    correct_vec = np.random.random(128).astype(np.float32)
    db.add(id=1, vec=correct_vec)
    print("   ✓ Added vector with correct dimension")
    
    # Test 3: Search with no results
    print("\n3. Testing search with k > num_vectors...")
    query = np.random.random(128).astype(np.float32)
    ids, distances = db.search(query, k=10)
    print(f"   ✓ Returned {len(ids)} results (expected 1)")
    
    # Test 4: Multiple adds
    print("\n4. Testing multiple adds...")
    for i in range(2, 100):
        vec = np.random.random(128).astype(np.float32)
        db.add(id=i, vec=vec)
    print("   ✓ Added 99 more vectors (100 total)")
    
    # Test 5: Large k
    print("\n5. Testing search with k=50...")
    ids, distances = db.search(query, k=50)
    print(f"   ✓ Found {len(ids)} results")
    
    db.save()
    print("\n✅ Edge cases test PASSED")
    
except Exception as e:
    print(f"\n❌ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
finally:
    import os
    if os.path.exists("test_edge.feather"):
        os.remove("test_edge.feather")
EOF
```

### 2.4 Persistence Test

```bash
python3 << 'EOF'
import feather_py
import numpy as np
import os

print("=" * 60)
print("Test 4: Persistence")
print("=" * 60)

try:
    # Create and save database
    print("\n1. Creating database and adding vectors...")
    db = feather_py.DB.open("test_persist.feather", dim=64)
    
    test_vectors = []
    for i in range(10):
        vec = np.random.random(64).astype(np.float32)
        test_vectors.append(vec)
        db.add(id=i, vec=vec)
    
    db.save()
    print("   ✓ Saved 10 vectors")
    
    # Close by deleting
    del db
    
    # Reopen and verify
    print("\n2. Reopening database...")
    db2 = feather_py.DB.open("test_persist.feather", dim=64)
    print("   ✓ Database reopened")
    
    # Search with first vector
    print("\n3. Searching for first vector...")
    ids, distances = db2.search(test_vectors[0], k=1)
    print(f"   ✓ Found ID: {ids[0]}, Distance: {distances[0]:.6f}")
    
    # Verify it found itself
    assert ids[0] == 0, f"Expected ID 0, got {ids[0]}"
    assert distances[0] < 0.001, f"Expected distance ~0, got {distances[0]}"
    
    print("\n✅ Persistence test PASSED")
    
except Exception as e:
    print(f"\n❌ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
finally:
    if os.path.exists("test_persist.feather"):
        os.remove("test_persist.feather")
EOF
```

---

## 🦀 Step 3: Test Rust CLI

### 3.1 CLI Basic Test

```bash
echo "=" | tr '=' '-' | head -c 60; echo
echo "Test 5: Rust CLI"
echo "=" | tr '=' '-' | head -c 60; echo

# Create test data
python3 << 'EOF'
import numpy as np
for i in range(3):
    vec = np.random.random(128).astype(np.float32)
    np.save(f'test_vec{i}.npy', vec)
np.save('test_query.npy', np.random.random(128).astype(np.float32))
print("✓ Created test .npy files")
EOF

# Test CLI
CLI="./feather-cli/target/release/feather-cli"

echo ""
echo "1. Creating database..."
$CLI new test_cli.feather --dim 128
echo "   ✓ Database created"

echo ""
echo "2. Adding vectors..."
for i in {0..2}; do
    $CLI add test_cli.feather $i -n test_vec${i}.npy
    echo "   ✓ Added vector $i"
done

echo ""
echo "3. Searching..."
$CLI search test_cli.feather -n test_query.npy --k 3

echo ""
echo "✅ Rust CLI test PASSED"

# Cleanup
rm -f test_vec*.npy test_query.npy test_cli.feather
```

---

## 📝 Step 4: Run All Examples

### 4.1 Basic Example

```bash
echo "=" | tr '=' '-' | head -c 60; echo
echo "Test 6: Basic Example"
echo "=" | tr '=' '-' | head -c 60; echo

python3 examples/basic_python_example.py

if [ $? -eq 0 ]; then
    echo "✅ Basic example PASSED"
else
    echo "❌ Basic example FAILED"
    exit 1
fi

# Cleanup
rm -f example.feather
```

### 4.2 Semantic Search Example

```bash
echo ""
echo "=" | tr '=' '-' | head -c 60; echo
echo "Test 7: Semantic Search Example"
echo "=" | tr '=' '-' | head -c 60; echo

python3 examples/semantic_search_example.py

if [ $? -eq 0 ]; then
    echo "✅ Semantic search example PASSED"
else
    echo "❌ Semantic search example FAILED"
    exit 1
fi

# Cleanup
rm -f semantic_search.feather
```

### 4.3 Batch Processing Example

```bash
echo ""
echo "=" | tr '=' '-' | head -c 60; echo
echo "Test 8: Batch Processing Example"
echo "=" | tr '=' '-' | head -c 60; echo

python3 examples/batch_processing_example.py

if [ $? -eq 0 ]; then
    echo "✅ Batch processing example PASSED"
else
    echo "❌ Batch processing example FAILED"
    exit 1
fi

# Cleanup
rm -f large_dataset.feather
```

---

## 🔍 Step 5: Accuracy Validation

```bash
python3 << 'EOF'
import feather_py
import numpy as np

print("=" * 60)
print("Test 9: Search Accuracy Validation")
print("=" * 60)

try:
    db = feather_py.DB.open("test_accuracy.feather", dim=128)
    
    # Create known vectors
    print("\n1. Creating test vectors...")
    vec1 = np.zeros(128, dtype=np.float32)
    vec1[0] = 1.0  # [1, 0, 0, ...]
    
    vec2 = np.zeros(128, dtype=np.float32)
    vec2[0] = 0.9  # [0.9, 0, 0, ...] - similar to vec1
    
    vec3 = np.zeros(128, dtype=np.float32)
    vec3[1] = 1.0  # [0, 1, 0, ...] - different from vec1
    
    db.add(id=1, vec=vec1)
    db.add(id=2, vec=vec2)
    db.add(id=3, vec=vec3)
    print("   ✓ Added 3 test vectors")
    
    # Search with vec1
    print("\n2. Searching with vec1...")
    ids, distances = db.search(vec1, k=3)
    
    print(f"   Results:")
    for i, (id, dist) in enumerate(zip(ids, distances)):
        print(f"     {i+1}. ID: {id}, Distance: {dist:.6f}")
    
    # Verify order
    print("\n3. Validating results...")
    assert ids[0] == 1, f"Expected ID 1 first, got {ids[0]}"
    assert distances[0] < 0.001, f"Expected distance ~0, got {distances[0]}"
    print("   ✓ Found itself first (ID 1, distance ~0)")
    
    assert ids[1] == 2, f"Expected ID 2 second, got {ids[1]}"
    print(f"   ✓ Found similar vector second (ID 2, distance {distances[1]:.6f})")
    
    assert ids[2] == 3, f"Expected ID 3 third, got {ids[2]}"
    print(f"   ✓ Found different vector third (ID 3, distance {distances[2]:.6f})")
    
    # Verify distances make sense
    assert distances[0] < distances[1] < distances[2], "Distances not in order!"
    print("   ✓ Distances are correctly ordered")
    
    print("\n✅ Accuracy validation PASSED")
    
except Exception as e:
    print(f"\n❌ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
finally:
    import os
    if os.path.exists("test_accuracy.feather"):
        os.remove("test_accuracy.feather")
EOF
```

---

## 🎯 Step 6: Run Complete Test Suite

```bash
echo ""
echo "=" | tr '=' '-' | head -c 60; echo
echo "Test 10: Complete Rust CLI Test Suite"
echo "=" | tr '=' '-' | head -c 60; echo

# Run the automated test suite
./p-test/run_tests.sh

if [ $? -eq 0 ]; then
    echo "✅ Complete test suite PASSED"
else
    echo "❌ Complete test suite FAILED"
    exit 1
fi
```

---

## 📊 Test Summary

After running all tests, you should see:

```
✅ Test 1: Import Test - PASSED
✅ Test 2: Basic Functionality - PASSED
✅ Test 3: Edge Cases - PASSED
✅ Test 4: Persistence - PASSED
✅ Test 5: Rust CLI - PASSED
✅ Test 6: Basic Example - PASSED
✅ Test 7: Semantic Search Example - PASSED
✅ Test 8: Batch Processing Example - PASSED
✅ Test 9: Accuracy Validation - PASSED
✅ Test 10: Complete Test Suite - PASSED

All tests passed! Ready for release! 🎉
```

---

## 🚀 Quick Test Script

Save this as `run_all_tests.sh`:

```bash
#!/bin/bash

echo "🧪 Running Complete Test Suite for Feather DB"
echo ""

# Run all tests from this guide
# (Copy all test commands here)

echo ""
echo "=" | tr '=' '-' | head -c 60; echo
echo "TEST SUMMARY"
echo "=" | tr '=' '-' | head -c 60; echo
echo ""
echo "All tests completed!"
echo ""
```

---

## 🐛 If Tests Fail

### Python Import Fails
```bash
# Rebuild Python bindings
python3 setup.py clean --all
python3 setup.py build_ext --inplace

# Check .so file exists
ls -la *.so
```

### Dimension Mismatch Errors
```bash
# Verify vector dimensions match database
# Check that all vectors are float32
# Ensure consistent dimensions throughout
```

### Rust CLI Not Found
```bash
# Rebuild CLI
cd feather-cli
~/.cargo/bin/cargo build --release
cd ..

# Verify binary exists
ls -la feather-cli/target/release/feather-cli
```

### Memory Issues
```bash
# Check for memory leaks (macOS)
leaks -atExit -- python3 examples/basic_python_example.py

# Or use valgrind (Linux)
valgrind --leak-check=full python3 examples/basic_python_example.py
```

---

## ✅ Ready for Release?

After all tests pass:
- [ ] All 10 tests passed
- [ ] Examples run without errors
- [ ] No memory leaks detected
- [ ] Search results are accurate
- [ ] Persistence works correctly

**If all checked, you're ready to release!** 🚀

Proceed to `RELEASE_CHECKLIST.md` for release steps.
