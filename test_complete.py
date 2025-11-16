#!/usr/bin/env python
"""
Complete local testing suite for Feather DB
Run this before release to verify everything works
"""

import sys
import os
import numpy as np

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")

def print_test(name):
    print(f"{Colors.YELLOW}Test: {name}{Colors.NC}")

def print_pass(msg="PASSED"):
    print(f"{Colors.GREEN}✅ {msg}{Colors.NC}")

def print_fail(msg="FAILED"):
    print(f"{Colors.RED}❌ {msg}{Colors.NC}")

def print_info(msg):
    print(f"   {msg}")

# Test counters
tests_passed = 0
tests_failed = 0

def run_test(test_func):
    """Decorator to run tests and track results"""
    global tests_passed, tests_failed
    try:
        test_func()
        tests_passed += 1
        print_pass()
    except Exception as e:
        tests_failed += 1
        print_fail(f"Error: {e}")
        import traceback
        traceback.print_exc()
    print()

print_header("🧪 Feather DB - Complete Local Testing Suite")

# Test 1: Import Test
print_test("1. Import Test")
@run_test
def test_import():
    import feather_py
    print_info(f"✓ feather_py module imported")
    print_info(f"✓ Module: {feather_py.__name__}")
    assert hasattr(feather_py, 'DB'), "DB class not found"
    print_info(f"✓ DB class available")

# Test 2: Basic Functionality
print_test("2. Basic Functionality")
@run_test
def test_basic():
    import feather_py
    
    # Create database
    db = feather_py.DB.open("test_basic.feather", dim=128)
    print_info("✓ Database created")
    
    # Check dimension
    dim = db.dim()
    assert dim == 128, f"Expected dim=128, got {dim}"
    print_info(f"✓ Dimension: {dim}")
    
    # Add vectors
    for i in range(5):
        vec = np.random.random(128).astype(np.float32)
        db.add(id=i, vec=vec)
    print_info("✓ Added 5 vectors")
    
    # Search
    query = np.random.random(128).astype(np.float32)
    ids, distances = db.search(query, k=3)
    assert len(ids) == 3, f"Expected 3 results, got {len(ids)}"
    print_info(f"✓ Search returned {len(ids)} results")
    
    # Save
    db.save()
    print_info("✓ Database saved")
    
    # Cleanup
    os.remove("test_basic.feather")

# Test 3: Edge Cases
print_test("3. Edge Cases")
@run_test
def test_edge_cases():
    import feather_py
    
    db = feather_py.DB.open("test_edge.feather", dim=128)
    
    # Test dimension mismatch
    try:
        wrong_vec = np.random.random(64).astype(np.float32)
        db.add(id=1, vec=wrong_vec)
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError:
        print_info("✓ Correctly rejected wrong dimension")
    
    # Test correct dimension
    correct_vec = np.random.random(128).astype(np.float32)
    db.add(id=1, vec=correct_vec)
    print_info("✓ Accepted correct dimension")
    
    # Add more vectors
    for i in range(2, 50):
        vec = np.random.random(128).astype(np.float32)
        db.add(id=i, vec=vec)
    print_info("✓ Added 49 vectors (50 total)")
    
    # Search with large k
    query = np.random.random(128).astype(np.float32)
    ids, distances = db.search(query, k=30)
    print_info(f"✓ Search with k=30 returned {len(ids)} results")
    
    db.save()
    os.remove("test_edge.feather")

# Test 4: Persistence
print_test("4. Persistence")
@run_test
def test_persistence():
    import feather_py
    
    # Create and save
    db = feather_py.DB.open("test_persist.feather", dim=64)
    test_vectors = []
    for i in range(10):
        vec = np.random.random(64).astype(np.float32)
        test_vectors.append(vec)
        db.add(id=i, vec=vec)
    db.save()
    print_info("✓ Saved 10 vectors")
    del db
    
    # Reopen
    db2 = feather_py.DB.open("test_persist.feather", dim=64)
    print_info("✓ Database reopened")
    
    # Search for first vector
    ids, distances = db2.search(test_vectors[0], k=1)
    assert ids[0] == 0, f"Expected ID 0, got {ids[0]}"
    assert distances[0] < 0.001, f"Expected distance ~0, got {distances[0]}"
    print_info(f"✓ Found vector 0 with distance {distances[0]:.6f}")
    
    os.remove("test_persist.feather")

# Test 5: Search Accuracy
print_test("5. Search Accuracy")
@run_test
def test_accuracy():
    import feather_py
    
    db = feather_py.DB.open("test_accuracy.feather", dim=128)
    
    # Create known vectors
    vec1 = np.zeros(128, dtype=np.float32)
    vec1[0] = 1.0  # [1, 0, 0, ...]
    
    vec2 = np.zeros(128, dtype=np.float32)
    vec2[0] = 0.9  # [0.9, 0, 0, ...] - similar to vec1
    
    vec3 = np.zeros(128, dtype=np.float32)
    vec3[1] = 1.0  # [0, 1, 0, ...] - different
    
    db.add(id=1, vec=vec1)
    db.add(id=2, vec=vec2)
    db.add(id=3, vec=vec3)
    print_info("✓ Added 3 test vectors")
    
    # Search with vec1
    ids, distances = db.search(vec1, k=3)
    
    # Verify order
    assert ids[0] == 1, f"Expected ID 1 first, got {ids[0]}"
    assert distances[0] < 0.001, f"Expected distance ~0, got {distances[0]}"
    print_info(f"✓ Found itself first (ID {ids[0]}, dist {distances[0]:.6f})")
    
    assert ids[1] == 2, f"Expected ID 2 second, got {ids[1]}"
    print_info(f"✓ Found similar second (ID {ids[1]}, dist {distances[1]:.6f})")
    
    assert ids[2] == 3, f"Expected ID 3 third, got {ids[2]}"
    print_info(f"✓ Found different third (ID {ids[2]}, dist {distances[2]:.6f})")
    
    assert distances[0] < distances[1] < distances[2], "Distances not ordered!"
    print_info("✓ Distances correctly ordered")
    
    os.remove("test_accuracy.feather")

# Test 6: Large Dataset
print_test("6. Large Dataset (1000 vectors)")
@run_test
def test_large():
    import feather_py
    import time
    
    db = feather_py.DB.open("test_large.feather", dim=256)
    
    # Add 1000 vectors
    start = time.time()
    for i in range(1000):
        vec = np.random.random(256).astype(np.float32)
        db.add(id=i, vec=vec)
    add_time = time.time() - start
    print_info(f"✓ Added 1000 vectors in {add_time:.2f}s ({1000/add_time:.0f} vec/s)")
    
    # Search
    query = np.random.random(256).astype(np.float32)
    start = time.time()
    ids, distances = db.search(query, k=10)
    search_time = (time.time() - start) * 1000
    print_info(f"✓ Search completed in {search_time:.2f}ms")
    
    assert len(ids) == 10, f"Expected 10 results, got {len(ids)}"
    print_info(f"✓ Returned {len(ids)} results")
    
    db.save()
    os.remove("test_large.feather")

# Test 7: File Format
print_test("7. File Format Validation")
@run_test
def test_file_format():
    import feather_py
    
    db = feather_py.DB.open("test_format.feather", dim=32)
    vec = np.random.random(32).astype(np.float32)
    db.add(id=1, vec=vec)
    db.save()
    
    # Check file exists and has content
    assert os.path.exists("test_format.feather"), "File not created"
    size = os.path.getsize("test_format.feather")
    print_info(f"✓ File created ({size} bytes)")
    
    # Check magic number
    with open("test_format.feather", "rb") as f:
        magic = f.read(4)
        # Should be "FEAT" or "TAEF" (byte order)
        assert len(magic) == 4, "Invalid file header"
        print_info(f"✓ File header present")
    
    os.remove("test_format.feather")

# Summary
print_header("📊 Test Summary")
print(f"{Colors.GREEN}Passed: {tests_passed}{Colors.NC}")
print(f"{Colors.RED}Failed: {tests_failed}{Colors.NC}")
print()

if tests_failed == 0:
    print(f"{Colors.GREEN}🎉 All tests passed! Ready for release!{Colors.NC}")
    sys.exit(0)
else:
    print(f"{Colors.RED}❌ Some tests failed. Please fix before release.{Colors.NC}")
    sys.exit(1)
