#!/bin/bash

# Feather DB - Local Testing Script
# Run this to test everything before release

set -e  # Exit on error

echo "🧪 Feather DB - Local Testing Suite"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${YELLOW}Running: $test_name${NC}"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAILED${NC}"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Test 1: Check build environment
echo "📋 Checking build environment..."
echo ""

if command -v g++ &> /dev/null; then
    echo "✓ g++ found: $(g++ --version | head -1)"
else
    echo "❌ g++ not found"
    exit 1
fi

if command -v python3 &> /dev/null; then
    echo "✓ Python found: $(python3 --version)"
else
    echo "❌ Python not found"
    exit 1
fi

if [ -f ~/.cargo/bin/cargo ]; then
    echo "✓ Cargo found: $(~/.cargo/bin/cargo --version)"
else
    echo "⚠️  Cargo not found (Rust CLI tests will be skipped)"
fi

echo ""

# Test 2: Build C++ core
echo "🔨 Building C++ core..."
g++ -O3 -std=c++17 -fPIC -c src/feather_core.cpp -o feather_core.o 2>&1 | grep -v "warning:" || true
ar rcs libfeather.a feather_core.o

if [ -f libfeather.a ]; then
    echo "✓ C++ core built successfully"
else
    echo "❌ C++ core build failed"
    exit 1
fi
echo ""

# Test 3: Build Python bindings
echo "🐍 Building Python bindings..."
python3 setup.py build_ext --inplace 2>&1 | grep -E "(running|creating|copying)" || true

if ls *.so 1> /dev/null 2>&1; then
    echo "✓ Python bindings built successfully"
else
    echo "❌ Python bindings build failed"
    exit 1
fi
echo ""

# Test 4: Python import test
run_test "Python Import Test" "python3 -c 'import feather; import numpy as np; print(\"Import successful\")'"

# Test 5: Basic functionality
run_test "Basic Functionality Test" "python3 << 'EOF'
import feather
import numpy as np
db = feather.DB.open(\"test_tmp.feather\", dim=128)
vec = np.random.random(128).astype(np.float32)
db.add(id=1, vec=vec)
query = np.random.random(128).astype(np.float32)
ids, distances = db.search(query, k=1)
db.save()
import os
os.remove(\"test_tmp.feather\")
print(\"Test passed\")
EOF
"

# Test 6: Run basic example
if [ -f examples/basic_python_example.py ]; then
    run_test "Basic Example" "python3 examples/basic_python_example.py > /dev/null 2>&1 && rm -f example.feather"
fi

# Test 7: Run semantic search example
if [ -f examples/semantic_search_example.py ]; then
    run_test "Semantic Search Example" "python3 examples/semantic_search_example.py > /dev/null 2>&1 && rm -f semantic_search.feather"
fi

# Test 8: Rust CLI (if available)
if [ -f ~/.cargo/bin/cargo ]; then
    echo "🦀 Building Rust CLI..."
    cd feather-cli
    ~/.cargo/bin/cargo build --release 2>&1 | grep -E "(Compiling|Finished)" || true
    cd ..
    
    if [ -f feather-cli/target/release/feather-cli ]; then
        echo "✓ Rust CLI built successfully"
        echo ""
        
        # Create test data
        python3 << 'EOF'
import numpy as np
np.save('test_vec.npy', np.random.random(128).astype(np.float32))
np.save('test_query.npy', np.random.random(128).astype(np.float32))
EOF
        
        run_test "Rust CLI Test" "./feather-cli/target/release/feather-cli new test_cli.feather --dim 128 > /dev/null 2>&1 && \
./feather-cli/target/release/feather-cli add test_cli.feather 1 -n test_vec.npy > /dev/null 2>&1 && \
./feather-cli/target/release/feather-cli search test_cli.feather -n test_query.npy --k 1 > /dev/null 2>&1 && \
rm -f test_vec.npy test_query.npy test_cli.feather"
    fi
fi

# Summary
echo "===================================="
echo "📊 Test Summary"
echo "===================================="
echo ""
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Ready for release!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please fix before release.${NC}"
    exit 1
fi
