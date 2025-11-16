#!/usr/bin/env python3
"""
Test script for Feather Rust CLI
Creates test .npy files and provides commands to test the CLI
"""

import numpy as np
import os

def create_test_data():
    """Create test .npy files for CLI testing"""
    print("Creating test data...")
    
    # Create test directory
    os.makedirs("p-test/test-data", exist_ok=True)
    
    # Create vectors with 128 dimensions
    dim = 128
    
    # Vector 1: Random normal distribution
    vec1 = np.random.randn(dim).astype(np.float32)
    np.save('p-test/test-data/vector1.npy', vec1)
    print(f"✓ Created vector1.npy (shape: {vec1.shape}, dtype: {vec1.dtype})")
    
    # Vector 2: Random normal distribution
    vec2 = np.random.randn(dim).astype(np.float32)
    np.save('p-test/test-data/vector2.npy', vec2)
    print(f"✓ Created vector2.npy (shape: {vec2.shape}, dtype: {vec2.dtype})")
    
    # Vector 3: Similar to vector1 (for testing similarity)
    vec3 = vec1 + np.random.randn(dim).astype(np.float32) * 0.1
    np.save('p-test/test-data/vector3.npy', vec3)
    print(f"✓ Created vector3.npy (shape: {vec3.shape}, dtype: {vec3.dtype})")
    
    # Query vector: Very similar to vector1
    query = vec1 + np.random.randn(dim).astype(np.float32) * 0.05
    np.save('p-test/test-data/query.npy', query)
    print(f"✓ Created query.npy (shape: {query.shape}, dtype: {query.dtype})")
    
    # Calculate expected distances for verification
    dist1 = np.sum((query - vec1) ** 2)
    dist2 = np.sum((query - vec2) ** 2)
    dist3 = np.sum((query - vec3) ** 2)
    
    print(f"\nExpected L2 distances from query:")
    print(f"  query → vector1: {dist1:.4f}")
    print(f"  query → vector2: {dist2:.4f}")
    print(f"  query → vector3: {dist3:.4f}")
    print(f"\nExpected order: vector1 (closest), vector3, vector2")
    
    return dim

def print_test_commands(dim):
    """Print the test commands to run"""
    print("\n" + "="*70)
    print("TEST COMMANDS FOR RUST CLI")
    print("="*70)
    
    print("\n1. Build the Rust CLI:")
    print("   cd feather-cli")
    print("   cargo build --release")
    print("   cd ..")
    
    print("\n2. Create a new database:")
    print(f"   ./feather-cli/target/release/feather-cli new p-test/test.feather --dim {dim}")
    
    print("\n3. Add vectors to database:")
    print("   ./feather-cli/target/release/feather-cli add p-test/test.feather 1 -n p-test/test-data/vector1.npy")
    print("   ./feather-cli/target/release/feather-cli add p-test/test.feather 2 -n p-test/test-data/vector2.npy")
    print("   ./feather-cli/target/release/feather-cli add p-test/test.feather 3 -n p-test/test-data/vector3.npy")
    
    print("\n4. Search for similar vectors:")
    print("   ./feather-cli/target/release/feather-cli search p-test/test.feather -n p-test/test-data/query.npy --k 3")
    
    print("\n5. Verify database file:")
    print("   ls -lh p-test/test.feather")
    print("   xxd p-test/test.feather | head -5")
    
    print("\n" + "="*70)
    print("Expected search result: ID 1 should have smallest distance")
    print("="*70 + "\n")

def create_test_script():
    """Create a bash script to run all tests"""
    script = """#!/bin/bash
# Automated test script for Feather Rust CLI

set -e  # Exit on error

echo "=========================================="
echo "Feather Rust CLI Test Suite"
echo "=========================================="

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Rust/Cargo not found"
    echo "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✓ Rust is installed"

# Build the CLI
echo ""
echo "Building Rust CLI..."
cd feather-cli
cargo build --release
cd ..
echo "✓ Build successful"

# Check if binary exists
if [ ! -f "feather-cli/target/release/feather-cli" ]; then
    echo "❌ Error: Binary not found at feather-cli/target/release/feather-cli"
    exit 1
fi

echo "✓ Binary found"

# Create test database
echo ""
echo "Test 1: Creating database..."
./feather-cli/target/release/feather-cli new p-test/test.feather --dim 128
if [ -f "p-test/test.feather" ]; then
    echo "✓ Database created"
else
    echo "❌ Database file not created"
    exit 1
fi

# Add vectors
echo ""
echo "Test 2: Adding vectors..."
./feather-cli/target/release/feather-cli add p-test/test.feather 1 -n p-test/test-data/vector1.npy
echo "✓ Added vector 1"

./feather-cli/target/release/feather-cli add p-test/test.feather 2 -n p-test/test-data/vector2.npy
echo "✓ Added vector 2"

./feather-cli/target/release/feather-cli add p-test/test.feather 3 -n p-test/test-data/vector3.npy
echo "✓ Added vector 3"

# Search
echo ""
echo "Test 3: Searching for similar vectors..."
./feather-cli/target/release/feather-cli search p-test/test.feather -n p-test/test-data/query.npy --k 3

# Verify file
echo ""
echo "Test 4: Verifying database file..."
ls -lh p-test/test.feather
echo ""
echo "File header (should start with FEAT magic number):"
xxd p-test/test.feather | head -3

echo ""
echo "=========================================="
echo "✓ All tests completed successfully!"
echo "=========================================="
"""
    
    with open('p-test/run_tests.sh', 'w') as f:
        f.write(script)
    
    os.chmod('p-test/run_tests.sh', 0o755)
    print("✓ Created executable test script: p-test/run_tests.sh")

if __name__ == "__main__":
    print("Feather Rust CLI Test Data Generator")
    print("=" * 70)
    
    # Create test data
    dim = create_test_data()
    
    # Create test script
    print()
    create_test_script()
    
    # Print commands
    print_test_commands(dim)
    
    print("\nQuick start:")
    print("  1. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
    print("  2. Run: ./p-test/run_tests.sh")
