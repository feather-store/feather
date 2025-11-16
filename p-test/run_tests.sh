#!/bin/bash
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
