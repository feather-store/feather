#!/bin/bash
set -e

echo "=== 🦀 Publishing Feather CLI to Crates.io ==="

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "✅ Cargo found: $(cargo --version)"

# Navigate to CLI directory
cd feather-cli

# Login to crates.io (you'll be prompted for token)
echo "Logging in to crates.io..."
read -p "Enter your crates.io token: " -s CRATES_TOKEN
echo ""
cargo login "$CRATES_TOKEN"

# Publish
echo "Publishing feather-db-cli v0.3.0..."
cargo publish

echo "✅ Successfully published to crates.io!"
echo "View at: https://crates.io/crates/feather-db-cli"
