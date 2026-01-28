#!/bin/bash
set -e

echo "🚀 Starting Deployment for Feather DB v0.2.0..."

# 1. Clean previous builds
echo "🧹 Cleaning up..."
rm -rf build/ dist/ *.egg-info
cd feather-cli && ~/.cargo/bin/cargo clean && cd ..

# 2. Build Python Package
echo "📦 Building Python Package..."
# Ensure dependencies
pip install build twine
# Build sdist and wheel
python3 -m build


echo "✅ Python build complete. Artifacts in dist/:"
ls -l dist/

# 3. Verify Rust Package
echo "🦀 Verifying Rust CLI..."
cd feather-cli
~/.cargo/bin/cargo check
cd ..

echo "=============================================="
echo "🎉 Ready to Publish!"
echo "=============================================="
echo ""
echo "prediction: To publish to PyPI, run:"
echo "  twine upload dist/*"
echo ""
echo "prediction: To publish to Crates.io, run:"
echo "  cd feather-cli && ~/.cargo/bin/cargo publish"
echo ""
echo "=============================================="
