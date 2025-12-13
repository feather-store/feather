#!/bin/bash

# Feather DB - PyPI Deployment Script
# This script will help you deploy to PyPI

set -e

echo "🚀 Feather DB - PyPI Deployment"
echo "================================"
echo ""

# Check if package is built
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "❌ No package found in dist/"
    echo "Building package..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python -m build
    else
        python3 -m build
    fi
    echo ""
fi

# Check package
echo "📦 Checking package..."
if [ -f ".venv/bin/twine" ]; then
    .venv/bin/twine check dist/*
else
    twine check dist/*
fi
echo ""

# Ask user which PyPI to upload to
echo "Where do you want to upload?"
echo "1) TestPyPI (test.pypi.org) - RECOMMENDED for first time"
echo "2) PyPI (pypi.org) - Production"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "📤 Uploading to TestPyPI..."
        echo ""
        echo "You'll need a TestPyPI account and token:"
        echo "1. Create account: https://test.pypi.org/account/register/"
        echo "2. Create token: https://test.pypi.org/manage/account/token/"
        echo ""
        echo "When prompted:"
        echo "  Username: __token__"
        echo "  Password: (paste your token)"
        echo ""
        if [ -f ".venv/bin/twine" ]; then
            .venv/bin/twine upload --repository testpypi dist/*
        else
            twine upload --repository testpypi dist/*
        fi
        
        echo ""
        echo "✅ Uploaded to TestPyPI!"
        echo ""
        echo "Test installation with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ feather-db"
        ;;
    2)
        echo ""
        echo "📤 Uploading to PyPI..."
        echo ""
        echo "You'll need a PyPI account and token:"
        echo "1. Create account: https://pypi.org/account/register/"
        echo "2. Create token: https://pypi.org/manage/account/token/"
        echo ""
        echo "When prompted:"
        echo "  Username: __token__"
        echo "  Password: (paste your token)"
        echo ""
        if [ -f ".venv/bin/twine" ]; then
            .venv/bin/twine upload dist/*
        else
            twine upload dist/*
        fi
        
        echo ""
        echo "✅ Uploaded to PyPI!"
        echo ""
        echo "Your package is now live at:"
        echo "  https://pypi.org/project/feather-db/"
        echo ""
        echo "Users can install with:"
        echo "  pip install feather-db"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment complete!"
