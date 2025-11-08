#!/bin/bash
# Build script for vogel-model-trainer
# This script builds the package for PyPI distribution

set -e

echo "🔨 Building vogel-model-trainer package..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info src/*.egg-info

# Install/upgrade build tools
echo "📦 Ensuring build tools are up to date..."
python -m pip install --upgrade pip
python -m pip install --upgrade build twine

# Build the package
echo "🏗️  Building package..."
python -m build

# Check the build
echo "✅ Checking package..."
twine check dist/*

echo ""
echo "✨ Build complete!"
echo "📦 Distribution files created in dist/"
ls -lh dist/

echo ""
echo "To test the package locally:"
echo "  pip install dist/vogel_model_trainer-*.whl"
echo ""
echo "To upload to Test PyPI:"
echo "  ./scripts/upload-testpypi.sh"
echo ""
echo "To upload to PyPI:"
echo "  ./scripts/upload-pypi.sh"
