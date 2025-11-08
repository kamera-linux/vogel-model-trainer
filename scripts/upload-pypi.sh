#!/bin/bash
# Upload script for PyPI
# Use this to publish your package to the official PyPI

set -e

echo "🚀 Uploading vogel-model-trainer to PyPI..."
echo "⚠️  WARNING: This will publish to the REAL PyPI!"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Upload cancelled"
    exit 0
fi

# Check if dist/ exists and has files
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "❌ No distribution files found in dist/"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

# Upload to PyPI
echo "📤 Uploading to PyPI..."
python -m twine upload dist/*

echo ""
echo "✨ Upload complete!"
echo ""
echo "To install the package:"
echo "  pip install vogel-model-trainer"
echo ""
echo "Visit your package at:"
echo "  https://pypi.org/project/vogel-model-trainer/"
