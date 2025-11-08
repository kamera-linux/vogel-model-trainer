#!/bin/bash
# Upload script for Test PyPI
# Use this to test your package before uploading to the real PyPI

set -e

echo "🚀 Uploading vogel-model-trainer to Test PyPI..."

# Check if dist/ exists and has files
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "❌ No distribution files found in dist/"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

# Upload to Test PyPI
echo "📤 Uploading to Test PyPI..."
python -m twine upload --repository testpypi dist/*

echo ""
echo "✨ Upload complete!"
echo ""
echo "To test the installation:"
echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vogel-model-trainer"
echo ""
echo "Visit your package at:"
echo "  https://test.pypi.org/project/vogel-model-trainer/"
