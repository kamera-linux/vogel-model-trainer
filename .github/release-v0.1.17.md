# Release v0.1.17 - Bug Fix Release

**Release Date:** November 18, 2025

## 🐛 Critical Bug Fixes

This is an important bug fix release that resolves issues with image extraction in the CLI.

### Fixed Issues

#### CLI Image Extraction Not Working
- **Problem:** The `vogel-trainer extract` command only processed video files, ignoring image files completely
- **Root Cause:** CLI's `extract_command()` function called `extract_birds_from_video()` directly instead of delegating to `extractor.main()`
- **Solution:** Refactored CLI to properly delegate to `extractor.main()` which handles videos, images, and convert mode
- **Impact:** All image extraction features (introduced in v0.1.16) now work correctly via CLI

#### Version Management
- **Problem:** CLI version string was hardcoded as "0.1.15" in `main.py`
- **Solution:** Version now dynamically loaded from `__version__.py` 
- **Benefit:** Single source of truth for version numbers across the package

#### Argument Mapping
- **Problem:** CLI arguments not properly mapped to `extractor.main()` parameters
- **Solution:** Implemented proper argument translation with `hasattr()` checks
- **Fixed:** Background color conversion (e.g., 'gray' → '128,128,128')

## 📋 What's Working Now

All v0.1.16 features are now fully functional via CLI:

### Image Extraction
```bash
# Single image with auto-classification
vogel-trainer extract photo.jpg \
  --folder ~/data/ \
  --species-model kamera-linux/german-bird-classifier \
  --threshold 0.25

# Directory with recursive search
vogel-trainer extract ~/photos/ \
  --folder ~/data/ \
  --bird blaumeise \
  --recursive

# With background removal and crop padding
vogel-trainer extract image.jpg \
  --folder ~/data/ \
  --bird rotkehlchen \
  --remove-background \
  --bg-transparent \
  --crop-padding 20
```

### Convert Mode
```bash
# Convert existing bird crops
vogel-trainer extract \
  --convert \
  --source ~/raw-data/ \
  --target ~/processed-data/ \
  --bg-remove \
  --bg-transparent
```

## 🔧 Technical Details

### Changes Made

**Modified Files:**
- `src/vogel_model_trainer/cli/main.py`:
  - Refactored `extract_command()` to delegate to `extractor.main()`
  - Added dynamic version loading from `__version__`
  - Implemented argument mapping with proper type conversions
  - Added color name to RGB conversion for background colors

**Version Updates:**
- `src/vogel_model_trainer/__version__.py`: 0.1.16 → 0.1.17
- `pyproject.toml`: version 0.1.16 → 0.1.17

### Upgrade Instructions

```bash
# From PyPI
pip install --upgrade vogel-model-trainer

# Verify version
vogel-trainer --version  # Should show: vogel-trainer 0.1.17

# Test image extraction
vogel-trainer extract test.jpg --folder output/ --bird test
```

## 🙏 Acknowledgments

Thanks to users who reported the CLI image extraction issue!

## 📚 Documentation

- [README.md](../README.md) - English documentation
- [README.de.md](../README.de.md) - German documentation  
- [README.ja.md](../README.ja.md) - Japanese documentation
- [CHANGELOG.md](../CHANGELOG.md) - Complete version history

## 🔗 Links

- **GitHub Release:** https://github.com/kamera-linux/vogel-model-trainer/releases/tag/v0.1.17
- **PyPI Package:** https://pypi.org/project/vogel-model-trainer/0.1.17/
- **Issues:** https://github.com/kamera-linux/vogel-model-trainer/issues
- **Discussions:** https://github.com/kamera-linux/vogel-model-trainer/discussions

---

**Full Changelog:** v0.1.16...v0.1.17
