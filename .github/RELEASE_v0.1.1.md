# 🔧 vogel-model-trainer v0.1.1

**Release Date:** November 8, 2025

Bug fix release with corrected CLI parameters to match original training scripts.

## 📦 Installation

```bash
pip install vogel-model-trainer==0.1.1
```

## 🐛 Bug Fixes

### CLI Parameter Corrections

Fixed CLI parameters to match the original `extract_birds.py` script:

- ✅ **Changed `--output/-o` to `--folder`** - Matches original parameter naming
- ✅ **Added `--bird` parameter** - Manual species naming (creates subdirectory)
- ✅ **Added `--species-model` parameter** - Auto-sorting with trained classifier
- ✅ **Added `--no-resize` flag** - Keep original image size
- ✅ **Renamed `--model` to `--detection-model`** - Clearer naming for YOLO model
- ✅ **Updated default `--threshold`** - Changed from 0.3 to 0.5 for higher quality
- ✅ **Updated default `--sample-rate`** - Changed from 10 to 3 for better detection
- ✅ **Added `--recursive/-r` flag** - Recursive directory search

### Simplified CLI Interface

- ✅ **Removed `extract-manual` command** - Now use `extract --bird species-name`
- ✅ **Removed `extract-auto` command** - Now use `extract --species-model path`
- ✅ **Single unified `extract` command** - Supports all three modes via flags

## 📚 Updated Documentation

- ✅ Updated README.md with correct CLI examples
- ✅ Updated README.de.md (German documentation)
- ✅ Corrected all code examples to use new parameter names

## 🔄 Migration Guide

### Before (v0.1.0):
```bash
# Manual extraction
vogel-trainer extract video.mp4 -o training-data/

# Manual labeling (separate command)
vogel-trainer extract-manual video.mp4 -o training-data/

# Auto-sorting (separate command)
vogel-trainer extract-auto video.mp4 -o training-data/ --classifier model/
```

### After (v0.1.1):
```bash
# Standard extraction (all birds to one directory)
vogel-trainer extract video.mp4 --folder training-data/

# Manual labeling (creates subdirectory)
vogel-trainer extract video.mp4 --folder training-data/ --bird rotkehlchen

# Auto-sorting (with trained classifier)
vogel-trainer extract video.mp4 --folder training-data/ --species-model model/final/
```

## 📖 Complete CLI Reference

### Extract Command

```bash
vogel-trainer extract VIDEO --folder FOLDER [OPTIONS]
```

**Positional Arguments:**
- `VIDEO` - Video file, directory, or glob pattern (e.g., `*.mp4`, `~/Videos/**/*.mp4`)

**Required Options:**
- `--folder` - Base directory for extracted bird images

**Optional Flags:**
- `--bird` - Manual bird species name (creates subdirectory)
- `--species-model` - Path to species classifier for automatic sorting
- `--no-resize` - Keep original image size instead of 224x224px
- `--detection-model` - YOLO model path (default: `yolov8n.pt`)
- `--threshold` - Detection confidence threshold (default: `0.5`)
- `--sample-rate` - Analyze every Nth frame (default: `3`)
- `--recursive, -r` - Search directories recursively

**Examples:**
```bash
# Standard mode
vogel-trainer extract video.mp4 --folder data/

# Manual mode with species
vogel-trainer extract video.mp4 --folder data/ --bird kohlmeise

# Auto-sort mode
vogel-trainer extract "*.mp4" --folder data/ --species-model ~/models/classifier/

# Recursive search
vogel-trainer extract ~/Videos/ --folder data/ --bird amsel --recursive

# Custom parameters
vogel-trainer extract video.mp4 --folder data/ --threshold 0.6 --sample-rate 5
```

## ⚙️ Technical Details

- **Changed Files:** 
  - `src/vogel_model_trainer/cli/main.py` - Complete CLI parameter overhaul
  - `README.md` - Updated all examples
  - `README.de.md` - Updated German examples

- **Backward Compatibility:** ⚠️ **Breaking changes** - v0.1.0 commands will not work with v0.1.1

## 🙏 Thank You

Thanks to users who reported the parameter inconsistencies between the CLI and original training scripts!

---

**Full Changelog:** https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.0...v0.1.1

**Questions?** Open an issue at https://github.com/kamera-linux/vogel-model-trainer/issues
