# Release v0.1.15 - Enhanced Training & PNG Support

**Release Date:** November 17, 2025

## 🎯 Highlights

This release brings significant improvements for training with transparent backgrounds, better background removal control, and full PNG support throughout the pipeline.

### Key Features

- **🎯 Crop Padding for Better Background Removal**: Preserve important bird details (feet, beak, feathers)
- **🎨 Random Background Augmentation**: Improved training for transparent PNG images
- **🖼️ Full PNG Pipeline Support**: From extraction to organization to training
- **🔧 ARM64/Raspberry Pi Compatibility**: Better support for embedded systems

## ✨ New Features

### Crop Padding (`--crop-padding`)

Expands the rembg foreground mask to preserve bird details that might otherwise be cut off:

```bash
vogel-trainer extract video.mp4 \
  --remove-background \
  --crop-padding 20 \
  --bg-transparent
```

- **Recommended values**: 5-20 pixels
- **Use case**: Prevents aggressive background removal from cutting feet, beaks, or wing tips
- **Works with**: `--remove-background` flag

### Random Background Augmentation

When training with transparent PNG images, the model now applies random backgrounds:

- **Training phase**: Random gray/black/white backgrounds
- **Validation phase**: Consistent neutral gray
- **Benefit**: Model focuses on bird features instead of background color
- **Automatic**: Applied when PNG/RGBA images detected

### ARM64/Raspberry Pi Support

- Added `onnxruntime>=1.15.0` as explicit dependency
- Fixes rembg installation issues on ARM64 architectures
- Dynamic import check for better error handling

## 🐛 Bug Fixes

### PNG Transparency with All Features

Fixed complex bug where transparent PNGs weren't created when combining:
- `--species-model` (classification)
- `--bg-transparent` (transparent background)
- `--deduplicate` (duplicate detection)

**Three separate fixes required:**

1. **Filename Extension Logic**: Fixed `.jpg` hardcoded in filename when species model used
2. **Deduplicate RGBA Handling**: Preserve alpha channel during duplicate detection
3. **Variable Caching**: Explicit deletion to prevent RGB mode being reused after background removal

**Verification:** All PNG images now correctly saved as RGBA format in all scenarios.

### Dataset Organization PNG Support

The `organize` command now supports PNG files:

```bash
vogel-trainer organize /raw/dataset/ -o /organized/ --max-images-per-class 100
```

- Previously only worked with JPG files
- Now detects and organizes both `.jpg` and `.png` files
- Essential for organizing transparent background datasets

## 📊 Testing Results

Comprehensive end-to-end testing performed:

### Test Setup
- **Videos**: 3 videos processed (20251020, 20251021, 20251015)
- **Species**: Blaumeise, Sumpfmeise
- **Features tested**: Transparent PNG + crop-padding + deduplicate + quality filters

### Results
- ✅ **192 PNG images** extracted with RGBA format
- ✅ **Dataset organized** with train/val split (80/20)
- ✅ **Training successful** with random background augmentation
- ✅ **92.9% validation accuracy** after 3 epochs
- ✅ **Model inference** correctly classifies species

### Format Verification
```bash
$ file dataset/train/sumpfmeise/*.png
PNG image data, 224 x 224, 8-bit/color RGBA, non-interlaced
```

## 📚 Documentation Updates

All documentation updated across 3 languages:
- ✅ English: README.md
- ✅ German: README.de.md  
- ✅ Japanese: README.ja.md

New sections added:
- "Training with Transparent Backgrounds" guide
- Crop padding usage examples
- PNG workflow documentation

## 🔄 Migration Guide

### For Users

**If you're using transparent backgrounds:**

```bash
# Old way (might create JPG instead of PNG)
vogel-trainer extract video.mp4 --remove-background --bg-transparent

# New way (guaranteed PNG with deduplicate)
vogel-trainer extract video.mp4 \
  --remove-background \
  --bg-transparent \
  --crop-padding 20 \
  --deduplicate
```

**If organizing datasets:**

```bash
# Now works with both JPG and PNG
vogel-trainer organize /dataset/raw/ -o /dataset/organized/
```

### For Developers

No breaking changes. All new features are opt-in via command-line flags.

## 🏗️ Technical Details

### Code Changes

**Modified Files:**
- `src/vogel_model_trainer/core/extractor.py`: PNG transparency fixes (3 separate changes)
- `src/vogel_model_trainer/core/organizer.py`: PNG support added
- `src/vogel_model_trainer/core/trainer.py`: Random background augmentation

**Affected Lines:**
- Extractor: Lines 547-552, 584-590, 609-619
- Organizer: Lines 42-56
- Trainer: Lines 43-53, 148-168

### Dependencies

No new dependencies added. `onnxruntime>=1.15.0` made explicit (was already required by rembg).

## 🙏 Acknowledgments

Special thanks to the community for reporting the PNG transparency issues and helping test the ARM64 improvements.

## 📦 Installation

```bash
# Upgrade to v0.1.15
pip install --upgrade vogel-model-trainer

# Or from source
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.15
pip install -e .
```

## 🔗 Links

- **Full Changelog**: [CHANGELOG.md](../CHANGELOG.md#0115---2025-11-17)
- **Documentation**: [README.md](../README.md)
- **Issues**: [GitHub Issues](https://github.com/kamera-linux/vogel-model-trainer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kamera-linux/vogel-model-trainer/discussions)

---

**Full Changelog**: v0.1.14...v0.1.15
