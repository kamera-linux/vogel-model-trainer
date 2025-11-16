# Release v0.1.14 - Gray Background Default & Clean-Gray Command

## 🎨 Gray Background as Default for Training

Changed default background from **transparent** to **gray (#808080)** for better model training:

- ✅ Most training frameworks ignore alpha channel (RGBA→RGB conversion)
- ✅ Neutral gray provides consistent background without affecting bird colors
- ✅ Reduces file size (JPEG vs PNG)
- ✅ No transparency artifacts or edge issues

### Migration Guide

**Old behavior (v0.1.13 and earlier):**
```bash
# Created PNG with transparent background by default
vogel-trainer extract video.mp4 --remove-background
```

**New behavior (v0.1.14+):**
```bash
# Creates JPEG with gray background by default (optimal for training)
vogel-trainer extract video.mp4 --remove-background

# Use --bg-transparent if you need transparent backgrounds
vogel-trainer extract video.mp4 --remove-background --bg-transparent
```

### Updated Defaults

| Parameter | Old Default | New Default | Reason |
|-----------|-------------|-------------|---------|
| `--bg-color` | `white` | `gray` | Neutral for training |
| `--bg-transparent` | `True` (PNG) | `False` (JPEG) | Better for training |
| `--bg-fill-black` | `True` | `False` | Not needed for gray |

## 🧹 New Clean-Gray Command

Added dataset validation for gray background images:

```bash
# Check dataset for invalid gray ratios
vogel-trainer clean-gray ~/dataset/sperling/ --mode report

# Remove images with too much/little gray
vogel-trainer clean-gray ~/dataset/ --mode delete --recursive

# Move invalid images to invalid_gray/ folder
vogel-trainer clean-gray ~/dataset/ --mode move
```

### Features

- **Too much gray** (>95%): Mostly background, no bird visible
- **Too little gray** (<5%): No background padding, bird fills entire image
- **Configurable tolerance**: `--gray-tolerance 30` (default)
- **Works with both**: JPEG and PNG images
- **Three modes**: report, move, delete

### Example Output

```
🔍 Checking 14 images for gray background...
📊 Thresholds:
   • Min gray ratio: 5%
   • Max gray ratio: 95%
   • Gray tolerance: ±30

❌ bird_001.jpg
   Reason: Too much gray background (96.5% > 95.0%)
   Gray: 96.5% (142274 pixels), Bird: 3.5% (5182 pixels)

======================================================================
📊 Summary:
   Total images checked: 14
   Valid images: 13
   Invalid (wrong gray ratio): 1
```

## 🐛 Bug Fixes

### Fixed Background Color Parameter

**Issue**: `--bg-color gray` was producing black backgrounds instead of gray

**Root causes**:
1. Padding color was hardcoded to `(0, 0, 0)` black
2. `--bg-fill-black` was applied to colored backgrounds

**Fixed**:
- Padding now uses `bg_color` parameter with BGR→RGB conversion
- `--bg-fill-black` only applies when `--bg-transparent` is enabled

**Before (v0.1.13)**:
```bash
vogel-trainer extract video.mp4 --bg-color gray --no-bg-transparent
# Produced: Black padding areas ❌
```

**After (v0.1.14)**:
```bash
vogel-trainer extract video.mp4 --bg-color gray --no-bg-transparent
# Produces: Gray padding areas ✅
```

## 📦 Installation

```bash
pip install --upgrade vogel-model-trainer
```

## 🔗 Related Issues

- Background color parameter not working correctly
- Training optimization: transparent vs colored backgrounds
- Dataset validation for gray backgrounds

## 🙏 Thanks

Thanks to the community for reporting the background color bug and suggesting gray as the optimal default for training!

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.13...v0.1.14
