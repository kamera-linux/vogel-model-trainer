# Release v0.1.12 - Transparent Background & Dataset Cleaning 🧹✨

## 🎯 Release Highlights

**Major Improvements**: Transparent background as default + new tool to clean fragmented images!

### 🎨 Transparent Background as DEFAULT (v0.1.12)

Background removal now creates **PNG with alpha channel** by default - perfect for clean, isolated training images!

```bash
# Simple - transparent background by default
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird rotkehlchen \
  --remove-background

# Use colored background instead
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird blaumeise \
  --remove-background \
  --no-bg-transparent \
  --bg-color black
```

**What Changed:**
- ✅ `--bg-transparent` is now **TRUE** by default (PNG with alpha)
- ✅ `--bg-fill-black` is now **TRUE** by default (black areas transparent)
- ✅ Automatically saves as **PNG** (transparent) or **JPEG** (opaque)
- ✅ Use `--no-bg-transparent` to switch back to colored backgrounds

**Benefits:**
- Better model training with clean, isolated birds
- No background artifacts or color bleeding
- Professional-looking dataset images
- Smaller file sizes with PNG compression
- Perfect for transfer learning and fine-tuning

### 🧹 New: Clean Transparent Command (v0.1.12)

After background removal, automatically detect and remove fragmented/incomplete bird images:

```bash
# Safe mode - preview only
vogel-trainer clean-transparent ~/training-data/ --mode report

# Move invalid images to separate folder
vogel-trainer clean-transparent ~/training-data/ --mode move

# Permanently delete invalid images
vogel-trainer clean-transparent ~/training-data/ --mode delete

# Scan recursively through subdirectories
vogel-trainer clean-transparent ~/training-data/ --mode move --recursive
```

**Detection Criteria:**

1. **Too Transparent** - Images with >95% transparency (adjustable with `--max-transparency`)
2. **Too Few Pixels** - Less than 500 visible pixels (adjustable with `--min-pixels`)
3. **Fragmented** - Largest connected region < 100 pixels (adjustable with `--min-region`)

**Example Output:**
```
🔍 Checking 156 PNG images...
📊 Thresholds:
   • Min visible pixels: 500
   • Max transparency: 95%
   • Min region size: 100

❌ bird_fragment_001.png
   Reason: Too transparent (97.2% > 95.0%)
   Visible pixels: 4104, Transparency: 97.2%, Largest region: 0

======================================================================
📊 Summary:
   Total images checked: 156
   Valid images: 143
   Invalid/Fragmented: 13

📁 Moving 13 invalid images to invalid_transparent/...
✅ Moved 13 images
```

**Use Cases:**
- Remove tiny fragments after AI background removal
- Clean up partial bird detections (bird flew out of frame)
- Eliminate heavily transparent images
- Find disconnected/scattered pixel groups
- Quality control for training datasets

## 🎬 Complete Workflow Example

```bash
# Step 1: Extract birds with transparent background (NEW DEFAULT)
vogel-trainer extract video.mp4 \
  --folder ~/raw-data/ \
  --species-model kamera-linux/german-bird-classifier \
  --remove-background \
  --min-sharpness 150 \
  --deduplicate

# Step 2: Clean up fragmented/incomplete images (NEW)
vogel-trainer clean-transparent ~/raw-data/ \
  --mode move \
  --recursive

# Step 3: Organize into train/val split
vogel-trainer organize ~/raw-data/ \
  -o ~/clean-dataset/ \
  --max-images-per-class 200

# Step 4: Train your model
vogel-trainer train ~/clean-dataset/ \
  -o ~/models/my-classifier/
```

## 📋 Full Changelog

### Added
- **🧹 Clean Transparent Command**: New `clean-transparent` command for dataset cleaning
  - Detects fragmented/incomplete transparent PNG images
  - Three validation criteria: transparency %, visible pixels, region size
  - Three modes: `report`, `move`, `delete`
  - Configurable thresholds: `--min-pixels`, `--max-transparency`, `--min-region`
  - Recursive directory scanning with `--recursive`
  - Automatic invalid image detection using connected component analysis

### Changed
- **🎨 Transparent Background as DEFAULT**: 
  - `--bg-transparent` now defaults to TRUE (creates PNG with alpha channel)
  - `--bg-fill-black` now defaults to TRUE (black areas become transparent)
  - Automatically saves as PNG when transparent, JPEG when opaque
  - Use `--no-bg-transparent` to revert to colored backgrounds
  - Use `--no-bg-fill-black` to keep black padding opaque

### Improved
- Better RGBA image handling during resizing and padding
- Automatic file format selection based on transparency
- Enhanced validation for transparent images
- Cleaner dataset creation with fragment detection

## 📦 Installation

### From PyPI

```bash
pip install vogel-model-trainer --upgrade
```

### With GPU Support (Recommended for Background Removal)

```bash
pip install vogel-model-trainer --upgrade
pip install rembg[gpu]
```

### From Source

```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.12
pip install -e .
```

## 💡 Tips & Best Practices

### Transparent Background

1. **Default is now transparent** - no need to specify `--bg-transparent`
2. **Black boxes are removed** - padding areas become transparent automatically
3. **Use colored backgrounds** for specific needs: `--no-bg-transparent --bg-color white`
4. **Green/Blue screen** for chroma keying: `--bg-color green-screen`

### Cleaning Datasets

1. **Always use report mode first**: `--mode report` to preview
2. **Adjust thresholds** based on your bird size: larger birds need higher `--min-pixels`
3. **Move, don't delete**: Use `--mode move` to review invalid images later
4. **Run recursively**: Add `--recursive` to scan all species subdirectories
5. **Quality control**: Check `invalid_transparent/` folder before training

### Image Quality Workflow

```bash
# 1. Extract with high quality filters
vogel-trainer extract video.mp4 \
  --folder data/ \
  --remove-background \
  --min-sharpness 150 \
  --min-edge-quality 80 \
  --deduplicate

# 2. Remove fragments
vogel-trainer clean-transparent data/ --mode move --recursive

# 3. Check for other quality issues
vogel-trainer quality-check data/ \
  --blur-threshold 120 \
  --mode report \
  --recursive

# 4. Organize clean dataset
vogel-trainer organize data/ -o clean-data/ --max-images-per-class 150
```

## ⚠️ Breaking Changes

None - all changes are backward compatible! Old commands still work.

## 🐛 Known Issues

None reported yet (new features)

## 📝 Documentation

- **English**: [README.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.md)
- **German**: [README.de.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.de.md)
- **Japanese**: [README.ja.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.ja.md)

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Issues**: https://github.com/kamera-linux/vogel-model-trainer/issues
- **Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.11...v0.1.12
