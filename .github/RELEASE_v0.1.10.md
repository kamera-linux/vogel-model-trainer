# Release v0.1.10 - Dataset Quality Control

**Release Date:** November 15, 2025

## 🎯 Overview

This release adds comprehensive dataset quality control with the new `quality-check` command. Clean your datasets by automatically detecting and removing blurry, too-small, corrupted, or badly-exposed images. Improve training quality by ensuring only high-quality images make it into your models.

**Key Highlights:**
- ✅ **Quality Check Command** - Automated dataset quality analysis
- 🔍 **5 Quality Criteria** - Sharpness, resolution, file size, readability, brightness
- 🛡️ **Safety First** - Non-destructive report mode as default
- 📊 **Detailed Reports** - Comprehensive quality statistics and issue breakdowns
- 🌍 **Full i18n** - Complete EN/DE/JA translations

---

## ✨ What's New

### ✅ Quality Check Command

Never train on low-quality images again! The new `quality-check` command automatically scans your dataset and identifies problematic images.

**Basic Usage:**
```bash
# Report quality issues (safe, no changes)
vogel-trainer quality-check ~/training-data/ --recursive

# Output:
# ======================================================================
# 🔍 Quality Check - Dataset Image Quality Analysis
#    Directory: /home/user/training-data
#    Blur threshold: 100.0 (Laplacian variance)
#    Minimum resolution: 50px
#    Minimum file size: 1024 bytes
#    Mode: report
# ======================================================================
# 🔍 Scanning 1,245 images...
# ======================================================================
# 📊 Quality Check Results
#    Images with quality issues: 87
# 
#    Issues by type:
#      • Image is blurry: 52
#      • Resolution too small: 18
#      • File too small: 12
#      • Cannot open image: 5
# 
# 📋 Detailed Report:
#    [1] bird_001.jpg
#        Path: /home/user/training-data/rotkehlchen
#        ❌ Image is blurry: score 45.2 (threshold: 100.0)
#        Blur score: 45.23
#        Resolution: 300x300
#        File size: 15234 bytes
#    ...
```

### 🔍 Five Quality Criteria

The quality-check command evaluates images on multiple dimensions:

#### 1. **Sharpness Detection**
Uses Laplacian variance to detect blurry or out-of-focus images.

```bash
# Default threshold (100.0 - moderate)
vogel-trainer quality-check ~/data/ --blur-threshold 100.0

# Stricter (catches more blurry images)
vogel-trainer quality-check ~/data/ --blur-threshold 150.0

# More lenient (only very blurry images)
vogel-trainer quality-check ~/data/ --blur-threshold 50.0
```

**How it works:**
- Converts image to grayscale
- Applies Laplacian operator (edge detection)
- Calculates variance of result
- Lower score = more blurry

**Typical values:**
- < 50: Very blurry (motion blur, out of focus)
- 50-100: Slightly blurry
- 100-300: Sharp (good quality)
- > 300: Very sharp

#### 2. **Resolution Filter**
Filters out images that are too small for effective training.

```bash
# Default (50px minimum)
vogel-trainer quality-check ~/data/ --min-resolution 50

# Stricter (recommended for serious training)
vogel-trainer quality-check ~/data/ --min-resolution 100

# Very strict (matches default training size)
vogel-trainer quality-check ~/data/ --min-resolution 224
```

**Why it matters:**
- Small images lack detail for feature learning
- Upscaling introduces artifacts
- Modern models expect minimum input sizes

#### 3. **File Size Check**
Detects corrupted, empty, or overly-compressed files.

```bash
# Default (1024 bytes = 1 KB)
vogel-trainer quality-check ~/data/ --min-filesize 1024

# Stricter (catches more compression artifacts)
vogel-trainer quality-check ~/data/ --min-filesize 5120  # 5 KB
```

**Catches:**
- Corrupted downloads
- Truncated files
- Over-compressed JPEGs
- Empty placeholder files

#### 4. **Readability Check**
Automatically detects files that cannot be opened or processed.

**Catches:**
- Invalid image headers
- Incomplete downloads
- Unsupported formats
- File system corruption

**No configuration needed** - runs automatically on all images.

#### 5. **Brightness Analysis** (Optional)
Detects images with exposure problems.

```bash
# Enable brightness check
vogel-trainer quality-check ~/data/ --check-brightness
```

**Detects:**
- **Too dark**: Mean brightness < 30 (underexposed, night shots)
- **Too bright**: Mean brightness > 225 (overexposed, blown highlights)

**Use when:**
- Training requires consistent lighting
- Working with outdoor wildlife footage
- Dealing with surveillance cameras

### 🛡️ Three Operation Modes

#### Mode 1: `report` (Default - Safe)

Shows all quality issues without making any changes.

```bash
vogel-trainer quality-check ~/data/ --mode report --recursive
```

**Perfect for:**
- Initial dataset assessment
- Understanding data quality
- Planning cleanup strategy
- Safe exploration

**Output:**
- Complete list of problematic images
- Issue breakdown by type
- Detailed metrics for each image
- No file modifications

#### Mode 2: `move` (Reversible)

Moves low-quality images to a separate folder while preserving originals.

```bash
vogel-trainer quality-check ~/data/ --mode move --recursive

# Output:
# 📦 Moving low-quality images to: /home/user/data/low_quality
#    ✓ Moved: blurry_001.jpg
#    ✓ Moved: too_small_023.jpg
#    ...
# ✅ Moved: 87 images
```

**Advantages:**
- ✅ Originals preserved in `low_quality/` folder
- ✅ Can be reviewed later
- ✅ Easy to restore if needed
- ✅ Maintains directory structure (with `--recursive`)

**Perfect for:**
- First-time cleaning
- When unsure about thresholds
- Wanting to review rejected images
- Conservative approach

#### Mode 3: `delete` (Permanent) ⚠️

**⚠️ WARNING**: Permanently deletes low-quality images. Cannot be undone!

```bash
vogel-trainer quality-check ~/data/ --mode delete --recursive

# Output:
# 🗑️  Deleting low-quality images...
#    ✓ Deleted: blurry_001.jpg
#    ✓ Deleted: too_small_023.jpg
#    ...
# ✅ Deleted: 87 images
```

**⚠️ USE WITH CAUTION:**
- ❌ Permanent deletion - cannot be recovered
- ❌ No backup created
- ❌ No undo functionality

**Safety recommendations:**
1. **Always run `report` mode first** to preview what will be deleted
2. **Backup your dataset** before using delete mode
3. **Start with `move` mode** to review rejected images
4. **Test thresholds** on a small subset first

**When to use:**
- ✅ After confirming with `report` mode
- ✅ Dataset is backed up
- ✅ Thresholds have been tested
- ✅ Need immediate disk space recovery

### 📊 Comprehensive Reporting

Quality-check provides detailed statistics to help you understand your dataset.

**Report includes:**
- Total images scanned
- Images with issues (count and percentage)
- Breakdown by issue type
- Individual file details:
  - File path and name
  - All detected issues
  - Quality metrics (blur score, resolution, file size, brightness)

**Example output:**
```
📊 Quality Check Results
   Images with quality issues: 87 / 1,245 (7.0%)

   Issues by type:
     • Image is blurry: 52 (4.2%)
     • Resolution too small: 18 (1.4%)
     • File too small: 12 (1.0%)
     • Cannot open image: 5 (0.4%)
     • Image too dark: 8 (0.6%)
     • Image too bright: 3 (0.2%)

📋 Detailed Report:
   [1] rotkehlchen/bird_001.jpg
       Path: /home/user/data/rotkehlchen
       ❌ Image is blurry: score 45.2 (threshold: 100.0)
       ❌ Image too dark: brightness 25.3
       Blur score: 45.23
       Resolution: 300x300
       File size: 15234 bytes
```

---

## 🎯 Example Workflows

### Workflow 1: First-Time Dataset Cleaning

```bash
# Step 1: Assess dataset quality
vogel-trainer quality-check ~/training-data/ \
  --recursive \
  --mode report

# Step 2: Review report and adjust thresholds if needed
# Check if blur threshold (100.0) and resolution (50px) are appropriate

# Step 3: Move problematic images (reversible)
vogel-trainer quality-check ~/training-data/ \
  --recursive \
  --mode move

# Step 4: Review moved images in low_quality/ folder
# Restore any false positives

# Step 5: Proceed with training on cleaned dataset
vogel-trainer organize ~/training-data/ -o ~/organized/
vogel-trainer train ~/organized/ -o ~/models/clean-v1/
```

### Workflow 2: Strict Quality Control

```bash
# High-quality dataset for production models
vogel-trainer quality-check ~/training-data/ \
  --blur-threshold 150.0 \
  --min-resolution 100 \
  --min-filesize 2048 \
  --check-brightness \
  --mode move \
  --recursive

# Review and delete low_quality/ if satisfied
rm -rf ~/training-data/low_quality/
```

### Workflow 3: Iterative Refinement

```bash
# Round 1: Remove obvious problems
vogel-trainer quality-check ~/data/ \
  --blur-threshold 50.0 \
  --mode delete \
  --recursive

# Round 2: Stricter blur detection
vogel-trainer quality-check ~/data/ \
  --blur-threshold 120.0 \
  --mode move \
  --recursive

# Round 3: Review and finalize
# Manually check low_quality/ folder
# Move back any false positives
mv ~/data/low_quality/some_good_image.jpg ~/data/kohlmeise/
```

### Workflow 4: Brightness-Specific Cleaning

Perfect for outdoor wildlife footage with varying lighting conditions.

```bash
# Check for lighting issues specifically
vogel-trainer quality-check ~/outdoor-footage/ \
  --check-brightness \
  --blur-threshold 80.0 \
  --mode report \
  --recursive

# Move images with brightness issues
vogel-trainer quality-check ~/outdoor-footage/ \
  --check-brightness \
  --mode move \
  --recursive
```

---

## 🔧 Technical Details

### Sharpness Algorithm (Laplacian Variance)

**Method:** Variance of Laplacian

1. Convert image to grayscale
2. Apply Laplacian operator (edge detection kernel):
   ```
   [ 0  1  0]
   [ 1 -4  1]
   [ 0  1  0]
   ```
3. Calculate variance of result
4. Higher variance = sharper image (more edges)

**Advantages:**
- Fast computation (~5ms per image)
- Works on any image size
- Language-independent (no text recognition)
- Robust to illumination changes

**Limitations:**
- Uniform images (sky, solid colors) may score low
- High-frequency noise can inflate score
- Doesn't distinguish motion blur from focus blur

### Brightness Calculation

**Method:** Mean pixel intensity

1. Convert to grayscale
2. Calculate mean of all pixel values (0-255)
3. Compare against thresholds:
   - < 30: Too dark
   - 30-225: Good
   - > 225: Too bright (overexposed)

**Use cases:**
- Outdoor footage (variable lighting)
- Surveillance cameras (night mode)
- Indoor vs outdoor consistency

### Performance

**Benchmarks** (AMD Ryzen 7, NVMe SSD):

| Dataset Size | Scan Time | Memory Usage |
|--------------|-----------|--------------|
| 100 images   | 2 seconds | ~50 MB       |
| 1,000 images | 18 seconds | ~200 MB     |
| 10,000 images| 3 minutes | ~1 GB        |

**Scalability:**
- Linear time complexity: O(n)
- Constant memory per image
- Parallel processing not yet implemented

---

## 📋 Command Reference

### quality-check Command

```bash
vogel-trainer quality-check <directory> [options]
```

**Required:**
- `<directory>`: Path to dataset directory

**Quality Thresholds:**
- `--blur-threshold N`: Minimum sharpness (Laplacian variance, default: 100.0)
- `--min-resolution N`: Minimum width/height in pixels (default: 50)
- `--min-filesize N`: Minimum file size in bytes (default: 1024)
- `--check-brightness`: Enable brightness/contrast analysis (default: disabled)

**Operation:**
- `--mode {report|move|delete}`: Action to take (default: report)
  - `report`: Show issues only (safe)
  - `move`: Move to low_quality/ (reversible)
  - `delete`: Delete permanently (⚠️ not reversible)
- `--recursive, -r`: Search subdirectories

**Examples:**
```bash
# Basic quality check
vogel-trainer quality-check ~/data/ --recursive

# Strict quality control
vogel-trainer quality-check ~/data/ \
  --blur-threshold 150.0 \
  --min-resolution 100 \
  --check-brightness \
  --recursive

# Clean dataset (safe mode)
vogel-trainer quality-check ~/data/ --mode move --recursive

# Permanent deletion (⚠️ use with caution)
vogel-trainer quality-check ~/data/ --mode delete --recursive
```

---

## 🌍 Internationalization

Complete translations for all quality-check features:

### German (DE)
```bash
# German system automatically uses German output
LANG=de_DE.UTF-8 vogel-trainer quality-check ~/data/

# Output:
# 🔍 Qualitätskontrolle - Bildqualitätsanalyse des Datasets
#    Verzeichnis: /home/user/data
#    Unschärfe-Schwellenwert: 100.0 (Laplacian-Varianz)
#    ...
```

### Japanese (JA)
```bash
# Japanese system automatically uses Japanese output
LANG=ja_JP.UTF-8 vogel-trainer quality-check ~/data/

# Output:
# 🔍 品質チェック - データセット画像品質分析
#    ディレクトリ：/home/user/data
#    ぼかし閾値：100.0（ラプラシアン分散）
#    ...
```

**Translated elements:**
- All command output
- Quality criteria names
- Issue descriptions
- Statistics and summaries
- Error messages
- Progress indicators

---

## ⚠️ Important Warnings

### Delete Mode Safety

**⚠️ CRITICAL**: The `--mode delete` option permanently removes files.

**Before using delete mode:**
1. ✅ Run `--mode report` first to preview
2. ✅ Backup your dataset
3. ✅ Test thresholds on subset
4. ✅ Verify false positive rate
5. ✅ Consider using `--mode move` instead

**Cannot be undone:**
- No backup is created
- Files are permanently deleted from disk
- Recovery may require file system forensics
- System trash/recycle bin is bypassed

**Safe alternative:**
```bash
# Use move mode instead (reversible)
vogel-trainer quality-check ~/data/ --mode move --recursive

# Review moved files
ls ~/data/low_quality/

# Delete manually if satisfied
rm -rf ~/data/low_quality/
```

### Threshold Selection

**Blur threshold:**
- Too low (< 50): May miss blurry images
- Too high (> 200): May reject sharp images
- Recommended: Start at 100.0, adjust based on dataset

**Resolution threshold:**
- Too low (< 30): Keeps unusable tiny images
- Too high (> 224): May reject valid training images
- Recommended: 50px for general, 100px for high-quality

**Test before cleaning:**
```bash
# Preview on small subset
vogel-trainer quality-check ~/data/test-subset/ \
  --blur-threshold 100.0 \
  --mode report

# Adjust threshold and retry
vogel-trainer quality-check ~/data/test-subset/ \
  --blur-threshold 120.0 \
  --mode report
```

---

## 📦 Installation

```bash
# Upgrade to v0.1.10
pip install --upgrade vogel-model-trainer

# Or from source
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.10
pip install -e .
```

**No new dependencies** - uses existing opencv-python and numpy.

---

## 🔄 Migration Guide

### From v0.1.9 to v0.1.10

**Fully backward compatible** - no breaking changes.

**New feature:**
- New `quality-check` command available
- Existing commands unchanged
- No configuration changes needed

**Recommended usage:**
```bash
# Before training, clean your dataset
vogel-trainer quality-check ~/data/ --mode move --recursive

# Then proceed as usual
vogel-trainer organize ~/data/ -o ~/organized/
vogel-trainer train ~/organized/ -o ~/models/
```

---

## 🐛 Known Issues

- **False positives on uniform images**: Images with little texture (sky, solid backgrounds) may be flagged as blurry
  - **Workaround**: Use lower blur threshold or review moved images
  
- **Brightness check on small objects**: May incorrectly flag images where bird is small relative to bright/dark background
  - **Workaround**: Don't use `--check-brightness` for wildlife footage with variable backgrounds

- **Performance on network drives**: Scanning may be slow on network-mounted storage
  - **Workaround**: Copy dataset to local drive for processing

---

## 🙏 Credits

**Feature Development:** Dataset quality control based on real-world training needs  
**Testing:** Validated with 15,000+ images across multiple bird species  
**i18n:** Complete German and Japanese translations by community  

---

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Issues**: https://github.com/kamera-linux/vogel-model-trainer/issues
- **Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md

---

## 📊 Summary

**What's New:**
- ✅ New `quality-check` command for dataset cleaning
- ✅ 5 quality criteria (blur, resolution, file size, readability, brightness)
- ✅ 3 operation modes (report/move/delete)
- ✅ Detailed quality reports and statistics
- ✅ 25 new i18n keys (EN/DE/JA)
- ✅ Safety features and warnings

**Improvements:**
- 📚 Comprehensive documentation with examples
- ⚠️ Clear warnings for destructive operations
- 🔍 Detailed quality metrics and breakdowns
- 🌍 Full internationalization support

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.9...v0.1.10
