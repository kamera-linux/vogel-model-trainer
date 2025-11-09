# Release v0.1.4 - Enhanced Extraction Statistics

**Release Date:** November 9, 2025

## 🎯 Overview

This bugfix release improves transparency in the extraction process by providing detailed statistics about detected, exported, and skipped birds. The bird counting bug when using `--species-threshold` has been fixed, and users now get a clear breakdown of quality control impact.

## 🐛 Bug Fixes

### Fixed Bird Count with Species Threshold

**Problem:** When using `--species-threshold`, the tool incorrectly counted ALL detected birds, including those that were skipped due to low confidence.

**Example of the bug:**
```bash
vogel-trainer extract video.mp4 \
  --species-model classifier/ \
  --species-threshold 0.85

# Output showed:
# ⏭️  Skipped: rotkehlchen (conf 0.37 < 0.85), frame 5200
# ⏭️  Skipped: rotkehlchen (conf 0.39 < 0.85), frame 5300
# ⏭️  Skipped: rotkehlchen (conf 0.39 < 0.85), frame 5500
# ⏭️  Skipped: rotkehlchen (conf 0.79 < 0.85), frame 5700
# ✅ Total birds extracted: 4  ← WRONG! Should be 0
```

**Fixed:** Now correctly counts only birds that were actually exported.

## ✨ What's New

### 📊 Detailed Extraction Statistics

The extraction summary now provides three distinct counters for complete transparency:

```bash
✅ Extraction complete!
   📁 Output directory: /home/user/data/
   🔍 Detected birds total: 4        # All birds found by YOLO
   🐦 Exported birds: 0              # Birds actually saved
   ⏭️  Skipped (< 0.85): 4          # Birds filtered by threshold
   🆔 Session ID: video_20251109_132913
```

**Benefits:**
- ✅ **Clear visibility** of quality control impact
- 📊 **Accurate statistics** showing what was actually saved
- 🎯 **Better understanding** of threshold effectiveness
- 🔍 **Debugging support** when adjusting threshold values

### 🌍 Multi-Language Support

All new statistics are fully translated in three languages:

**English:**
```
   🔍 Detected birds total: 4
   🐦 Exported birds: 0
   ⏭️  Skipped (< 0.85): 4
```

**German:**
```
   🔍 Erkannte Vögel gesamt: 4
   🐦 Exportierte Vögel: 0
   ⏭️  Übersprungen (< 0.85): 4
```

**Japanese:**
```
   🔍 検出された鳥の総数：4
   🐦 エクスポートされた鳥：0
   ⏭️  スキップされた (< 0.85)：4
```

## 🔄 Changes

### Code Changes
- **extractor.py**: Fixed bird counting logic to increment counter AFTER threshold check
- **extractor.py**: Added `detected_count` and `skipped_count` tracking
- **extractor.py**: Enhanced output with three separate statistics
- **i18n.py**: Added new translation keys for detailed statistics

### Translation Keys Added
- `detected_birds_total`: Total birds detected by YOLO
- `exported_birds_total`: Birds successfully exported
- `skipped_birds_total`: Birds filtered by threshold

## 📋 Use Cases

### Use Case 1: Quality Control Monitoring

```bash
# Strict threshold for high-quality dataset
vogel-trainer extract videos/*.mp4 \
  --species-model classifier/ \
  --species-threshold 0.90

# Output shows exactly how strict your filter is:
# 🔍 Detected birds total: 50
# 🐦 Exported birds: 30
# ⏭️  Skipped (< 0.90): 20
# → 60% pass rate, adjust threshold if needed
```

### Use Case 2: Threshold Tuning

```bash
# Test different thresholds to find optimal balance

# Very strict (0.95):
# 🔍 Detected: 100 | 🐦 Exported: 20 | ⏭️ Skipped: 80  ← Too strict

# Moderate (0.85):
# 🔍 Detected: 100 | 🐦 Exported: 65 | ⏭️ Skipped: 35  ← Good balance

# Lenient (0.70):
# 🔍 Detected: 100 | 🐦 Exported: 90 | ⏭️ Skipped: 10  ← More data, lower quality
```

### Use Case 3: Model Quality Assessment

```bash
# See how confident your classifier is
vogel-trainer extract new-videos/*.mp4 \
  --species-model classifier/v1/ \
  --species-threshold 0.80

# High skip rate might indicate:
# - Poor lighting in videos
# - Model needs more training data
# - Unfamiliar bird poses/angles
# - Wrong species in videos (not in training set)
```

## 🔧 Technical Details

### Bug Fix Implementation

**Before (buggy):**
```python
bird_count += 1  # Counted too early!
# ... species classification ...
if species_conf < threshold:
    continue  # Skip, but already counted
```

**After (fixed):**
```python
detected_count += 1  # Track all detections
# ... species classification ...
if species_conf < threshold:
    skipped_count += 1
    continue  # Skip before counting
bird_count += 1  # Count only exported birds
```

### Statistics Logic

```python
detected_count = 0   # All birds found by YOLO
bird_count = 0       # Birds actually exported
skipped_count = 0    # Birds filtered by threshold

# Invariant: detected_count = bird_count + skipped_count
```

## 📦 Installation

```bash
# Upgrade to v0.1.4
pip install --upgrade vogel-model-trainer

# Or from source
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.4
pip install -e .
```

## 🔄 Migration Guide

**No migration required!** This release is fully backward compatible:

- All existing commands work exactly as before
- Only the output statistics have been improved
- No changes to command-line arguments
- No changes to file formats or APIs

**If you were parsing the old output:**
```python
# Old output format (still works, but deprecated):
# "🐦 Total birds extracted: 4"

# New output format (recommended):
# "🔍 Detected birds total: 4"
# "🐦 Exported birds: 0"
# "⏭️  Skipped (< 0.85): 4"
```

## 🎯 Example Workflow

```bash
# Extract with quality control
vogel-trainer extract videos/*.mp4 \
  --folder data/ \
  --species-model classifier/final/ \
  --species-threshold 0.85

# Output shows complete picture:
# ✅ Extraction complete!
#    📁 Output directory: data/
#    🔍 Detected birds total: 120      ← YOLO found 120 birds
#    🐦 Exported birds: 95             ← 95 met confidence threshold
#    ⏭️  Skipped (< 0.85): 25          ← 25 filtered out
#    🆔 Session ID: batch_20251109_140523

# Now you know:
# - Detection is working well (120 birds found)
# - Classifier is confident on 79% of detections (95/120)
# - Quality control filtered 21% uncertain predictions
```

## 🐛 Known Issues

None currently identified.

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Issues**: https://github.com/kamera-linux/vogel-model-trainer/issues
- **Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md

## 🙏 Thank You

Thank you for using vogel-model-trainer! This bugfix was identified through real-world usage and improves the transparency and accuracy of the extraction process.

If you notice the skipped count being unexpectedly high, consider:
- Reviewing your `--species-threshold` value (lower = more inclusive)
- Checking if your classifier needs more training data
- Verifying video quality and bird visibility

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.3...v0.1.4
