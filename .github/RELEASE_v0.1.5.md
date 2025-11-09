# Release v0.1.5 - Training Fix & Complete i18n

**Release Date:** November 9, 2025

## 🎯 Overview

This release fixes a critical training bug that prevented model training from working and completes the internationalization (i18n) coverage for all core components. Users can now train models successfully and experience fully translated output in their preferred language.

## 🐛 Critical Bug Fix

### Fixed Training Error

**Problem:** Training command failed with cryptic PyTorch error preventing any model training:
```
Error: expected Tensor as element 0 in argument 0, but got list
```

**Root Cause:** The data transformation pipeline was returning a Python list of PyTorch tensors, but HuggingFace datasets expected numpy arrays for proper serialization.

**Solution:** 
1. Convert tensors to numpy arrays in `transform_function()` using `.numpy()`
2. Add `set_format(type="torch")` to convert back to tensors for training
3. Simplified `collate_fn()` to handle the corrected data flow

**Impact:** Training now works correctly! Users can train custom bird classifiers without errors.

**Example (now working):**
```bash
vogel-trainer train ~/organized-data/ -o ~/models/my-classifier/
# ✅ Training completes successfully with proper accuracy metrics
```

## ✨ What's New

### 🌍 Complete Internationalization Coverage

All user-facing output is now fully translated in **English**, **German**, and **Japanese**.

#### Training Module (trainer.py)
- Model loading messages
- Dataset information and statistics  
- Species detection and label mapping
- Training progress indicators
- Graceful shutdown handling (Ctrl+C)
- Model saving and completion messages

**Example output (German):**
```
============================================================
Vogel-Artenerkennung Training
============================================================
(Drücke Strg+C zum sauberen Beenden)
============================================================

🔍 Erkenne Vogelarten aus Verzeichnisstruktur...
📁 Ausgabe-Verzeichnis: /home/user/models/classifier-20251109_153609
🐦 Vogelarten: blaumeise, grünling, haussperling, ...
🔢 Anzahl Klassen: 8

📂 Lade Dataset...
   Training:   879 Bilder
   Validation: 224 Bilder

🚀 Starte Training...
📦 Batch Size: 16
📈 Learning Rate: 0.0002
🔄 Epochen: 50
```

#### Testing Module (tester.py)
- Model loading status
- Image classification results
- Validation set testing output
- Accuracy metrics and summaries
- Usage instructions

**Example output (German):**
```bash
vogel-trainer test models/final/ image.jpg

🤖 Lade Modell: models/final/
🖼️  Klassifiziere Bild: image.jpg

Ergebnisse:
==================================================
1. rotkehlchen    - 0.9234 (92.3%)
2. blaumeise      - 0.0456 (4.6%)
```

#### CLI Commands
All command-line interface output fully translated:
- **Extract**: Video processing, bird detection, output summaries
- **Organize**: Dataset organization, train/val split information
- **Train**: Training initialization, progress, results

### 📋 New Translation Keys

Added **42 new translation keys** across all languages:

**Training (30 keys):**
- `train_loading_model`, `train_header`, `train_ctrl_c_hint`
- `train_detecting_species`, `train_output_dir`, `train_species`
- `train_loading_dataset`, `train_dataset_size`, `train_val_size`
- `train_starting`, `train_batch_size`, `train_learning_rate`
- `train_complete`, `train_model_saved`
- And more...

**Testing (12 keys):**
- `test_loading_model`, `test_classifying_image`, `test_results`
- `test_on_validation`, `test_no_images_found`
- `test_overall_accuracy`, `test_usage`, `test_error_no_input`
- And more...

## 🔧 Technical Details

### Data Pipeline Fix

**Before (broken):**
```python
def transform_function(examples, processor, is_training=True):
    pixel_values = []
    for img in examples["image"]:
        processed = processor(img, return_tensors="pt")
        pixel_values.append(processed["pixel_values"][0])  # ❌ PyTorch tensor
    examples["pixel_values"] = pixel_values  # ❌ List of tensors
    return examples
```

**After (fixed):**
```python
def transform_function(examples, processor, is_training=True):
    pixel_values = []
    for img in examples["image"]:
        processed = processor(img, return_tensors="pt")
        pixel_values.append(processed["pixel_values"][0].numpy())  # ✅ NumPy array
    examples["pixel_values"] = pixel_values  # ✅ List of numpy arrays
    return examples

# Then set format for PyTorch:
dataset["train"].set_format(type="torch", columns=["pixel_values", "label"])
```

**Data Flow:**
1. PIL Image → Processor → PyTorch Tensor
2. PyTorch Tensor → NumPy Array (for HF datasets storage)
3. NumPy Array → PyTorch Tensor (via set_format for training)

### i18n Architecture

All print statements now use the translation system:
```python
from vogel_model_trainer.i18n import _

# Before:
print(f"🤖 Lade Modell: {model_name}")

# After:
print(_('train_loading_model', model=model_name))
```

Language detection is automatic via `LANG` environment variable.

## 🔄 Changes

### Training Module
- Added i18n import to all functions with user-facing output
- Replaced 25+ hardcoded strings with translation calls
- Enhanced Ctrl+C handling with translated messages
- Improved error messages with language support

### Testing Module  
- Added i18n import to test functions
- Replaced 15+ hardcoded strings with translation calls
- Translated usage instructions
- Better error reporting in user's language

### CLI Module
- Completed i18n coverage for all three commands
- No more mixed English/German output
- Consistent user experience across all operations

## 📦 Installation

```bash
# Upgrade to v0.1.5
pip install --upgrade vogel-model-trainer

# Or from source
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.5
pip install -e .
```

## 🔄 Migration Guide

**No migration required!** This release is fully backward compatible:

- All existing commands work exactly as before
- Training now works (was broken in v0.1.4)
- Output messages are now in your system language
- No changes to command-line arguments or file formats

**If you experienced training errors in v0.1.4:**
Simply upgrade to v0.1.5 and training will work correctly.

## 🎯 Example Workflows

### Complete Training Workflow (Now Working!)

```bash
# 1. Extract birds from videos
vogel-trainer extract videos/*.mp4 \
  --folder data/ \
  --species-model classifier/ \
  --species-threshold 0.85

# 2. Organize into train/val split
vogel-trainer organize data/ -o organized-data/

# 3. Train model (✅ NOW WORKS!)
vogel-trainer train organized-data/ -o models/my-classifier/

# Output in German:
# ============================================================
# Vogel-Artenerkennung Training
# ============================================================
# 🔍 Erkenne Vogelarten aus Verzeichnisstruktur...
# 📁 Ausgabe-Verzeichnis: models/my-classifier/bird-classifier-20251109_153609
# ...
# ✅ Training abgeschlossen!
# 📁 Modell gespeichert in: models/my-classifier/bird-classifier-20251109_153609/final
```

### Language Support

The tool automatically uses your system language:
```bash
# German system (LANG=de_DE.UTF-8)
vogel-trainer train data/ -o models/
# Output: "🎓 Trainiere Modell auf Dataset: data/"

# English system (LANG=en_US.UTF-8)
vogel-trainer train data/ -o models/
# Output: "🎓 Training model on dataset: data/"

# Japanese system (LANG=ja_JP.UTF-8)
vogel-trainer train data/ -o models/
# Output: "🎓 データセットでモデルをトレーニング中：data/"
```

## 🐛 Known Issues

None currently identified.

## 🙏 Credits

**Bug Discovery:** Training error found during real-world usage testing  
**i18n Completion:** Systematic review identified remaining hardcoded strings  
**Testing:** Verified with 879 training images, 224 validation images, 8 bird species

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Issues**: https://github.com/kamera-linux/vogel-model-trainer/issues
- **Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md

## 📊 Summary

**What's Fixed:**
- ✅ Training now works (critical tensor conversion bug fixed)
- ✅ No more "expected Tensor but got list" errors
- ✅ Proper data pipeline with HuggingFace datasets

**What's New:**
- ✅ Complete i18n for training module (30+ keys)
- ✅ Complete i18n for testing module (12+ keys)
- ✅ All CLI commands fully translated
- ✅ Consistent language experience in English, German, Japanese

**Breaking Changes:**
- None - fully backward compatible

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.4...v0.1.5
