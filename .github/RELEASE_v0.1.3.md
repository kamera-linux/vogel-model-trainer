# Release v0.1.3 - Internationalization & Quality Control

**Release Date:** November 9, 2025

## 🎯 Overview

This release adds multi-language support for CLI output and introduces quality control features for auto-classification. Users can now experience vogel-model-trainer in their native language (English, German, or Japanese), and the new `--species-threshold` parameter helps maintain high-quality training datasets.

## ✨ What's New

### 🌍 Multi-Language Support (i18n)

vogel-model-trainer now speaks your language! The CLI automatically detects your system language and displays output in:
- 🇬🇧 **English** (en)
- 🇩🇪 **German** (de)
- 🇯🇵 **Japanese** (ja)

**How it works:**
```bash
# Automatically uses system language
export LANG=de_DE.UTF-8
vogel-trainer extract video.mp4 --folder output/
# Output: "🤖 Lade YOLO-Modell: ..."

export LANG=ja_JP.UTF-8
vogel-trainer extract video.mp4 --folder output/
# Output: "🤖 YOLOモデルを読み込んでいます："
```

**What's translated:**
- Loading messages and progress indicators
- Video processing output
- Species detection results
- Dataset organization summaries
- Error messages and warnings

### 🎯 Quality Control: Species Confidence Threshold

New `--species-threshold` parameter allows filtering auto-classified birds by confidence level:

```bash
# Only export birds classified with ≥85% confidence
vogel-trainer extract video.mp4 \
  --folder output/ \
  --species-model ~/models/classifier/ \
  --species-threshold 0.85
```

**Benefits:**
- ✅ **Higher quality datasets** - Exclude uncertain predictions
- 🎯 **Better training results** - Less noise in training data
- 📊 **Flexible quality control** - Adjust threshold based on your needs
- ⏭️ **Clear feedback** - Shows skipped detections with confidence scores

**Example output:**
```
✅ Bird #120: grünling (conf 0.89), frame 11600  ← Exported
✅ Bird #121: grünling (conf 0.91), frame 11700  ← Exported
⏭️  Skipped: grünling (conf 0.79 < 0.85), frame 11800  ← Skipped
```

### 📊 Visual Workflow Diagrams

All READMEs now include detailed ASCII diagrams showing the iterative training workflow:

```
Phase 1: Initial Model (Manual) → 92% accuracy
         ↓
Phase 2: Auto-Classification + Review → 96% accuracy
         ↓
       Repeat for further improvement
```

Shows concrete examples of accuracy improvements through iterative refinement.

### 🔧 Virtual Environment Recommendations

Installation sections now recommend using virtual environments:

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install vogel-model-trainer
pip install vogel-model-trainer
```

Available in all three language versions of the README.

## 🔄 Changes

### Internationalization
- **New module**: `src/vogel_model_trainer/i18n.py`
  - 100+ translation keys
  - Automatic language detection
  - Fallback to English
- **extractor.py**: All output messages translated
- **organizer.py**: Core messages translated
- **Future-ready**: Framework in place for trainer.py and tester.py

### Documentation Improvements
- **Workflow diagrams** in README.md, README.de.md, README.ja.md
- **Practical examples** showing 92% → 96% accuracy improvement
- **Phase-by-phase breakdown** of iterative training
- **Virtual environment setup** in installation sections

## 🚀 Usage Examples

### Multi-Language Output

```bash
# German
export LANG=de_DE.UTF-8
vogel-trainer organize data/ -o organized/
# 📊 Organisiere Dataset: data/
# ✅ Dataset organisiert!

# Japanese  
export LANG=ja_JP.UTF-8
vogel-trainer organize data/ -o organized/
# 📊 データセットを整理中：data/
# ✅ データセット整理完了！
```

### Quality-Controlled Auto-Classification

```bash
# Phase 1: Train initial model manually
vogel-trainer extract videos1/*.mp4 --folder data/ --bird kohlmeise
vogel-trainer organize data/ -o organized/
vogel-trainer train organized/ -o models/v1/
# Result: 92% accuracy

# Phase 2: Expand with high-confidence predictions
vogel-trainer extract videos2/*.mp4 \
  --folder data-v2/ \
  --species-model models/v1/final/ \
  --species-threshold 0.85

# Review and retrain
cp -r data-v2/* data/
vogel-trainer organize data/ -o organized-v2/
vogel-trainer train organized-v2/ -o models/v2/
# Result: 96% accuracy! 🎉
```

### Iterative Improvement Workflow

```bash
# Iteration 1: Small dataset, manual labels
vogel-trainer extract batch1/*.mp4 --folder data/ --bird blaumeise
vogel-trainer train data/ -o models/v1/

# Iteration 2: Auto-classify with quality control
vogel-trainer extract batch2/*.mp4 \
  --folder data/ \
  --species-model models/v1/final/ \
  --species-threshold 0.90  # Very strict

# Iteration 3: Expand with moderate threshold
vogel-trainer extract batch3/*.mp4 \
  --folder data/ \
  --species-model models/v2/final/ \
  --species-threshold 0.80  # More inclusive

# Each iteration improves the model!
```

## 🔨 Technical Details

### i18n Architecture

```python
from vogel_model_trainer.i18n import _

# Automatic language detection
print(_('loading_yolo') + f" yolov26n.pt")
# de: "🤖 Lade YOLO-Modell: yolov26n.pt"
# ja: "🤖 YOLOモデルを読み込んでいます：yolov26n.pt"

# With parameters
print(_('bird_extracted', count=120, species='grünling', 
        conf=0.89, frame=11600))
```

### Translation Coverage

| Module | Status | Keys |
|--------|--------|------|
| extractor.py | ✅ Complete | 25+ |
| organizer.py | ✅ Complete | 15+ |
| trainer.py | 🔜 Planned | 20+ |
| tester.py | 🔜 Planned | 15+ |
| cli/main.py | 🔜 Planned | 10+ |

### Language Detection

```python
# Automatic via LANG environment variable
LANG=de_DE.UTF-8  → German
LANG=ja_JP.UTF-8  → Japanese
LANG=en_US.UTF-8  → English (default)
```

## 📦 Installation

```bash
# Upgrade to v0.1.3
pip install --upgrade vogel-model-trainer

# Or from source
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.3
pip install -e .
```

## 🌐 Language Support

This release includes:
- 🇬🇧 **English README** with full documentation
- 🇩🇪 **German README** (README.de.md) with workflow diagrams
- 🇯🇵 **Japanese README** (README.ja.md) with complete translation
- 🌍 **CLI i18n** for extractor and organizer modules

## 📝 Migration Guide

**No migration required!** This release is fully backward compatible:

- All existing commands work exactly as before
- `--species-threshold` is optional (defaults to no filtering)
- Language selection is automatic
- No breaking changes to APIs or CLI

**New features are opt-in:**
```bash
# Use new quality control (optional)
--species-threshold 0.85

# Language is auto-detected (no action needed)
export LANG=de_DE.UTF-8  # Optional override
```

## 🎯 Use Cases

### Use Case 1: High-Quality Dataset Curation
```bash
# Only accept very confident predictions
vogel-trainer extract videos/*.mp4 \
  --species-model classifier/ \
  --species-threshold 0.90 \
  --folder high-quality/
```

### Use Case 2: Iterative Bootstrapping
```bash
# Start strict, gradually include more data
vogel-trainer extract batch1/ --species-threshold 0.90  # v1: strict
vogel-trainer extract batch2/ --species-threshold 0.85  # v2: moderate  
vogel-trainer extract batch3/ --species-threshold 0.80  # v3: inclusive
```

### Use Case 3: Multi-Language Teams
```bash
# German user
LANG=de_DE.UTF-8 vogel-trainer organize data/ -o organized/

# Japanese user  
LANG=ja_JP.UTF-8 vogel-trainer organize data/ -o organized/

# Same workflow, different languages! 🌍
```

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Issues**: https://github.com/kamera-linux/vogel-model-trainer/issues
- **Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md

## 🙏 Thank You

Thank you for using vogel-model-trainer! This release makes the tool more accessible to international users and provides better control over dataset quality.

Special thanks to the community for feedback on workflow improvements and quality control needs.

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.2...v0.1.3
