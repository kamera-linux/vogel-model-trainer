# Release v0.1.2 - Core Module Refactoring & Japanese Documentation

**Release Date:** November 9, 2025

## 🎯 Overview

This release focuses on improving the library architecture and expanding documentation. All core modules have been refactored from script-only implementations to proper library functions, making vogel-model-trainer more suitable for programmatic use while maintaining full backward compatibility for direct script execution.

## ✨ What's New

### 📚 Japanese Documentation
- **Complete Japanese Translation**: Added `README.ja.md` with full documentation in Japanese
- **Language Selection**: All READMEs now include links to English, German, and Japanese versions
- **Internationalization**: Expanding accessibility for non-English speaking users

### 🔧 Library-Style Architecture
All core modules now provide dedicated functions for programmatic use:

#### `extractor.py`
```python
from vogel_model_trainer.core.extractor import extract_birds_from_video

extract_birds_from_video(
    video_path="video.mp4",
    output_dir="output/",
    bird_species="Blaumeise",
    detection_model="yolov8n.pt",
    species_model=None,
    threshold=0.5,
    sample_rate=3,
    resize_to_target=True
)
```

#### `organizer.py`
```python
from vogel_model_trainer.core.organizer import organize_dataset

organize_dataset(
    source_dir="extracted_images/",
    output_dir="dataset/",
    train_ratio=0.8
)
```

#### `trainer.py`
```python
from vogel_model_trainer.core.trainer import train_model

train_model(
    data_dir="dataset/",
    output_dir="models/",
    model_name="google/efficientnet-b0",
    batch_size=16,
    num_epochs=50,
    learning_rate=3e-4
)
```

#### `tester.py`
```python
from vogel_model_trainer.core.tester import test_model

# Test on validation set
test_model(
    model_path="models/bird_classifier_20251109/",
    data_dir="dataset/"
)

# Test single image
test_model(
    model_path="models/bird_classifier_20251109/",
    image_path="test_image.jpg"
)
```

## 🔄 Changes

### Architecture Improvements
- **Library+Script Hybrid Pattern**: All core modules now follow a consistent pattern:
  - Dedicated functions for library use (e.g., `organize_dataset()`, `train_model()`)
  - `main()` functions as thin wrappers for direct script execution
  - CLI commands updated to call library functions
  
### Core Module Updates
- **organizer.py**: New `organize_dataset()` function for dataset organization
- **trainer.py**: Extracted training logic into configurable `train_model()` function
- **tester.py**: Unified testing interface supporting both validation sets and single images
- **extractor.py**: CLI integration improved with correct parameter mapping

### Documentation
- Updated language selection in all README files
- Removed email addresses from support sections
- Consistent formatting across all language versions

## 🐛 Bug Fixes

- **CLI Integration**: Fixed parameter mapping between CLI commands and core functions
- **Function Signatures**: Ensured all CLI calls use correct parameter names matching function definitions
- **extract_command**: Updated to properly handle glob patterns and call `extract_birds_from_video()` with correct parameters

## 🔨 Technical Details

### Function Signatures

**organizer.py**
```python
def organize_dataset(source_dir: str, output_dir: str, train_ratio: float = 0.8)
```

**trainer.py**
```python
def train_model(
    data_dir: str,
    output_dir: str,
    model_name: str = "google/efficientnet-b0",
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 3e-4
)
```

**tester.py**
```python
def test_model(
    model_path: str,
    data_dir: Optional[str] = None,
    image_path: Optional[str] = None
) -> dict
```

### Backward Compatibility
- All modules can still be executed directly as scripts
- CLI interface remains unchanged
- No breaking changes for existing users

## 📦 Installation

```bash
pip install --upgrade vogel-model-trainer
```

Or with pip3:
```bash
pip3 install --upgrade vogel-model-trainer
```

## 🚀 Usage Examples

### Using as Library
```python
from vogel_model_trainer.core import extractor, organizer, trainer, tester

# Extract birds from videos
extractor.extract_birds_from_video("video.mp4", "output/", "Blaumeise")

# Organize into train/val splits
organizer.organize_dataset("output/", "dataset/", train_ratio=0.8)

# Train model
trainer.train_model("dataset/", "models/", num_epochs=30)

# Test model
results = tester.test_model("models/bird_classifier_20251109/", data_dir="dataset/")
print(f"Accuracy: {results['accuracy']:.2%}")
```

### Using CLI (unchanged)
```bash
# Extract birds
vogel-trainer extract video.mp4 --folder output/ --bird Blaumeise

# Organize dataset
vogel-trainer organize output/ -o dataset/

# Train model
vogel-trainer train dataset/ -o models/ --epochs 30

# Test model
vogel-trainer test models/bird_classifier_20251109/ -d dataset/
```

## 🌐 Multi-Language Support

This release includes complete documentation in three languages:
- 🇬🇧 [English](README.md)
- 🇩🇪 [Deutsch](README.de.md)
- 🇯🇵 [日本語](README.ja.md)

## 📝 Migration Guide

**No migration required!** This release is fully backward compatible:
- CLI commands work exactly as before
- Direct script execution (`python extractor.py ...`) still supported
- New library functions are additions, not replacements

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Issues**: https://github.com/kamera-linux/vogel-model-trainer/issues
- **Documentation**: See README files in your preferred language

## 🙏 Thank You

Thank you for using vogel-model-trainer! This release makes the library more flexible and accessible to a wider audience.

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md
