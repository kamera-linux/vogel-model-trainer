# Release v0.1.20 - Bulk Image Classification

**Release Date:** November 20, 2025

## 🎯 Major New Feature: Batch Image Classification

This release introduces a powerful new **bulk classification** feature that allows you to automatically classify thousands of bird images with your trained models.

### Key Features

#### 🔍 Batch Classification
- Classify large batches of bird images automatically
- Process thousands of images in seconds (~100 images/sec on GPU)
- CSV export with detailed classification results
- Auto-sorting by species into separate folders
- Configurable confidence thresholds for quality filtering

#### 📁 Advanced File Management
- **Copy mode** (default): Original files remain untouched
- **Move mode** (`--move`): Save disk space by moving instead of copying
- **Delete source** (`--delete-source`): Clean up source directory after processing
- **Dry run** (`--dry-run`): Preview operations without making changes
- **Force mode** (`--force`): Skip confirmation prompts for automation

#### 📊 Rich Output Options
- **CSV Export**: Detailed results with Top-K predictions per image
- **Species Statistics**: Distribution analysis with average confidence scores
- **Unknown Folder**: Images below confidence threshold separated automatically
- **Progress Tracking**: Real-time progress bars with tqdm

### Usage Examples

```bash
# Simple classification with CSV export
vogel-trainer classify ~/models/my-classifier/ ~/images/ \
  --csv-report results.csv

# Auto-sort by species with confidence threshold
vogel-trainer classify ~/models/my-classifier/ ~/images/ \
  --sort-output ~/sorted/ \
  --min-confidence 0.85

# Full workflow: CSV + sorting + Top-3 predictions
vogel-trainer classify ~/models/my-classifier/ ~/images/ \
  --csv-report results.csv \
  --sort-output ~/sorted/ \
  --top-k 3

# Space-saving: Move files and cleanup
vogel-trainer classify ~/models/my-classifier/ ~/images/ \
  --sort-output ~/sorted/ \
  --move \
  --delete-source \
  --force

# Safe testing: Dry run first
vogel-trainer classify ~/models/my-classifier/ ~/images/ \
  --sort-output ~/sorted/ \
  --delete-source \
  --dry-run
```

### CSV Output Format

```csv
filename,predicted_species,confidence,top_2_species,top_2_confidence,top_3_species,top_3_confidence
bird001.jpg,kohlmeise,0.9750,blaumeise,0.0180,rotkehlchen,0.0045
bird002.jpg,amsel,0.9200,rotkehlchen,0.0520,buchfink,0.0210
```

### Real-World Performance

Test with 2,651 bird images (8 species):
- **Processing time**: 26 seconds (~98 images/second)
- **GPU acceleration**: Automatic detection and usage
- **Classification accuracy**: 66.8% - 99.4% depending on species
- **Average confidence**: 59.42% - 90.85% per species

### Use Cases

- 📸 **Camera Trap Analysis**: Automatically identify species in thousands of photos
- 🔍 **Citizen Science**: Hobby ornithologists can classify their photo collections
- 📊 **Monitoring Projects**: Time-series analysis of bird populations
- ✅ **Dataset Quality**: Check existing datasets for misclassifications
- 🗂️ **Photo Organization**: Automatically sort and organize bird photos

## 🌍 Internationalization

All new features include complete translations:
- 🇬🇧 English
- 🇩🇪 German (Deutsch)
- 🇯🇵 Japanese (日本語)

37 new translation keys added for classification features.

## 📚 Documentation

Complete documentation added to all READMEs:
- Comprehensive usage examples
- Parameter reference with defaults
- File management safety guidelines
- CSV format specification
- Real-world use cases

## 🔧 Technical Details

**New Files:**
- `src/vogel_model_trainer/core/classifier.py` - Classification engine
- CLI integration in `src/vogel_model_trainer/cli/main.py`
- 37 i18n translation keys in `src/vogel_model_trainer/i18n.py`

**Parameters:**
- `model`: Path to trained model (required)
- `input`: Directory with images to classify (required)
- `--sort-output, -s`: Output directory for sorted images
- `--min-confidence`: Minimum confidence threshold (0.0-1.0, default: 0.0)
- `--csv-report, -c`: CSV file for classification results
- `--top-k, -k`: Number of top predictions (1-5, default: 1)
- `--batch-size, -b`: Processing batch size (default: 32)
- `--move`: Move files instead of copying
- `--delete-source`: Delete source directory after processing
- `--force, -f`: Skip confirmation prompts
- `--dry-run`: Simulate without file changes
- `--no-recursive`: Only process top-level images

## ⚠️ Safety Features

- Confirmation prompts for destructive operations (`--delete-source`)
- Dry-run mode to preview changes
- Automatic filename conflict resolution
- Unknown folder for low-confidence classifications

## 🚀 Upgrade Guide

```bash
pip install --upgrade vogel-model-trainer
```

All existing features remain fully functional. No breaking changes.

## 📦 Full Changelog

See [CHANGELOG.md](../CHANGELOG.md) for complete list of changes.

## 🙏 Acknowledgments

Thanks to the community for feature requests and feedback!

---

**Previous Release:** [v0.1.19 - Hardware Auto-Detection](release-v0.1.19.md)
