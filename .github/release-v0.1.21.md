# Release v0.1.21 - Hugging Face Integration

**Release Date:** November 20, 2025

## 🎯 What's New

### 🤗 Hugging Face Model Support

The `classify` command now supports **Hugging Face model IDs** in addition to local model paths:

```bash
# Use Hugging Face model (downloads automatically)
vogel-trainer classify --species-model kamera-linux/german-bird-classifier ~/images/ \
  --sort-output ~/sorted/ \
  --min-confidence 0.85

# Local models still work as before
vogel-trainer classify --species-model ~/models/final/ ~/images/ \
  --sort-output ~/sorted/
```

**Benefits:**
- 🚀 **Instant access** to pre-trained models without manual download
- 🔄 **Automatic caching** - models are downloaded once and reused
- 🌍 **Community models** - use any compatible model from Hugging Face Hub
- 💾 **Space efficient** - share models across projects

### 🔧 Improved Parameter Consistency

Changed model parameter to `--species-model` for consistency with other commands:

**Before (v0.1.20):**
```bash
vogel-trainer classify ~/models/final/ ~/images/ --sort-output ~/sorted/
```

**Now (v0.1.21):**
```bash
vogel-trainer classify --species-model ~/models/final/ ~/images/ --sort-output ~/sorted/
```

This matches the naming convention used in `extract` and `test` commands.

## 📦 Available Pre-trained Models

### German Garden Birds Classifier
- **Model ID:** `kamera-linux/german-bird-classifier`
- **Size:** 16 MB
- **Species:** 8 common German garden birds
  - Blaumeise (Blue Tit)
  - Grünling (Greenfinch)
  - Haussperling (House Sparrow)
  - Kernbeißer (Hawfinch)
  - Kleiber (Nuthatch)
  - Kohlmeise (Great Tit)
  - Rotkehlchen (Robin)
  - Sumpfmeise (Marsh Tit)

**Performance:** ~100 images/second on GPU (tested with 2,651 images)

## 📚 Usage Examples

### Quick Start with Pre-trained Model

```bash
# Install the package
pip install vogel-model-trainer

# Classify images with Hugging Face model
vogel-trainer classify \
  --species-model kamera-linux/german-bird-classifier \
  ~/my-bird-photos/ \
  --sort-output ~/sorted-birds/ \
  --min-confidence 0.85 \
  --csv-report results.csv
```

### Advanced Usage

```bash
# High-confidence sorting with Top-3 predictions
vogel-trainer classify \
  --species-model kamera-linux/german-bird-classifier \
  ~/camera-trap/ \
  --sort-output ~/sorted/ \
  --min-confidence 0.90 \
  --csv-report analysis.csv \
  --top-k 3

# Batch processing with file cleanup
vogel-trainer classify \
  --species-model kamera-linux/german-bird-classifier \
  ~/unsorted-images/ \
  --sort-output ~/sorted/ \
  --move \
  --delete-source \
  --force
```

## 🔄 Migration Guide

If you're upgrading from v0.1.20, update your scripts to use `--species-model`:

```bash
# Old syntax (v0.1.20)
vogel-trainer classify ~/models/final/ ~/images/ --sort-output ~/sorted/

# New syntax (v0.1.21+)
vogel-trainer classify --species-model ~/models/final/ ~/images/ --sort-output ~/sorted/
```

## 📊 Technical Details

### Model Loading Strategy

The classifier now intelligently detects model sources:

1. **Local Path Check:** If path exists locally → load from filesystem
2. **Hugging Face ID:** Otherwise → download from Hugging Face Hub
3. **Automatic Caching:** Models are cached in `~/.cache/huggingface/`

### Performance

Tested with 2,651 images on GPU:
- **Classification Speed:** ~106 images/second
- **Total Time:** ~25 seconds
- **Memory Usage:** ~2 GB GPU RAM

## 🐛 Bug Fixes

- Fixed model loading to support both local and remote models
- Improved error messages for model not found scenarios

## 📖 Documentation

All documentation has been updated:
- ✅ English README with Hugging Face examples
- ✅ German README with Hugging Face examples
- ✅ Japanese README with Hugging Face examples
- ✅ Updated CHANGELOG with v0.1.21 entry

## 🔗 Links

- **GitHub Repository:** https://github.com/kamera-linux/vogel-model-trainer
- **Hugging Face Model:** https://huggingface.co/kamera-linux/german-bird-classifier
- **PyPI Package:** https://pypi.org/project/vogel-model-trainer/

## 🙏 Acknowledgments

Thanks to the Hugging Face team for providing an excellent platform for sharing machine learning models!

---

**Full Changelog:** https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md
