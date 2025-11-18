# v0.1.16 - Image Support & Convert Mode

Release v0.1.16 extends the extractor with comprehensive image support and introduces a new convert mode for normalizing existing training datasets.

## 🎯 Highlights

- **🖼️ Full Image Support**: Extract bird crops from static images (JPG, PNG, BMP, TIFF)
- **🔄 Convert Mode**: Process existing bird datasets without detection
- **📊 Advanced Quality Filtering**: Sharpness, edge quality, and deduplication
- **🤖 GitHub Automation**: Complete release workflow automation

## ✨ New Features

### Image Support in Extractor

The extractor now processes both videos and images with the same powerful pipeline:

```bash
# Single image
python -m vogel_model_trainer.core.extractor photo.jpg \
  --folder output/ \
  --bg-remove --bg-transparent

# Batch processing
python -m vogel_model_trainer.core.extractor "~/Photos/*.jpg" \
  --folder output/ --recursive \
  --species-model ~/models/classifier/

# Mixed (videos + images)
python -m vogel_model_trainer.core.extractor "~/Media/**/*" \
  --folder output/ --recursive
```

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` (case-insensitive)

### Convert Mode

New `--convert` mode for normalizing existing bird crop datasets:

```bash
# Convert with transparent background
python -m vogel_model_trainer.core.extractor \
  --convert \
  --source ~/vogel-training-data-species \
  --target ~/vogel-training-data-transparent \
  --bg-remove \
  --bg-transparent \
  --crop-padding 10 \
  --min-sharpness 80 \
  --quality-report

# Convert with colored background
python -m vogel_model_trainer.core.extractor \
  --convert \
  --source ~/data-original \
  --target ~/data-gray \
  --bg-remove \
  --bg-color 128,128,128 \
  --deduplicate
```

**Features**:
- ✅ No YOLO detection (direct processing)
- ✅ Maintains folder structure (species subdirectories)
- ✅ Batch processing with quality filtering
- ✅ Perfect for dataset normalization

**Use Cases**:
- Normalize existing training datasets for model comparability
- Apply consistent background removal across all images
- Filter out low-quality images from existing datasets
- Create multiple dataset variants (transparent, gray, white backgrounds)

### Quality Filtering

New parameters for advanced image quality control:

```bash
--min-sharpness 80        # Filter blurry images (Laplacian variance)
--min-edge-quality 50     # Filter poor edge quality (Sobel gradient)
--quality-report          # Generate quality statistics
--deduplicate             # Skip duplicate/similar images
--similarity-threshold 5  # Hamming distance for duplicates (0-64)
```

**Example with quality filtering**:
```bash
python -m vogel_model_trainer.core.extractor \
  "~/Videos/*.mp4" \
  --folder output/ \
  --min-sharpness 80 \
  --deduplicate \
  --quality-report
```

### GitHub Automation Tools

Complete workflow automation infrastructure for releases:

**1. Automated Release Script** (`scripts/create_github_release.py`):
```bash
# Auto-detect latest tag
python scripts/create_github_release.py

# Specific tag with workflow monitoring
python scripts/create_github_release.py v0.1.16

# Draft release
python scripts/create_github_release.py --draft
```

**Features**:
- Auto-detects latest Git tag
- Finds and uses release notes automatically
- Creates GitHub release via gh CLI
- Monitors PyPI publish workflow in real-time
- Matches workflow by release tag (prevents version confusion)
- Shows workflow status: queued ⏳, in_progress 🔄, completed ✅
- Supports draft and pre-release modes

**2. GitHub CLI Guide** (`.github/GITHUB_CLI_GUIDE.md`):
- 700+ lines comprehensive documentation
- Complete gh CLI reference (releases, PRs, workflows, issues)
- Practical examples and production scripts
- Best practices and useful aliases
- German language documentation

## 🔧 Improvements

### Enhanced Extractor Parameters

All new background and quality parameters now available:

```bash
# Background processing
--bg-remove              # AI background removal (rembg)
--bg-transparent         # Transparent PNG output
--bg-color R,G,B         # Custom background color
--bg-model u2net         # rembg model selection
--bg-fill-black          # Make black areas transparent
--crop-padding 10        # Extra pixels around bird

# Quality control
--quality 95             # JPEG quality (1-100)
--min-sharpness 80       # Sharpness filter
--min-edge-quality 50    # Edge quality filter
--quality-report         # Statistics report
--deduplicate            # Duplicate detection
```

### Flexible File Handling

Enhanced input handling for videos and images:

```bash
# Glob patterns
python -m vogel_model_trainer.core.extractor "*.jpg" --folder output/

# Recursive search
python -m vogel_model_trainer.core.extractor "~/Media/" --folder output/ --recursive

# Mixed media types
python -m vogel_model_trainer.core.extractor "~/Media/**/*" --folder output/ --recursive
```

## 🐛 Bug Fixes

### Release Script Workflow Matching

Fixed workflow version detection in release automation:
- **Before**: Showed wrong workflow version (e.g., v0.1.14 instead of v0.1.15)
- **After**: Searches last 5 workflow runs for matching release tag
- Matches both `v0.1.16` and `0.1.16` in workflow title
- Provides accurate real-time PyPI publish status

## 📊 Technical Details

### New Functions

- `extract_birds_from_image()`: Process single images without video decoding
- `convert_bird_images()`: Batch process existing bird crops
- `wait_for_workflow_trigger()`: Monitor GitHub Actions with tag matching
- `check_workflow_status()`: Display workflow status with proper version

### Architecture

**Image Pipeline**:
```
Image File → cv2.imread() → YOLO Detection → Crop → Processing → Save
```

**Convert Pipeline**:
```
Existing Crops → Load → Processing (no detection) → Save (maintain structure)
```

**Video Pipeline** (unchanged):
```
Video → Frame Extraction → YOLO Detection → Crop → Processing → Save
```

### Performance

- Image processing: ~0.5-2s per image (depending on size and features)
- Convert mode: ~0.3-1s per image (no detection overhead)
- Batch processing: Efficient with minimal memory overhead
- Quality filtering: Negligible performance impact

## 📚 Documentation Updates

### Updated Documentation

- **README.md**: Added image support and convert mode examples
- **README.de.md**: German documentation updated
- **README.ja.md**: Japanese documentation updated
- **scripts/README.md**: Release automation documentation
- **GITHUB_CLI_GUIDE.md**: New comprehensive gh CLI guide

### Usage Examples

**Convert Mode Examples**:
```bash
# Transparent backgrounds for training
python -m vogel_model_trainer.core.extractor \
  --convert \
  --source ~/data-original \
  --target ~/data-transparent \
  --bg-remove --bg-transparent

# Gray backgrounds with quality filtering
python -m vogel_model_trainer.core.extractor \
  --convert \
  --source ~/data-original \
  --target ~/data-gray \
  --bg-remove --bg-color 128,128,128 \
  --min-sharpness 80 \
  --quality-report

# Deduplicate and normalize
python -m vogel_model_trainer.core.extractor \
  --convert \
  --source ~/data-mixed \
  --target ~/data-clean \
  --deduplicate \
  --crop-padding 10
```

**Image Extraction Examples**:
```bash
# Single bird photo
python -m vogel_model_trainer.core.extractor \
  bird-photo.jpg \
  --folder output/ \
  --bird rotkehlchen

# Batch bird photos
python -m vogel_model_trainer.core.extractor \
  "~/BirdPhotos/*.jpg" \
  --folder output/ \
  --recursive \
  --species-model ~/models/classifier/ \
  --bg-remove --bg-transparent

# Mixed videos and photos
python -m vogel_model_trainer.core.extractor \
  "~/Media/**/*" \
  --folder output/ \
  --recursive \
  --threshold 0.6
```

## 🔄 Migration Guide

### From v0.1.15 to v0.1.16

No breaking changes. All existing workflows continue to work.

**New capabilities**:
1. Can now process images directly (previously videos only)
2. Can normalize existing datasets with `--convert` mode
3. Enhanced quality filtering options available
4. Automated release workflow available

**Recommended actions**:
1. Try convert mode on existing datasets for consistency
2. Use quality filtering for better training data
3. Process static bird photos for additional training data
4. Use automation scripts for future releases

## 🙏 Acknowledgments

Thank you to all users and contributors who requested image support and dataset normalization features!

## 📦 Installation

```bash
pip install vogel-model-trainer==0.1.16
```

Or upgrade:
```bash
pip install --upgrade vogel-model-trainer
```

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Documentation**: See README.md
- **Issues**: https://github.com/kamera-linux/vogel-model-trainer/issues
