# 🐦 Vogel Model Trainer

**Languages:** [🇬🇧 English](README.md) | [🇩🇪 Deutsch](README.de.md) | [🇯🇵 日本語](README.ja.md)

<p align="left">
  <a href="https://pypi.org/project/vogel-model-trainer/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/vogel-model-trainer.svg"></a>
  <a href="https://pypi.org/project/vogel-model-trainer/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/vogel-model-trainer.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://pypi.org/project/vogel-model-trainer/"><img alt="PyPI Status" src="https://img.shields.io/pypi/status/vogel-model-trainer.svg"></a>
  <a href="https://pepy.tech/project/vogel-model-trainer"><img alt="Downloads" src="https://static.pepy.tech/badge/vogel-model-trainer"></a>
</p>

**Train custom bird species classifiers from your own video footage using YOLOv8 and EfficientNet.**

A specialized toolkit for creating high-accuracy bird species classifiers tailored to your specific monitoring setup. Extract training data from videos, organize datasets, and train custom models with >96% accuracy.

---

## ✨ Features

- 🎯 **YOLO-based Bird Detection** - Automated bird cropping from videos and images using YOLOv8
- 🖼️ **Image Support** - Extract birds from static images (JPG, PNG, BMP, TIFF)
- 🔄 **Convert Mode** - Normalize existing bird datasets without detection
- 🤖 **Three Extraction Modes** - Manual labeling, auto-sorting, or standard extraction
- 📁 **Wildcard Support** - Batch process multiple videos/images with glob patterns
- 🖼️ **Flexible Image Sizes** - 224/384/448px or keep original size
- 🔍 **Advanced Filtering** - Box size, blur detection, sharpness, edge quality thresholds
- 🔄 **Duplicate Detection** - Perceptual hashing removes similar images
- ✅ **Quality Check** - Find blurry, too-small, corrupted, or badly-exposed images
- 🎨 **AI Background Removal** - Remove backgrounds with gray default for optimal training
- 🧹 **Dataset Validation** - Clean transparent/gray datasets with automated checks
- 🧠 **EfficientNet-B0 Training** - Lightweight yet powerful classification model
- 🎨 **4-Level Data Augmentation** - None/light/medium/heavy intensity options
- ⚡ **Mixed Precision Training** - FP16/BF16 support for faster GPU training
- 📊 **Advanced Training Options** - 13 configurable parameters for fine-tuning
- 🔧 **Dataset Deduplication** - Clean existing datasets with perceptual hashing
- ⏸️ **Graceful Shutdown** - Save model state on Ctrl+C interruption
- 🌍 **Full i18n Support** - English, German, Japanese translations
- 📈 **Per-Species Metrics** - Detailed accuracy breakdown by species

## 🤖 Pre-trained Model

**German Garden Birds Classifier** - Ready to use!

We provide a pre-trained model on Hugging Face that can classify 8 common German garden bird species with 100% validation accuracy:

🔗 **[kamera-linux/german-bird-classifier](https://huggingface.co/kamera-linux/german-bird-classifier)**

**Supported Species:**
- Blaumeise (Blue Tit)
- Grünling (European Greenfinch)
- Haussperling (House Sparrow)
- Kernbeißer (Hawfinch)
- Kleiber (Eurasian Nuthatch)
- Kohlmeise (Great Tit)
- Rotkehlchen (European Robin)
- Sumpfmeise (Marsh Tit)

**Usage during extraction:**
```bash
vogel-trainer extract video.mp4 \
  --species-model kamera-linux/german-bird-classifier \
  --remove-background \
  --crop-padding 20 \
  --sample-rate 20 --skip-blurry --deduplicate \
  --min-sharpness 150 --min-edge-quality 80
```

The model will automatically classify detected birds during extraction!

---

## 🚀 Quick Start

### Installation

#### Recommended: Using Virtual Environment

```bash
# Install venv if needed (Debian/Ubuntu)
sudo apt install python3-venv

# Create virtual environment
python3 -m venv ~/venv-vogel

# Activate it
source ~/venv-vogel/bin/activate  # On Windows: ~/venv-vogel\Scripts\activate

# Install package
pip install vogel-model-trainer
```

#### Quick Install

```bash
# Install from PyPI
pip install vogel-model-trainer

# Or install from source
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
pip install -e .
```

### 🎥 Video Tutorials

Learn vogel-model-trainer with step-by-step video guides:

- **Getting Started** - Installation and first extraction (5 min)
- **Extracting Birds** - Quality filters, deduplication, species classification (10 min)
- **Organizing Datasets** - Train/val splits, class balance management (8 min) **NEW in v0.1.8**
- **Training Models** - Custom classifier training and parameters (12 min)
- **Testing & Evaluation** - Model testing and performance analysis (7 min)

📺 *Video tutorials coming soon!*

### Basic Workflow

```bash
# 1. Extract bird images from videos
vogel-trainer extract video.mp4 --folder ~/training-data/ --bird kohlmeise

# 2. Organize into train/validation split
vogel-trainer organize ~/training-data/ -o ~/organized-data/

# 3. Train custom classifier
vogel-trainer train ~/organized-data/ -o ~/models/my-classifier/

# 4. Test the trained model
vogel-trainer test ~/models/my-classifier/ -d ~/organized-data/
```

---

## 📖 Usage Guide

### Using as a Library (New in v0.1.2)

All core functions can now be used programmatically in your Python code:

```python
from vogel_model_trainer.core import extractor, organizer, trainer, tester

# Extract birds from video
extractor.extract_birds_from_video(
    video_path="video.mp4",
    output_dir="output/",
    bird_species="great-tit",
    detection_model="yolov8n.pt",
    species_model=None,
    threshold=0.5,
    sample_rate=3,
    resize_to_target=True
)

# Extract birds from static images (New in v0.1.16)
extractor.extract_birds_from_image(
    image_path="photo.jpg",
    output_dir="output/",
    bird_species="great-tit",
    detection_model="yolov8n.pt",
    remove_bg=True,
    bg_transparent=True
)

# Convert existing bird crops (New in v0.1.16)
stats = extractor.convert_bird_images(
    source_dir="raw-data/",
    target_dir="processed-data/",
    remove_bg=True,
    bg_transparent=True,
    crop_padding=10,
    min_sharpness=80,
    deduplicate=True
)
print(f"Converted: {stats['converted']}, Skipped: {stats['skipped_quality']}")

# Organize into train/val splits
organizer.organize_dataset(
    source_dir="output/",
    output_dir="dataset/",
    train_ratio=0.8
)

# Train model
trainer.train_model(
    data_dir="dataset/",
    output_dir="models/",
    model_name="google/efficientnet-b0",
    batch_size=16,
    num_epochs=50,
    learning_rate=3e-4
)

# Test model
results = tester.test_model(
    model_path="models/bird_classifier/",
    data_dir="dataset/"
)
print(f"Accuracy: {results['accuracy']:.2%}")
```

### 1. Extract Training Images

vogel-model-trainer now supports both **videos** and **static images** as input sources.

#### 🎬 Video Extraction

Extract bird crops from video files with YOLO detection:

##### Manual Mode (Recommended for Initial Collection)

When you know the species in your video:

```bash
vogel-trainer extract ~/Videos/great-tit.mp4 \
  --folder ~/training-data/ \
  --bird great-tit \
  --threshold 0.5 \
  --sample-rate 3
```

##### Auto-Sort Mode (For Iterative Training)

Use an existing model to automatically classify and sort:

```bash
vogel-trainer extract ~/Videos/mixed.mp4 \
  --folder ~/training-data/ \
  --species-model ~/models/classifier/final/ \
  --threshold 0.5
```

#### Batch Processing with Wildcards

```bash
# Process all videos in a directory
vogel-trainer extract "~/Videos/*.mp4" --folder ~/data/ --bird blue-tit

# Recursive directory search
vogel-trainer extract ~/Videos/ \
  --folder ~/data/ \
  --bird amsel \
  --recursive
```

**Parameters:**
- `--folder`: Base directory for extracted images (required)
- `--bird`: Manual species label (creates subdirectory)
- `--species-model`: Path to trained model for auto-classification
- `--species-threshold`: Minimum confidence for species classification (e.g., 0.85 for 85%)
- `--threshold`: YOLO confidence threshold (default: 0.5)
- `--sample-rate`: Process every Nth frame (default: 3)
- `--detection-model`: YOLO model path (default: yolov8n.pt)
- `--image-size`: Target image size in pixels (default: 224, use 0 for original size)
- `--max-detections`: Maximum detections per frame (default: 10)
- `--min-box-size`: Minimum bounding box size in pixels (default: 50)
- `--max-box-size`: Maximum bounding box size in pixels (default: 800)
- `--quality`: JPEG quality 1-100 (default: 95)
- `--skip-blurry`: Skip blurry/out-of-focus images (experimental)
- `--deduplicate`: Skip duplicate/similar images using perceptual hashing
- `--similarity-threshold`: Similarity threshold for duplicates - Hamming distance 0-64 (default: 5)
- `--min-sharpness`: **NEW v0.1.9** - Minimum sharpness score (Laplacian variance, typical: 100-300)
- `--min-edge-quality`: **NEW v0.1.9** - Minimum edge quality (Sobel gradient, typical: 50-150)
- `--save-quality-report`: **NEW v0.1.9** - Generate detailed quality statistics report
- `--remove-background`: **🧪 EXPERIMENTAL v0.1.11** - Remove background using AI (rembg)
- `--crop-padding`: **NEW v0.1.15** - Pixels to expand mask around bird (preserves details like feet/beak)
- `--bg-color [white|black|gray]`: **🧪 EXPERIMENTAL v0.1.11** - Background color (default: white)
- `--bg-model [u2net|u2netp|isnet-general-use]`: **🧪 EXPERIMENTAL v0.1.11** - AI model for background removal (default: u2net)
- `--recursive, -r`: Search directories recursively
- `--log`: Save console output to log file (`/var/log/vogel-kamera-linux/YYYY/KWXX/`)

**Advanced Filtering Examples:**

```bash
# High-quality extraction with all filters (v0.1.15)
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird rotkehlchen \
  --threshold 0.6 \
  --min-box-size 80 \
  --max-box-size 600 \
  --min-sharpness 150 \
  --min-edge-quality 80 \
  --skip-blurry \
  --deduplicate \
  --save-quality-report \
  --remove-background \
  --crop-padding 20 \
  --bg-color gray \
  --bg-model u2net

# Background removal with detail preservation (recommended)
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird blaumeise \
  --remove-background \
  --crop-padding 20 \
  --bg-color gray \
  --bg-model isnet-general-use
```

#### 🖼️ Image Extraction (New in v0.1.16)

Extract bird crops from static images (JPG, PNG, BMP, TIFF) using YOLO detection:

```bash
# Single image
vogel-trainer extract photo.jpg --folder ~/training-data/ --bird amsel

# Multiple images with glob pattern
vogel-trainer extract "~/photos/*.jpg" --folder ~/training-data/ --bird rotkehlchen

# Recursive directory search
vogel-trainer extract ~/photos/ \
  --folder ~/training-data/ \
  --bird blaumeise \
  --recursive

# With background removal and quality filtering
vogel-trainer extract photo.jpg \
  --folder ~/training-data/ \
  --bird kohlmeise \
  --bg-remove \
  --bg-transparent \
  --crop-padding 10 \
  --min-sharpness 100 \
  --quality-report

# Auto-classification with trained model
vogel-trainer extract photo.jpg \
  --folder ~/training-data/ \
  --species-model ~/models/classifier/final/ \
  --species-threshold 0.85

# Batch processing with auto-sort
vogel-trainer extract "~/photos/*.jpg" \
  --folder ~/training-data/ \
  --species-model kamera-linux/german-bird-classifier \
  --recursive
```

**Note:** All video extraction parameters (filtering, background removal, quality control) are available for image extraction.

#### 🔄 Convert Mode (New in v0.1.16)

Process existing bird crop images **without YOLO detection**. Useful for normalizing existing datasets:

```bash
# Convert existing crops to transparent background
vogel-trainer extract \
  --convert \
  --source ~/raw-bird-data/ \
  --target ~/transparent-bird-data/ \
  --bg-remove \
  --bg-transparent \
  --crop-padding 10

# Normalize with quality filtering and deduplication
vogel-trainer extract \
  --convert \
  --source ~/existing-dataset/ \
  --target ~/normalized-dataset/ \
  --bg-remove \
  --bg-color 128,128,128 \
  --min-sharpness 80 \
  --min-edge-quality 50 \
  --deduplicate \
  --quality-report

# Convert to gray background (optimal for training)
vogel-trainer extract \
  --convert \
  --source ~/vogel-training-data-species/ \
  --target ~/vogel-training-data-gray/ \
  --bg-remove \
  --bg-color 128,128,128 \
  --crop-padding 15
```

**Convert Mode Features:**
- ✅ Maintains folder structure (species subdirectories preserved)
- ✅ Direct image processing (no YOLO overhead, ~0.3-1s per image)
- ✅ All quality filters available (sharpness, edge quality, blur detection)
- ✅ Background removal with transparent or custom color
- ✅ Deduplication with perceptual hashing
- ✅ Batch statistics and quality reports

**Use Cases:**
- Normalize existing datasets from different sources
- Add transparent backgrounds to legacy datasets
- Apply consistent quality filtering across old data
- Remove duplicates from merged datasets
- Prepare datasets for model comparability
```

**🧪 Background Removal (EXPERIMENTAL v0.1.11+, Stable v0.1.14):**

The `--remove-background` feature uses AI-powered rembg library to automatically segment birds from backgrounds.

**NEW in v0.1.14:** Gray background is now the DEFAULT for optimal training! Smaller JPEG files, better compatibility.

**NEW in v0.1.15:** Crop padding feature to preserve bird details (feet, beak, feathers)!

- **Crop Padding (v0.1.15+):**
  - `--crop-padding N`: Expand foreground mask by N pixels around detected bird
  - Prevents loss of important details (feet, beak, tail feathers) during background removal
  - Recommended value: `20` pixels for optimal results
  - Only works with `--remove-background` flag
  - Example: `--crop-padding 20` keeps 20 more pixels around the bird

- **Models:**
  - `u2net` (default): Best overall quality, ~180MB download
  - `u2netp`: Faster, smaller model for quick processing
  - `isnet-general-use`: Best edge quality for detailed feathers

- **Background Colors (NEW DEFAULT v0.1.14):**
  - `gray` (DEFAULT): Neutral gray background (#808080) - best for training
  - `white`: Clean white background (#FFFFFF)
  - `black`: High contrast black background (#000000)
  - `green-screen`: Chroma key green (#00FF00)
  - `blue-screen`: Chroma key blue (#0000FF)

- **Transparency Options:**
  - `--bg-transparent`: Create PNG with alpha channel (flexible but larger files)
  - `--no-bg-transparent` (DEFAULT): Use colored background (smaller JPEG files)
  - `--bg-fill-black`: Makes black box areas transparent (requires --bg-transparent)
  - `--no-bg-fill-black` (DEFAULT): Keep padding areas with background color

- **Features:**
  - AI-based U²-Net segmentation for accurate bird isolation
  - Alpha matting for smooth, professional edges
  - Post-processing with morphological operations
  - Handles complex backgrounds (branches, leaves, buildings)
  - Works with varied bird plumage and fine feather details
  - Automatically saves as PNG (transparent) or JPEG (opaque)

- **Note:** First use downloads ~180MB model (cached afterward), requires `rembg>=2.0.50` dependency

**💡 Training with Transparent Backgrounds (NEW v0.1.15):**

When training with PNG images that have transparent backgrounds, the trainer automatically applies **random background augmentation**:
- During training: Each image gets a random gray/black/white background
- During validation/testing: Consistent neutral gray background
- **Benefit**: Model learns to focus on bird features, not background color
- **Result**: More robust classifier that works with any background

To use this feature, simply extract with `--remove-background --bg-transparent`:
```bash
# Extract with transparent backgrounds
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird rotkehlchen \
  --remove-background \
  --crop-padding 20 \
  --bg-transparent \
  --sample-rate 30

# Train - random backgrounds applied automatically!
vogel-trainer train data/ --output-dir models/
```

**💡 Best Practice for Public Models:**
```bash
# Recommended settings for neutral dataset (v0.1.15)
# Using opaque gray background (smaller files, consistent)
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird rotkehlchen \
  --remove-background \
  --crop-padding 20 \
  --bg-color gray \
  --sample-rate 30 \
  --skip-blurry \
  --deduplicate \
  --save-quality-report \
  --quality 98
```

# Extract with high quality settings
vogel-trainer extract video.mp4 \
  --folder data/ \
  --skip-blurry \
  --deduplicate \
  --save-quality-report \
  --remove-background \
  --quality 98

# Extract with duplicate detection (prevents similar images)
vogel-trainer extract ~/Videos/*.mp4 \
  --folder data/ \
  --bird kohlmeise \
  --deduplicate \
  --similarity-threshold 3  # Stricter duplicate detection

# Large image size for high-detail training
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird amsel \
  --image-size 384  # Larger images for better quality

# Auto-sort with confidence filter (only high-confidence classifications)
vogel-trainer extract video.mp4 \
  --folder data/ \
  --species-model ~/models/classifier/ \
  --species-threshold 0.90 \
  --deduplicate
```

**Logging Example:**

```bash
# Save output to log file
vogel-trainer extract ~/Videos/great-tit.mp4 \
  --folder ~/data/ \
  --bird great-tit \
  --log

# Log file path: /var/log/vogel-kamera-linux/2025/KW45/20251109_160000_extract.log
```

### 1b. Clean Dataset Images (NEW v0.1.12+) 🧹

**Clean Transparent Images** - For transparent PNG datasets:

```bash
# Safe mode: Report only (no files modified)
vogel-trainer clean-transparent ~/training-data/ --mode report

# Move invalid images to invalid_transparent/ folder
vogel-trainer clean-transparent ~/training-data/ --mode move --recursive
```

**Clean Gray Background Images (NEW v0.1.14)** - For gray background datasets:

```bash
# Check gray background ratio
vogel-trainer clean-gray ~/training-data/ --mode report

# Move images with wrong gray ratio to invalid_gray/
vogel-trainer clean-gray ~/training-data/ --mode move --recursive

# Custom thresholds
vogel-trainer clean-gray ~/training-data/ \
  --min-gray 0.10 \
  --max-gray 0.90 \
  --gray-tolerance 30 \
  --mode move
```

**Detection Criteria:**

*clean-transparent:*
- **Min Visible Pixels** (`--min-pixels`, default: 500): Minimum non-transparent pixels
- **Max Transparency** (`--max-transparency`, default: 0.95): Maximum 95% transparency allowed
- **Min Region Size** (`--min-region`, default: 100): Minimum size of largest connected object

*clean-gray:*
- **Min Gray Ratio** (`--min-gray`, default: 0.05): Minimum 5% gray background required
- **Max Gray Ratio** (`--max-gray`, default: 0.95): Maximum 95% gray allowed (need visible bird)
- **Gray Tolerance** (`--gray-tolerance`, default: 30): Tolerance for gray detection (R≈G≈B±30)

**Use Cases:**
- Remove tiny fragments after background removal
- Clean up partial detections (bird flew out of frame)
- Eliminate images with too much/little background
- Find images where bird is barely visible or missing

### 2. Organize Dataset

```bash
# Basic organization (80/20 split)
vogel-trainer organize ~/training-data/ -o ~/organized-data/

# With class balance control (NEW in v0.1.8)
vogel-trainer organize ~/training-data/ -o ~/organized-data/ \
  --max-images-per-class 100 \
  --tolerance 15.0
```

**Class Balance Features:**
- `--max-images-per-class N`: Limit to N images per class, delete excess
- `--tolerance N`: Maximum allowed imbalance % (default: 15)
  - < 10%: ✅ Perfect
  - 10-15%: ⚠️ Warning
  - > 15%: ❌ Error with recommendations

Creates an 80/20 train/validation split:
```
organized/
├── train/
│   ├── great-tit/
│   ├── blue-tit/
│   └── robin/
└── val/
    ├── great-tit/
    ├── blue-tit/
    └── robin/
```

### 3. Train Classifier

```bash
vogel-trainer train ~/organized-data/ -o ~/models/my-classifier/
```

**Training Parameters:**
- `--batch-size`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--early-stopping-patience`: Early stopping patience in epochs (default: 5, 0 to disable)
- `--weight-decay`: Weight decay for regularization (default: 0.01)
- `--warmup-ratio`: Learning rate warmup ratio (default: 0.1)
- `--label-smoothing`: Label smoothing factor (default: 0.1, 0 to disable)
- `--save-total-limit`: Maximum checkpoints to keep (default: 3)
- `--augmentation-strength`: Data augmentation intensity: `none`, `light`, `medium` (default), `heavy`
- `--image-size`: Input image size in pixels (default: 224, supports 224/384/448)
- `--scheduler`: Learning rate scheduler: `cosine` (default), `linear`, `constant`
- `--seed`: Random seed for reproducibility (default: 42)
- `--resume-from-checkpoint`: Path to checkpoint to resume training
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 1)
- `--mixed-precision`: Mixed precision training: `no` (default), `fp16`, `bf16`
- `--push-to-hub`: Push model to HuggingFace Hub (default: False)
- `--log`: Save console output to log file

**Augmentation Strength Levels:**

- **none**: No augmentation (only normalization)
- **light**: Small rotations (±10°), minimal color jitter
- **medium** (default): Moderate rotations (±20°), affine transforms, color jitter, gaussian blur
- **heavy**: Strong rotations (±30°), aggressive transforms, strong color variations

**Advanced Training Examples:**

```bash
# High-accuracy training with large images and heavy augmentation
vogel-trainer train ~/organized-data/ \
  -o ~/models/high-accuracy/ \
  --image-size 384 \
  --augmentation-strength heavy \
  --epochs 100 \
  --early-stopping-patience 10 \
  --batch-size 8

# Fast training with mixed precision (requires GPU)
vogel-trainer train ~/organized-data/ \
  -o ~/models/fast/ \
  --mixed-precision fp16 \
  --batch-size 32 \
  --gradient-accumulation-steps 2

# Reproducible training with fixed seed
vogel-trainer train ~/organized-data/ \
  -o ~/models/reproducible/ \
  --seed 12345 \
  --augmentation-strength light

# Resume interrupted training
vogel-trainer train ~/organized-data/ \
  -o ~/models/continued/ \
  --resume-from-checkpoint ~/models/my-classifier/checkpoints/checkpoint-1000

# Training with logging
vogel-trainer train ~/organized-data/ \
  -o ~/models/logged/ \
  --log
```

**Training Configuration:**
- Base Model: `google/efficientnet-b0` (8.5M parameters)
- Optimizer: AdamW with configurable LR schedule
- Augmentation: 4 intensity levels (none/light/medium/heavy)
- Regularization: Weight decay, label smoothing, early stopping
- Mixed Precision: FP16/BF16 support for faster training on GPU

**Output:**
```
~/models/my-classifier/
├── checkpoints/     # Intermediate checkpoints
├── logs/           # TensorBoard logs
└── final/          # Final trained model
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

### 4. Deduplicate Dataset

Remove duplicate or very similar images from your dataset to improve training quality:

```bash
# Report duplicates without deleting
vogel-trainer deduplicate ~/training-data/ --recursive

# Delete duplicates (keep first occurrence)
vogel-trainer deduplicate ~/training-data/ \
  --mode delete \
  --recursive

# Move duplicates to separate folder
vogel-trainer deduplicate ~/training-data/ \
  --mode move \
  --recursive

# Stricter duplicate detection
vogel-trainer deduplicate ~/training-data/ \
  --threshold 3 \
  --recursive

# Keep largest file instead of first
vogel-trainer deduplicate ~/training-data/ \
  --mode delete \
  --keep largest \
  --recursive
```

**Deduplication Parameters:**
- `--threshold`: Similarity threshold - Hamming distance 0-64, lower=stricter (default: 5)
- `--method`: Hash method: `phash` (default, recommended), `dhash`, `whash`, `average_hash`
- `--mode`: Action: `report` (show only, default), `delete` (remove), `move` (to duplicates/)
- `--keep`: Which duplicate to keep: `first` (chronological, default) or `largest` (file size)
- `--recursive, -r`: Search recursively through subdirectories

**How it works:**
- Uses perceptual hashing (pHash) to detect visually similar images
- Robust against resizing, cropping, and minor color changes
- Threshold of 5 = very similar, 10 = similar, >15 = different
- Safe default: `report` mode prevents accidental deletion

### 5. Quality Check Dataset (New!)

Check your dataset for low-quality images (blurry, too small, corrupted, brightness issues):

```bash
# Report quality issues without deleting
vogel-trainer quality-check ~/training-data/ --recursive

# Delete low-quality images
vogel-trainer quality-check ~/training-data/ \
  --mode delete \
  --recursive

# Move low-quality images to separate folder
vogel-trainer quality-check ~/training-data/ \
  --mode move \
  --recursive

# Stricter blur detection
vogel-trainer quality-check ~/training-data/ \
  --blur-threshold 150.0 \
  --recursive

# Check for brightness/contrast issues
vogel-trainer quality-check ~/training-data/ \
  --check-brightness \
  --recursive

# Comprehensive quality check with custom thresholds
vogel-trainer quality-check ~/training-data/ \
  --blur-threshold 120.0 \
  --min-resolution 100 \
  --min-filesize 2048 \
  --check-brightness \
  --mode move \
  --recursive
```

**Quality Check Parameters:**
- `--blur-threshold`: Minimum blur score (Laplacian variance), lower=more blurry (default: 100.0)
- `--min-resolution`: Minimum image width/height in pixels (default: 50)
- `--min-filesize`: Minimum file size in bytes (default: 1024)
- `--check-brightness`: Also check for brightness/contrast issues (too dark or overexposed)
- `--mode`: Action: `report` (show only, default), `delete` (remove), `move` (to low_quality/)
- `--recursive, -r`: Search recursively through subdirectories

**⚠️ WARNING - Delete Mode:**
- The `--mode delete` option **permanently removes files** without backup
- **Always run `--mode report` first** to preview what will be deleted
- **Backup your dataset** before using delete mode
- Consider using `--mode move` instead (keeps files in `low_quality/` folder)

**What is checked:**
- ✅ **Sharpness**: Detects blurry/out-of-focus images using Laplacian variance
- ✅ **Resolution**: Filters out too-small images that hurt training
- ✅ **File size**: Detects corrupted or empty files
- ✅ **Readability**: Checks if images can be opened and processed
- ✅ **Brightness** (optional): Detects too-dark or overexposed images

**Typical thresholds:**
- Blur: 100.0 (default) = moderate, 150.0 = stricter, 50.0 = more lenient
- Resolution: 50px (default) = very permissive, 100px = recommended, 224px = strict
- Filesize: 1024 bytes (default) = catches corrupted files

**Recommended workflow:**
```bash
# 1. Preview issues first (safe)
vogel-trainer quality-check ~/data/ --mode report --recursive

# 2. Move problematic images (reversible)
vogel-trainer quality-check ~/data/ --mode move --recursive

# 3. Review moved files in low_quality/ folder
# 4. Delete manually if satisfied: rm -rf ~/data/low_quality/
```

### 6. Test Model

```bash
# Test on validation dataset
vogel-trainer test ~/models/my-classifier/ -d ~/organized-data/

# Output:
# 🧪 Testing model on validation set...
#    🐦 Predicted: great-tit (98.5% confidence)
```

---

## 🔄 Iterative Training Workflow
Improve your model accuracy through iterative refinement using auto-classification:

```mermaid
flowchart TD
    Start([📋 Phase 1: Initial Model<br/>Manual Labeling]) --> Extract1[1️⃣ Extract with manual labels<br/><code>vogel-trainer extract video.mp4<br/>--folder data/ --bird kohlmeise</code>]
    
    Extract1 --> Organize1[2️⃣ Organize dataset 80/20 split<br/><code>vogel-trainer organize data/<br/>-o organized/</code>]
    
    Organize1 --> Train1[3️⃣ Train initial model<br/><code>vogel-trainer train organized/<br/>-o models/v1/</code><br/>✅ <b>Result: 92% accuracy</b>]
    
    Train1 --> Phase2([🔄 Phase 2: Model Improvement<br/>Auto-Classification])
    
    Phase2 --> Extract2[4️⃣ Auto-extract with trained model<br/><code>vogel-trainer extract new-videos/<br/>--folder data-v2/<br/>--species-model models/v1/final/<br/>--species-threshold 0.85</code><br/>🎯 <b>Automatically sorted by species!</b>]
    
    Extract2 --> Review[5️⃣ Manual review & corrections<br/>• Check auto-classifications<br/>• Move misclassified images<br/>• Merge with previous dataset]
    
    Review --> Train2[6️⃣ Retrain with expanded dataset<br/><code>vogel-trainer organize data-v2/<br/>-o organized-v2/<br/>vogel-trainer train organized-v2/<br/>-o models/v2/</code><br/>🎉 <b>Result: 96% accuracy!</b>]
    
    Train2 --> Repeat{♻️ Continue<br/>improving?}
    Repeat -->|Yes| Extract2
    Repeat -->|No| End([✅ Final Model])
    
    style Start fill:#e1f5ff,stroke:#0066cc,stroke-width:3px
    style Phase2 fill:#e1f5ff,stroke:#0066cc,stroke-width:3px
    style Train1 fill:#d4edda,stroke:#28a745,stroke-width:2px
    style Train2 fill:#d4edda,stroke:#28a745,stroke-width:2px
    style End fill:#d4edda,stroke:#28a745,stroke-width:3px
    style Extract2 fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    style Review fill:#f8d7da,stroke:#dc3545,stroke-width:2px
```

**Key Benefits:**
- 🚀 **Faster labeling**: Auto-classification saves manual work
- 📈 **Better accuracy**: More training data = better model
- 🎯 **Quality control**: `--species-threshold` filters uncertain predictions
- 🔄 **Continuous improvement**: Each iteration improves the model

**Example Commands:**

```bash
# Phase 1: Manual training (initial dataset)
vogel-trainer extract ~/Videos/batch1/*.mp4 --folder ~/data/ --bird great-tit
vogel-trainer organize ~/data/ -o ~/data/organized/
vogel-trainer train ~/data/organized/ -o ~/models/v1/

# Phase 2: Auto-classification with trained model
vogel-trainer extract ~/Videos/batch2/*.mp4 \
  --folder ~/data-v2/ \
  --species-model ~/models/v1/final/ \
  --species-threshold 0.85

# Review classifications in ~/data-v2/<species>/ folders
# Move any misclassified images to correct species folders

# Merge datasets and retrain
cp -r ~/data-v2/* ~/data/
vogel-trainer organize ~/data/ -o ~/data/organized-v2/
vogel-trainer train ~/data/organized-v2/ -o ~/models/v2/
```

---

##  Performance & Best Practices

### Dataset Size Recommendations

| Quality | Images per Species | Expected Accuracy |
|---------|-------------------|-------------------|
| Minimum | 20-30            | ~85-90%          |
| Good    | 50-100           | ~92-96%          |
| Optimal | 100+             | >96%             |

### Tips for Better Results

1. **Dataset Diversity**
   - Include various lighting conditions
   - Capture different poses (side, front, back)
   - Cover different seasons (plumage changes)

2. **Class Balance**
   - Aim for similar image counts per species
   - Avoid having one dominant class

3. **Quality Over Quantity**
   - Use threshold 0.5-0.6 for clear detections
   - Manual review of auto-sorted images improves quality

4. **Monitor Training**
   - Check per-class accuracy for weak species
   - Use confusion matrix to identify similar species
   - Add more data for low-performing classes

---

## 🔗 Integration with vogel-video-analyzer

Use your trained model for species identification:

```bash
vogel-analyze --identify-species \
  --species-model ~/models/final/ \
  --species-threshold 0.3 \
  video.mp4
```

---

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Credits

- **YOLO** by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **EfficientNet** by [Google Research](https://github.com/google/automl)
- **Transformers** by [Hugging Face](https://huggingface.co/transformers)

---

## 📮 Support & Contributing

- **Issues**: [GitHub Issues](https://github.com/kamera-linux/vogel-model-trainer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kamera-linux/vogel-model-trainer/discussions)
- **Pull Requests**: Contributions welcome!

---

Made with ❤️ for bird watching enthusiasts 🐦
