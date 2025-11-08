# 🐦 Vogel Model Trainer

**Languages:** [🇬🇧 English](README.md) | [🇩🇪 Deutsch](README.de.md)

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

- 🎯 **YOLO-based Bird Detection** - Automated bird cropping from videos using YOLOv8
- 🤖 **Three Extraction Modes** - Manual labeling, auto-sorting, or standard extraction
- 📁 **Wildcard Support** - Batch process multiple videos with glob patterns
- 🖼️ **Auto-Resize to 224x224** - Optimal image size for training
- 🧠 **EfficientNet-B0 Training** - Lightweight yet powerful classification model
- 🎨 **Enhanced Data Augmentation** - Rotation, affine transforms, color jitter, gaussian blur
- 📊 **Optimized Training** - Cosine LR scheduling, label smoothing, early stopping
- ⏸️ **Graceful Shutdown** - Save model state on Ctrl+C interruption
- 🔄 **Iterative Training** - Use trained models to expand your dataset
- 📈 **Per-Species Metrics** - Detailed accuracy breakdown by species

---

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install vogel-model-trainer

# Or install from source
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
pip install -e .
```

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

### 1. Extract Training Images

#### Manual Mode (Recommended for Initial Collection)

When you know the species in your video:

```bash
vogel-trainer extract ~/Videos/great-tit.mp4 \
  --folder ~/training-data/ \
  --bird great-tit \
  --threshold 0.5 \
  --sample-rate 3
```

#### Auto-Sort Mode (For Iterative Training)

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
- `--threshold`: YOLO confidence threshold (default: 0.5)
- `--sample-rate`: Process every Nth frame (default: 3)
- `--detection-model`: YOLO model path (default: yolov8n.pt)
- `--no-resize`: Keep original image size (default: resize to 224x224)
- `--recursive, -r`: Search directories recursively

### 2. Organize Dataset

```bash
vogel-trainer organize ~/training-data/ -o ~/organized-data/
```

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

**Training Configuration:**
- Base Model: `google/efficientnet-b0` (8.5M parameters)
- Optimizer: AdamW with cosine LR schedule
- Augmentation: Rotation, affine, color jitter, gaussian blur
- Regularization: Weight decay 0.01, label smoothing 0.1
- Early Stopping: Patience of 7 epochs

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

### 4. Test Model

```bash
# Test on validation dataset
vogel-trainer test ~/models/my-classifier/ -d ~/organized-data/

# Output:
# 🧪 Testing model on validation set...
#    🐦 Predicted: great-tit (98.5% confidence)
```

---

## 🔄 Iterative Training Workflow

Improve your model by iteratively expanding your dataset:

```bash
# 1. Initial training with manual labels
vogel-trainer extract ~/Videos/batch1/*.mp4 --bird great-tit --output ~/data/
vogel-trainer organize --source ~/data/ --output ~/data/organized/
vogel-trainer train --data ~/data/organized/ --output ~/models/v1/

# 2. Use trained model to extract more data
vogel-trainer extract ~/Videos/batch2/*.mp4 \
  --species-model ~/models/v1/final/ \
  --output ~/data/iteration2/

# 3. Review and correct misclassifications manually
# Move incorrect predictions to correct species folders

# 4. Combine datasets and retrain
cp -r ~/data/iteration2/* ~/data/
vogel-trainer organize --source ~/data/ --output ~/data/organized/
vogel-trainer train --data ~/data/organized/ --output ~/models/v2/

# Result: Higher accuracy! 🎉
```

---

## 📊 Performance & Best Practices

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
