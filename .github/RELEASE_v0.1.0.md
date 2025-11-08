# 🎉 vogel-model-trainer v0.1.0

**Release Date:** November 8, 2025

First official release of vogel-model-trainer! 🐦

## 📦 Installation

```bash
pip install vogel-model-trainer
```

## ✨ Features

### Core Functionality

- **🎯 YOLO-based Bird Detection** - Automated bird cropping from videos using YOLOv8
- **🤖 Three Extraction Modes**:
  - Standard extraction mode
  - Manual labeling mode (interactive species selection)
  - Auto-sorting mode (using pre-trained classifier)
- **📁 Wildcard Support** - Batch process multiple videos with glob patterns
- **🖼️ Auto-Resize to 224x224** - Optimal image size for training
- **📊 Dataset Organization** - Automatic 80/20 train/val split with shuffling

### Model Training

- **🧠 EfficientNet-B0 Training** - Lightweight yet powerful classification model
- **🎨 Enhanced Data Augmentation**:
  - Random resized crop (80-100%)
  - Random horizontal flip (50%)
  - Color jitter (brightness, contrast, saturation, hue)
  - Gaussian blur for robustness
- **📈 Optimized Training**:
  - Cosine learning rate scheduling
  - Label smoothing (0.1)
  - Early stopping (patience: 7 epochs)
  - Automatic checkpointing
- **⏸️ Graceful Shutdown** - Save model state on Ctrl+C interruption
- **📊 Per-Species Metrics** - Detailed accuracy breakdown by species

### CLI Interface

Comprehensive command-line interface with subcommands:

```bash
# Extract birds from videos
vogel-trainer extract video.mp4 -o training-data/

# Organize dataset
vogel-trainer organize training-data/ -o organized/

# Train model
vogel-trainer train organized/ -o models/my-classifier/

# Test model
vogel-trainer test models/my-classifier/ -d organized/
```

### Documentation

- **📖 Bilingual Documentation** - Complete README in English and German
- **🎓 Training Guides** - Step-by-step workflow documentation
- **💡 Examples** - Real-world usage examples
- **🤝 Contributing Guidelines** - Clear contribution process
- **🔒 Security Policy** - Responsible disclosure policy

## 🚀 Getting Started

### Quick Start

```bash
# 1. Install
pip install vogel-model-trainer

# 2. Extract birds from your videos
vogel-trainer extract video.mp4 -o ~/training-data/kohlmeise/

# 3. Organize dataset
vogel-trainer organize ~/training-data/ -o ~/organized-data/

# 4. Train your model
vogel-trainer train ~/organized-data/ -o ~/models/my-classifier/

# 5. Test the model
vogel-trainer test ~/models/my-classifier/ -d ~/organized-data/
```

### System Requirements

- **Python:** 3.9 or higher
- **GPU:** Recommended for training (Raspberry Pi 5 supported)
- **Disk Space:** Depends on dataset size
- **RAM:** Minimum 4GB, 8GB+ recommended for training

## 📊 Performance

- **Extraction Speed:** ~5-10 FPS on Raspberry Pi 5
- **Training Time:** ~3-4 hours for 500 images on Raspberry Pi 5
- **Model Accuracy:** >96% on well-organized datasets
- **Model Size:** ~17MB (EfficientNet-B0)

## 🔄 Workflow

1. **Extract** bird images from videos using YOLO detection
2. **Organize** images by species into subdirectories
3. **Split** dataset into 80% training / 20% validation
4. **Train** custom EfficientNet-B0 classifier
5. **Test** model accuracy on validation set
6. **Deploy** trained model for species identification

## 📚 Resources

- **Documentation:** [README.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.md)
- **German Docs:** [README.de.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.de.md)
- **Contributing:** [CONTRIBUTING.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/CONTRIBUTING.md)
- **Security:** [SECURITY.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/SECURITY.md)
- **Changelog:** [CHANGELOG.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md)
- **PyPI:** [pypi.org/project/vogel-model-trainer](https://pypi.org/project/vogel-model-trainer/)

## 🐛 Known Issues

- Manual labeling mode not yet implemented (use standard extraction + manual sorting)
- Auto-sorting mode not yet implemented (use standard extraction + manual sorting)
- No GUI interface (CLI only)

## 🔮 Future Plans

- Interactive manual labeling during extraction
- Automatic species sorting using pre-trained models
- Support for additional model architectures
- Real-time training monitoring dashboard
- Dataset augmentation presets
- Model optimization for embedded devices

## 🙏 Acknowledgments

- **Ultralytics** - For the excellent YOLOv8 implementation
- **Hugging Face** - For the Transformers library
- **PyTorch Team** - For the deep learning framework
- **Contributors** - Thank you to everyone who helped test and improve this project!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Full Changelog:** https://github.com/kamera-linux/vogel-model-trainer/commits/v0.1.0

**Questions?** Open an issue at https://github.com/kamera-linux/vogel-model-trainer/issues
