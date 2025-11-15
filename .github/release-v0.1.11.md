# Release v0.1.11 - AI-Powered Background Removal 🧪

## 🎯 Release Highlights

**Experimental Feature**: AI-powered automatic background removal using state-of-the-art deep learning segmentation.

### 🧪 Background Removal (EXPERIMENTAL)

Isolate birds from complex backgrounds with AI-powered segmentation using the rembg library:

```bash
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird rotkehlchen \
  --remove-background \
  --bg-color white \
  --bg-model u2net
```

#### New CLI Parameters

- `--remove-background`: Enable AI-powered background removal
- `--bg-color [white|black|gray]`: Choose background color (default: white)
- `--bg-model [u2net|u2netp|isnet-general-use]`: Select AI model (default: u2net)

#### Features

- **AI-Based Segmentation**: Uses U²-Net deep learning model for accurate bird isolation
- **Alpha Matting**: Smooth, professional edges around feathers and fine details
- **Post-Processing**: Morphological operations and Gaussian blur for clean results
- **Multiple Models**:
  - `u2net` (default): Best overall quality (~180MB)
  - `u2netp`: Faster, smaller model for quick processing
  - `isnet-general-use`: Best edge quality for detailed feathers
- **Background Colors**:
  - `white`: Clean white background (#FFFFFF)
  - `black`: High contrast black background (#000000)
  - `gray`: Neutral gray background (#808080)
- **Complex Backgrounds**: Works with branches, leaves, buildings, and varied environments
- **Varied Plumage**: Handles different bird species, feather colors, and fine details

#### Technical Details

- **Model Download**: ~180MB on first use (cached afterward)
- **Dependency**: Requires `rembg>=2.0.50`
- **GPU Support**: Optional GPU acceleration via `rembg[gpu]`
- **Processing**: Alpha matting with foreground/background threshold tuning
- **Quality**: Morphological close/open operations + Gaussian blur for smooth edges

### 🔄 Changes from v0.1.10

- **Improved Algorithm**: Switched from OpenCV GrabCut to AI-based rembg for better quality
- **Enhanced Edges**: Alpha matting technique for professional-looking segmentation
- **Better Reliability**: Works consistently with complex backgrounds and varied bird plumage
- **More Options**: Model selection and background color customization

### 📖 Documentation Updates

- Updated all README files (EN/DE/JA) with comprehensive background removal documentation
- Added EXPERIMENTAL markers to clearly indicate the feature status
- Updated i18n strings to reflect rembg/AI implementation
- Updated CHANGELOG.md with detailed release notes

## 🎬 Example Use Cases

### High-Quality Dataset Creation

```bash
# Extract with all quality filters + background removal
vogel-trainer extract video.mp4 \
  --folder ~/clean-dataset/ \
  --species-model kamera-linux/german-bird-classifier \
  --threshold 0.7 \
  --species-threshold 0.85 \
  --min-sharpness 150 \
  --min-edge-quality 80 \
  --deduplicate \
  --remove-background \
  --bg-color white \
  --bg-model u2net
```

### Black Background for Contrast

```bash
# Ideal for training models with high-contrast images
vogel-trainer extract video.mp4 \
  --folder ~/data-black-bg/ \
  --bird blaumeise \
  --remove-background \
  --bg-color black \
  --bg-model isnet-general-use
```

### Fast Processing with Smaller Model

```bash
# Quick extraction with smaller, faster model
vogel-trainer extract video.mp4 \
  --folder ~/data-quick/ \
  --bird rotkehlchen \
  --remove-background \
  --bg-model u2netp
```

## 📦 Installation

### From PyPI

```bash
pip install vogel-model-trainer
```

### With GPU Support (Recommended)

```bash
pip install vogel-model-trainer
pip install rembg[gpu]
```

### From Source

```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.11
pip install -e .
```

## ⚠️ Important Notes

### Experimental Status

This feature is marked as **EXPERIMENTAL** because:
- It's a new addition requiring real-world testing with diverse videos
- Model performance may vary across different bird species and backgrounds
- Processing time is significantly longer than without background removal
- Model download (~180MB) required on first use

### Requirements

- Python 3.9+
- ~180MB disk space for model cache
- rembg>=2.0.50 (installed automatically)
- Optional: CUDA-capable GPU for faster processing

### Performance Considerations

- **First Run**: Downloads ~180MB model (one-time, cached afterward)
- **Processing Speed**: ~2-5x slower than without background removal
- **GPU Acceleration**: Significantly faster with CUDA GPU
- **Memory Usage**: Increased memory footprint (~500MB-1GB)

## 🐛 Known Issues

- None reported yet (new feature)

## 📝 Feedback Welcome

As this is an experimental feature, we welcome feedback on:
- Quality results with different bird species
- Performance on different hardware configurations
- Edge cases or failure scenarios
- Suggestions for improvement

Please report issues or share results at: https://github.com/kamera-linux/vogel-model-trainer/issues

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Documentation**: https://github.com/kamera-linux/vogel-model-trainer#readme
- **Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md

## 👥 Credits

- **rembg Library**: https://github.com/danielgatis/rembg
- **U²-Net Model**: Qin, X., et al. "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection"

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.10...v0.1.11
