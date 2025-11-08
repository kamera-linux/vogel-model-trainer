# Changelog

All notable changes to vogel-model-trainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-11-08

### Fixed
- **CLI Parameters**: Corrected extract command parameters to match original `extract_birds.py` script
  - Changed `--output/-o` to `--folder` for consistency
  - Renamed `--model` to `--detection-model` for clarity
  - Updated default `--threshold` from 0.3 to 0.5 for higher quality
  - Updated default `--sample-rate` from 10 to 3 for better detection

### Added
- `--bird` parameter for manual species naming (creates subdirectory)
- `--species-model` parameter for auto-sorting with trained classifier
- `--no-resize` flag to keep original image size
- `--recursive/-r` flag for recursive directory search

### Changed
- Simplified CLI interface: removed separate `extract-manual` and `extract-auto` commands
- All extraction modes now unified under single `extract` command with flags
- Updated documentation (README.md, README.de.md) with correct parameter examples

### Breaking Changes
- ⚠️ CLI parameter names changed - v0.1.0 commands will not work with v0.1.1
- Migration required for existing scripts using the CLI

## [0.1.0] - 2025-11-08

### Added
- Initial release of vogel-model-trainer
- **CLI Commands**: `vogel-trainer extract`, `organize`, `train`, `test`
- **Bird Detection**: YOLO-based bird detection and cropping from videos
- **Extraction Modes**:
  - Manual labeling mode (interactive species selection)
  - Auto-sorting mode (using pre-trained classifier)
  - Standard extraction mode
- **Video Processing**:
  - Wildcard and recursive video processing
  - Batch processing support
  - Sample rate configuration
- **Image Processing**:
  - Automatic 224x224 image resizing
  - Quality filtering
- **Model Training**:
  - EfficientNet-B0 based architecture
  - Enhanced data augmentation pipeline (rotation, affine, color jitter, blur)
  - Optimized hyperparameters (cosine LR scheduling, label smoothing)
  - Early stopping support
  - Automatic train/val split
- **Features**:
  - Graceful shutdown with Ctrl+C (saves model state)
  - Automatic species detection from directory structure
  - Per-species accuracy metrics
  - Model testing and evaluation tools
- **Documentation**:
  - Comprehensive README (English and German)
  - Usage examples and workflows
  - Installation instructions

[Unreleased]: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kamera-linux/vogel-model-trainer/releases/tag/v0.1.0
