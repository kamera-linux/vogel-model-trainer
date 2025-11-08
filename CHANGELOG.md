# Changelog

All notable changes to vogel-model-trainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PyPI publishing infrastructure
  - MANIFEST.in for package distribution
  - Build and upload scripts
  - GitHub Actions workflow for automated publishing
  - Comprehensive PUBLISHING.md documentation
- Community health files
  - CONTRIBUTING.md with contribution guidelines
  - SECURITY.md with security policy and best practices
  - Enhanced issue templates (bug report, feature request, question)

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

[Unreleased]: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kamera-linux/vogel-model-trainer/releases/tag/v0.1.0
