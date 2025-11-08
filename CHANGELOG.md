# Changelog

All notable changes to vogel-model-trainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-08

### Added
- Initial release of vogel-model-trainer
- CLI commands: `vogel-trainer extract`, `organize`, `train`, `test`
- YOLO-based bird detection and cropping
- 3 extraction modes: manual, auto-sort, standard
- Wildcard and recursive video processing
- Automatic 224x224 image resizing
- EfficientNet-B0 based training
- Enhanced data augmentation pipeline
- Optimized training hyperparameters
- Graceful shutdown with Ctrl+C
- Automatic species detection from directory structure
- Per-species accuracy metrics
- Model testing and evaluation

[0.1.0]: https://github.com/kamera-linux/vogel-model-trainer/releases/tag/v0.1.0
