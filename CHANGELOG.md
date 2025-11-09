# Changelog

All notable changes to vogel-model-trainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2025-11-09

### Added
- **Japanese Documentation**: Complete Japanese translation of README (README.ja.md)
- **Library Functions**: Core modules now provide dedicated library functions
  - `extractor.extract_birds_from_video()` - Library-ready extraction function
  - `organizer.organize_dataset()` - Dataset organization function
  - `trainer.train_model()` - Model training function with configurable parameters
  - `tester.test_model()` - Unified testing function for validation sets and single images

### Changed
- **Core Module Architecture**: Converted from script-only to library+script hybrid pattern
  - All core modules now have dedicated functions for programmatic use
  - `main()` functions kept as wrappers for direct script execution
  - CLI commands updated to call library functions
- **Documentation**: Updated language selection in all READMEs to include Japanese
- **tester.py**: Unified `test_model()` function now handles both validation set testing and single image prediction

### Fixed
- **CLI Integration**: Improved parameter mapping between CLI commands and core functions
- **Function Signatures**: Ensured all CLI calls match function parameter names exactly

### Technical Details
- `organizer.py`: Added `organize_dataset(source_dir, output_dir, train_ratio=0.8)` function
- `trainer.py`: Extracted training logic into `train_model(data_dir, output_dir, model_name, batch_size, num_epochs, learning_rate)` function
- `tester.py`: Refactored to support both operational modes through single function interface
- All modules maintain backward compatibility for direct script execution

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
