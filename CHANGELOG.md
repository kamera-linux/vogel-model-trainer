# Changelog

All notable changes to vogel-model-trainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.16] - 2025-11-18

### Added
- **🖼️ Image Support in Extractor**: Extended bird crop extraction to support static images
  - New `extract_birds_from_image()` function for processing single images
  - Automatic detection of video vs. image files based on extension
  - Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` (and uppercase variants)
  - Same features as video extraction: background removal, species classification, quality filtering
  - Example: `python -m vogel_model_trainer.core.extractor photo.jpg --folder output/ --bg-remove --bg-transparent`
- **🔄 Convert Mode**: New `--convert` mode for processing existing bird crop datasets
  - Process already-extracted bird images without YOLO detection
  - Maintains folder structure (species subdirectories)
  - Applies consistent processing: background removal, quality filtering, deduplication
  - Use case: Normalize existing training datasets for model comparability
  - Example: `--convert --source ~/data-original --target ~/data-transparent --bg-remove --bg-transparent`
- **📊 Quality Filtering Parameters**: New parameters for image quality control
  - `--min-sharpness`: Filter out blurry images (Laplacian variance threshold)
  - `--min-edge-quality`: Filter out poor edge quality (Sobel gradient threshold)
  - `--quality-report`: Generate detailed quality statistics
  - `--deduplicate`: Skip duplicate/similar images using perceptual hashing
  - `--similarity-threshold`: Hamming distance for duplicate detection (default: 5)
- **🎨 Background Processing Options**: Extended background removal parameters
  - `--bg-remove`: Enable AI-based background removal (rembg)
  - `--bg-transparent`: Use transparent background (PNG with alpha channel)
  - `--bg-color R,G,B`: Custom background color (default: 128,128,128 = gray)
  - `--bg-model`: Choose rembg model (u2net, u2netp, isnet-general-use)
  - `--bg-fill-black`: Make black background/padding areas transparent
  - `--crop-padding N`: Extra pixels around detected bird (default: 0)
  - `--quality`: JPEG quality 1-100 (default: 95)
- **🤖 GitHub Automation Tools**: Complete workflow automation infrastructure
  - `scripts/create_github_release.py`: Automated release creation from Git tags
    - Auto-detects latest tag or uses specified version
    - Finds and uses release notes automatically
    - Monitors PyPI publish workflow with real-time status
    - Matches workflow by release tag to prevent version mismatches
    - Support for draft and pre-release modes
  - `.github/GITHUB_CLI_GUIDE.md`: Comprehensive GitHub CLI documentation (700+ lines)
    - Complete gh CLI reference for releases, PRs, workflows, issues
    - Practical examples and production scripts
    - Best practices and useful aliases
    - German language documentation

### Changed
- **📝 Extractor Help Text**: Updated to reflect video and image support
  - New examples for single images, batch processing, mixed media
  - Clearer parameter descriptions for new features
  - Convert mode usage examples added
- **🗂️ Flexible Input Handling**: Enhanced file detection and processing
  - Supports glob patterns for both videos and images (`*.mp4`, `*.jpg`, `~/Media/**/*`)
  - Recursive directory search with `--recursive` flag
  - Mixed video and image processing in single run
  - Categorizes files by type and shows summary before processing

### Fixed
- **🎯 Release Script Workflow Matching**: Fixed workflow version detection
  - Previously showed wrong workflow version (e.g., v0.1.14 instead of v0.1.15)
  - Now searches last 5 workflow runs for matching release tag
  - Matches both `v0.1.16` and `0.1.16` in workflow title
  - Provides accurate real-time workflow status

## [0.1.15] - 2025-11-17

### Added
- **🎯 Crop Padding Feature**: New `--crop-padding` parameter for better background removal
  - Expands rembg foreground mask to preserve bird details (feet, beak, feathers)
  - Prevents important details from being cut off by aggressive background removal
  - Recommended values: 5-20 pixels for optimal results
  - Example: `--crop-padding 20` keeps 20px more background around detected bird
  - Works only with `--remove-background` flag
- **🎨 Random Background Augmentation**: Training now uses random backgrounds for transparent images
  - When training with PNG/transparent images, applies random gray/black/white backgrounds
  - Forces model to focus on bird features instead of background color
  - Improves model robustness against different backgrounds
  - Automatically applied during training phase only (validation uses neutral gray)
- **🔧 ARM64/Raspberry Pi Support**: Added `onnxruntime` as explicit dependency
  - Fixes rembg installation issues on ARM64 architectures (Raspberry Pi, etc.)
  - `onnxruntime>=1.15.0` now automatically installed with package
  - Dynamic rembg import check at runtime for better error handling

### Changed
- **📝 Improved Help Text**: Updated `--crop-padding` help documentation
  - Clearer explanation of mask expansion functionality
  - Better guidance for recommended values
  - Explains interaction with `--remove-background` flag

### Fixed
- **🐛 PNG Transparency with Species Classification**: Fixed transparent PNG export when using `--species-model` + `--bg-transparent` + `--deduplicate`
  - Fixed filename extension determination happening after filename creation (was hardcoded to .jpg)
  - Fixed deduplicate creating RGB PIL images before background removal (now preserves RGBA)
  - Fixed Python variable caching preventing proper RGBA mode after background removal (explicit deletion)
  - All three fixes required for full PNG transparency support with all feature combinations
  - Verified: PNG images now correctly saved with RGBA format in all scenarios
- **🖼️ PNG Support in Dataset Organizer**: Added PNG file support to `vogel-trainer organize`
  - Previously only searched for .jpg files, now supports both .jpg and .png
  - Enables organizing datasets created with `--bg-transparent` flag
  - Both direct species folders and legacy video folders now support PNG
- **🐛 Dynamic Import**: Fixed rembg import detection for runtime-installed packages
  - Moved import check inside function instead of module-level
  - Allows pip install after module import without restart
  - Better error messages when rembg is not available

## [0.1.14] - 2025-11-16

### Changed
- **🎨 Gray Background as Default**: Changed default background from transparent to gray (#808080)
  - Better for model training - most frameworks ignore alpha channel anyway
  - Neutral gray provides consistent background without affecting bird colors
  - Use `--bg-transparent` to enable transparent backgrounds if needed
  - Changed `--bg-fill-black` default to `False` (not needed for gray backgrounds)

### Added
- **🧹 Clean Gray Command**: New `clean-gray` command to validate gray background datasets
  - Detects images with too much gray (>95% = mostly background, no bird)
  - Detects images with too little gray (<5% = no background padding)
  - Configurable gray tolerance for detection (default: ±30 RGB difference)
  - Three modes: `report` (preview), `move` (to invalid_gray/), `delete` (permanent)
  - Works with both JPEG and PNG images
  - Example: `vogel-trainer clean-gray ~/dataset/sperling/ --mode report`

### Fixed
- **🎯 Background Color Bug**: Fixed `--bg-color` parameter not being applied to padding areas
  - Padding color was hardcoded to black (0,0,0) instead of using bg_color parameter
  - Now correctly converts BGR→RGB and applies chosen background color
  - Fixed `--bg-fill-black` being applied to colored backgrounds (should only affect transparent)

## [0.1.13] - 2025-11-16

### Fixed
- **🐦 Black Feather Preservation**: Fixed `--bg-fill-black` to preserve black feathers on birds
  - Now only makes black areas transparent that are ALREADY identified as background by rembg (alpha < 0.1)
  - Black feathers, Kohlmeisen caps, Raben, Amseln are now correctly preserved
  - Only removes black padding/box areas that were already marked as background
  - Prevents false transparency on birds with black plumage

### Improved
- Better documentation for `--bg-fill-black` parameter explaining that black bird features are preserved
- Updated CLI help text to clarify that only padding/background areas are affected

## [0.1.12] - 2025-11-16

### Added
- **🧹 Clean Transparent Command**: New `clean-transparent` command to detect and remove fragmented/incomplete transparent PNG images
  - Detects images with too much transparency (>95% default)
  - Identifies fragmented objects with small connected regions
  - Checks for minimum visible pixels (500 default)
  - Three modes: `report` (safe preview), `move` (to invalid_transparent/), `delete` (permanent)
  - Configurable thresholds: `--min-pixels`, `--max-transparency`, `--min-region`
  - Recursive directory scanning with `--recursive`
  - Helps clean up datasets after AI-based background removal

### Changed
- **🎨 Transparent Background as DEFAULT**: Background removal now creates PNG with alpha channel by default
  - `--bg-transparent` is now TRUE by default (use `--no-bg-transparent` to disable)
  - `--bg-fill-black` is now TRUE by default (black areas become transparent)
  - Automatically saves as PNG when transparent, JPEG when opaque
  - Better for training models with clean, isolated bird images
  - File extension automatically set based on transparency (.png vs .jpg)

### Improved
- Better handling of RGBA images during resizing and padding
- Automatic file format detection based on transparency
- Enhanced validation for transparent images with connected component analysis

## [0.1.11] - 2025-11-16

### Added
- **🧪 Background Removal (EXPERIMENTAL)**: AI-powered automatic background removal using rembg
  - `--remove-background`: Enable background removal feature
  - `--bg-color [white|black|gray]`: Choose background color (default: white)
  - `--bg-model [u2net|u2netp|isnet-general-use]`: Select AI model (default: u2net)
  - Uses U²-Net deep learning model for accurate bird segmentation
  - Alpha matting for smooth, professional edges
  - Post-processing with morphological operations and Gaussian blur
  - Helps create cleaner training datasets by isolating birds from complex backgrounds
  - **Note**: Downloads ~180MB model on first use (cached afterward)
  - **Note**: Requires additional dependency `rembg>=2.0.50`
  - Full i18n support (EN/DE/JA)

### Changed
- Improved background removal algorithm from OpenCV GrabCut to AI-based rembg for better quality
- Enhanced edge quality in extracted images with alpha matting technique

### Fixed
- Background removal now works reliably with complex backgrounds and varied bird plumage
- Better handling of feathered edges and fine details

## [0.1.10] - 2025-11-15

### Added
- **Background Removal**: Initial implementation (superseded by v0.1.11)

- **Quality Check Command**: New `quality-check` command to find and remove low-quality images from datasets
  - `--blur-threshold N`: Minimum sharpness score using Laplacian variance (default: 100.0)
  - `--min-resolution N`: Minimum image width/height in pixels (default: 50)
  - `--min-filesize N`: Minimum file size in bytes to detect corrupted files (default: 1024)
  - `--check-brightness`: Optional brightness/contrast analysis (detects too-dark or overexposed images)
  - `--mode`: Three operation modes: `report` (safe, default), `move` (to low_quality/), `delete` (permanent)
  - `--recursive`: Search through subdirectories
  
- **Comprehensive Quality Checks**: Five quality criteria for dataset cleaning
  - **Sharpness**: Laplacian variance detects blurry/out-of-focus images
  - **Resolution**: Filters images below minimum dimensions
  - **File Size**: Detects corrupted or empty files
  - **Readability**: Checks if images can be opened and processed
  - **Brightness** (optional): Detects too-dark (<30) or overexposed (>225) images

- **Safety Features**: Multiple safeguards against accidental data loss
  - Default `report` mode is non-destructive
  - Detailed preview of all issues before deletion
  - `move` mode preserves originals in separate folder
  - Clear warnings for `delete` mode in documentation

- **Full i18n Coverage**: Complete translations for all quality-check features
  - 25 new translation keys added (EN/DE/JA)
  - Detailed quality reports in user's language
  - Consistent multilingual error messages

### Improved
- **Documentation**: Comprehensive quality-check examples and warnings
  - README.md: Added complete quality-check section with 6 examples
  - README.de.md: Full German translation with safety warnings
  - README.ja.md: Japanese documentation updated
  - Feature list expanded to 16 features

## [0.1.9] - 2025-11-14

### Added
- **Motion Quality Filtering**: Advanced blur and motion detection for better image quality
  - `--min-sharpness N`: Filter images by sharpness score (Laplacian variance, typical: 100-300)
  - `--min-edge-quality N`: Filter images by edge clarity (Sobel gradient, typical: 50-150)
  - `--save-quality-report`: Generate detailed quality statistics after extraction
  - Automatic rejection of motion-blurred and out-of-focus images
  - Quality metrics: sharpness, edge quality, overall score

- **Quality Report**: Detailed statistics about extraction quality
  - Total detections processed
  - Accepted vs rejected counts and percentages
  - Average quality scores for accepted and rejected images
  - Breakdown by rejection reason (motion blur, poor edges)

- **Full Internationalization**: All new motion quality features support EN/DE/JA
  - Complete translations for quality filters and reports
  - Consistent multilingual experience

### Changed
- **Extract Command**: Enhanced with motion quality analysis
- **Image Quality**: Better automatic filtering of unusable images
- **Fewer False Positives**: Motion-blurred birds automatically rejected

### Improved
- Reduced manual review time by automatically filtering low-quality detections
- Better training data quality leads to more accurate models
- Reproducible quality decisions based on quantifiable metrics

## [0.1.8] - 2025-11-13

### Added
- **Class Balance Management**: New parameters for `organize` command to ensure balanced datasets
  - `--max-images-per-class N`: Automatically limit images per class (100, 200, 300, etc.) and delete excess
  - `--tolerance N`: Set maximum allowed class imbalance (default: 15%)
  - Displays warnings at 10-15% imbalance, errors and exits above tolerance
  - Shows detailed statistics: minimum, maximum, average, and affected classes
  - Provides recommendations for improving balance

- **Full Internationalization**: All new features support EN/DE/JA
  - Complete translations for all class balance messages
  - Consistent multilingual experience across all commands

### Changed
- **Organize Command**: Enhanced with automatic class balancing and limiting
- **Dataset Quality**: Better control over dataset composition and balance

## [0.1.7] - 2025-11-10

### Fixed
- **Cross-Session Deduplication**: Fixed bug where `--deduplicate` only worked within a single extraction session
  - Now pre-loads existing images from output directory into hash cache at startup
  - Prevents re-extracting identical birds when running the same command multiple times
  - Displays loading progress: "Lade existierende Bilder für Deduplizierung..." and "X existierende Bilder in Hash-Cache geladen"

## [0.1.6] - 2025-11-10

### Added
- **Duplicate Detection in Extract**: Prevent extracting similar/duplicate images
  - `--deduplicate`: Enable perceptual hashing to skip duplicate images
  - `--similarity-threshold N`: Configure duplicate detection sensitivity (0-64, default: 5)
  - Uses pHash algorithm, robust against resize/crop/minor color changes
  - Session-level cache prevents duplicates within same extraction run
  - Statistics show number of skipped duplicates

- **New Deduplicate Command**: Clean existing datasets from duplicates
  - `vogel-trainer deduplicate <data_dir>`: Find and remove duplicate images
  - Three modes: `report` (show only), `delete` (remove), `move` (to duplicates/)
  - Four hash methods: `phash` (default), `dhash`, `whash`, `average_hash`
  - Keep strategies: `first` (chronological) or `largest` (file size)
  - Recursive directory scanning with `--recursive`

- **Advanced Extraction Filters**: 8 new quality control parameters
  - `--species-threshold`: Minimum confidence for species classification (e.g., 0.85)
  - `--max-detections N`: Limit detections per frame (default: 10)
  - `--min-box-size N`: Filter out small/distant birds (default: 50px)
  - `--max-box-size N`: Filter out large false positives (default: 800px)
  - `--quality N`: JPEG quality 1-100 (default: 95)
  - `--skip-blurry`: Skip out-of-focus images using Laplacian variance
  - `--image-size N`: Consistent with train command (224/384/448 or 0 for original)
  - All filters fully internationalized (EN/DE/JA)

- **13 New Training Parameters**: Professional ML workflow control
  - `--early-stopping-patience N`: Stop when validation plateaus (default: 5)
  - `--weight-decay N`: L2 regularization strength (default: 0.01)
  - `--warmup-ratio N`: Learning rate warmup (default: 0.1)
  - `--label-smoothing N`: Label smoothing factor (default: 0.1)
  - `--save-total-limit N`: Maximum checkpoints to keep (default: 3)
  - `--augmentation-strength`: none/light/medium/heavy intensity levels
  - `--image-size N`: Support for 224/384/448px images
  - `--scheduler`: cosine/linear/constant LR schedules
  - `--seed N`: Reproducible training with fixed random seed
  - `--resume-from-checkpoint`: Continue interrupted training
  - `--gradient-accumulation-steps N`: Simulate larger batch sizes
  - `--mixed-precision`: fp16/bf16 support for faster GPU training
  - `--push-to-hub`: Automatic HuggingFace Hub upload

- **4-Level Data Augmentation System**: Configurable augmentation intensity
  - `none`: No augmentation (only normalization)
  - `light`: Minimal transforms (±10° rotation, minimal color jitter)
  - `medium`: Balanced transforms (±20° rotation, affine, color jitter, blur) - default
  - `heavy`: Aggressive transforms (±30° rotation, strong variations)

- **Extended i18n Coverage**: 28 new translation keys
  - 22 keys for deduplication (scanning, progress, results, statistics)
  - 6 keys for extraction filters (detections, box size, quality, blur, dedup)
  - All translations in English, German, Japanese
  - Total i18n coverage: 180+ keys across all modules

### Changed
- **Extract `--image-size` Parameter**: Replaced `--no-resize` boolean
  - Old: `--no-resize` (boolean flag)
  - New: `--image-size N` (integer, default: 224, use 0 for original)
  - Consistent with train command for easier workflows
  - Breaking change: Users using `--no-resize` must switch to `--image-size 0`

### Improved
- **Extraction Quality**: Multiple filters prevent low-quality training data
  - Box size filtering removes distant birds and false positives
  - Blur detection skips out-of-focus images
  - Duplicate detection prevents redundant similar images
  - Configurable JPEG quality balances size vs quality

- **Training Flexibility**: Professional-grade hyperparameter control
  - Fine-tune augmentation intensity for dataset size
  - Mixed precision training for 2x speed on modern GPUs
  - Reproducible experiments with seed parameter
  - Resume training from any checkpoint

- **Documentation**: Comprehensive README updates
  - All 8 new extraction parameters documented with examples
  - All 13 new training parameters explained
  - New deduplicate command section with usage examples
  - Feature list expanded to 14 items
  - Both English and German READMEs fully updated

### Dependencies
- Added `imagehash>=4.3.0` for perceptual hashing

## [0.1.5] - 2025-11-09

### Fixed
- **Training Error**: Fixed critical PyTorch tensor conversion bug that prevented training
  - Error: "expected Tensor as element 0 in argument 0, but got list"
  - Root cause: Transform function returned list of tensors instead of numpy arrays
  - Solution: Added `.numpy()` conversion and `set_format(type="torch")` for proper data pipeline
  - Training now works correctly with HuggingFace datasets

### Added
- **Complete i18n for Training Module**: Fully translated training output
  - All training messages now use i18n system (English, German, Japanese)
  - Includes: model loading, dataset info, progress messages, completion status
  - Graceful Ctrl+C handling messages translated
  - 30+ new translation keys for training workflow

- **Complete i18n for Testing Module**: Fully translated testing output  
  - Test result messages now in user's language
  - Single image classification and validation set testing
  - Usage instructions translated
  - 12+ new translation keys for testing workflow

### Improved
- **CLI i18n Coverage**: Fixed remaining hardcoded English strings
  - Extract command: All startup messages translated
  - Organize command: Fully translated output
  - Train command: Fully translated CLI interface
  - Consistent language experience across all commands

## [0.1.4] - 2025-11-09

### Added
- **Detailed Extraction Statistics**: New comprehensive bird counting system
  - `detected_birds_total`: Shows all birds detected by YOLO
  - `exported_birds_total`: Shows birds actually saved (after threshold filtering)
  - `skipped_birds_total`: Shows birds filtered out by `--species-threshold`
  - Translated labels for all three languages (English, German, Japanese)

### Fixed
- **Bird Count Bug**: Fixed incorrect counting when using `--species-threshold`
  - Previously counted all detected birds, even those skipped due to low confidence
  - Now correctly counts only exported birds
  - Statistics now accurately reflect what was actually saved vs. what was filtered

### Changed
- **Extraction Output**: Enhanced summary with three distinct counters for better transparency
  - Clear visibility of quality control impact
  - Users can now see how many birds were filtered vs. accepted
  - Example output:
    ```
    🔍 Detected birds total: 4
    🐦 Exported birds: 0
    ⏭️  Skipped (< 0.85): 4
    ```

## [0.1.3] - 2025-11-09

### Added
- **Internationalization (i18n)**: Multi-language support for CLI output
  - New `i18n.py` module with automatic language detection
  - Support for English, German, and Japanese
  - Automatic language selection based on LANG environment variable
  - Translated output for extractor and organizer modules
- **Species Confidence Filtering**: New `--species-threshold` parameter
  - Filter auto-classified birds by confidence level (e.g., 0.85 for 85%)
  - Only exports birds meeting minimum confidence threshold
  - Improves dataset quality by excluding uncertain predictions
- **Enhanced Documentation**: 
  - Visual workflow diagrams for iterative training in all READMEs
  - Phase 1 (manual labeling) and Phase 2 (auto-classification) clearly illustrated
  - Virtual environment recommendations added to installation sections
  - Practical examples showing accuracy improvements (92% → 96%)

### Changed
- **extractor.py**: All print statements now use i18n translations
- **organizer.py**: Core output messages translated
- **README files**: Added ASCII workflow diagrams showing iterative model improvement

### Technical Details
- `i18n.py`: 100+ translation keys covering all major operations
- Language detection via `locale` module and LANG environment variable
- Translation keys for extraction, organization, training, and testing workflows
- Backward compatible - English used as fallback if language not supported

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
