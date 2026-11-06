# Release v0.1.29 - Training, CLI & Data Safety Fixes

**Release Date:** November 6, 2026

## 🎯 Overview

This release fixes several commands that were broken or silently ignored options, removes two ways of accidentally losing data, and adds a real test suite with CI.

**Please read the "Behavior Changes" section before upgrading.**

## 🐛 Critical Fixes

- **`train` works with transformers 5.x again** - the image processor is passed as `processing_class` (older transformers versions still work)
- **`train --model` is respected** - previously `google/efficientnet-b0` was always loaded
- **Data augmentation now varies per epoch** - it was applied once and cached
- **Ctrl+C during training** stops cleanly after the current step and saves the model (`"interrupted": true` in `training_config.json`)
- **`clean-gray` checks images again** - the check loop had become dead code
- **`vogel-trainer extract`**: `--log` no longer aborts; `--image-size`, `--max-detections`, `--min-box-size`, `--max-box-size`, `--skip-blurry` and `--convert` now work
- **`--bg-model` takes effect** and the rembg model is loaded once instead of for every crop (large speed-up with `--remove-background`)
- **`--species-threshold` without `--species-model`** no longer discards every bird

## 🛡️ Data Safety

- **`organize --max-images-per-class` no longer deletes source images** - it only limits what is copied. The class balance check runs before anything is written
- **`classify --delete-source`** only deletes images that were successfully sorted (requires `--sort-output`); unclassified files and a sort output inside the input folder are kept
- `deduplicate`, `quality-check` and `clean-*` no longer re-process their own output folders (`duplicates/`, `low_quality/`, `invalid_*`)

## 🔧 Improvements

- **Train/val split by video session** (`organize --split-by session`, default) prevents near-identical frames from appearing in both train and val. New `--seed` and `--clean` options
- Video, image and convert mode share one processing pipeline: consistent `--crop-padding`, aspect-ratio-preserving resize, deduplication on the final image
- Models are loaded once per extraction run; skipped video frames are not decoded; the species classifier uses the GPU when available
- `classify` and `evaluate` process images in batches; `evaluate` reports low-confidence correct predictions separately
- Transparent images use the same gray background for validation, `test`, `classify` and `evaluate`
- `--log` falls back to `~/.local/state/vogel-model-trainer/logs/` when `/var/log/vogel-kamera-linux/` is not writable (`VOGEL_LOG_DIR` overrides)
- Global `--debug` flag for full tracebacks; non-zero exit codes on errors

## ⚠️ Behavior Changes

- **Python ≥ 3.10** is required (Python 3.9 is end-of-life)
- `organize` splits by session by default - validation accuracy may be lower but more realistic. Use `--split-by image` for the old behavior
- `--crop-padding` in image mode is now a percentage (as in video mode), not pixels
- `test --data` evaluates all validation images (use `--samples N` for a random subset)
- Image and convert mode keep the aspect ratio (padding instead of stretching)

## 📦 Dependencies

- `transformers>=4.46.0` (was `>=4.30.0`)
- `tqdm>=4.65.0` added (was used but not declared)

## 🧪 Testing

- Test suite grew from 13 to 59 tests (pipeline, organizer, classify/evaluate, dataset tools, trainer, translations)
- New `Tests` workflow runs pytest and ruff on every push and pull request; the PyPI publish workflow reuses it

## 🚀 Installation

```bash
pip install --upgrade vogel-model-trainer==0.1.29
```

Or from source:

```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.29
pip install -e .
```

---

**Full Changelog:** See `CHANGELOG.md`
