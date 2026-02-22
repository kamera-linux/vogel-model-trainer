# Release v0.1.27 - YOLO Default Update & Dependency Baseline

**Release Date:** February 22, 2026

## 🎯 Overview

This release standardizes the repository on `yolov26n.pt` as the default detection model and aligns the Ultralytics dependency baseline for consistent runtime behavior.

## ✨ What's New

### 🐦 Default Detection Model Updated

The default YOLO detection model has been switched from `yolov8n.pt` to `yolov26n.pt` across the project.

**Updated areas:**
- Core extractor default model constant
- CLI default value and help text (`--detection-model`)
- Test expectations for extractor constants
- Documentation examples and parameter references (EN/DE/JA)
- Release-note snippets referencing default detection model

### 📦 Dependency Baseline Raised

`pyproject.toml` now requires:

```toml
ultralytics>=8.4.14
```

This ensures a consistent minimum feature baseline for model loading behavior and aligns with the documented dependency stack.

### 🧾 Version Bump

Project version is now:
- `pyproject.toml`: `0.1.27`
- `src/vogel_model_trainer/__version__.py`: `0.1.27`
- `CHANGELOG.md`: added `0.1.27` entry

## 🔄 File-Level Summary

### Core / CLI
- `src/vogel_model_trainer/core/extractor.py`
- `src/vogel_model_trainer/cli/main.py`

### Versioning / Packaging
- `pyproject.toml`
- `src/vogel_model_trainer/__version__.py`
- `CHANGELOG.md`

### Documentation
- `README.md`
- `README.de.md`
- `README.ja.md`
- `SECURITY.md`
- Existing release notes updated where default model was documented

### Tests
- `tests/test_extractor_constants.py`

## ✅ Compatibility

- No breaking CLI parameter changes
- Existing explicit `--detection-model` usage still works
- Users relying on implicit default now get `yolov26n.pt`

## 🚀 Installation

```bash
pip install --upgrade vogel-model-trainer==0.1.27
```

Or from source:

```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.27
pip install -e .
```

---

**Full Changelog:** See `CHANGELOG.md`
