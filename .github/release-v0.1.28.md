# Release v0.1.28 - YOLO v26 Default Model Loading Fix

**Release Date:** February 22, 2026

## 🎯 Overview

This patch release fixes default model loading edge cases for `yolov26n.pt` so extraction works reliably across different runtime environments and working directories.

## ✅ What Was Fixed

### 🛠️ Robust Detection Model Resolution

`v0.1.27` introduced `yolov26n.pt` as the default detection model. In some environments this could fail with:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'yolov26n.pt'
```

`v0.1.28` now resolves model references more robustly:

- Tries explicit user path and current working directory
- Tries common project-relative locations
- Preserves model IDs/names for Ultralytics resolution when not a local file

### 🔁 Alias Fallback for YOLO v26 Naming

Some setups resolve the model family as `yolo26n.pt` rather than `yolov26n.pt`.

When default loading fails, the extractor now automatically retries:

- `yolov26n.pt` → `yolo26n.pt`

### 🎬 Applied to Both Extraction Modes

The fix is active in:

- `extract_birds_from_video()`
- `extract_birds_from_image()`

So video and static-image workflows behave consistently.

## 📦 Version Updates

- `pyproject.toml` → `0.1.28`
- `src/vogel_model_trainer/__version__.py` → `0.1.28`
- `CHANGELOG.md` updated with this fix

## 🚀 Installation

```bash
pip install --upgrade vogel-model-trainer==0.1.28
```

Or from source:

```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.28
pip install -e .
```

---

**Full Changelog:** See `CHANGELOG.md`
