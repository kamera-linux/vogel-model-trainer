# Release v0.1.23 - Warning Suppression Fix

**Release Date:** December 16, 2025

## 🐛 Bug Fixes

### Silent Cholesky Warnings

Fixed annoying performance warnings that appeared during bird extraction with species classification:

**Before:**
```
⏳ Fortschritt: 62.1% (1600/2576 Frames)
✅ Vogel #1: kohlmeise (Konf 0.91), Frame 1665
PERFORMANCE WARNING:
Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A with parameters:
    discard_threshold = 1.000000e-04
    shift = 0.000000e+00
Try decreasing discard_threshold or start with a larger shift
✅ Vogel #2: kohlmeise (Konf 0.96), Frame 1707
```

**After:**
```
⏳ Fortschritt: 62.1% (1600/2576 Frames)
✅ Vogel #1: kohlmeise (Konf 0.91), Frame 1665
✅ Vogel #2: kohlmeise (Konf 0.96), Frame 1707
✅ Vogel #3: kohlmeise (Konf 0.93), Frame 1710
```

### Technical Details

- **Issue**: PyTorch optimizer emitted harmless Cholesky decomposition warnings during transformers model inference
- **Impact**: No functional issues, but cluttered output during video processing
- **Fix**: Added warning filters to `extractor.py` (previously only in `evaluator.py`)
- **Affected Commands**: `extract` with `--species-model` parameter

## 📝 Changes

### Modified Files
- `src/vogel_model_trainer/core/extractor.py`: Added warning suppression for Cholesky and positive-definiteness messages

## 🔧 Usage

No changes to command-line interface. Update as usual:

```bash
pip install --upgrade vogel-model-trainer
```

Or with pip from GitHub:

```bash
pip install git+https://github.com/kamera-linux/vogel-model-trainer.git@v0.1.23
```

## 📊 Testing

Verified with:
- 56 video files from AI-HAD camera system
- `--species-model kamera-linux/german-bird-classifier-v2`
- Background removal enabled (`--remove-background --bg-transparent`)
- High thresholds (`--threshold 0.6 --species-threshold 0.50`)
- Result: Clean output without performance warnings

## 🙏 Credits

Thanks to the PyTorch and transformers teams for the excellent underlying libraries. The warnings were cosmetic only and indicated normal fallback behavior in the optimizer.

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.22...v0.1.23
