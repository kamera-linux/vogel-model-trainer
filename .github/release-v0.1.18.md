# Release v0.1.18 - Bug Fix Release

**Release Date:** November 19, 2025

## 🐛 Bug Fixes

### Critical Fixes
- **Fixed `resize_to_target` parameter error** in `extractor.py`
  - Removed invalid `resize_to_target` parameter from `extract_params` dictionary
  - Changed to use `target_image_size` parameter with conditional value (224 or 0)
  - This fix resolves the error: `extract_birds_from_video() got an unexpected keyword argument 'resize_to_target'`
  
- **Fixed string formatting errors in i18n.py**
  - Corrected malformed string formatting `{"="*70}` and `{'='*70}` in translation strings
  - Removed separator lines from translation strings (now handled separately in code)
  - Applied fixes to all three languages: English (en), German (de), Japanese (ja)
  - Affected strings: `processing_video`, `all_videos_processed`

## 📝 Technical Details

### Files Modified
- `src/vogel_model_trainer/core/extractor.py` - Line 1412: Changed `resize_to_target` to `target_image_size`
- `src/vogel_model_trainer/i18n.py` - Lines 65-68 (en), 340-343 (de), 615-618 (ja): Removed malformed separator formatting

### Impact
These fixes resolve critical issues that prevented the extraction command from working properly:
1. Users can now successfully run `vogel-model-trainer extract` without parameter errors
2. Console output now displays correct separator lines instead of literal format strings

## 🔍 Testing
- Tested extraction with online YOLO model (yolov8n.pt)
- Verified correct output formatting in all three languages
- Confirmed successful video processing and bird detection

## 📦 Installation

```bash
pip install --upgrade vogel-model-trainer==0.1.18
```

Or from source:
```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.18
pip install -e .
```

## 🙏 Acknowledgments
Thank you to all users who reported these issues!

---

**Full Changelog:** [v0.1.17...v0.1.18](https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.17...v0.1.18)
