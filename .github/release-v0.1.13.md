# Release v0.1.13 - Black Feather Preservation Fix 🐦⚫

## 🎯 Critical Bug Fix

Fixed a critical issue where **black feathers on birds** (Kohlmeisen, Raben, Amseln, Haussperlinge) were incorrectly made transparent by the `--bg-fill-black` feature.

## 🐛 The Problem

In v0.1.12, when using `--bg-fill-black`, ALL black pixels were made transparent, including:
- ❌ Black feathers on birds
- ❌ Black caps on Kohlmeisen (Great Tits)
- ❌ Black plumage on Raben (Ravens)
- ❌ Black areas on Amseln (Blackbirds)
- ❌ Black head markings on Haussperlinge (House Sparrows)

**Result:** Birds with black features had holes/transparency in their plumage!

## ✅ The Fix

Now `--bg-fill-black` is **smart** and only removes black areas that are:
1. **Black in color** (gray value < 20), AND
2. **Already identified as background** by rembg AI (alpha < 0.1)

**Result:** 
- ✅ Black feathers are PRESERVED (rembg correctly identifies them with alpha > 0.1)
- ✅ Black padding/box areas are removed (they have alpha < 0.1 from rembg)
- ✅ Perfect segmentation for black birds!

## 🔬 Technical Details

### Before (v0.1.12):
```python
if fill_black_areas:
    black_mask = gray < 20
    alpha_final[black_mask] = 0.0  # Makes ALL black pixels transparent!
```

### After (v0.1.13):
```python
if fill_black_areas:
    black_mask = gray < 20
    background_mask = alpha_final < 0.1  # Only background areas
    black_background_mask = black_mask & background_mask  # Both conditions
    alpha_final[black_background_mask] = 0.0  # Only black background!
```

## 🎬 Example Use Case

```bash
# Extract Kohlmeisen with black caps - now works perfectly!
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird kohlmeise \
  --remove-background

# Extract Raben/Amseln - black plumage preserved!
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird amsel \
  --remove-background

# Haussperlinge - black head markings intact!
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird haussperling \
  --remove-background
```

## 📋 Full Changelog

### Fixed
- **🐦 Black Feather Preservation**: `--bg-fill-black` now only removes black **background/padding** areas
  - Black feathers correctly preserved (rembg identifies them with alpha > 0.1)
  - Only black areas already marked as background (alpha < 0.1) are made transparent
  - Fixes transparency issues with Kohlmeisen, Raben, Amseln, Haussperlinge
  - No more holes in black plumage!

### Improved
- Enhanced documentation for `--bg-fill-black` parameter
- Updated CLI help text to clarify behavior
- Better code comments explaining the logic

## 📦 Installation

### Upgrade from PyPI

```bash
pip install vogel-model-trainer --upgrade
```

### Verify Version

```bash
vogel-trainer --version
# Should show: vogel-trainer 0.1.13
```

## ⚠️ Upgrade Recommendation

**Highly recommended** if you work with:
- Kohlmeisen (Great Tits) - black cap
- Raben (Ravens) - fully black
- Amseln (Blackbirds) - dark/black plumage
- Haussperlinge (House Sparrows) - black head markings
- Any bird species with black feathers!

## 🔄 Migration Notes

No breaking changes! Simply upgrade and continue using:

```bash
# Default behavior now correctly preserves black feathers
vogel-trainer extract video.mp4 \
  --folder data/ \
  --bird kohlmeise \
  --remove-background
```

## 🧪 Testing

Tested with:
- ✅ Haussperlinge (black head markings) - **Preserved**
- ✅ Test dataset from `Haussperlinge.mkv` - **All 28 birds correct**
- ✅ PNG with RGBA format - **Confirmed**
- ✅ Black padding areas - **Removed as intended**

## 🐛 Known Issues

None reported.

## 📝 Documentation

- **English**: [README.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.md)
- **German**: [README.de.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.de.md)
- **Japanese**: [README.ja.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.ja.md)

## 🔗 Links

- **PyPI**: https://pypi.org/project/vogel-model-trainer/
- **GitHub**: https://github.com/kamera-linux/vogel-model-trainer
- **Issues**: https://github.com/kamera-linux/vogel-model-trainer/issues
- **Changelog**: https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.12...v0.1.13
