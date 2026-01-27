# Release v0.1.25 - Code Quality Improvements

**Release Date:** January 27, 2026

## 🔧 What's New

### Code Maintainability Enhancements

Refactored `extractor.py` to improve code quality and maintainability by replacing magic numbers with named constants and adding type hints for better IDE support.

## ✅ Changes

### Named Constants

Extracted 8 hardcoded values into well-documented configuration constants:

```python
# Image processing constants
MIN_IMAGE_SIZE = 50  # Minimum image dimension for background removal
ALPHA_TRANSPARENCY_THRESHOLD = 10  # Alpha value below which pixels are considered transparent
BLACK_PIXEL_THRESHOLD = 20  # Grayscale value below which pixels are considered black
PROGRESS_LOG_INTERVAL = 100  # Log progress every N frames/images

# Background removal constants (rembg/alpha matting)
ALPHA_MATTING_FOREGROUND_THRESHOLD = 240  # Threshold for foreground in alpha matting
ALPHA_MATTING_BACKGROUND_THRESHOLD = 10  # Threshold for background in alpha matting
ALPHA_MATTING_ERODE_SIZE = 10  # Erosion kernel size for alpha matting
GRAY_BACKGROUND_DEFAULT = (128, 128, 128)  # Default gray background color (BGR)
```

### Type Hints

Added type annotations to quality filter functions:

```python
def calculate_motion_quality(image: np.ndarray) -> Dict[str, float]:
    """Calculate motion/blur quality metrics for an image."""
    ...

def is_motion_acceptable(
    quality_metrics: Dict[str, float], 
    min_sharpness: Optional[float] = None, 
    min_edge_quality: Optional[float] = None
) -> Tuple[bool, str]:
    """Check if motion quality metrics meet minimum thresholds."""
    ...
```

### Typing Imports

Added proper typing support:

```python
from typing import Dict, Optional, Tuple
```

## 🎯 Benefits

- **Improved Maintainability**: Constants can be changed in one central location
- **Better Readability**: Descriptive names instead of magic numbers
- **IDE Support**: Type hints enable better autocomplete and error checking
- **Documentation**: Inline comments explain the purpose of each constant
- **Future-Proof**: Easier to adjust parameters for different use cases

## 📊 Impact

### No Behavioral Changes
- ✅ All functionality remains identical
- ✅ API is fully backward compatible
- ✅ No translation updates required
- ✅ Tests confirm correct behavior

### Code Statistics
- **1 file changed**: `src/vogel_model_trainer/core/extractor.py`
- **+27 lines / -12 lines**
- **12 locations** where magic numbers were replaced
- **2 functions** received type hints
- **8 new constants** added

## 🔬 Testing

New unit tests verify the refactoring:

```bash
# Test constants exist and have correct values
pytest tests/test_extractor_constants.py -v

# All tests pass:
# ✅ All new constants exist with correct values
# ✅ All existing constants still present
# ✅ Type hints added successfully
```

## 🛠️ Usage

No changes to end-user functionality. Update as usual:

```bash
pip install --upgrade vogel-model-trainer
```

All existing scripts and workflows continue to work without modification.

## 📝 Technical Details

### Files Modified
- `src/vogel_model_trainer/core/extractor.py`: Refactored magic numbers and added type hints
- `src/vogel_model_trainer/__version__.py`: Version bump to 0.1.25
- `pyproject.toml`: Version bump to 0.1.25
- `CHANGELOG.md`: Added v0.1.25 entry

### Files Added
- `tests/test_extractor_constants.py`: Unit tests for new constants and type hints

### Constants Replaced
| Location | Old Value | New Constant |
|----------|-----------|--------------|
| `remove_background()` size check | `50` | `MIN_IMAGE_SIZE` |
| Alpha matting foreground | `240` | `ALPHA_MATTING_FOREGROUND_THRESHOLD` |
| Alpha matting background | `10` | `ALPHA_MATTING_BACKGROUND_THRESHOLD` |
| Alpha matting erosion | `10` | `ALPHA_MATTING_ERODE_SIZE` |
| Alpha transparency check | `10` | `ALPHA_TRANSPARENCY_THRESHOLD` |
| Black pixel threshold | `20` | `BLACK_PIXEL_THRESHOLD` |
| Progress log interval | `100` | `PROGRESS_LOG_INTERVAL` |

## 🚀 Future Improvements

This refactoring lays groundwork for:
- Configuration files for adjusting processing parameters
- Per-project settings overrides
- A/B testing different parameter values
- Better documentation of tuning options

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.24...v0.1.25
