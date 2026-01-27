"""
Unit tests for extractor.py constants and optimizations.
Tests that all magic numbers have been replaced with named constants.
"""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_constants_exist():
    """Test that all new constants are defined."""
    from vogel_model_trainer.core import extractor
    
    # Test image processing constants
    assert hasattr(extractor, 'MIN_IMAGE_SIZE')
    assert extractor.MIN_IMAGE_SIZE == 50
    
    assert hasattr(extractor, 'ALPHA_TRANSPARENCY_THRESHOLD')
    assert extractor.ALPHA_TRANSPARENCY_THRESHOLD == 10
    
    assert hasattr(extractor, 'BLACK_PIXEL_THRESHOLD')
    assert extractor.BLACK_PIXEL_THRESHOLD == 20
    
    assert hasattr(extractor, 'PROGRESS_LOG_INTERVAL')
    assert extractor.PROGRESS_LOG_INTERVAL == 100
    
    # Test background removal constants
    assert hasattr(extractor, 'ALPHA_MATTING_FOREGROUND_THRESHOLD')
    assert extractor.ALPHA_MATTING_FOREGROUND_THRESHOLD == 240
    
    assert hasattr(extractor, 'ALPHA_MATTING_BACKGROUND_THRESHOLD')
    assert extractor.ALPHA_MATTING_BACKGROUND_THRESHOLD == 10
    
    assert hasattr(extractor, 'ALPHA_MATTING_ERODE_SIZE')
    assert extractor.ALPHA_MATTING_ERODE_SIZE == 10
    
    assert hasattr(extractor, 'GRAY_BACKGROUND_DEFAULT')
    assert extractor.GRAY_BACKGROUND_DEFAULT == (128, 128, 128)


def test_existing_constants():
    """Test that existing constants are still present."""
    from vogel_model_trainer.core import extractor
    
    assert hasattr(extractor, 'DEFAULT_THRESHOLD')
    assert extractor.DEFAULT_THRESHOLD == 0.5
    
    assert hasattr(extractor, 'DEFAULT_SAMPLE_RATE')
    assert extractor.DEFAULT_SAMPLE_RATE == 3
    
    assert hasattr(extractor, 'DEFAULT_MODEL')
    assert extractor.DEFAULT_MODEL == "yolov8n.pt"
    
    assert hasattr(extractor, 'TARGET_IMAGE_SIZE')
    assert extractor.TARGET_IMAGE_SIZE == 224


def test_type_hints_exist():
    """Test that type hints were added to quality functions."""
    from vogel_model_trainer.core import extractor
    import inspect
    
    # Check calculate_motion_quality has type hints
    sig = inspect.signature(extractor.calculate_motion_quality)
    assert 'image' in sig.parameters
    # Return annotation should be Dict[str, float]
    assert sig.return_annotation != inspect.Signature.empty
    
    # Check is_motion_acceptable has type hints
    sig = inspect.signature(extractor.is_motion_acceptable)
    assert 'quality_metrics' in sig.parameters
    assert 'min_sharpness' in sig.parameters
    assert 'min_edge_quality' in sig.parameters
    # Return annotation should be Tuple[bool, str]
    assert sig.return_annotation != inspect.Signature.empty


if __name__ == "__main__":
    print("Testing extractor constants...")
    
    try:
        test_constants_exist()
        print("✅ All new constants exist with correct values")
    except AssertionError as e:
        print(f"❌ Constants test failed: {e}")
        sys.exit(1)
    
    try:
        test_existing_constants()
        print("✅ All existing constants still present")
    except AssertionError as e:
        print(f"❌ Existing constants test failed: {e}")
        sys.exit(1)
    
    try:
        test_type_hints_exist()
        print("✅ Type hints added successfully")
    except AssertionError as e:
        print(f"❌ Type hints test failed: {e}")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
