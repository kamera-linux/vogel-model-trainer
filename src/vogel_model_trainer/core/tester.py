#!/usr/bin/env python3
"""
Script to test the trained custom bird classifier on sample images.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

def validate_gray_background(image_path: str, min_gray_ratio: float = 0.05, 
                             max_gray_ratio: float = 0.95, gray_tolerance: int = 30):
    """
    Validate images with gray background to detect problems.
    
    Args:
        image_path: Path to image (JPEG or PNG)
        min_gray_ratio: Minimum required gray pixels (default: 0.05 = 5%)
        max_gray_ratio: Maximum allowed gray pixels (default: 0.95 = 95%)
        gray_tolerance: Tolerance for gray detection (default: 30)
                       Pixels with R≈G≈B±tolerance are considered gray
    
    Returns:
        dict: Validation results with:
            - valid: bool (True if image is valid)
            - reason: str (reason for rejection if invalid)
            - gray_pixels: int (number of gray background pixels)
            - gray_ratio: float (ratio of gray pixels)
            - bird_pixels: int (number of non-gray pixels = bird)
            - bird_ratio: float (ratio of bird pixels)
    """
    try:
        # Load image
        with Image.open(image_path) as src_img:
            img = src_img.copy()
        
        # Convert to RGB if needed (handle RGBA)
        if img.mode == 'RGBA':
            # Gray background for transparent areas
            background = Image.new('RGB', img.size, (128, 128, 128))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Detect gray pixels: R≈G≈B (within tolerance)
        # Gray is where max(R,G,B) - min(R,G,B) < tolerance
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        max_channel = np.maximum(np.maximum(r, g), b)
        min_channel = np.minimum(np.minimum(r, g), b)
        color_diff = max_channel - min_channel
        
        # Gray pixels are those with low color difference
        gray_mask = color_diff < gray_tolerance
        
        gray_pixels = np.sum(gray_mask)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        gray_ratio = gray_pixels / total_pixels
        
        bird_pixels = total_pixels - gray_pixels
        bird_ratio = bird_pixels / total_pixels
        
        # Check 1: Too few gray pixels (mostly bird = no background)
        if gray_ratio < min_gray_ratio:
            return {
                'valid': False,
                'reason': f'Too little gray background ({gray_ratio:.1%} < {min_gray_ratio:.1%})',
                'gray_pixels': gray_pixels,
                'gray_ratio': gray_ratio,
                'bird_pixels': bird_pixels,
                'bird_ratio': bird_ratio
            }
        
        # Check 2: Too many gray pixels (mostly background = no bird)
        if gray_ratio > max_gray_ratio:
            return {
                'valid': False,
                'reason': f'Too much gray background ({gray_ratio:.1%} > {max_gray_ratio:.1%})',
                'gray_pixels': gray_pixels,
                'gray_ratio': gray_ratio,
                'bird_pixels': bird_pixels,
                'bird_ratio': bird_ratio
            }
        
        # All checks passed
        return {
            'valid': True,
            'reason': 'Valid image',
            'gray_pixels': gray_pixels,
            'gray_ratio': gray_ratio,
            'bird_pixels': bird_pixels,
            'bird_ratio': bird_ratio
        }
        
    except Exception as e:
        return {
            'valid': False,
            'reason': f'Error: {str(e)}',
            'gray_pixels': 0,
            'gray_ratio': 0.0,
            'bird_pixels': 0,
            'bird_ratio': 0.0
        }

def validate_transparent_image(image_path: str, min_visible_pixels: int = 500, 
                               max_transparency: float = 0.95, min_region_size: int = 100):
    """
    Validate transparent PNG images to detect incomplete/fragmented birds.
    
    Args:
        image_path: Path to PNG image with alpha channel
        min_visible_pixels: Minimum number of non-transparent pixels (default: 500)
        max_transparency: Maximum allowed transparency ratio (default: 0.95 = 95%)
        min_region_size: Minimum size of largest connected region (default: 100)
    
    Returns:
        dict: Validation results with:
            - valid: bool (True if image is valid)
            - reason: str (reason for rejection if invalid)
            - visible_pixels: int (number of visible pixels)
            - transparency_ratio: float (ratio of transparent pixels)
            - largest_region: int (size of largest connected region)
    """
    try:
        # Load image with PIL
        with Image.open(image_path) as src_img:
            img = src_img.copy()
        
        # Check if image has alpha channel
        if img.mode != 'RGBA':
            return {
                'valid': True,
                'reason': 'No alpha channel (not transparent)',
                'visible_pixels': img.width * img.height,
                'transparency_ratio': 0.0,
                'largest_region': img.width * img.height
            }
        
        # Convert to numpy array
        img_array = np.array(img)
        alpha = img_array[:, :, 3]
        
        # Count visible pixels (alpha > 10)
        visible_mask = alpha > 10
        visible_pixels = np.sum(visible_mask)
        total_pixels = alpha.size
        transparency_ratio = 1.0 - (visible_pixels / total_pixels)
        
        # Check 1: Too few visible pixels
        if visible_pixels < min_visible_pixels:
            return {
                'valid': False,
                'reason': f'Too few visible pixels ({visible_pixels} < {min_visible_pixels})',
                'visible_pixels': visible_pixels,
                'transparency_ratio': transparency_ratio,
                'largest_region': 0
            }
        
        # Check 2: Too much transparency
        if transparency_ratio > max_transparency:
            return {
                'valid': False,
                'reason': f'Too transparent ({transparency_ratio:.1%} > {max_transparency:.1%})',
                'visible_pixels': visible_pixels,
                'transparency_ratio': transparency_ratio,
                'largest_region': 0
            }
        
        # Check 3: Fragmentation - find largest connected region
        # Convert visible mask to uint8
        visible_uint8 = (visible_mask * 255).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(visible_uint8, connectivity=8)
        
        # Get size of largest region (excluding background label 0)
        if num_labels > 1:
            largest_region = np.max(stats[1:, cv2.CC_STAT_AREA])
        else:
            largest_region = 0
        
        # Check if largest region is too small
        if largest_region < min_region_size:
            return {
                'valid': False,
                'reason': f'Fragmented object (largest region {largest_region} < {min_region_size})',
                'visible_pixels': visible_pixels,
                'transparency_ratio': transparency_ratio,
                'largest_region': largest_region
            }
        
        # All checks passed
        return {
            'valid': True,
            'reason': 'Valid image',
            'visible_pixels': visible_pixels,
            'transparency_ratio': transparency_ratio,
            'largest_region': largest_region
        }
        
    except Exception as e:
        return {
            'valid': False,
            'reason': f'Error loading image: {e}',
            'visible_pixels': 0,
            'transparency_ratio': 1.0,
            'largest_region': 0
        }

def test_model(model_path: str, data_dir: str = None, image_path: str = None,
               samples: int = 0, seed: int = 42):
    """
    Test the model on the validation dataset or a single image.

    Args:
        model_path: Path to trained model directory
        data_dir: Path to organized dataset directory (with val/ folder)
        image_path: Path to single image file (alternative to data_dir)
        samples: Images per species to test (0 = all images)
        seed: Random seed used when sampling

    Returns:
        dict: Test results with accuracy metrics
    """
    import random
    import torch
    from vogel_model_trainer.i18n import _
    from vogel_model_trainer.utils.image import find_images, load_rgb

    model_path = str(Path(model_path).expanduser())

    print(_('test_loading_model', path=model_path))
    classifier = pipeline(
        "image-classification",
        model=model_path,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Single image test
    if image_path:
        image_path = str(Path(image_path).expanduser())
        print(_('test_classifying_image', path=image_path))
        results = classifier(load_rgb(image_path), top_k=5)

        print(_('test_results'))
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['label']:15s} - {result['score']:.4f} ({result['score']*100:.1f}%)")

        return {"image": image_path, "predictions": results}

    if not data_dir:
        raise ValueError("Either data_dir or image_path must be given")

    data_dir = Path(data_dir).expanduser()
    val_dir = data_dir / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    print(_('test_on_validation', path=val_dir))
    print("=" * 70)

    rng = random.Random(seed)
    species = sorted(d.name for d in val_dir.iterdir() if d.is_dir())
    correct = 0
    total = 0
    results_by_species = {}

    for sp in species:
        images = find_images(val_dir / sp, recursive=False)
        if not images:
            print(_('test_no_images_found', species=sp))
            continue

        if samples and samples < len(images):
            images = rng.sample(images, samples)  # nosec B311 - local testing, not cryptographic

        sp_correct = 0
        for img_path in images:
            prediction = classifier(load_rgb(img_path), top_k=1)[0]
            predicted, confidence = prediction['label'], prediction['score']
            if predicted == sp:
                sp_correct += 1
            else:  # only show errors
                print(f"❌ {sp:12s} -> {predicted:12s} ({confidence:.4f})  {img_path.name}")
            total += 1

        correct += sp_correct
        sp_accuracy = sp_correct / len(images)
        results_by_species[sp] = sp_accuracy
        print(f"   {sp:12s}: {sp_correct}/{len(images)} = {sp_accuracy*100:.1f}%")

    overall_accuracy = correct / total if total > 0 else 0

    print("=" * 70)
    print(_('test_overall_accuracy', correct=correct, total=total, accuracy=overall_accuracy*100))

    return {
        "overall_accuracy": overall_accuracy,
        "by_species": results_by_species,
        "correct": correct,
        "total": total
    }


if __name__ == "__main__":
    from vogel_model_trainer.i18n import _
    
    if len(sys.argv) < 2:
        print(_('test_usage'))
        print(_('test_usage_single'))
        print(_('test_usage_single_cmd'))
        print()
        print(_('test_usage_validation'))
        print(_('test_usage_validation_cmd'))
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if len(sys.argv) == 3 and not sys.argv[2].startswith("--"):
        image_path = sys.argv[2]
        test_model(model_path, image_path=image_path)
    elif "--data-dir" in sys.argv:
        data_dir_idx = sys.argv.index("--data-dir") + 1
        data_dir = sys.argv[data_dir_idx] if data_dir_idx < len(sys.argv) else None
        test_model(model_path, data_dir=data_dir)
    else:
        print(_('test_error_no_input'))
        sys.exit(1)
