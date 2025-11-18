#!/usr/bin/env python3
"""
Script to extract bird crops from videos for training data collection.
Extracts detected birds and saves them as individual images.
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import sys
import uuid
from datetime import datetime
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import glob
import imagehash
import numpy as np

# Import i18n for translations
from vogel_model_trainer.i18n import _

# Default configuration (can be overridden via command line)
DEFAULT_THRESHOLD = 0.5  # Higher threshold for better quality birds
DEFAULT_SAMPLE_RATE = 3  # Check more frames for better coverage
DEFAULT_MODEL = "yolov8n.pt"
TARGET_IMAGE_SIZE = 224  # Optimal size for EfficientNet-B0 training

def calculate_motion_quality(image):
    """
    Calculate motion/blur quality metrics for an image.
    
    Args:
        image: BGR image (numpy array)
        
    Returns:
        dict: Quality metrics including:
            - sharpness: Laplacian variance (higher = sharper)
            - edge_quality: Sobel gradient magnitude (higher = clearer edges)
            - overall: Combined quality score
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Laplacian Variance (Sharpness measure)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    
    # 2. Sobel Gradient Magnitude (Edge quality)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2).mean()
    
    # 3. Combined overall score (weighted average)
    overall_score = laplacian_var * 0.7 + gradient_mag * 0.3
    
    return {
        'sharpness': laplacian_var,
        'edge_quality': gradient_mag,
        'overall': overall_score
    }

def is_motion_acceptable(quality_metrics, min_sharpness=None, min_edge_quality=None):
    """
    Check if motion quality metrics meet minimum thresholds.
    
    Args:
        quality_metrics: Dict from calculate_motion_quality()
        min_sharpness: Minimum sharpness score (default: None = no filter)
        min_edge_quality: Minimum edge quality score (default: None = no filter)
        
    Returns:
        tuple: (is_acceptable: bool, reason: str)
    """
    if min_sharpness is not None and quality_metrics['sharpness'] < min_sharpness:
        return False, 'low_sharpness'
    
    if min_edge_quality is not None and quality_metrics['edge_quality'] < min_edge_quality:
        return False, 'poor_edges'
    
    return True, 'accepted'

def remove_background(image, margin=10, iterations=10, bg_color=(128, 128, 128), model_name='u2net', 
                     transparent=False, fill_black_areas=False, expand_mask=0):
    """
    Remove background from bird image using rembg (AI-based segmentation).
    This provides professional-quality background removal using deep learning.
    
    Args:
        image: BGR image (numpy array)
        margin: Not used, kept for API compatibility
        iterations: Not used, kept for API compatibility
        bg_color: Background color as BGR tuple (default: (255, 255, 255) = white)
                 Special values: (0, 255, 0) = green-screen, (255, 0, 0) = blue-screen
                 Ignored if transparent=True
        model_name: rembg model to use (default: 'u2net')
                   Options: 'u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use'
        transparent: If True, return PNG with transparent background (alpha channel) - DEFAULT
        fill_black_areas: If True, make black BACKGROUND/PADDING areas transparent - DEFAULT
                         Only affects areas already identified as background by rembg (alpha < 0.1)
                         BLACK FEATHERS/BIRDS are preserved! (they have alpha > 0.1 from rembg)
        expand_mask: Pixels to expand the foreground mask (keeps more background around bird details like feet/beak)
                     Positive values = keep more background, 0 = default rembg behavior
        
    Returns:
        numpy array: Image with replaced background (BGRA if transparent=True, BGR otherwise)
    """
    if image is None or image.size == 0:
        return image
    
    # Check if rembg is available (dynamically check at runtime)
    try:
        from rembg import remove as rembg_remove
    except ImportError:
        print("⚠️ Warning: rembg not installed. Background removal disabled. Install with: pip install rembg")
        return image
    
    height, width = image.shape[:2]
    
    # Minimum size check
    if height < 50 or width < 50:
        return image
    
    try:
        # Convert BGR to RGB for rembg
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Remove background using rembg with specified model
        # alpha_matting improves edge quality
        output = rembg_remove(
            pil_image,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_process_mask=True
        )
        
        # Convert back to numpy array
        output_np = np.array(output)
        
        # Split into RGB and Alpha channels
        rgb = output_np[:, :, :3]
        alpha = output_np[:, :, 3]
        
        # Post-processing: Smooth alpha channel for better edges
        alpha_float = alpha.astype(np.float32) / 255.0
        
        # Expand mask if requested to keep more background around bird details
        if expand_mask > 0:
            # Dilate the mask to include more pixels around the bird
            kernel_size = expand_mask * 2 + 1  # Convert to odd kernel size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            alpha_binary = (alpha > 10).astype(np.uint8) * 255  # Threshold to binary
            alpha_expanded = cv2.dilate(alpha_binary, kernel, iterations=1)
            alpha_float = alpha_expanded.astype(np.float32) / 255.0
        
        # Apply slight Gaussian blur to alpha for smoother edges
        alpha_smooth = cv2.GaussianBlur(alpha_float, (3, 3), 0)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        alpha_cleaned = cv2.morphologyEx((alpha_smooth * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Back to float
        alpha_final = alpha_cleaned.astype(np.float32) / 255.0
        
        # If transparent background requested, return BGRA image
        if transparent:
            # If fill_black_areas is enabled, detect and make black PADDING areas transparent
            # Only removes black areas that are ALREADY in the background (alpha < 0.1)
            # This preserves black feathers/birds that rembg correctly identified as foreground
            # NOTE: Only applies when transparent=True! With colored backgrounds, this is skipped.
            if fill_black_areas:
                # Convert RGB to grayscale
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                # Detect very dark pixels (black box/padding areas)
                black_mask = gray < 20  # Threshold for "black" pixels
                # Only set alpha to 0 for black areas that are ALREADY mostly transparent (background)
                # This way, black feathers with alpha > 0.1 are preserved
                background_mask = alpha_final < 0.1
                black_background_mask = black_mask & background_mask
                alpha_final[black_background_mask] = 0.0
            
            # Create BGRA image with alpha channel
            alpha_channel = (alpha_final * 255).astype(np.uint8)
            result_rgba = np.dstack((rgb, alpha_channel))
            # Convert RGB to BGR for OpenCV
            result_bgra = cv2.cvtColor(result_rgba, cv2.COLOR_RGBA2BGRA)
            return result_bgra
        
        # Create background with specified color (RGB)
        # NOTE: fill_black_areas is NOT applied here - we want the full colored background
        bg_rgb = np.array([bg_color[2], bg_color[1], bg_color[0]], dtype=np.uint8)  # BGR to RGB
        background = np.full((height, width, 3), bg_rgb, dtype=np.uint8)
        
        # Blend foreground and background using alpha
        alpha_3channel = alpha_final[:, :, np.newaxis]
        result_rgb = (rgb * alpha_3channel + background * (1 - alpha_3channel)).astype(np.uint8)
        
        # Convert back to BGR for OpenCV
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        return result_bgr
        
    except Exception as e:
        # If rembg fails, return original image
        print(f"Warning: Background removal failed: {e}")
        print("Make sure rembg is installed: pip install rembg")
        return image

def auto_crop_to_content(image, margin_percent=10):
    """
    Automatically crop image to non-transparent content with a margin.
    
    Args:
        image: Image with alpha channel (BGRA) or BGR image
        margin_percent: Percentage margin to add around content (default: 10%)
    
    Returns:
        Cropped image with margin
    """
    if image is None or image.size == 0:
        return image
    
    # Check if image has alpha channel
    if image.shape[2] == 4:
        # Use alpha channel to find content
        alpha = image[:, :, 3]
        # Find non-transparent pixels
        coords = cv2.findNonZero(alpha)
    else:
        # For BGR images, find non-black pixels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
    
    if coords is None:
        return image
    
    # Get bounding box of content
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add margin
    margin_x = int(w * margin_percent / 100)
    margin_y = int(h * margin_percent / 100)
    
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)
    
    # Crop to content with margin
    return image[y1:y2, x1:x2]

def convert_bird_images(source_dir, target_dir, remove_bg=True, bg_color=(128, 128, 128),
                       bg_model='u2net', bg_transparent=True, bg_fill_black=False,
                       crop_padding=0, min_sharpness=None, min_edge_quality=None,
                       quality_report=False, quality=95, deduplicate=False,
                       similarity_threshold=5, resize_to_target=True, target_image_size=224):
    """
    Convert already-extracted bird images with consistent processing.
    Useful for normalizing existing training datasets.
    
    This function:
    - Loads existing bird crop images (no detection needed)
    - Applies background removal, quality filtering, etc.
    - Maintains original folder structure (species subdirectories)
    - Saves to new target directory
    
    Args:
        source_dir: Source directory with bird images (can have subdirectories for species)
        target_dir: Target directory for converted images (will mirror source structure)
        remove_bg: Apply background removal (default: True)
        bg_color: Background color as BGR tuple (default: gray)
        bg_model: rembg model to use (default: 'u2net')
        bg_transparent: Use transparent background (default: True)
        bg_fill_black: Make black areas transparent (default: False)
        crop_padding: Extra pixels around image (default: 0)
        min_sharpness: Minimum sharpness filter (default: None)
        min_edge_quality: Minimum edge quality filter (default: None)
        quality_report: Generate quality statistics (default: False)
        quality: JPEG quality for non-transparent (default: 95)
        deduplicate: Skip duplicate images (default: False)
        similarity_threshold: Hamming distance for duplicates (default: 5)
        resize_to_target: Resize images (default: True)
        target_image_size: Target size in pixels (default: 224)
    """
    source_path = Path(source_dir).expanduser()
    target_path = Path(target_dir).expanduser()
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Find all image files recursively
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(source_path.rglob(f'*{ext}'))
    
    if not image_files:
        print(f"❌ No images found in: {source_dir}")
        return
    
    print(f"🖼️  Found {len(image_files)} image(s) to convert")
    print(f"📁 Source: {source_path}")
    print(f"📁 Target: {target_path}")
    print()
    
    # Statistics
    converted_count = 0
    skipped_quality = 0
    skipped_duplicate = 0
    hash_cache = {}
    quality_stats = {
        'accepted': [],
        'rejected_blur': [],
        'rejected_edges': []
    }
    
    # Process each image
    for idx, img_file in enumerate(image_files, 1):
        # Calculate relative path to maintain folder structure
        rel_path = img_file.relative_to(source_path)
        target_file = target_path / rel_path
        
        # Create target directory
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Progress
        if idx % 100 == 0 or idx == 1:
            print(f"Processing {idx}/{len(image_files)}: {rel_path}")
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"⚠️  Failed to load: {rel_path}")
            continue
        
        # Apply crop padding if specified
        if crop_padding > 0:
            # Add padding by creating larger canvas
            h, w = image.shape[:2]
            new_h, new_w = h + 2 * crop_padding, w + 2 * crop_padding
            
            if image.shape[2] == 4:  # BGRA
                padded = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            else:  # BGR
                padded = np.full((new_h, new_w, 3), bg_color, dtype=np.uint8)
            
            # Place original image in center
            padded[crop_padding:crop_padding+h, crop_padding:crop_padding+w] = image
            image = padded
        
        # Quality filtering
        if min_sharpness is not None or min_edge_quality is not None:
            quality_metrics = calculate_motion_quality(image)
            is_acceptable, reason = is_motion_acceptable(quality_metrics, min_sharpness, min_edge_quality)
            
            if not is_acceptable:
                skipped_quality += 1
                if reason == 'low_sharpness':
                    quality_stats['rejected_blur'].append(quality_metrics['overall'])
                elif reason == 'poor_edges':
                    quality_stats['rejected_edges'].append(quality_metrics['overall'])
                continue
            else:
                quality_stats['accepted'].append(quality_metrics['overall'])
        
        # Background removal
        if remove_bg:
            image = remove_background(
                image,
                bg_color=bg_color,
                model_name=bg_model,
                transparent=bg_transparent,
                fill_black_areas=bg_fill_black,
                expand_mask=0
            )
        
        # Resize if needed
        if resize_to_target and target_image_size > 0:
            image = cv2.resize(image, (target_image_size, target_image_size), interpolation=cv2.INTER_LANCZOS4)
        
        # Deduplication
        if deduplicate:
            if image.shape[2] == 4:
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
            else:
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            img_hash = imagehash.phash(img_pil)
            
            # Check against existing hashes
            is_duplicate = False
            for cached_path, cached_hash in hash_cache.items():
                if img_hash - cached_hash <= similarity_threshold:
                    is_duplicate = True
                    skipped_duplicate += 1
                    break
            
            if is_duplicate:
                continue
            
            hash_cache[str(target_file)] = img_hash
        
        # Save image
        if bg_transparent and remove_bg:
            # Save as PNG with transparency
            save_path = target_file.with_suffix('.png')
            cv2.imwrite(str(save_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            # Save as JPEG
            save_path = target_file.with_suffix('.jpg')
            # Convert BGRA to BGR if needed
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(str(save_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        converted_count += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("=" * 60)
    print(f"📁 Target directory: {target_path}")
    print(f"🖼️  Total processed: {len(image_files)}")
    print(f"✅ Converted: {converted_count}")
    
    if skipped_quality > 0:
        print(f"⏭️  Skipped (quality): {skipped_quality}")
    
    if skipped_duplicate > 0:
        print(f"⏭️  Skipped (duplicate): {skipped_duplicate}")
    
    # Quality report
    if quality_report and quality_stats['accepted']:
        print("\n" + "=" * 60)
        print("📊 Quality Report")
        print("=" * 60)
        
        total = len(image_files)
        accepted = len(quality_stats['accepted'])
        rejected = skipped_quality
        
        print(f"Total images: {total}")
        print(f"Accepted: {accepted} ({accepted/total*100:.1f}%)")
        print(f"Rejected: {rejected} ({rejected/total*100:.1f}%)")
        
        if quality_stats['accepted']:
            avg_quality = np.mean(quality_stats['accepted'])
            print(f"Average quality (accepted): {avg_quality:.2f}")
        
        if quality_stats['rejected_blur']:
            avg_blur = np.mean(quality_stats['rejected_blur'])
            print(f"Average quality (rejected blur): {avg_blur:.2f}")
        
        if quality_stats['rejected_edges']:
            avg_edges = np.mean(quality_stats['rejected_edges'])
            print(f"Average quality (rejected edges): {avg_edges:.2f}")


def extract_birds_from_video(video_path, output_dir, bird_species=None, 
                             detection_model=None, species_model=None,
                             threshold=None, sample_rate=None, target_image_size=224,
                             species_threshold=None, target_class=14,
                             max_detections=10, min_box_size=50, max_box_size=800,
                             quality=95, skip_blurry=False,
                             deduplicate=False, similarity_threshold=5,
                             min_sharpness=None, min_edge_quality=None,
                             save_quality_report=False, remove_bg=False,
                             bg_color=(128, 128, 128), bg_model='u2net',
                             bg_transparent=False, bg_fill_black=False,
                             crop_padding=0):
    """
    Extract bird crops from video and save as images
    
    Args:
        video_path: Path to video file
        output_dir: Base directory to save extracted bird images
        bird_species: Manually specified bird species (if known, e.g., 'rotkehlchen')
        detection_model: YOLO model path for bird detection (default: yolov8n.pt)
        species_model: Custom species classifier model path for automatic sorting
        threshold: Detection confidence threshold (default: 0.5 for high quality)
        sample_rate: Analyze every Nth frame (default: 3)
        target_image_size: Target image size in pixels (default: 224, 0 for original)
        species_threshold: Minimum confidence for species classification (default: None, no filter)
        target_class: COCO class for bird (14)
        max_detections: Maximum number of detections per frame (default: 10)
        min_box_size: Minimum bounding box size in pixels (default: 50)
        max_box_size: Maximum bounding box size in pixels (default: 800)
        quality: JPEG quality 1-100 (default: 95)
        skip_blurry: Skip blurry images (default: False)
        deduplicate: Skip duplicate/similar images (default: False)
        similarity_threshold: Hamming distance threshold for duplicates 0-64 (default: 5)
        min_sharpness: Minimum sharpness score (default: None = no filter)
        min_edge_quality: Minimum edge quality score (default: None = no filter)
        save_quality_report: Save detailed quality report (default: False)
        crop_padding: Extra pixels to include around detected bird (default: 0)
    """
    # Use defaults if not specified
    detection_model = detection_model or DEFAULT_MODEL
    threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
    sample_rate = sample_rate if sample_rate is not None else DEFAULT_SAMPLE_RATE
    resize_to_target = (target_image_size > 0)  # If 0, keep original size
    
    # Load species classifier if provided
    classifier = None
    processor = None
    if species_model:
        print(_('loading_species') + f" {species_model}")
        processor = AutoImageProcessor.from_pretrained(species_model)
        classifier = AutoModelForImageClassification.from_pretrained(species_model)
        classifier.eval()
        print(_('loaded_species_classes', count=len(classifier.config.id2label)))
    
    # Load YOLO model
    print(_('loading_yolo') + f" {detection_model}")
    model = YOLO(detection_model)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(_('cannot_open_video', path=video_path))
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(_('video_info') + f" {Path(video_path).name}")
    print(_('total_frames', total=total_frames, fps=fps))
    print(_('analyzing_every_nth', n=sample_rate))
    print(_('detection_threshold', threshold=threshold))
    if species_threshold is not None:
        print(_('species_threshold', threshold=species_threshold))
    
    if resize_to_target:
        print(_('image_size', size=target_image_size))
    else:
        print(_('image_size_original'))
    
    # Print additional filter settings
    if max_detections < 999:
        print(_('max_detections_per_frame', max=max_detections))
    if min_box_size > 0:
        print(_('min_box_size_filter', size=min_box_size))
    if max_box_size < 9999:
        print(_('max_box_size_filter', size=max_box_size))
    if quality < 95:
        print(_('jpeg_quality_filter', quality=quality))
    if skip_blurry:
        print(_('blur_detection_filter'))
    if deduplicate:
        print(_('dedup_filter', threshold=similarity_threshold))
    if min_sharpness is not None:
        print(_('motion_sharpness_filter', threshold=min_sharpness))
    if min_edge_quality is not None:
        print(_('motion_edge_filter', threshold=min_edge_quality))
    if remove_bg:
        print(_('background_removal_enabled'))
    
    # Determine output mode
    if species_model:
        print(_('mode_autosorting'))
    elif bird_species:
        print(_('mode_manual', species=bird_species))
    else:
        print(_('mode_standard'))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique session ID for this video extraction
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{video_name}_{timestamp}"
    
    bird_count = 0  # Successfully exported birds
    detected_count = 0  # Total detected birds (including skipped)
    skipped_count = 0  # Birds skipped due to threshold
    duplicate_count = 0  # Birds skipped due to duplication
    motion_rejected_count = 0  # Birds rejected due to motion/blur
    species_counts = {}
    frame_num = 0
    
    # Quality report statistics
    quality_stats = {
        'accepted': [],
        'rejected_motion': [],
        'rejected_blur': [],
        'rejected_edges': []
    } if save_quality_report else None
    
    # Initialize hash cache for deduplication
    hash_cache = {} if deduplicate else None
    
    # Pre-load existing images into hash cache for cross-session deduplication
    if deduplicate:
        print(_('dedup_loading_existing'))
        existing_images = list(output_path.rglob("*.jpg")) + list(output_path.rglob("*.jpeg")) + list(output_path.rglob("*.png"))
        for img_path in existing_images:
            try:
                img = Image.open(img_path)
                img_hash = imagehash.phash(img)
                hash_cache[str(img_path)] = img_hash
            except Exception as e:
                # Skip corrupted images
                pass
        if len(hash_cache) > 0:
            print(_('dedup_loaded_existing', count=len(hash_cache)))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame
            if frame_num % sample_rate != 0:
                frame_num += 1
                continue
            
            # Run detection
            results = model(frame, verbose=False)
            
            # Extract birds
            detection_in_frame = 0  # Track detections per frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if it's a bird with sufficient confidence
                    if cls == target_class and conf >= threshold:
                        # Get bounding box
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        
                        # Calculate box size
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_size = max(box_width, box_height)
                        
                        # Filter by box size
                        if box_size < min_box_size:
                            continue  # Too small, likely distant bird
                        if box_size > max_box_size:
                            continue  # Too large, likely false positive
                        
                        # Limit detections per frame
                        if detection_in_frame >= max_detections:
                            break
                        detection_in_frame += 1
                        
                        # Ensure coordinates are within frame
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Apply padding/cropping to adjust context
                        # Positive values = expand crop (more background)
                        # Negative values = shrink crop (tighter around bird, less background)
                        # Value is percentage-based: e.g., 50 = 50% larger, -20 = 20% smaller
                        if crop_padding != 0:
                            box_width = x2 - x1
                            box_height = y2 - y1
                            
                            # Calculate padding as percentage of box size
                            padding_factor = crop_padding / 100.0
                            pad_x = int(box_width * padding_factor)
                            pad_y = int(box_height * padding_factor)
                            
                            # Apply padding (can be negative to shrink)
                            x1 = max(0, x1 - pad_x)
                            y1 = max(0, y1 - pad_y)
                            x2 = min(w, x2 + pad_x)
                            y2 = min(h, y2 + pad_y)
                        
                        # Crop bird
                        bird_crop = frame[y1:y2, x1:x2]
                        
                        if bird_crop.size > 0:
                            # Calculate motion quality metrics
                            motion_quality = calculate_motion_quality(bird_crop)
                            
                            # Check motion quality thresholds
                            motion_ok, motion_reason = is_motion_acceptable(
                                motion_quality,
                                min_sharpness=min_sharpness,
                                min_edge_quality=min_edge_quality
                            )
                            
                            if not motion_ok:
                                motion_rejected_count += 1
                                if save_quality_report:
                                    if motion_reason == 'low_sharpness':
                                        quality_stats['rejected_blur'].append(motion_quality['sharpness'])
                                    elif motion_reason == 'poor_edges':
                                        quality_stats['rejected_edges'].append(motion_quality['edge_quality'])
                                continue
                            
                            # Skip blurry images if requested (legacy method)
                            if skip_blurry:
                                blur_score = cv2.Laplacian(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                                if blur_score < 100:  # Threshold for blur detection
                                    continue
                            
                            # Store quality stats if requested
                            if save_quality_report:
                                quality_stats['accepted'].append(motion_quality['overall'])
                            
                            # Count all detected birds
                            detected_count += 1
                            
                            # Generate unique ID for this bird image
                            unique_id = uuid.uuid4().hex[:8]  # 8-character unique ID
                            
                            # Determine species and output directory
                            species_name = None
                            species_conf = 0.0
                            
                            if species_model and classifier and processor:
                                # Auto-classify species
                                bird_image = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB))
                                inputs = processor(bird_image, return_tensors="pt")
                                
                                with torch.no_grad():
                                    outputs = classifier(**inputs)
                                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                    species_conf = probs.max().item()
                                    predicted_class = outputs.logits.argmax(-1).item()
                                    species_name = classifier.config.id2label[predicted_class]
                                
                            elif bird_species:
                                # Manual species
                                species_name = bird_species
                                species_conf = 1.0
                            
                            # Apply species confidence filter if specified
                            if species_threshold is not None and species_conf < species_threshold:
                                skipped_count += 1
                                print(_('bird_skipped', species=species_name, conf=species_conf, threshold=species_threshold, frame=frame_num))
                                continue
                            
                            # Check for duplicates if enabled
                            if deduplicate:
                                # Check if image has alpha channel (BGRA) - preserve it!
                                if bird_crop.shape[2] == 4:
                                    bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGRA2RGBA))
                                else:
                                    bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB))
                                img_hash = imagehash.phash(bird_pil)
                                
                                # Check if similar image already exists
                                is_duplicate = False
                                similar_to = None
                                min_distance = float('inf')
                                
                                for existing_path, existing_hash in hash_cache.items():
                                    distance = img_hash - existing_hash
                                    if distance <= similarity_threshold:
                                        is_duplicate = True
                                        similar_to = Path(existing_path).name
                                        min_distance = distance
                                        break
                                
                                if is_duplicate:
                                    duplicate_count += 1
                                    print(_('dedup_skipped_duplicate', filename=similar_to, distance=min_distance))
                                    continue
                            
                            # Only count birds that passed all filters
                            bird_count += 1
                            
                            # Create species subdirectory if needed
                            if species_name:
                                species_dir = output_path / species_name
                                species_dir.mkdir(exist_ok=True)
                                save_dir = species_dir
                                
                                # Track species counts
                                species_counts[species_name] = species_counts.get(species_name, 0) + 1
                            else:
                                save_dir = output_path
                            
                            # Use PNG for transparent background, JPG otherwise
                            file_ext = "png" if (remove_bg and bg_transparent) else "jpg"
                            
                            # Filename: session_id + unique_id + metadata
                            if species_name and species_model:
                                filename = f"{session_id}_{unique_id}_f{frame_num:06d}_det{conf:.2f}_cls{species_conf:.2f}.{file_ext}"
                            else:
                                filename = f"{session_id}_{unique_id}_f{frame_num:06d}_c{conf:.2f}.{file_ext}"
                            
                            save_path = save_dir / filename
                            
                            # Apply background removal if requested
                            if remove_bg:
                                # Use crop_padding to control mask expansion
                                # Positive values = keep more background around bird details
                                expand_pixels = max(0, crop_padding) if crop_padding > 0 else 0
                                bird_crop = remove_background(bird_crop, bg_color=bg_color, model_name=bg_model,
                                                            transparent=bg_transparent, fill_black_areas=bg_fill_black,
                                                            expand_mask=expand_pixels)
                            
                            # Resize to target size for optimal training
                            if resize_to_target:
                                # IMPORTANT: (Re)create PIL image after potential background removal
                                # to ensure correct color mode (RGBA for transparent, RGB otherwise)
                                # Delete old bird_pil from deduplication check if it exists
                                if 'bird_pil' in locals():
                                    del bird_pil
                                
                                # Check if image has alpha channel (BGRA) after background removal
                                if bird_crop.shape[2] == 4:
                                    bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGRA2RGBA))
                                else:
                                    bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB))
                                # Resize maintaining aspect ratio with padding (better quality than distortion)
                                bird_pil.thumbnail((target_image_size, target_image_size), Image.Resampling.LANCZOS)
                                
                                # Create square image with padding
                                # Use RGBA for transparent images, RGB for opaque with bg_color
                                if bird_pil.mode == 'RGBA':
                                    new_img = Image.new('RGBA', (target_image_size, target_image_size), (0, 0, 0, 0))
                                else:
                                    # Use background color for padding (BGR -> RGB conversion)
                                    padding_color = (bg_color[2], bg_color[1], bg_color[0])  # BGR to RGB
                                    new_img = Image.new('RGB', (target_image_size, target_image_size), padding_color)
                                # Center the image
                                x_offset = (target_image_size - bird_pil.width) // 2
                                y_offset = (target_image_size - bird_pil.height) // 2
                                new_img.paste(bird_pil, (x_offset, y_offset))
                                
                                # Save with PIL (better quality)
                                # Determine file extension based on transparency
                                if new_img.mode == 'RGBA':
                                    save_path = save_path.with_suffix('.png')
                                    new_img.save(save_path, 'PNG', compress_level=6)
                                else:
                                    save_path = save_path.with_suffix('.jpg')
                                    new_img.save(save_path, 'JPEG', quality=quality)
                                
                                # Add to hash cache if deduplication is enabled
                                if deduplicate:
                                    # Recompute hash for saved image (in case resizing changed it)
                                    saved_hash = imagehash.phash(new_img)
                                    hash_cache[str(save_path)] = saved_hash
                            else:
                                # Save original size
                                if remove_bg and bg_transparent:
                                    # Save as PNG with transparency
                                    save_path = save_path.with_suffix('.png')
                                    cv2.imwrite(str(save_path), bird_crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                                else:
                                    # Save as JPEG
                                    save_path = save_path.with_suffix('.jpg')
                                    cv2.imwrite(str(save_path), bird_crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                                
                                # Add to hash cache if deduplication is enabled
                                if deduplicate:
                                    if not 'bird_pil' in locals():  # Only create if not already created
                                        if bird_crop.shape[2] == 4:
                                            bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGRA2RGBA))
                                        else:
                                            bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB))
                                    hash_cache[str(save_path)] = img_hash
                            
                            if species_name:
                                print(_('bird_extracted', count=bird_count, species=species_name, conf=species_conf, frame=frame_num))
                            else:
                                print(_('bird_extracted_simple', count=bird_count, frame=frame_num, conf=conf))
            
            frame_num += 1
            
            # Progress
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100
                print(_('progress', percent=progress, current=frame_num, total=total_frames))
    
    except KeyboardInterrupt:
        print(_('extraction_interrupted'))
    
    finally:
        cap.release()
    
    print(_('extraction_complete'))
    print(_('output_directory', path=output_path))
    print(_('detected_birds_total', count=detected_count))
    print(_('exported_birds_total', count=bird_count))
    
    # Show skipped count if threshold was applied
    if species_threshold is not None and skipped_count > 0:
        print(_('skipped_birds_total', count=skipped_count, threshold=species_threshold))
    
    # Show deduplication statistics if enabled
    if deduplicate and duplicate_count > 0:
        total_checked = bird_count + duplicate_count
        percent = (duplicate_count / total_checked * 100) if total_checked > 0 else 0
        print(_('dedup_stats'))
        print(_('dedup_stats_checked', count=total_checked))
        print(_('dedup_stats_skipped', count=duplicate_count, percent=percent))
    
    # Show motion quality rejection statistics if applicable
    if motion_rejected_count > 0:
        print(_('motion_rejected_stats', count=motion_rejected_count))
    
    # Show quality report if requested
    if save_quality_report and quality_stats:
        print("\n" + _('quality_report_title'))
        print("━" * 60)
        
        total_processed = detected_count
        accepted = len(quality_stats['accepted'])
        rejected_blur = len(quality_stats['rejected_blur'])
        rejected_edges = len(quality_stats['rejected_edges'])
        
        print(_('quality_report_total', count=total_processed))
        print(_('quality_report_accepted', count=accepted, percent=accepted/total_processed*100 if total_processed > 0 else 0))
        print(_('quality_report_rejected', count=motion_rejected_count, percent=motion_rejected_count/total_processed*100 if total_processed > 0 else 0))
        
        if quality_stats['accepted']:
            avg_quality = np.mean(quality_stats['accepted'])
            print(_('quality_report_avg_accepted', score=avg_quality))
        
        if quality_stats['rejected_blur']:
            avg_blur = np.mean(quality_stats['rejected_blur'])
            print(_('quality_report_avg_rejected_blur', score=avg_blur))
        
        if quality_stats['rejected_edges']:
            avg_edges = np.mean(quality_stats['rejected_edges'])
            print(_('quality_report_avg_rejected_edges', score=avg_edges))
        
        print("━" * 60)
    
    # Show species breakdown if applicable
    if species_counts:
        print(_('species_breakdown'))
        for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
            print(_('species_count', species=species, count=count))
    
    print(_('session_id', id=session_id))
    
    if species_model:
        print(_('filename_format', format=f"{session_id}_<id>_f<frame>_det<det-conf>_cls<species-conf>.jpg"))
    else:
        print(_('filename_format', format=f"{session_id}_<unique-id>_f<frame>_c<confidence>.jpg"))
    
    print(_('next_steps'))
    if species_model or bird_species:
        print(_('next_step_review', path=output_path))
        print(_('next_step_verify'))
        print(_('next_step_organize'))
        print(_('next_step_train'))
    else:
        print(_('next_step_review', path=output_path))
        print(_('next_step_manual_sort'))
        print(_('next_step_organize'))
        print(_('next_step_train'))


def extract_birds_from_image(image_path, output_dir, bird_species=None,
                            detection_model=None, species_model=None,
                            threshold=None, target_image_size=224,
                            species_threshold=None, target_class=14,
                            max_detections=10, min_box_size=50, max_box_size=800,
                            quality=95, deduplicate=False, similarity_threshold=5,
                            min_sharpness=None, min_edge_quality=None,
                            save_quality_report=False, remove_bg=False,
                            bg_color=(128, 128, 128), bg_model='u2net',
                            bg_transparent=False, bg_fill_black=False,
                            crop_padding=0, resize_to_target=True):
    """
    Extract bird crops from a single image and save as images.
    Similar to extract_birds_from_video() but for static images.
    
    Args:
        image_path: Path to image file
        (all other args same as extract_birds_from_video)
    """
    # Use defaults if not specified
    detection_model = detection_model or DEFAULT_MODEL
    threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
    
    # Load image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    # Initialize YOLO model
    print(_('loading_model', model=detection_model))
    model = YOLO(detection_model)
    
    # Generate session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup output directory
    if bird_species:
        output_path = Path(output_dir) / bird_species
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize species classifier if provided
    species_processor = None
    species_classifier = None
    if species_model:
        print(_('loading_species_model', model=species_model))
        species_processor = AutoImageProcessor.from_pretrained(species_model)
        species_classifier = AutoModelForImageClassification.from_pretrained(species_model)
    
    # Statistics
    bird_count = 0
    detected_count = 0
    skipped_count = 0
    duplicate_count = 0
    motion_rejected_count = 0
    species_counts = {}
    hash_cache = {}
    quality_stats = {
        'accepted': [],
        'rejected_blur': [],
        'rejected_edges': []
    }
    
    # Run detection on the image
    results = model(frame, conf=threshold, classes=[target_class], max_det=max_detections, verbose=False)
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Validate box size
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < min_box_size or box_height < min_box_size:
                continue
            if box_width > max_box_size or box_height > max_box_size:
                continue
            
            detected_count += 1
            
            # Apply crop padding
            if crop_padding > 0:
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0, x1 - crop_padding)
                y1 = max(0, y1 - crop_padding)
                x2 = min(frame_w, x2 + crop_padding)
                y2 = min(frame_h, y2 + crop_padding)
            
            # Extract crop
            bird_crop = frame[y1:y2, x1:x2].copy()
            
            # Check motion/blur quality if filters enabled
            if min_sharpness is not None or min_edge_quality is not None:
                quality_metrics = calculate_motion_quality(bird_crop)
                is_acceptable, reason = is_motion_acceptable(quality_metrics, min_sharpness, min_edge_quality)
                
                if not is_acceptable:
                    motion_rejected_count += 1
                    if reason == 'low_sharpness':
                        quality_stats['rejected_blur'].append(quality_metrics['overall'])
                    elif reason == 'poor_edges':
                        quality_stats['rejected_edges'].append(quality_metrics['overall'])
                    continue
                else:
                    quality_stats['accepted'].append(quality_metrics['overall'])
            
            # Background removal
            if remove_bg:
                bird_crop = remove_background(
                    bird_crop,
                    bg_color=bg_color,
                    model_name=bg_model,
                    transparent=bg_transparent,
                    fill_black_areas=bg_fill_black,
                    expand_mask=0
                )
            
            # Species classification
            species_name = None
            species_conf = 0.0
            
            if species_classifier is not None:
                bird_pil = Image.fromarray(cv2.cvtColor(bird_crop if bird_crop.shape[2] == 3 else bird_crop[:,:,:3], cv2.COLOR_BGR2RGB))
                inputs = species_processor(images=bird_pil, return_tensors="pt")
                with torch.no_grad():
                    outputs = species_classifier(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    predicted_class_idx = logits.argmax(-1).item()
                    species_conf = probs[0][predicted_class_idx].item()
                    species_name = species_classifier.config.id2label[predicted_class_idx]
                
                if species_threshold is not None and species_conf < species_threshold:
                    skipped_count += 1
                    continue
                
                species_counts[species_name] = species_counts.get(species_name, 0) + 1
                save_dir = Path(output_dir) / species_name
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = output_path
            
            # Deduplicate
            skip_duplicate = False
            if deduplicate:
                if bird_crop.shape[2] == 4:
                    bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGRA2RGBA))
                else:
                    bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB))
                
                img_hash = imagehash.phash(bird_pil)
                
                for cached_path, cached_hash in hash_cache.items():
                    if img_hash - cached_hash <= similarity_threshold:
                        skip_duplicate = True
                        duplicate_count += 1
                        break
            
            if not skip_duplicate:
                bird_count += 1
                unique_id = str(uuid.uuid4())[:8]
                
                if species_name:
                    filename = f"{session_id}_{unique_id}_det{conf:.2f}_cls{species_conf:.2f}"
                else:
                    filename = f"{session_id}_{unique_id}_c{conf:.2f}"
                
                save_path = save_dir / filename
                
                # Resize if needed
                if resize_to_target and target_image_size > 0:
                    bird_crop = cv2.resize(bird_crop, (target_image_size, target_image_size), interpolation=cv2.INTER_LANCZOS4)
                
                # Save
                if remove_bg and bg_transparent:
                    save_path = save_path.with_suffix('.png')
                    cv2.imwrite(str(save_path), bird_crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                else:
                    save_path = save_path.with_suffix('.jpg')
                    cv2.imwrite(str(save_path), bird_crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                if deduplicate:
                    if not 'bird_pil' in locals():
                        if bird_crop.shape[2] == 4:
                            bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGRA2RGBA))
                        else:
                            bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB))
                    hash_cache[str(save_path)] = img_hash
                
                if species_name:
                    print(_('bird_extracted', count=bird_count, species=species_name, conf=species_conf, frame=Path(image_path).name))
                else:
                    print(_('bird_extracted_simple', count=bird_count, frame=Path(image_path).name, conf=conf))
    
    # Print summary
    print(_('extraction_complete'))
    print(_('output_directory', path=output_path))
    print(_('detected_birds_total', count=detected_count))
    print(_('exported_birds_total', count=bird_count))
    
    if species_threshold is not None and skipped_count > 0:
        print(_('skipped_birds_total', count=skipped_count, threshold=species_threshold))
    
    if deduplicate and duplicate_count > 0:
        total_checked = bird_count + duplicate_count
        percent = (duplicate_count / total_checked * 100) if total_checked > 0 else 0
        print(_('dedup_stats'))
        print(_('dedup_stats_checked', count=total_checked))
        print(_('dedup_stats_skipped', count=duplicate_count, percent=percent))
    
    if motion_rejected_count > 0:
        print(_('motion_rejected_stats', count=motion_rejected_count))
    
    if species_counts:
        print(_('species_breakdown'))
        for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
            print(_('species_count', species=species, count=count))


def main():
    parser = argparse.ArgumentParser(
        description='Extract bird crops from videos or images for training data collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Videos - Single video file
  python extract_birds.py video.mp4 --folder data/ --bird rotkehlchen

  # Videos - Multiple videos with wildcards
  python extract_birds.py "~/Videos/*.mp4" --folder data/ --species-model ~/vogel-models/bird-classifier-*/final/

  # Videos - Recursive directory search
  python extract_birds.py "~/Videos/**/*.mp4" --folder data/ --bird kohlmeise

  # Images - Single image file
  python extract_birds.py photo.jpg --folder data/ --bird rotkehlchen

  # Images - Multiple images with wildcards
  python extract_birds.py "~/Photos/*.jpg" --folder data/ --species-model ~/models/classifier/

  # Images - Recursive directory search for images
  python extract_birds.py "~/Photos/**/*.jpg" --folder data/ --bird kohlmeise

  # Mixed - Process both videos and images
  python extract_birds.py "~/Media/**/*" --folder data/ --recursive

  # Extract with custom detection parameters
  python extract_birds.py video.mp4 --folder data/ --bird kohlmeise --threshold 0.6
  
  # Extract in original size (no resize)
  python extract_birds.py video.mp4 --folder data/ --bird rotkehlchen --no-resize

  # Convert mode - Process existing bird images
  python extract_birds.py --convert \
    --source ~/vogel-training-data-species \
    --target ~/vogel-training-data-species-transparent \
    --bg-remove --bg-transparent --crop-padding 10 --min-sharpness 80
        """
    )
    
    # Mode selection
    parser.add_argument('--convert', action='store_true',
                       help='Convert mode: Process existing bird images (no detection). Requires --source and --target.')
    parser.add_argument('--source', help='Source directory with existing bird images (for --convert mode)')
    parser.add_argument('--target', help='Target directory for converted images (for --convert mode)')
    
    # Input/Output for extraction mode
    parser.add_argument('input', nargs='?', help='Video/image file, directory, or glob pattern (e.g., "*.mp4", "*.jpg", "~/Media/**/*")')
    parser.add_argument('--folder', help='Base directory for extracted bird images (for extraction mode)')
    parser.add_argument('--bird', help='Manual bird species name (e.g., rotkehlchen, kohlmeise). Creates subdirectory.')
    parser.add_argument('--species-model', help='Path to custom species classifier for automatic sorting')
    parser.add_argument('--no-resize', action='store_true',
                       help=f'Keep original image size instead of resizing to {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}px')
    parser.add_argument('--detection-model', default=None, help=f'YOLO detection model path (default: {DEFAULT_MODEL})')
    parser.add_argument('--threshold', type=float, default=None, 
                       help=f'Detection confidence threshold (default: {DEFAULT_THRESHOLD} for high quality)')
    parser.add_argument('--species-threshold', type=float, default=None,
                       help='Minimum confidence for species classification (e.g., 0.85 for 85%%). Only exports birds with confidence >= this value.')
    parser.add_argument('--sample-rate', type=int, default=None, 
                       help=f'Analyze every Nth frame for videos (default: {DEFAULT_SAMPLE_RATE})')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search directories recursively for files')
    
    # Background removal options
    parser.add_argument('--bg-remove', action='store_true',
                       help='Remove background from bird images using AI')
    parser.add_argument('--bg-color', type=str, default='128,128,128',
                       help='Background color as R,G,B (default: 128,128,128 = gray). Ignored if --bg-transparent is used.')
    parser.add_argument('--bg-model', default='u2net',
                       choices=['u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use'],
                       help='Background removal model (default: u2net)')
    parser.add_argument('--bg-transparent', action='store_true',
                       help='Use transparent background (PNG with alpha channel)')
    parser.add_argument('--bg-fill-black', action='store_true',
                       help='Make black background/padding areas transparent')
    
    # Quality and deduplication
    parser.add_argument('--deduplicate', action='store_true',
                       help='Skip duplicate/similar images using perceptual hashing')
    parser.add_argument('--similarity-threshold', type=int, default=5,
                       help='Hamming distance threshold for duplicates (0-64, default: 5)')
    parser.add_argument('--min-sharpness', type=float, default=None,
                       help='Minimum sharpness score (Laplacian variance)')
    parser.add_argument('--min-edge-quality', type=float, default=None,
                       help='Minimum edge quality score (Sobel gradient)')
    parser.add_argument('--quality-report', action='store_true',
                       help='Save detailed quality statistics report')
    parser.add_argument('--crop-padding', type=int, default=0,
                       help='Extra pixels to include around detected bird (default: 0)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality 1-100 (default: 95)')
    
    # Keep -o as alias for backwards compatibility
    parser.add_argument('-o', '--output', dest='folder_alias', help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Convert mode - process existing bird images
    if args.convert:
        if not args.source or not args.target:
            parser.error("--convert mode requires both --source and --target")
        
        # Parse background color
        bg_color_rgb = tuple(map(int, args.bg_color.split(',')))
        bg_color_bgr = (bg_color_rgb[2], bg_color_rgb[1], bg_color_rgb[0])
        
        print("=" * 70)
        print("🔄 CONVERT MODE: Processing existing bird images")
        print("=" * 70)
        print()
        
        convert_bird_images(
            source_dir=args.source,
            target_dir=args.target,
            remove_bg=args.bg_remove,
            bg_color=bg_color_bgr,
            bg_model=args.bg_model,
            bg_transparent=args.bg_transparent,
            bg_fill_black=args.bg_fill_black,
            crop_padding=args.crop_padding,
            min_sharpness=args.min_sharpness,
            min_edge_quality=args.min_edge_quality,
            quality_report=args.quality_report,
            quality=args.quality,
            deduplicate=args.deduplicate,
            similarity_threshold=args.similarity_threshold,
            resize_to_target=not args.no_resize,
            target_image_size=TARGET_IMAGE_SIZE
        )
        
        return  # Exit after conversion
    
    # Extraction mode (normal operation)
    if not args.input:
        parser.error("input is required for extraction mode (or use --convert)")
    
    # Handle backwards compatibility for -o
    output_dir = args.folder or args.folder_alias
    if not output_dir:
        parser.error("--folder is required for extraction mode")
    
    # Define supported extensions
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF'}
    
    # Collect files
    input_files = []
    input_path = Path(args.input).expanduser()
    
    # Check if it's a glob pattern
    if '*' in args.input or '?' in args.input:
        # Expand glob pattern
        input_files = [Path(p) for p in glob.glob(str(input_path), recursive=args.recursive)]
    elif input_path.is_dir():
        # Directory - search for video and image files
        if args.recursive:
            patterns = ['**/*']
        else:
            patterns = ['*']
        
        for pattern in patterns:
            for file_path in input_path.glob(pattern):
                if file_path.is_file() and file_path.suffix in (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS):
                    input_files.append(file_path)
    elif input_path.is_file():
        # Single file
        input_files = [input_path]
    else:
        print(f"❌ File/directory not found: {args.input}")
        sys.exit(1)
    
    # Remove duplicates and sort
    input_files = sorted(set(input_files))
    
    if not input_files:
        print(f"❌ No video or image files found matching: {args.input}")
        sys.exit(1)
    
    # Categorize files by type
    video_files = [f for f in input_files if f.suffix in VIDEO_EXTENSIONS]
    image_files = [f for f in input_files if f.suffix in IMAGE_EXTENSIONS]
    
    # Show what will be processed
    total_files = len(video_files) + len(image_files)
    print(f"📁 Found {total_files} file(s) to process:")
    if video_files:
        print(f"   🎬 {len(video_files)} video(s)")
    if image_files:
        print(f"   🖼️  {len(image_files)} image(s)")
    
    # Show first 10 files
    all_files = video_files + image_files
    for i, f in enumerate(all_files[:10], 1):
        file_type = "🎬" if f.suffix in VIDEO_EXTENSIONS else "🖼️"
        print(f"   {i}. {file_type} {f.name}")
    if len(all_files) > 10:
        print(f"   ... and {len(all_files) - 10} more")
    print()
    
    # Validate that only one sorting method is used
    if args.bird and args.species_model:
        print("⚠️  Warning: Both --bird and --species-model specified. Using auto-classification.")
    
    # Parse background color
    bg_color_rgb = tuple(map(int, args.bg_color.split(',')))
    # Convert RGB to BGR for OpenCV
    bg_color_bgr = (bg_color_rgb[2], bg_color_rgb[1], bg_color_rgb[0])
    
    # Common extraction parameters
    extract_params = {
        'output_dir': output_dir,
        'bird_species': args.bird,
        'detection_model': args.detection_model,
        'species_model': args.species_model,
        'threshold': args.threshold,
        'target_image_size': TARGET_IMAGE_SIZE if not args.no_resize else 0,
        'species_threshold': args.species_threshold,
        'deduplicate': args.deduplicate,
        'similarity_threshold': args.similarity_threshold,
        'min_sharpness': args.min_sharpness,
        'min_edge_quality': args.min_edge_quality,
        'save_quality_report': args.quality_report,
        'remove_bg': args.bg_remove,
        'bg_color': bg_color_bgr,
        'bg_model': args.bg_model,
        'bg_transparent': args.bg_transparent,
        'bg_fill_black': args.bg_fill_black,
        'crop_padding': args.crop_padding,
        'quality': args.quality
    }
    
    # Process video files
    if video_files:
        print(f"\n{'='*70}")
        print(f"🎬 Processing {len(video_files)} video file(s)")
        print(f"{'='*70}\n")
        
        for idx, video_file in enumerate(video_files, 1):
            print(f"\n{'='*70}")
            print(f"📹 Processing video {idx}/{len(video_files)}: {video_file.name}")
            print(f"{'='*70}")
            
            try:
                extract_birds_from_video(
                    video_path=str(video_file),
                    sample_rate=args.sample_rate,
                    **extract_params
                )
            except Exception as e:
                print(_('error_processing', name=video_file.name, error=e))
                print(_('continuing'))
                continue
    
    # Process image files
    if image_files:
        print(f"\n{'='*70}")
        print(f"🖼️  Processing {len(image_files)} image file(s)")
        print(f"{'='*70}\n")
        
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n{'='*70}")
            print(f"📸 Processing image {idx}/{len(image_files)}: {image_file.name}")
            print(f"{'='*70}")
            
            try:
                extract_birds_from_image(
                    image_path=str(image_file),
                    **extract_params
                )
            except Exception as e:
                print(_('error_processing', name=image_file.name, error=e))
                print(_('continuing'))
                continue
    
    print(f"\n{'='*70}")
    print(_('all_videos_processed', path=output_dir))
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
