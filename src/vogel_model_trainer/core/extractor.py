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

# Import i18n for translations
from vogel_model_trainer.i18n import _

# Default configuration (can be overridden via command line)
DEFAULT_THRESHOLD = 0.5  # Higher threshold for better quality birds
DEFAULT_SAMPLE_RATE = 3  # Check more frames for better coverage
DEFAULT_MODEL = "yolov8n.pt"
TARGET_IMAGE_SIZE = 224  # Optimal size for EfficientNet-B0 training

def extract_birds_from_video(video_path, output_dir, bird_species=None, 
                             detection_model=None, species_model=None,
                             threshold=None, sample_rate=None, target_image_size=224,
                             species_threshold=None, target_class=14,
                             max_detections=10, min_box_size=50, max_box_size=800,
                             quality=95, skip_blurry=False,
                             deduplicate=False, similarity_threshold=5):
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
    """
    # Use defaults if not specified
    detection_model = detection_model or DEFAULT_MODEL
    threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
    sample_rate = sample_rate if sample_rate is not None else DEFAULT_SAMPLE_RATE
    resize_to_target = (target_image_size > 0)  # If 0, keep original size

        # Determine if we should resize
    resize_to_target = (target_image_size > 0)
    
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
    species_counts = {}
    frame_num = 0
    
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
                        
                        # Crop bird
                        bird_crop = frame[y1:y2, x1:x2]
                        
                        if bird_crop.size > 0:
                            # Skip blurry images if requested
                            if skip_blurry:
                                blur_score = cv2.Laplacian(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                                if blur_score < 100:  # Threshold for blur detection
                                    continue
                            
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
                            
                            # Filename: session_id + unique_id + metadata
                            if species_name and species_model:
                                filename = f"{session_id}_{unique_id}_f{frame_num:06d}_det{conf:.2f}_cls{species_conf:.2f}.jpg"
                            else:
                                filename = f"{session_id}_{unique_id}_f{frame_num:06d}_c{conf:.2f}.jpg"
                            
                            save_path = save_dir / filename
                            
                            # Resize to target size for optimal training
                            if resize_to_target:
                                if not deduplicate:  # Only create if not already created for dedup check
                                    bird_pil = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB))
                                # Resize maintaining aspect ratio with padding (better quality than distortion)
                                bird_pil.thumbnail((target_image_size, target_image_size), Image.Resampling.LANCZOS)
                                
                                # Create square image with padding
                                new_img = Image.new('RGB', (target_image_size, target_image_size), (0, 0, 0))
                                # Center the image
                                x_offset = (target_image_size - bird_pil.width) // 2
                                y_offset = (target_image_size - bird_pil.height) // 2
                                new_img.paste(bird_pil, (x_offset, y_offset))
                                
                                # Save with PIL (better quality)
                                new_img.save(str(save_path), 'JPEG', quality=quality)
                                
                                # Add to hash cache if deduplication is enabled
                                if deduplicate:
                                    # Recompute hash for saved image (in case resizing changed it)
                                    saved_hash = imagehash.phash(new_img)
                                    hash_cache[str(save_path)] = saved_hash
                            else:
                                # Save original size with OpenCV
                                cv2.imwrite(str(save_path), bird_crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                                
                                # Add to hash cache if deduplication is enabled
                                if deduplicate:
                                    if not 'bird_pil' in locals():  # Only create if not already created
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


def main():
    parser = argparse.ArgumentParser(
        description='Extract bird crops from videos for training data collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard mode: Extract all birds to one directory
  python extract_birds.py video.mp4 --folder training_data/

  # Manual mode: Specify bird species (creates subdirectory)
  python extract_birds.py rotkehlchen_video.mp4 --folder data/ --bird rotkehlchen

Examples:
  # Single video file
  python extract_birds.py video.mp4 --folder data/ --bird rotkehlchen

  # Multiple videos with wildcards
  python extract_birds.py "~/Videos/*.mp4" --folder data/ --species-model ~/vogel-models/bird-classifier-*/final/

  # Recursive directory search
  python extract_birds.py "~/Videos/**/*.mp4" --folder data/ --bird kohlmeise

  # Auto-sort mode with wildcard
  python extract_birds.py "/media/videos/vogelhaus_*.mp4" --folder data/ --species-model ~/vogel-models/bird-classifier-*/final/

  # Extract with custom detection parameters
  python extract_birds.py video.mp4 --folder data/ --bird kohlmeise --threshold 0.6 --sample-rate 2
  
  # Extract in original size (no resize)
  python extract_birds.py video.mp4 --folder data/ --bird rotkehlchen --no-resize
        """
    )
    
    parser.add_argument('video', help='Video file, directory, or glob pattern (e.g., "*.mp4", "~/Videos/**/*.mp4")')
    parser.add_argument('--folder', required=True, help='Base directory for extracted bird images')
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
                       help=f'Analyze every Nth frame (default: {DEFAULT_SAMPLE_RATE})')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search directories recursively for video files')
    
    # Keep -o as alias for backwards compatibility
    parser.add_argument('-o', '--output', dest='folder_alias', help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle backwards compatibility for -o
    output_dir = args.folder or args.folder_alias
    if not output_dir:
        parser.error("--folder is required")
    
    # Collect video files
    video_files = []
    video_path = Path(args.video).expanduser()
    
    # Check if it's a glob pattern
    if '*' in args.video or '?' in args.video:
        # Expand glob pattern
        video_files = [Path(p) for p in glob.glob(str(video_path), recursive=args.recursive)]
    elif video_path.is_dir():
        # Directory - search for video files
        if args.recursive:
            patterns = ['**/*.mp4', '**/*.avi', '**/*.mov', '**/*.mkv', '**/*.MP4']
        else:
            patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']
        
        for pattern in patterns:
            video_files.extend(video_path.glob(pattern))
    elif video_path.is_file():
        # Single file
        video_files = [video_path]
    else:
        print(f"❌ Video file/directory not found: {args.video}")
        sys.exit(1)
    
    # Remove duplicates and sort
    video_files = sorted(set(video_files))
    
    if not video_files:
        print(f"❌ No video files found matching: {args.video}")
        sys.exit(1)
    
    # Show what will be processed
    print(f"🎬 Found {len(video_files)} video file(s) to process:")
    for i, vf in enumerate(video_files[:10], 1):  # Show first 10
        print(f"   {i}. {vf.name}")
    if len(video_files) > 10:
        print(f"   ... and {len(video_files) - 10} more")
    print()
    
    # Validate that only one sorting method is used
    if args.bird and args.species_model:
        print("⚠️  Warning: Both --bird and --species-model specified. Using auto-classification.")
    
    # Process each video file
    total_birds = 0
    for idx, video_file in enumerate(video_files, 1):
        print(f"\n{'='*70}")
        print(f"📹 Processing video {idx}/{len(video_files)}: {video_file.name}")
        print(f"{'='*70}")
        
        try:
            extract_birds_from_video(
                video_path=str(video_file),
                output_dir=output_dir,
                bird_species=args.bird,
                detection_model=args.detection_model,
                species_model=args.species_model,
                threshold=args.threshold,
                sample_rate=args.sample_rate,
                resize_to_target=not args.no_resize,
                species_threshold=args.species_threshold
            )
        except Exception as e:
            print(_('error_processing', name=video_file.name, error=e))
            print(_('continuing'))
            continue
    
    print(_('all_videos_processed', path=output_dir))


if __name__ == '__main__':
    main()
