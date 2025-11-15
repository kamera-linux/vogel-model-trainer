#!/usr/bin/env python3
"""
Deduplication module for removing duplicate images from datasets.
Uses perceptual hashing to find visually similar images.
"""

import imagehash
from PIL import Image
from pathlib import Path
from collections import defaultdict
import shutil
import cv2
import numpy as np
from vogel_model_trainer.i18n import _


def compute_image_hash(image_path, hash_method='phash'):
    """
    Compute perceptual hash for an image.
    
    Args:
        image_path: Path to image file
        hash_method: Hash method - 'phash', 'dhash', 'whash', or 'average_hash'
    
    Returns:
        ImageHash object or None if error
    """
    try:
        img = Image.open(image_path)
        
        if hash_method == 'phash':
            return imagehash.phash(img)
        elif hash_method == 'dhash':
            return imagehash.dhash(img)
        elif hash_method == 'whash':
            return imagehash.whash(img)
        elif hash_method == 'average_hash':
            return imagehash.average_hash(img)
        else:
            return imagehash.phash(img)  # Default
    except Exception as e:
        print(f"⚠️  Error hashing {image_path}: {e}")
        return None


def find_duplicates(data_dir, similarity_threshold=5, hash_method='phash', recursive=True):
    """
    Find duplicate images in a directory.
    
    Args:
        data_dir: Directory to scan
        similarity_threshold: Hamming distance threshold (0-64, lower = more similar)
        hash_method: Hash method to use
        recursive: Search recursively
    
    Returns:
        dict: {original_image: [duplicate1, duplicate2, ...]}
    """
    data_dir = Path(data_dir)
    
    # Collect all image files
    if recursive:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(data_dir.rglob(ext))
    else:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(data_dir.glob(ext))
    
    print(f"🔍 {_('dedup_scanning', count=len(image_files))}")
    
    # Compute hashes
    hashes = {}
    for i, img_path in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"   ⏳ {_('dedup_progress', current=i+1, total=len(image_files))}")
        
        img_hash = compute_image_hash(img_path, hash_method)
        if img_hash:
            hashes[img_path] = img_hash
    
    print(f"✅ {_('dedup_hashed', count=len(hashes))}")
    
    # Find duplicates
    duplicates = defaultdict(list)
    processed = set()
    
    hash_list = list(hashes.items())
    for i, (path1, hash1) in enumerate(hash_list):
        if path1 in processed:
            continue
        
        for path2, hash2 in hash_list[i+1:]:
            if path2 in processed:
                continue
            
            distance = hash1 - hash2
            if distance <= similarity_threshold:
                duplicates[path1].append(path2)
                processed.add(path2)
    
    return duplicates


def deduplicate_dataset(data_dir, similarity_threshold=5, hash_method='phash', 
                        mode='report', keep='first', recursive=True):
    """
    Remove duplicate images from dataset.
    
    Args:
        data_dir: Directory to deduplicate
        similarity_threshold: Hamming distance threshold (0-64)
        hash_method: Hash method - 'phash' (recommended), 'dhash', 'whash', 'average_hash'
        mode: 'report' (show only), 'delete' (remove), 'move' (move to duplicates/)
        keep: 'first' or 'largest' (which duplicate to keep)
        recursive: Search recursively through subdirectories
    
    Returns:
        dict: Statistics about deduplication
    """
    data_dir = Path(data_dir)
    
    print("="*70)
    print(_('dedup_header'))
    print("="*70)
    print(f"📁 {_('dedup_directory', path=data_dir)}")
    print(f"🎯 {_('dedup_threshold', threshold=similarity_threshold)}")
    print(f"🔍 {_('dedup_method', method=hash_method)}")
    print(f"⚙️  {_('dedup_mode', mode=mode)}")
    print(f"📦 {_('dedup_recursive', recursive='Yes' if recursive else 'No')}")
    print("="*70)
    
    # Find duplicates
    duplicates = find_duplicates(data_dir, similarity_threshold, hash_method, recursive)
    
    if not duplicates:
        print(f"\n✅ {_('dedup_no_duplicates')}")
        return {'total_images': 0, 'duplicate_groups': 0, 'duplicates_found': 0}
    
    print(f"\n🔍 {_('dedup_found_groups', count=len(duplicates))}")
    
    total_duplicates = sum(len(dups) for dups in duplicates.values())
    print(f"📊 {_('dedup_found_total', count=total_duplicates)}")
    
    # Show duplicate groups
    if mode == 'report':
        print(f"\n📋 {_('dedup_report_header')}")
        for i, (original, dups) in enumerate(duplicates.items(), 1):
            print(f"\n   Group {i} ({len(dups) + 1} images):")
            print(f"      ✅ Keep: {original.name}")
            for dup in dups:
                print(f"      ❌ Duplicate: {dup.name}")
        
        print(f"\n💡 {_('dedup_hint_delete')}")
        print(f"   vogel-trainer deduplicate {data_dir} --mode delete")
        
        return {
            'total_images': len(duplicates) + total_duplicates,
            'duplicate_groups': len(duplicates),
            'duplicates_found': total_duplicates
        }
    
    # Process duplicates (delete or move)
    duplicates_dir = data_dir / "duplicates"
    if mode == 'move':
        duplicates_dir.mkdir(exist_ok=True)
        print(f"\n📁 {_('dedup_move_to', path=duplicates_dir)}")
    
    deleted_count = 0
    moved_count = 0
    
    for original, dups in duplicates.items():
        # Determine which file to keep
        if keep == 'largest':
            all_files = [original] + dups
            sizes = [(f, f.stat().st_size) for f in all_files]
            keep_file = max(sizes, key=lambda x: x[1])[0]
            delete_files = [f for f in all_files if f != keep_file]
        else:  # keep == 'first'
            keep_file = original
            delete_files = dups
        
        for dup_path in delete_files:
            try:
                if mode == 'delete':
                    dup_path.unlink()
                    deleted_count += 1
                    print(f"   ❌ Deleted: {dup_path.name}")
                elif mode == 'move':
                    # Preserve directory structure
                    rel_path = dup_path.relative_to(data_dir)
                    target_path = duplicates_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dup_path), str(target_path))
                    moved_count += 1
                    print(f"   📦 Moved: {dup_path.name}")
            except Exception as e:
                print(f"   ⚠️  Error processing {dup_path.name}: {e}")
    
    print("\n" + "="*70)
    if mode == 'delete':
        print(f"✅ {_('dedup_deleted', count=deleted_count)}")
    elif mode == 'move':
        print(f"✅ {_('dedup_moved', count=moved_count)}")
    print("="*70)
    
    return {
        'total_images': len(duplicates) + total_duplicates,
        'duplicate_groups': len(duplicates),
        'duplicates_found': total_duplicates,
        'deleted': deleted_count if mode == 'delete' else 0,
        'moved': moved_count if mode == 'move' else 0
    }


def is_duplicate_of_existing(new_image_path, existing_hashes, similarity_threshold=5, hash_method='phash'):
    """
    Check if a new image is a duplicate of any existing image.
    
    Args:
        new_image_path: Path to new image
        existing_hashes: Dict of {path: hash} for existing images
        similarity_threshold: Hamming distance threshold
        hash_method: Hash method to use
    
    Returns:
        tuple: (is_duplicate: bool, similar_to: Path or None, distance: int or None)
    """
    new_hash = compute_image_hash(new_image_path, hash_method)
    if not new_hash:
        return False, None, None
    
    for existing_path, existing_hash in existing_hashes.items():
        distance = new_hash - existing_hash
        if distance <= similarity_threshold:
            return True, existing_path, distance
    
    return False, None, None


def check_image_quality(image_path, blur_threshold=100.0, min_resolution=50, min_filesize=1024, check_brightness=False):
    """
    Check image quality based on multiple criteria.
    
    Args:
        image_path: Path to image file
        blur_threshold: Minimum Laplacian variance for sharpness (lower = more blurry)
        min_resolution: Minimum width/height in pixels
        min_filesize: Minimum file size in bytes
        check_brightness: Whether to check brightness/contrast
    
    Returns:
        dict: {
            'is_valid': bool,
            'issues': list of str (reasons why invalid),
            'blur_score': float or None,
            'resolution': tuple (width, height) or None,
            'filesize': int,
            'brightness': float or None
        }
    """
    issues = []
    result = {
        'is_valid': True,
        'issues': [],
        'blur_score': None,
        'resolution': None,
        'filesize': 0,
        'brightness': None
    }
    
    try:
        # Check file size
        filesize = image_path.stat().st_size
        result['filesize'] = filesize
        if filesize < min_filesize:
            issues.append(_('quality_filesize_too_small', size=filesize, min_size=min_filesize))
        
        # Try to open image (check readability)
        try:
            img_pil = Image.open(image_path)
            width, height = img_pil.size
            result['resolution'] = (width, height)
            
            # Check resolution
            if width < min_resolution or height < min_resolution:
                issues.append(_('quality_resolution_too_small', width=width, height=height, min_res=min_resolution))
            
            # Check blur with OpenCV
            img_cv = cv2.imread(str(image_path))
            if img_cv is not None:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                result['blur_score'] = laplacian_var
                
                if laplacian_var < blur_threshold:
                    issues.append(_('quality_image_blurry', score=laplacian_var, threshold=blur_threshold))
                
                # Check brightness/contrast if requested
                if check_brightness:
                    mean_brightness = np.mean(gray)
                    result['brightness'] = mean_brightness
                    
                    # Too dark (mean < 30)
                    if mean_brightness < 30:
                        issues.append(_('quality_image_too_dark', brightness=mean_brightness))
                    # Too bright/overexposed (mean > 225)
                    elif mean_brightness > 225:
                        issues.append(_('quality_image_too_bright', brightness=mean_brightness))
            else:
                issues.append(_('quality_cannot_read_opencv'))
                
        except Exception as e:
            issues.append(_('quality_cannot_open_image', error=str(e)))
    
    except Exception as e:
        issues.append(_('quality_file_error', error=str(e)))
    
    result['issues'] = issues
    result['is_valid'] = len(issues) == 0
    
    return result


def find_low_quality_images(data_dir, blur_threshold=100.0, min_resolution=50, min_filesize=1024, 
                           check_brightness=False, recursive=True):
    """
    Find low-quality images in a directory.
    
    Args:
        data_dir: Directory to scan
        blur_threshold: Minimum blur score
        min_resolution: Minimum image resolution
        min_filesize: Minimum file size in bytes
        check_brightness: Check for brightness issues
        recursive: Search subdirectories
    
    Returns:
        dict: {image_path: quality_result}
    """
    data_dir = Path(data_dir)
    low_quality = {}
    
    # Find all image files
    pattern = "**/*" if recursive else "*"
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.glob(f"{pattern}{ext}"))
        image_files.extend(data_dir.glob(f"{pattern}{ext.upper()}"))
    
    print(f"🔍 {_('quality_scanning', count=len(image_files))}")
    print("="*70)
    
    for idx, img_path in enumerate(image_files, 1):
        if idx % 100 == 0:
            print(f"   {_('quality_progress', current=idx, total=len(image_files))}")
        
        quality_result = check_image_quality(
            img_path, 
            blur_threshold=blur_threshold,
            min_resolution=min_resolution,
            min_filesize=min_filesize,
            check_brightness=check_brightness
        )
        
        if not quality_result['is_valid']:
            low_quality[img_path] = quality_result
    
    return low_quality


def quality_check_dataset(data_dir, blur_threshold=100.0, min_resolution=50, min_filesize=1024,
                         check_brightness=False, mode='report', recursive=True):
    """
    Check dataset for low-quality images and optionally delete/move them.
    
    Args:
        data_dir: Directory to check
        blur_threshold: Minimum blur score (Laplacian variance)
        min_resolution: Minimum image width/height in pixels
        min_filesize: Minimum file size in bytes
        check_brightness: Check for brightness/contrast issues
        mode: 'report', 'delete', or 'move'
        recursive: Search subdirectories
    
    Returns:
        dict: Statistics about the operation
    """
    data_dir = Path(data_dir)
    
    print("="*70)
    print(f"🔍 {_('quality_check_header')}")
    print(f"   {_('quality_check_directory', directory=data_dir)}")
    print(f"   {_('quality_blur_threshold', threshold=blur_threshold)}")
    print(f"   {_('quality_min_resolution', resolution=min_resolution)}")
    print(f"   {_('quality_min_filesize', size=min_filesize)}")
    if check_brightness:
        print(f"   {_('quality_brightness_check_enabled')}")
    print(f"   {_('quality_mode', mode=mode)}")
    print("="*70)
    
    # Find low-quality images
    low_quality = find_low_quality_images(
        data_dir,
        blur_threshold=blur_threshold,
        min_resolution=min_resolution,
        min_filesize=min_filesize,
        check_brightness=check_brightness,
        recursive=recursive
    )
    
    print("="*70)
    print(f"📊 {_('quality_results')}")
    print(f"   {_('quality_total_issues', count=len(low_quality))}")
    
    if len(low_quality) == 0:
        print(f"\n✅ {_('quality_no_issues')}")
        print("="*70)
        return {
            'total_checked': 0,
            'low_quality_found': 0,
            'deleted': 0,
            'moved': 0
        }
    
    # Group by issue type
    issue_counts = defaultdict(int)
    for result in low_quality.values():
        for issue in result['issues']:
            # Extract issue type (first part before ':')
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_counts[issue_type] += 1
    
    print(f"\n   {_('quality_issues_breakdown')}:")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     • {issue_type}: {count}")
    
    # Report mode: show details
    if mode == 'report':
        print(f"\n📋 {_('quality_detailed_report')}:")
        for idx, (img_path, result) in enumerate(low_quality.items(), 1):
            print(f"\n   [{idx}] {img_path.name}")
            print(f"       {_('quality_path')}: {img_path.parent}")
            for issue in result['issues']:
                print(f"       ❌ {issue}")
            if result['blur_score'] is not None:
                print(f"       {_('quality_blur_score')}: {result['blur_score']:.2f}")
            if result['resolution']:
                print(f"       {_('quality_resolution')}: {result['resolution'][0]}x{result['resolution'][1]}")
            if result['filesize']:
                print(f"       {_('quality_filesize')}: {result['filesize']} bytes")
    
    deleted_count = 0
    moved_count = 0
    
    # Delete mode
    if mode == 'delete':
        print(f"\n🗑️  {_('quality_deleting')}")
        for img_path in low_quality.keys():
            try:
                img_path.unlink()
                deleted_count += 1
                print(f"   ✓ {_('quality_deleted_file', file=img_path.name)}")
            except Exception as e:
                print(f"   ⚠️  {_('quality_delete_error', file=img_path.name, error=str(e))}")
    
    # Move mode
    elif mode == 'move':
        low_quality_dir = data_dir / 'low_quality'
        low_quality_dir.mkdir(exist_ok=True)
        print(f"\n📦 {_('quality_moving', directory=low_quality_dir)}")
        
        for img_path in low_quality.keys():
            try:
                # Preserve subdirectory structure
                if recursive and img_path.parent != data_dir:
                    rel_path = img_path.relative_to(data_dir)
                    target_dir = low_quality_dir / rel_path.parent
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / img_path.name
                else:
                    target_path = low_quality_dir / img_path.name
                
                # Handle name conflicts
                counter = 1
                original_target = target_path
                while target_path.exists():
                    target_path = original_target.parent / f"{original_target.stem}_{counter}{original_target.suffix}"
                    counter += 1
                
                shutil.move(str(img_path), str(target_path))
                moved_count += 1
                print(f"   ✓ {_('quality_moved_file', file=img_path.name)}")
            except Exception as e:
                print(f"   ⚠️  {_('quality_move_error', file=img_path.name, error=str(e))}")
    
    print("="*70)
    print(f"📊 {_('quality_summary')}")
    print(f"   {_('quality_total_issues', count=len(low_quality))}")
    if mode == 'delete':
        print(f"   ✅ {_('quality_deleted_count', count=deleted_count)}")
    elif mode == 'move':
        print(f"   ✅ {_('quality_moved_count', count=moved_count)}")
    print("="*70)
    
    return {
        'total_checked': len(low_quality),
        'low_quality_found': len(low_quality),
        'deleted': deleted_count,
        'moved': moved_count
    }
