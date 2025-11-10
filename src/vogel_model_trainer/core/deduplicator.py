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
