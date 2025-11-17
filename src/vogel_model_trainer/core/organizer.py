#!/usr/bin/env python3
"""
Script to organize extracted bird images into train/val split.
Performs 80/20 split with random shuffling.
Works with both old (species_video*/) and new (species/) directory structures.
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# Import i18n for translations
from vogel_model_trainer.i18n import _

# Default configuration
DEFAULT_SOURCE_DIR = Path("/home/imme/vogel-training-data")
TRAIN_RATIO = 0.8

def collect_images_by_species(source_dir):
    """
    Collect all images grouped by species.
    Supports multiple directory structures:
    - New: source_dir/species/*.jpg (direct species folders)
    - Old: source_dir/species_video*/*.jpg (legacy video folders)
    """
    images_by_species = {}
    
    print(_('scanning_directory', path=source_dir))
    print()
    
    # Find all subdirectories that might contain species images
    for item in source_dir.iterdir():
        if not item.is_dir():
            continue
        
        species_name = None
        images = []
        
        # Check if it's a direct species folder (new format)
        # Support both JPG and PNG files
        jpg_files = list(item.glob("*.jpg"))
        png_files = list(item.glob("*.png"))
        if jpg_files or png_files:
            species_name = item.name
            images = jpg_files + png_files
        
        # Also check for old format: species_video* folders
        if not species_name:
            # Extract species name from folder like "kohlmeise_video20241108"
            for potential_species in ["blaumeise", "kleiber", "kohlmeise", "rotkehlchen", "sumpfmeise", 
                                     "amsel", "buchfink", "erlenzeisig", "feldsperling", "gimpel",
                                     "gruenfink", "haussperling", "kernbeisser", "star", "stieglitz"]:
                if item.name.startswith(potential_species):
                    species_name = potential_species
                    images = list(item.glob("*.jpg")) + list(item.glob("*.png"))
                    break
        
        if species_name and images:
            if species_name not in images_by_species:
                images_by_species[species_name] = []
            images_by_species[species_name].extend(images)
    
    # Print summary
    for species, imgs in sorted(images_by_species.items()):
        print(_('species_found', species=species, count=len(imgs)))
    
    if not images_by_species:
        print(_('no_images_found'))
    
    print()
    return images_by_species

def check_class_balance(images_by_species, max_images_per_class=None, tolerance_percent=15.0):
    """
    Check class balance and enforce limits.
    
    Args:
        images_by_species: Dict mapping species to list of image paths
        max_images_per_class: Maximum images per class (None = no limit, or 100, 200, 300, etc.)
        tolerance_percent: Maximum allowed difference between classes (default: 15%)
    
    Returns:
        tuple: (modified_images_by_species, deleted_files)
    """
    if not images_by_species:
        return images_by_species, []
    
    deleted_files = []
    modified_species = []
    
    # Step 1: Enforce max_images_per_class limit
    if max_images_per_class and max_images_per_class > 0:
        print(_('enforcing_max_images', max=max_images_per_class))
        
        for species, images in list(images_by_species.items()):
            if len(images) > max_images_per_class:
                # Randomly select which images to keep
                random.shuffle(images)
                images_to_keep = images[:max_images_per_class]
                images_to_delete = images[max_images_per_class:]
                
                # Delete excess images
                for img_path in images_to_delete:
                    try:
                        img_path.unlink()
                        deleted_files.append(str(img_path))
                    except Exception as e:
                        print(_('delete_error', filename=img_path.name, error=e))
                
                images_by_species[species] = images_to_keep
                modified_species.append(species)
                print(_('images_deleted', species=species, deleted=len(images_to_delete), before=len(images), after=len(images_to_keep)))
        
        if deleted_files:
            print(_('total_deleted', count=len(deleted_files)))
            print(_('affected_species', species=', '.join(modified_species)))
    
    # Step 2: Check class balance
    if len(images_by_species) < 2:
        return images_by_species, deleted_files
    
    image_counts = {species: len(images) for species, images in images_by_species.items()}
    min_count = min(image_counts.values())
    max_count = max(image_counts.values())
    avg_count = sum(image_counts.values()) / len(image_counts)
    
    # Calculate imbalance percentage
    if min_count > 0:
        imbalance_percent = ((max_count - min_count) / min_count) * 100
    else:
        imbalance_percent = 0
    
    print(_('class_balance_check'))
    print(_('balance_minimum', count=min_count))
    print(_('balance_maximum', count=max_count))
    print(_('balance_average', avg=avg_count))
    print(_('balance_difference', percent=imbalance_percent))
    
    # Check tolerance
    if imbalance_percent > tolerance_percent:
        print(_('balance_error'))
        print(_('balance_max_tolerance', percent=tolerance_percent))
        print(_('balance_current_diff', percent=imbalance_percent))
        print(_('balance_affected_classes'))
        for species, count in sorted(image_counts.items(), key=lambda x: x[1]):
            diff_from_avg = ((count - avg_count) / avg_count) * 100
            print(_('balance_class_item', species=species, count=count, diff=diff_from_avg))
        print(_('balance_recommendation'))
        print(_('balance_rec_collect'))
        print(_('balance_rec_limit', limit=int(min_count * 1.15), tolerance=tolerance_percent))
        raise ValueError(f"Class imbalance {imbalance_percent:.1f}% exceeds tolerance {tolerance_percent}%")
    
    elif imbalance_percent >= tolerance_percent * 0.67:  # Warning at 10% (67% of 15%)
        print(_('balance_warning'))
        print(_('balance_warning_diff', percent=imbalance_percent, tolerance=tolerance_percent))
        print(_('balance_warning_classes'))
        for species, count in sorted(image_counts.items(), key=lambda x: x[1]):
            diff_from_max = ((max_count - count) / max_count) * 100
            if diff_from_max > 5:  # Show classes that differ significantly
                print(_('balance_warning_item', species=species, count=count, diff=diff_from_max))
    else:
        print(_('balance_good', tolerance=tolerance_percent))
    
    return images_by_species, deleted_files

def split_and_copy(images_by_species, output_dir, train_ratio=TRAIN_RATIO):
    """Split images 80/20 and copy to train/val folders."""
    stats = {}
    
    for species, images in images_by_species.items():
        if len(images) == 0:
            print(f"⚠️  {species}: Keine Bilder gefunden, überspringe...")
            continue
        
        # Shuffle images randomly
        random.shuffle(images)
        
        # Calculate split point
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train folder
        train_dir = output_dir / "train" / species
        train_dir.mkdir(parents=True, exist_ok=True)
        for img_path in train_images:
            shutil.copy2(img_path, train_dir / img_path.name)
        
        # Copy to val folder
        val_dir = output_dir / "val" / species
        val_dir.mkdir(parents=True, exist_ok=True)
        for img_path in val_images:
            shutil.copy2(img_path, val_dir / img_path.name)
        
        stats[species] = {
            "total": len(images),
            "train": len(train_images),
            "val": len(val_images)
        }
        
        print(f"✓ {species}: {len(train_images)} train, {len(val_images)} val")
    
    return stats

def print_summary(stats, output_dir):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("Dataset Organisation abgeschlossen")
    print("="*50)
    
    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    total = sum(s["total"] for s in stats.values())
    
    print(f"\nGesamt: {total} Bilder")
    print(f"  Training:   {total_train} ({total_train/total*100:.1f}%)")
    print(f"  Validation: {total_val} ({total_val/total*100:.1f}%)")
    
    print("\nPro Vogelart:")
    for species, s in stats.items():
        print(f"  {species:12s}: {s['total']:3d} gesamt ({s['train']:3d} train, {s['val']:2d} val)")
    
    print(f"\nDataset Ordner: {output_dir}")
    print(f"  Training:   {output_dir}/train/")
    print(f"  Validation: {output_dir}/val/")

def organize_dataset(source_dir, output_dir, train_ratio=TRAIN_RATIO, max_images_per_class=None, tolerance_percent=15.0):
    """
    Organize bird images into train/val split.
    
    Args:
        source_dir: Source directory with species folders (str or Path)
        output_dir: Output directory for organized dataset (str or Path)
        train_ratio: Train/val split ratio (default: 0.8)
        max_images_per_class: Maximum images per class (None, 100, 200, 300, etc.)
        tolerance_percent: Maximum allowed class imbalance (default: 15%)
    
    Returns:
        dict: Statistics about the organized dataset
    """
    from pathlib import Path
    
    source_dir = Path(source_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    print(_('organizing_dataset', path=source_dir))
    print(_('output_dir', path=output_dir))
    print(_('train_ratio', ratio=train_ratio, val=1-train_ratio))
    if max_images_per_class:
        print(f"   🔢 Max images per class: {max_images_per_class}")
    print(f"   ⚖️  Class balance tolerance: {tolerance_percent}%")
    print()
    
    print(_('creating_splits'))
    images_by_species = collect_images_by_species(source_dir)
    
    if not images_by_species:
        raise ValueError("No images found!")
    
    # Check class balance and enforce limits
    images_by_species, deleted_files = check_class_balance(
        images_by_species, 
        max_images_per_class=max_images_per_class,
        tolerance_percent=tolerance_percent
    )
    
    print(_('creating_splits'))
    stats = split_and_copy(images_by_species, output_dir, train_ratio)
    
    print(_('organization_complete'))
    print(_('dataset_summary'))
    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    total = sum(s["total"] for s in stats.values())
    print(_('total_images', count=total))
    print(_('training_images', count=total_train))
    print(_('validation_images', count=total_val))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Organize bird images into train/val split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize images from default directory
  python organize_dataset.py
  
  # Organize images from custom directory
  python organize_dataset.py --source ~/my-birds/ --output ~/my-dataset/
  
  # Custom train/val split (70/30)
  python organize_dataset.py --train-ratio 0.7
        """
    )
    
    parser.add_argument('--source', '-s', type=Path, default=DEFAULT_SOURCE_DIR,
                       help=f'Source directory with species folders (default: {DEFAULT_SOURCE_DIR})')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output directory for organized dataset (default: <source>/organized)')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO,
                       help=f'Train/val split ratio (default: {TRAIN_RATIO})')
    parser.add_argument('--max-images-per-class', type=int, default=None,
                       help='Maximum images per class (e.g., 100, 200, 300). Excess images will be deleted. (default: None = no limit)')
    parser.add_argument('--tolerance', type=float, default=15.0,
                       help='Maximum allowed class imbalance in percent (default: 15.0). Warning at 10%%, error above tolerance.')
    
    args = parser.parse_args()
    
    try:
        organize_dataset(
            source_dir=args.source,
            output_dir=args.output or args.source / "organized",
            train_ratio=args.train_ratio,
            max_images_per_class=args.max_images_per_class,
            tolerance_percent=args.tolerance
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
