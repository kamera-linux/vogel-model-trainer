#!/usr/bin/env python3
"""
Script to organize extracted bird images into train/val split.

Images extracted from the same video (same session ID in the file name) are
nearly identical neighbouring frames. They are kept together in either train
or val so that validation accuracy is not inflated by near-duplicates
(``split_by='session'``, the default).

Works with both old (species_video*/) and new (species/) directory structures.
Source images are never modified or deleted.
"""

import argparse
import random
import re
import shutil
import sys
from pathlib import Path

from vogel_model_trainer.i18n import _
from vogel_model_trainer.utils.image import is_image_file

# Default configuration
DEFAULT_SOURCE_DIR = Path.home() / "vogel-training-data"
TRAIN_RATIO = 0.8
DEFAULT_SEED = 42

# Extractor file names: <video>_<YYYYmmdd>_<HHMMSS>_<8 hex id>_...
SESSION_PATTERN = re.compile(r'^(?P<session>.+_\d{8}_\d{6})_[0-9a-f]{8}(?:_|$)')
# Legacy folder names like "kohlmeise_video20241108"
LEGACY_FOLDER_PATTERN = re.compile(r'^(?P<species>.+?)_video')


def session_key(image_path):
    """
    Return the grouping key for an image.

    Images created by the extractor share a session ID (video name + timestamp);
    any other image forms its own group.
    """
    match = SESSION_PATTERN.match(Path(image_path).stem)
    if match:
        return match.group('session')
    return Path(image_path).stem


def _list_images(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and is_image_file(p))


def collect_images_by_species(source_dir, exclude_dirs=()):
    """
    Collect all images grouped by species.
    Supports multiple directory structures:
    - New: source_dir/species/*.jpg (direct species folders)
    - Old: source_dir/species_video*/*.jpg (legacy video folders)
    """
    source_dir = Path(source_dir)
    excluded = {Path(d).expanduser().resolve() for d in exclude_dirs}
    images_by_species = {}

    print(_('scanning_directory', path=source_dir))
    print()

    for item in sorted(source_dir.iterdir()):
        if not item.is_dir() or item.resolve() in excluded:
            continue

        images = _list_images(item)
        if not images:
            continue

        legacy = LEGACY_FOLDER_PATTERN.match(item.name)
        species_name = legacy.group('species') if legacy else item.name
        images_by_species.setdefault(species_name, []).extend(images)

    for species, imgs in sorted(images_by_species.items()):
        print(_('species_found', species=species, count=len(imgs)))

    if not images_by_species:
        print(_('no_images_found'))

    print()
    return images_by_species


def limit_images_per_class(images_by_species, max_images_per_class, rng=None):
    """
    Limit every class to ``max_images_per_class`` images.

    Only the selection is reduced - no file is deleted. Whole sessions are
    kept together where possible.

    Returns:
        tuple: (limited images_by_species, number of skipped images)
    """
    rng = rng or random.Random(DEFAULT_SEED)
    if not max_images_per_class or max_images_per_class <= 0:
        return images_by_species, 0

    print(_('enforcing_max_images', max=max_images_per_class))
    limited = {}
    skipped_total = 0
    for species, images in images_by_species.items():
        if len(images) <= max_images_per_class:
            limited[species] = list(images)
            continue
        shuffled = list(images)
        rng.shuffle(shuffled)
        keep = shuffled[:max_images_per_class]
        limited[species] = sorted(keep)
        skipped = len(images) - len(keep)
        skipped_total += skipped
        print(_('images_limited', species=species, skipped=skipped,
                before=len(images), after=len(keep)))

    if skipped_total:
        print(_('total_limited', count=skipped_total))
    return limited, skipped_total


def check_class_balance(images_by_species, max_images_per_class=None, tolerance_percent=15.0, rng=None):
    """
    Check class balance (after applying the optional per-class limit).

    Args:
        images_by_species: Dict mapping species to list of image paths
        max_images_per_class: Maximum images per class (None = no limit)
        tolerance_percent: Maximum allowed difference between classes (default: 15%)

    Returns:
        tuple: (images_by_species to use, list of skipped files - always empty, nothing is deleted)

    Raises:
        ValueError: If the imbalance exceeds ``tolerance_percent``
    """
    if not images_by_species:
        return images_by_species, []

    images_by_species, _skipped = limit_images_per_class(images_by_species, max_images_per_class, rng)

    if len(images_by_species) < 2:
        return images_by_species, []

    image_counts = {species: len(images) for species, images in images_by_species.items()}
    min_count = min(image_counts.values())
    max_count = max(image_counts.values())
    avg_count = sum(image_counts.values()) / len(image_counts)

    imbalance_percent = ((max_count - min_count) / min_count) * 100 if min_count > 0 else 0

    print(_('class_balance_check'))
    print(_('balance_minimum', count=min_count))
    print(_('balance_maximum', count=max_count))
    print(_('balance_average', avg=avg_count))
    print(_('balance_difference', percent=imbalance_percent))

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
        limit = int(min_count * (1 + tolerance_percent / 100))
        print(_('balance_rec_limit', limit=limit, tolerance=tolerance_percent))
        raise ValueError(f"Class imbalance {imbalance_percent:.1f}% exceeds tolerance {tolerance_percent}%")

    elif imbalance_percent >= tolerance_percent * 0.67:
        print(_('balance_warning'))
        print(_('balance_warning_diff', percent=imbalance_percent, tolerance=tolerance_percent))
        print(_('balance_warning_classes'))
        for species, count in sorted(image_counts.items(), key=lambda x: x[1]):
            diff_from_max = ((max_count - count) / max_count) * 100
            if diff_from_max > 5:
                print(_('balance_warning_item', species=species, count=count, diff=diff_from_max))
    else:
        print(_('balance_good', tolerance=tolerance_percent))

    return images_by_species, []


def split_images(images, train_ratio=TRAIN_RATIO, split_by='session', rng=None):
    """
    Split a list of images into train and val.

    With ``split_by='session'`` all images of one session (video) end up in
    the same split. If a class has only a single session, the split falls back
    to per-image splitting so that val is not empty.

    Returns:
        tuple: (train_images, val_images, used_mode)
    """
    rng = rng or random.Random(DEFAULT_SEED)
    images = sorted(images)

    if split_by == 'session':
        groups = {}
        for img in images:
            groups.setdefault(session_key(img), []).append(img)

        if len(groups) >= 2:
            keys = sorted(groups)
            rng.shuffle(keys)
            target_train = len(images) * train_ratio
            train, val = [], []
            for key in keys:
                # Fill train up to the target ratio, rest goes to val; keep at least one group in val
                if len(train) + len(groups[key]) / 2 <= target_train and key != keys[-1]:
                    train.extend(groups[key])
                else:
                    val.extend(groups[key])
            if train and val:
                return train, val, 'session'

    shuffled = list(images)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:], 'image'


def _clear_split_dirs(output_dir):
    for split in ("train", "val"):
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def _has_existing_files(output_dir):
    for split in ("train", "val"):
        split_dir = output_dir / split
        if split_dir.exists() and any(p.is_file() for p in split_dir.rglob('*')):
            return True
    return False


def split_and_copy(images_by_species, output_dir, train_ratio=TRAIN_RATIO, split_by='session', rng=None):
    """Split images per species and copy them to train/val folders."""
    rng = rng or random.Random(DEFAULT_SEED)
    output_dir = Path(output_dir)
    stats = {}

    for species, images in sorted(images_by_species.items()):
        if len(images) == 0:
            print(_('organize_species_empty', species=species))
            continue

        train_images, val_images, used_mode = split_images(images, train_ratio, split_by, rng)
        if split_by == 'session' and used_mode == 'image':
            print(_('organize_single_session_fallback', species=species))

        for split_name, split_images_list in (("train", train_images), ("val", val_images)):
            split_dir = output_dir / split_name / species
            split_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images_list:
                shutil.copy2(img_path, split_dir / img_path.name)

        stats[species] = {
            "total": len(images),
            "train": len(train_images),
            "val": len(val_images),
            "split_by": used_mode,
        }

        print(f"✓ {species}: {len(train_images)} train, {len(val_images)} val")

    return stats


def organize_dataset(source_dir, output_dir, train_ratio=TRAIN_RATIO, max_images_per_class=None,
                     tolerance_percent=15.0, split_by='session', seed=DEFAULT_SEED, clean=False):
    """
    Organize bird images into train/val split.

    Args:
        source_dir: Source directory with species folders (str or Path)
        output_dir: Output directory for organized dataset (str or Path)
        train_ratio: Train/val split ratio (default: 0.8)
        max_images_per_class: Use at most this many images per class (source files are kept)
        tolerance_percent: Maximum allowed class imbalance (default: 15%)
        split_by: 'session' (keep frames of one video together, default) or 'image'
        seed: Random seed for a reproducible split (default: 42)
        clean: Remove existing train/ and val/ folders in output_dir first

    Returns:
        dict: Statistics about the organized dataset
    """
    source_dir = Path(source_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if split_by not in ('session', 'image'):
        raise ValueError(f"split_by must be 'session' or 'image', got {split_by}")

    rng = random.Random(seed)

    print(_('organizing_dataset', path=source_dir))
    print(_('output_dir', path=output_dir))
    print(_('train_ratio', ratio=train_ratio, val=1 - train_ratio))
    print(_('organize_split_mode', mode=split_by, seed=seed))
    if max_images_per_class:
        print(_('organize_max_images', max=max_images_per_class))
    print(_('organize_tolerance', tolerance=tolerance_percent))
    print()

    # Never pick up a previous output inside the source directory as a species
    images_by_species = collect_images_by_species(source_dir, exclude_dirs=[output_dir])

    if not images_by_species:
        raise ValueError("No images found!")

    # Check class balance and apply limit (raises before anything is written)
    images_by_species, _skipped = check_class_balance(
        images_by_species,
        max_images_per_class=max_images_per_class,
        tolerance_percent=tolerance_percent,
        rng=rng,
    )

    if _has_existing_files(output_dir):
        if clean:
            print(_('organize_cleaning_output', path=output_dir))
            _clear_split_dirs(output_dir)
        else:
            print(_('organize_output_not_empty', path=output_dir))

    print(_('creating_splits'))
    stats = split_and_copy(images_by_species, output_dir, train_ratio, split_by, rng)

    print(_('organization_complete'))
    print(_('dataset_summary'))
    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    total = sum(s["total"] for s in stats.values())
    print(_('total_images', count=total))
    print(_('training_images', count=total_train))
    print(_('validation_images', count=total_val))

    return stats


def add_organize_arguments(parser, source_positional=False):
    """Add organize options to an argparse parser (shared with the main CLI)."""
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO,
                        help=f'Training set ratio (default: {TRAIN_RATIO} = 80%%)')
    parser.add_argument('--max-images-per-class', type=int, default=None,
                        help='Use at most N images per class (random selection). '
                             'Source files are NOT deleted. (default: no limit)')
    parser.add_argument('--tolerance', type=float, default=15.0,
                        help='Maximum allowed class imbalance in percent (default: 15.0). '
                             'Warning at 10%%, error above tolerance.')
    parser.add_argument('--split-by', choices=['session', 'image'], default='session',
                        help='session: keep all frames of one video in the same split to avoid '
                             'train/val leakage (default); image: random split per image')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed for a reproducible split (default: {DEFAULT_SEED})')
    parser.add_argument('--clean', action='store_true',
                        help='Remove existing train/ and val/ folders in the output directory first')


def main():
    parser = argparse.ArgumentParser(
        description='Organize bird images into train/val split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize images from default directory
  python organizer.py

  # Organize images from custom directory
  python organizer.py --source ~/my-birds/ --output ~/my-dataset/

  # Custom train/val split (70/30)
  python organizer.py --train-ratio 0.7
        """
    )

    parser.add_argument('--source', '-s', type=Path, default=DEFAULT_SOURCE_DIR,
                        help=f'Source directory with species folders (default: {DEFAULT_SOURCE_DIR})')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output directory for organized dataset (default: <source>/organized)')
    add_organize_arguments(parser)

    args = parser.parse_args()

    try:
        organize_dataset(
            source_dir=args.source,
            output_dir=args.output or args.source / "organized",
            train_ratio=args.train_ratio,
            max_images_per_class=args.max_images_per_class,
            tolerance_percent=args.tolerance,
            split_by=args.split_by,
            seed=args.seed,
            clean=args.clean,
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
