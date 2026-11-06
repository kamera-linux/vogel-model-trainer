#!/usr/bin/env python3
"""
Command-line interface for vogel-model-trainer.

Provides commands for:
- extract: Extract bird images from videos (with optional manual or auto-sorting)
- organize: Organize dataset into train/val splits
- train: Train a custom bird species classifier
- test: Test and evaluate a trained model
"""

import argparse
import contextlib
import shutil
import sys
from pathlib import Path

from vogel_model_trainer import __version__
from vogel_model_trainer.utils.logging import Tee, capture_output  # noqa: F401  (Tee re-exported)


def extract_command(args):
    """Execute the extract command (video/image extraction or --convert)."""
    from vogel_model_trainer.core import extractor
    return extractor.run(args, args._parser)


def organize_command(args):
    """Execute the organize command."""
    from vogel_model_trainer.core import organizer
    from vogel_model_trainer.i18n import _

    print(_('cli_organizing_dataset', path=args.source))
    print(_('cli_output_directory', path=args.output))

    organizer.organize_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        max_images_per_class=args.max_images_per_class,
        tolerance_percent=args.tolerance,
        split_by=args.split_by,
        seed=args.seed,
        clean=args.clean,
    )
    return 0


def train_command(args):
    """Execute the train command."""
    from vogel_model_trainer.core import trainer
    from vogel_model_trainer.i18n import _

    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(capture_output('train', enabled=args.log))
        except PermissionError:
            print(f"⚠️  {_('log_permission_denied')}", file=sys.stderr)
            print(f"   {_('log_permission_hint')}", file=sys.stderr)
            return 1

        print(_('cli_training_model', path=args.data))
        print(_('cli_output_directory', path=args.output))

        trainer.train_model(
            data_dir=args.data,
            output_dir=args.output,
            model_name=args.model,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            label_smoothing=args.label_smoothing,
            save_total_limit=args.save_total_limit,
            augmentation_strength=args.augmentation_strength,
            image_size=args.image_size,
            scheduler=args.scheduler,
            seed=args.seed,
            resume_from_checkpoint=args.resume_from_checkpoint,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            push_to_hub=args.push_to_hub
        )
    return 0


def test_command(args):
    """Execute the test command."""
    from vogel_model_trainer.core import tester

    if not args.data and not args.image:
        print("❌ Error: use --data <dataset> or --image <file>", file=sys.stderr)
        return 2

    print(f"🧪 Testing model: {args.model}")
    tester.test_model(
        model_path=args.model,
        data_dir=args.data,
        image_path=args.image,
        samples=args.samples,
        seed=args.seed,
    )
    return 0


def deduplicate_command(args):
    """Execute the deduplicate command."""
    from vogel_model_trainer.core import deduplicator

    deduplicator.deduplicate_dataset(
        data_dir=args.data_dir,
        similarity_threshold=args.threshold,
        hash_method=args.method,
        mode=args.mode,
        keep=args.keep,
        recursive=args.recursive
    )
    return 0


def quality_check_command(args):
    """Execute the quality-check command."""
    from vogel_model_trainer.core import deduplicator

    deduplicator.quality_check_dataset(
        data_dir=args.data_dir,
        blur_threshold=args.blur_threshold,
        min_resolution=args.min_resolution,
        min_filesize=args.min_filesize,
        check_brightness=args.check_brightness,
        mode=args.mode,
        recursive=args.recursive
    )
    return 0


def _clean_images(data_dir, image_files, validate, describe, mode, move_dirname, invalid_label, all_valid_msg):
    """
    Shared implementation of clean-gray / clean-transparent.

    Args:
        validate: callable(path) -> result dict with 'valid' and 'reason'
        describe: callable(result) -> detail line for the report
        mode: 'report', 'delete' or 'move'
        move_dirname: folder (inside data_dir) for --mode move; subfolders are preserved
    """
    from vogel_model_trainer.utils.image import unique_path

    invalid_images = []
    for img_path in image_files:
        result = validate(str(img_path))
        if not result['valid']:
            invalid_images.append((img_path, result))
            if mode != 'delete':
                print(f"❌ {img_path.relative_to(data_dir)}")
                print(f"   Reason: {result['reason']}")
                print(f"   {describe(result)}")
                print()

    total_checked = len(image_files)
    print("=" * 70)
    print("📊 Summary:")
    print(f"   Total images checked: {total_checked}")
    print(f"   Valid images: {total_checked - len(invalid_images)}")
    print(f"   {invalid_label}: {len(invalid_images)}")
    print()

    if not invalid_images:
        print(all_valid_msg)
        return 0

    if mode == 'report':
        print("ℹ️  Mode: REPORT only (no files modified)")
        print("💡 Use --mode delete to remove invalid images")
        print(f"💡 Use --mode move to move them to {move_dirname}/")
        return 0

    errors = 0
    if mode == 'delete':
        print(f"🗑️  Deleting {len(invalid_images)} invalid images...")
        for img_path, _result in invalid_images:
            try:
                img_path.unlink()
                print(f"   Deleted: {img_path.name}")
            except OSError as exc:
                errors += 1
                print(f"   ⚠️  {img_path.name}: {exc}")
        print(f"✅ Deleted {len(invalid_images) - errors} images")

    elif mode == 'move':
        move_dir = data_dir / move_dirname
        print(f"📁 Moving {len(invalid_images)} invalid images to {move_dir}...")
        for img_path, _result in invalid_images:
            dest = unique_path(move_dir / img_path.relative_to(data_dir))
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest))
                print(f"   Moved: {img_path.name}")
            except OSError as exc:
                errors += 1
                print(f"   ⚠️  {img_path.name}: {exc}")
        print(f"✅ Moved {len(invalid_images) - errors} images to {move_dir}")

    return 1 if errors else 0


def clean_gray_command(args):
    """Execute the clean-gray command to remove images with too much/little gray background."""
    from vogel_model_trainer.core import tester
    from vogel_model_trainer.utils.image import find_images

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return 1

    image_files = [p for p in find_images(data_dir, recursive=args.recursive)
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not image_files:
        print(f"ℹ️  No image files found in {data_dir}")
        return 0

    print(f"🔍 Checking {len(image_files)} images for gray background...")
    print("📊 Thresholds:")
    print(f"   • Min gray ratio: {args.min_gray*100:.0f}%")
    print(f"   • Max gray ratio: {args.max_gray*100:.0f}%")
    print(f"   • Gray tolerance: ±{args.gray_tolerance}")
    print()

    return _clean_images(
        data_dir, image_files,
        validate=lambda path: tester.validate_gray_background(
            path, min_gray_ratio=args.min_gray, max_gray_ratio=args.max_gray,
            gray_tolerance=args.gray_tolerance),
        describe=lambda r: (f"Gray: {r['gray_ratio']:.1%} ({r['gray_pixels']} pixels), "
                            f"Bird: {r['bird_ratio']:.1%} ({r['bird_pixels']} pixels)"),
        mode=args.mode,
        move_dirname="invalid_gray",
        invalid_label="Invalid (wrong gray ratio)",
        all_valid_msg="✅ All images have valid gray background ratio!",
    )


def clean_transparent_command(args):
    """Execute the clean-transparent command to remove fragmented/incomplete transparent images."""
    from vogel_model_trainer.core import tester
    from vogel_model_trainer.utils.image import find_images

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return 1

    png_files = [p for p in find_images(data_dir, recursive=args.recursive) if p.suffix.lower() == '.png']
    if not png_files:
        print(f"ℹ️  No PNG files found in {data_dir}")
        return 0

    print(f"🔍 Checking {len(png_files)} PNG images...")
    print("📊 Thresholds:")
    print(f"   • Min visible pixels: {args.min_pixels}")
    print(f"   • Max transparency: {args.max_transparency*100:.0f}%")
    print(f"   • Min region size: {args.min_region}")
    print()

    return _clean_images(
        data_dir, png_files,
        validate=lambda path: tester.validate_transparent_image(
            path, min_visible_pixels=args.min_pixels, max_transparency=args.max_transparency,
            min_region_size=args.min_region),
        describe=lambda r: (f"Visible pixels: {r['visible_pixels']}, "
                            f"Transparency: {r['transparency_ratio']:.1%}, "
                            f"Largest region: {r['largest_region']}"),
        mode=args.mode,
        move_dirname="invalid_transparent",
        invalid_label="Invalid/Fragmented",
        all_valid_msg="✅ All images are valid!",
    )


def classify_command(args):
    """Execute the classify command for bulk bird image classification."""
    from vogel_model_trainer.core import classifier

    classifier.classify_images(
        model_path=args.species_model,
        input_dir=args.input,
        sort_output=args.sort_output,
        min_confidence=args.min_confidence,
        csv_report=args.csv_report,
        top_k=args.top_k,
        batch_size=args.batch_size,
        move=args.move,
        delete_source=args.delete_source,
        force=args.force,
        dry_run=args.dry_run,
        recursive=not args.no_recursive
    )
    return 0


def evaluate_command(args):
    """Execute the evaluate command for model evaluation and analytics."""
    from vogel_model_trainer.core import evaluator

    evaluator.evaluate_model(
        model_path=args.species_model,
        test_dir=args.test_dir,
        export_misclassified=args.export_misclassified,
        export_json=args.export_json,
        min_confidence=args.min_confidence,
        batch_size=args.batch_size,
    )
    return 0


def main(argv=None):
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="vogel-trainer",
        description="Train custom bird species classifiers from video footage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard mode: Extract all birds to one directory
  vogel-trainer extract video.mp4 --folder training-data/

  # Manual mode: Specify bird species (creates subdirectory)
  vogel-trainer extract video.mp4 --folder data/ --bird rotkehlchen

  # Multiple videos with wildcards
  vogel-trainer extract "~/Videos/*.mp4" --folder data/ --bird kohlmeise

  # Auto-sort mode with species classifier
  vogel-trainer extract video.mp4 --folder data/ --species-model ~/models/classifier/

  # Recursive directory search
  vogel-trainer extract "~/Videos/" --folder data/ --bird amsel --recursive

  # Organize dataset (80/20 train/val split)
  vogel-trainer organize training-data/ -o organized-data/

  # Train a model
  vogel-trainer train organized-data/ -o models/my-classifier/

  # Test a trained model
  vogel-trainer test models/my-classifier/ -d organized-data/

For more information, visit:
  https://github.com/kamera-linux/vogel-model-trainer
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full tracebacks on errors"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )
    
    # ========== EXTRACT COMMAND ==========
    from vogel_model_trainer.core import extractor
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract bird images from videos or images",
        description="Extract bird crops from videos/images using YOLO detection "
                    "(or process existing images with --convert)"
    )
    extractor.add_extract_arguments(extract_parser)
    extract_parser.set_defaults(func=extract_command, _parser=extract_parser)

    # ========== ORGANIZE COMMAND ==========
    organize_parser = subparsers.add_parser(
        "organize",
        help="Organize dataset into train/val splits",
        description="Split dataset into training and validation sets (default: 80/20)"
    )
    organize_parser.add_argument(
        "source",
        help="Source directory with species subdirectories"
    )
    organize_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for organized dataset"
    )
    from vogel_model_trainer.core import organizer
    organizer.add_organize_arguments(organize_parser)
    organize_parser.set_defaults(func=organize_command)
    
    # ========== TRAIN COMMAND ==========
    train_parser = subparsers.add_parser(
        "train",
        help="Train a custom bird species classifier",
        description="Train an EfficientNet-based classifier on your organized dataset"
    )
    train_parser.add_argument(
        "data",
        help="Path to organized dataset (with train/ and val/ subdirs)"
    )
    train_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for trained model"
    )
    train_parser.add_argument(
        "--model",
        default="google/efficientnet-b0",
        help="Base model for fine-tuning (default: google/efficientnet-b0)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs (default: 5, 0 to disable)"
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization (default: 0.01)"
    )
    train_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Learning rate warmup ratio (default: 0.1)"
    )
    train_parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1, 0 to disable)"
    )
    train_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep (default: 3)"
    )
    train_parser.add_argument(
        "--augmentation-strength",
        choices=["none", "light", "medium", "heavy"],
        default="medium",
        help="Data augmentation intensity (default: medium)"
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size in pixels (default: 224)"
    )
    train_parser.add_argument(
        "--scheduler",
        choices=["cosine", "linear", "constant"],
        default="cosine",
        help="Learning rate scheduler type (default: cosine)"
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    train_parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    train_parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Mixed precision training (default: no)"
    )
    train_parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub"
    )
    train_parser.add_argument(
        "--log",
        action="store_true",
        help="Also write console output to a log file (/var/log/vogel-kamera-linux/YYYY/KWxx/, "
             "falls back to ~/.local/state/vogel-model-trainer/logs/ if not writable)"
    )
    train_parser.set_defaults(func=train_command)
    
    # ========== TEST COMMAND ==========
    test_parser = subparsers.add_parser(
        "test",
        help="Test and evaluate a trained model",
        description="Evaluate model accuracy on validation set or test image"
    )
    test_parser.add_argument(
        "model",
        help="Path to trained model directory"
    )
    test_parser.add_argument(
        "-d", "--data",
        help="Path to organized dataset for validation testing"
    )
    test_parser.add_argument(
        "-i", "--image",
        help="Path to single image for testing"
    )
    test_parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Images per species to test with --data (default: 0 = all images)"
    )
    test_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --samples (default: 42)"
    )
    test_parser.set_defaults(func=test_command)
    
    # ========== DEDUPLICATE COMMAND ==========
    deduplicate_parser = subparsers.add_parser(
        "deduplicate",
        help="Find and remove duplicate images from dataset",
        description="Use perceptual hashing to detect and remove duplicate/similar images"
    )
    deduplicate_parser.add_argument(
        "data_dir",
        help="Directory containing images to deduplicate"
    )
    deduplicate_parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Similarity threshold - Hamming distance 0-64, lower=stricter (default: 5)"
    )
    deduplicate_parser.add_argument(
        "--method",
        choices=["phash", "dhash", "whash", "average_hash"],
        default="phash",
        help="Perceptual hash method (default: phash - recommended)"
    )
    deduplicate_parser.add_argument(
        "--mode",
        choices=["report", "delete", "move"],
        default="report",
        help="Action: report (show only), delete (remove), move (to duplicates/) - default: report"
    )
    deduplicate_parser.add_argument(
        "--keep",
        choices=["first", "largest"],
        default="first",
        help="Which duplicate to keep: first (chronological) or largest (file size) - default: first"
    )
    deduplicate_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search recursively through subdirectories"
    )
    deduplicate_parser.set_defaults(func=deduplicate_command)
    
    # ===== Quality Check Command =====
    quality_check_parser = subparsers.add_parser(
        "quality-check",
        help="Check dataset for low-quality images",
        description="Check images for blur, low resolution, file corruption, and brightness issues"
    )
    quality_check_parser.add_argument(
        "data_dir",
        help="Directory containing images to check"
    )
    quality_check_parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Minimum blur score (Laplacian variance), lower=more blurry (default: 100.0)"
    )
    quality_check_parser.add_argument(
        "--min-resolution",
        type=int,
        default=50,
        help="Minimum image width/height in pixels (default: 50)"
    )
    quality_check_parser.add_argument(
        "--min-filesize",
        type=int,
        default=1024,
        help="Minimum file size in bytes (default: 1024)"
    )
    quality_check_parser.add_argument(
        "--check-brightness",
        action="store_true",
        help="Also check for brightness/contrast issues (too dark or overexposed)"
    )
    quality_check_parser.add_argument(
        "--mode",
        choices=["report", "delete", "move"],
        default="report",
        help="Action: report (show only), delete (remove), move (to low_quality/) - default: report"
    )
    quality_check_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search recursively through subdirectories"
    )
    quality_check_parser.set_defaults(func=quality_check_command)
    
    # ===== Clean Transparent Command =====
    clean_transparent_parser = subparsers.add_parser(
        "clean-transparent",
        help="Remove fragmented/incomplete transparent PNG images",
        description="Detect and remove PNG images with too much transparency or fragmented objects"
    )
    clean_transparent_parser.add_argument(
        "data_dir",
        help="Directory containing transparent PNG images to check"
    )
    clean_transparent_parser.add_argument(
        "--min-pixels",
        type=int,
        default=500,
        help="Minimum number of visible (non-transparent) pixels (default: 500)"
    )
    clean_transparent_parser.add_argument(
        "--max-transparency",
        type=float,
        default=0.95,
        help="Maximum transparency ratio 0.0-1.0 (default: 0.95 = 95%%)"
    )
    clean_transparent_parser.add_argument(
        "--min-region",
        type=int,
        default=100,
        help="Minimum size of largest connected region in pixels (default: 100)"
    )
    clean_transparent_parser.add_argument(
        "--mode",
        choices=["report", "delete", "move"],
        default="report",
        help="Action: report (show only), delete (remove), move (to invalid_transparent/) - default: report"
    )
    clean_transparent_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search recursively through subdirectories"
    )
    clean_transparent_parser.set_defaults(func=clean_transparent_command)
    
    # ===== Clean Gray Command =====
    clean_gray_parser = subparsers.add_parser(
        "clean-gray",
        help="Remove images with incorrect gray background ratio",
        description="Detect and remove images with too much/little gray background (for gray background datasets)"
    )
    clean_gray_parser.add_argument(
        "data_dir",
        help="Directory containing images to check"
    )
    clean_gray_parser.add_argument(
        "--min-gray",
        type=float,
        default=0.05,
        help="Minimum gray background ratio 0.0-1.0 (default: 0.05 = 5%%)"
    )
    clean_gray_parser.add_argument(
        "--max-gray",
        type=float,
        default=0.95,
        help="Maximum gray background ratio 0.0-1.0 (default: 0.95 = 95%%)"
    )
    clean_gray_parser.add_argument(
        "--gray-tolerance",
        type=int,
        default=30,
        help="Tolerance for gray detection: max(R,G,B)-min(R,G,B) threshold (default: 30)"
    )
    clean_gray_parser.add_argument(
        "--mode",
        choices=["report", "delete", "move"],
        default="report",
        help="Action: report (show only), delete (remove), move (to invalid_gray/) - default: report"
    )
    clean_gray_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search recursively through subdirectories"
    )
    clean_gray_parser.set_defaults(func=clean_gray_command)
    
    # ========== CLASSIFY COMMAND ==========
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify bird images in bulk with trained model",
        description="Batch classification of bird images with CSV export and auto-sorting"
    )
    classify_parser.add_argument(
        "--species-model",
        required=True,
        help="Path to trained model directory or Hugging Face model ID (e.g. kamera-linux/german-bird-classifier)"
    )
    classify_parser.add_argument(
        "input",
        help="Input directory containing images to classify"
    )
    classify_parser.add_argument(
        "--sort-output", "-s",
        help="Output directory for sorted images (by species)"
    )
    classify_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold for sorting (0.0-1.0, default: 0.0)"
    )
    classify_parser.add_argument(
        "--csv-report", "-c",
        help="Export classification results to CSV file"
    )
    classify_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=1,
        help="Number of top predictions to report (1-5, default: 1)"
    )
    classify_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Processing batch size (default: 32)"
    )
    classify_parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying when sorting (saves disk space)"
    )
    classify_parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source directory after successful processing (use with caution!)"
    )
    classify_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompts (for scripting)"
    )
    classify_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without moving/copying files"
    )
    classify_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively (only process top-level images)"
    )
    classify_parser.set_defaults(func=classify_command)
    
    # ========== EVALUATE COMMAND ==========
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance with test dataset",
        description="Comprehensive model evaluation with confusion matrix, per-species metrics, and misclassification analysis"
    )
    evaluate_parser.add_argument(
        "--species-model",
        required=True,
        help="Path to trained model directory or Hugging Face model ID"
    )
    evaluate_parser.add_argument(
        "--test-dir",
        required=True,
        help="Test directory with species subfolders (e.g., test/kohlmeise/, test/blaumeise/)"
    )
    evaluate_parser.add_argument(
        "--export-misclassified",
        help="Export misclassified images to CSV file"
    )
    evaluate_parser.add_argument(
        "--export-json",
        help="Export all metrics to JSON file"
    )
    evaluate_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold for evaluation (0.0-1.0, default: 0.0)"
    )
    evaluate_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Inference batch size (default: 32)"
    )
    evaluate_parser.set_defaults(func=evaluate_command)
    
    # Parse arguments and execute command
    args = parser.parse_args(argv)

    try:
        exit_code = args.func(args)
    except KeyboardInterrupt:
        print("\n\n⏸️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        if args.debug:
            raise
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("   (use --debug for the full traceback)", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
