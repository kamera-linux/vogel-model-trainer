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
import sys
from pathlib import Path
from datetime import datetime


class Tee:
    """Redirect output to multiple streams (console and file)."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def extract_command(args):
    """Execute the extract command."""
    from vogel_model_trainer.core import extractor
    from vogel_model_trainer.i18n import _
    
    # Setup logging if requested
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if args.log:
        try:
            # Create log directory structure: /var/log/vogel-kamera-linux/YYYY/KWXX/
            now = datetime.now()
            year = now.strftime('%Y')
            week = now.strftime('%V')
            timestamp = now.strftime('%Y%m%d_%H%M%S')
            
            log_dir = Path(f'/var/log/vogel-kamera-linux/{year}/KW{week}')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file_path = log_dir / f'{timestamp}_extract.log'
            log_file = open(log_file_path, 'w', encoding='utf-8')
            
            # Redirect stdout and stderr to both console and file
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)
            
            print(_('log_file', path=str(log_file_path)))
            
        except PermissionError:
            print(f"⚠️  {_('log_permission_denied')}", file=sys.stderr)
            print(f"   {_('log_permission_hint')}", file=sys.stderr)
            print("   sudo mkdir -p /var/log/vogel-kamera-linux && sudo chown $USER /var/log/vogel-kamera-linux")
            return 1
    
    try:
        print(_('cli_extracting_from', path=args.video))
        print(_('cli_output_folder', path=args.folder))
        
        if args.bird:
            print(_('cli_species', species=args.bird))
        if args.species_model:
            print(_('cli_using_classifier', path=args.species_model))
        
        # Handle glob patterns and recursive search
        import glob
        from pathlib import Path
        
        video_files = []
        if args.recursive:
            # Recursive search
            video_path = Path(args.video)
            if video_path.is_dir():
                for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                    video_files.extend(video_path.rglob(ext))
            else:
                video_files = [args.video]
        else:
            # Glob pattern or single file
            matches = glob.glob(args.video, recursive=False)
            video_files = matches if matches else [args.video]
        
        # Process each video file
        for video_file in video_files:
            print(_('cli_processing_video', path=video_file))
            extractor.extract_birds_from_video(
                video_path=str(video_file),
                output_dir=args.folder,
                bird_species=args.bird,
                detection_model=args.detection_model,
                species_model=args.species_model,
                threshold=args.threshold,
                sample_rate=args.sample_rate,
                target_image_size=args.image_size,
                species_threshold=args.species_threshold,
                max_detections=args.max_detections,
                min_box_size=args.min_box_size,
                max_box_size=args.max_box_size,
                quality=args.quality,
                skip_blurry=args.skip_blurry,
                deduplicate=args.deduplicate,
                similarity_threshold=args.similarity_threshold
            )
    
    finally:
        # Restore original stdout/stderr and close log file
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


def organize_command(args):
    """Execute the organize command."""
    from vogel_model_trainer.core import organizer
    from vogel_model_trainer.i18n import _
    
    print(_('cli_organizing_dataset', path=args.source))
    print(_('cli_output_directory', path=args.output))
    
    # Call the organization function
    organizer.organize_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        max_images_per_class=args.max_images_per_class,
        tolerance_percent=args.tolerance
    )


def train_command(args):
    """Execute the train command."""
    from vogel_model_trainer.core import trainer
    from vogel_model_trainer.i18n import _
    
    # Setup logging if requested
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if args.log:
        try:
            # Create log directory structure: /var/log/vogel-kamera-linux/YYYY/KWXX/
            now = datetime.now()
            year = now.strftime('%Y')
            week = now.strftime('%V')
            timestamp = now.strftime('%Y%m%d_%H%M%S')
            
            log_dir = Path(f'/var/log/vogel-kamera-linux/{year}/KW{week}')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file_path = log_dir / f'{timestamp}_train.log'
            log_file = open(log_file_path, 'w', encoding='utf-8')
            
            # Redirect stdout and stderr to both console and file
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)
            
            print(_('log_file', path=str(log_file_path)))
            
        except PermissionError:
            print(f"⚠️  {_('log_permission_denied')}", file=sys.stderr)
            print(f"   {_('log_permission_hint')}", file=sys.stderr)
            print("   sudo mkdir -p /var/log/vogel-kamera-linux && sudo chown $USER /var/log/vogel-kamera-linux")
            return 1
    
    try:
        print(_('cli_training_model', path=args.data))
        print(_('cli_output_directory', path=args.output))
        
        # Call the training function
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
    
    finally:
        # Restore original stdout/stderr and close log file
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()



def test_command(args):
    """Execute the test command."""
    from vogel_model_trainer.core import tester
    
    print(f"🧪 Testing model: {args.model}")
    
    # Call the testing function
    tester.test_model(
        model_path=args.model,
        data_dir=args.data,
        image_path=args.image
    )


def deduplicate_command(args):
    """Execute the deduplicate command."""
    from vogel_model_trainer.core import deduplicator
    
    # Run deduplication
    stats = deduplicator.deduplicate_dataset(
        data_dir=args.data_dir,
        similarity_threshold=args.threshold,
        hash_method=args.method,
        mode=args.mode,
        keep=args.keep,
        recursive=args.recursive
    )
    
    return 0


def main():
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
        version="%(prog)s 0.1.5"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )
    
    # ========== EXTRACT COMMAND ==========
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract bird images from videos",
        description="Extract bird crops from videos using YOLO detection"
    )
    extract_parser.add_argument(
        "video",
        help="Video file, directory, or glob pattern (e.g., '*.mp4', '~/Videos/**/*.mp4')"
    )
    extract_parser.add_argument(
        "--folder",
        required=True,
        help="Base directory for extracted bird images"
    )
    extract_parser.add_argument(
        "--bird",
        help="Manual bird species name (e.g., rotkehlchen, kohlmeise). Creates subdirectory."
    )
    extract_parser.add_argument(
        "--species-model",
        help="Path to custom species classifier for automatic sorting"
    )
    extract_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size in pixels (default: 224, use 0 for original size)"
    )
    extract_parser.add_argument(
        "--detection-model",
        default="yolov8n.pt",
        help="YOLO detection model path (default: yolov8n.pt)"
    )
    extract_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5 for high quality)"
    )
    extract_parser.add_argument(
        "--species-threshold",
        type=float,
        default=None,
        help="Minimum confidence for species classification (e.g., 0.85 for 85%%). Only exports birds with confidence >= this value."
    )
    extract_parser.add_argument(
        "--sample-rate",
        type=int,
        default=3,
        help="Analyze every Nth frame (default: 3)"
    )
    extract_parser.add_argument(
        "--max-detections",
        type=int,
        default=10,
        help="Maximum number of bird detections per frame (default: 10)"
    )
    extract_parser.add_argument(
        "--min-box-size",
        type=int,
        default=50,
        help="Minimum bounding box size in pixels (default: 50)"
    )
    extract_parser.add_argument(
        "--max-box-size",
        type=int,
        default=800,
        help="Maximum bounding box size in pixels (default: 800)"
    )
    extract_parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for saved images 1-100 (default: 95)"
    )
    extract_parser.add_argument(
        "--skip-blurry",
        action="store_true",
        help="Skip blurry/out-of-focus images (experimental)"
    )
    extract_parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Skip duplicate/similar images using perceptual hashing"
    )
    extract_parser.add_argument(
        "--similarity-threshold",
        type=int,
        default=5,
        help="Similarity threshold for duplicates - Hamming distance 0-64, lower=stricter (default: 5)"
    )
    extract_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search directories recursively for video files"
    )
    extract_parser.add_argument(
        "--log",
        action="store_true",
        help="Save console output to log file in /var/log/vogel-kamera-linux/YYYY/KWXX/"
    )
    extract_parser.set_defaults(func=extract_command)
    
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
    organize_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8 = 80%%)"
    )
    organize_parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Maximum images per class (e.g., 100, 200, 300). Excess images will be deleted."
    )
    organize_parser.add_argument(
        "--tolerance",
        type=float,
        default=15.0,
        help="Maximum allowed class imbalance in percent (default: 15.0)"
    )
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
        help="Save console output to log file in /var/log/vogel-kamera-linux/YYYY/KWXX/"
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
    
    # Parse arguments and execute command
    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n⏸️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
