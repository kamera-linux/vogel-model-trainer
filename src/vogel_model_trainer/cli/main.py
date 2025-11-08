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


def extract_command(args):
    """Execute the extract command."""
    from vogel_model_trainer.core import extractor
    
    print(f"� Extracting birds from: {args.video}")
    print(f"📁 Output folder: {args.folder}")
    
    if args.bird:
        print(f"🐦 Species: {args.bird}")
    if args.species_model:
        print(f"🤖 Using species classifier: {args.species_model}")
    
    # Call the extraction function with all parameters
    extractor.extract_birds_from_video(
        video_pattern=args.video,
        base_folder=args.folder,
        bird_name=args.bird,
        species_model_path=args.species_model,
        resize=(not args.no_resize),
        detection_model_path=args.detection_model,
        confidence_threshold=args.threshold,
        sample_rate=args.sample_rate,
        recursive=args.recursive
    )


def organize_command(args):
    """Execute the organize command."""
    from vogel_model_trainer.core import organizer
    
    print(f"📊 Organizing dataset: {args.source}")
    print(f"📁 Output directory: {args.output}")
    
    # Call the organization function
    organizer.organize_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio
    )


def train_command(args):
    """Execute the train command."""
    from vogel_model_trainer.core import trainer
    
    print(f"🎓 Training model on dataset: {args.data}")
    print(f"📁 Output directory: {args.output}")
    
    # Call the training function
    trainer.train_model(
        data_dir=args.data,
        output_dir=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )


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
        version="%(prog)s 0.1.1"
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
        "--no-resize",
        action="store_true",
        help="Keep original image size instead of resizing to 224x224px"
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
        "--sample-rate",
        type=int,
        default=3,
        help="Analyze every Nth frame (default: 3)"
    )
    extract_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search directories recursively for video files"
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
