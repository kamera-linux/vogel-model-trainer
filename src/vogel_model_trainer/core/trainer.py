#!/usr/bin/env python3
"""
Training script for custom bird species classifier.
Fine-tunes an EfficientNet model on the extracted bird images.
"""

import os
import warnings
import torch
import signal
import sys
from pathlib import Path
from datetime import datetime
import logging

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*pin_memory.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# Reduce transformers logging verbosity
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    RandomRotation,
    RandomAffine,
    GaussianBlur,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
)
import numpy as np
from PIL import Image
import json
import random

# Configuration
DATA_DIR = Path("/home/imme/vogel-training-data/organized")
OUTPUT_DIR = Path("/home/imme/vogel-models")
MODEL_NAME = "google/efficientnet-b0"  # Base model for fine-tuning
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 2e-4
IMAGE_SIZE = 224

def get_species_from_directory(data_dir):
    """Automatically detect species from train directory structure."""
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    # Get all subdirectories (each is a species)
    species = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    
    if not species:
        raise ValueError(f"No species directories found in {train_dir}")
    
    return species

def prepare_model_and_processor(species):
    """Load base model and processor."""
    from vogel_model_trainer.i18n import _
    print(_('train_loading_model', model=MODEL_NAME))
    
    # Create label mappings
    id2label = {i: sp for i, sp in enumerate(species)}
    label2id = {sp: i for i, sp in enumerate(species)}
    
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(species),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    return model, processor

def get_augmentation_transforms(strength="medium", image_size=224):
    """
    Get data augmentation transforms based on strength level.
    
    Args:
        strength: Augmentation intensity - "none", "light", "medium", "heavy"
        image_size: Target image size for RandomResizedCrop
        
    Returns:
        Compose: Torchvision transform composition
    """
    if strength == "none":
        # No augmentation, just basic resize
        return Compose([
            RandomResizedCrop(image_size, scale=(1.0, 1.0)),  # No scale variation
        ])
    
    elif strength == "light":
        # Minimal augmentation for high-quality datasets
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            RandomHorizontalFlip(p=0.3),
            RandomRotation(degrees=5),
        ])
    
    elif strength == "medium":
        # Balanced augmentation (default)
        return Compose([
            RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])
    
    elif strength == "heavy":
        # Aggressive augmentation for small datasets
        return Compose([
            RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}. Choose from: none, light, medium, heavy")

def transform_function(examples, processor, is_training=True, augmentation_strength="medium", image_size=224):
    """Transform function for dataset mapping.
    
    Uses processor for normalization to match inference behavior.
    Only applies data augmentation for training.
    
    Args:
        examples: Batch of examples from dataset
        processor: HuggingFace image processor
        is_training: Whether to apply augmentation
        augmentation_strength: Augmentation intensity level
        image_size: Target image size
    """
    # Define background color palette for random backgrounds
    BACKGROUND_COLORS = [
        (128, 128, 128),  # Gray (neutral)
        (255, 255, 255),  # White
        (0, 0, 0),        # Black
        (200, 200, 200),  # Light gray
        (50, 50, 50),     # Dark gray
        (150, 150, 150),  # Medium gray
        (100, 120, 140),  # Blue-gray
        (140, 130, 120),  # Brown-gray
        (120, 140, 120),  # Green-gray
    ]
    
    # Process each image and collect pixel values
    pixel_values = []
    
    for img in examples["image"]:
        # Handle transparent images with random background augmentation
        if img.mode == "RGBA" and is_training:
            # Random background color for training (helps model ignore background)
            bg_color = random.choice(BACKGROUND_COLORS)
            background = Image.new("RGB", img.size, bg_color)
            # Paste bird using alpha channel as mask
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            # Convert to RGB (for non-transparent or validation/test)
            img = img.convert("RGB")
        
        # Apply data augmentation for training only
        if is_training and augmentation_strength != "none":
            augmentation = get_augmentation_transforms(augmentation_strength, image_size)
            img = augmentation(img)
        
        # Use processor for final preprocessing (resize, normalize)
        # This ensures train/val/test preprocessing is consistent!
        processed = processor(img, return_tensors="pt")
        # Extract the tensor and convert to numpy for HF datasets
        pixel_values.append(processed["pixel_values"][0].numpy())
    
    # Return as numpy arrays (HF datasets will handle conversion)
    examples["pixel_values"] = pixel_values
    return examples

def create_compute_metrics(species):
    """Create compute_metrics function with species list."""
    id2label = {i: sp for i, sp in enumerate(species)}
    
    def compute_metrics(eval_pred):
        """Compute accuracy metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = (predictions == labels).mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for i, sp in id2label.items():
            mask = labels == i
            if mask.sum() > 0:
                per_class_acc[sp] = (predictions[mask] == labels[mask]).mean()
        
        return {
            "accuracy": accuracy,
            **{f"acc_{sp}": acc for sp, acc in per_class_acc.items()}
        }
    
    return compute_metrics

def collate_fn(examples):
    """Custom collate function for DataLoader.
    
    Since we use set_format(type="torch"), the dataset already returns tensors.
    We just need to stack them into batches.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Global flag for graceful shutdown
interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    from vogel_model_trainer.i18n import _
    global interrupted
    if not interrupted:
        interrupted = True
        print("\n\n" + "="*60)
        print(_('train_interrupted'))
        print(_('train_waiting_clean_exit'))
        print("="*60 + "\n")
        print(_('train_force_exit_hint'))
    else:
        print(_('train_force_exit'))
        sys.exit(1)

def train_model(data_dir, output_dir, model_name="google/efficientnet-b0", 
                batch_size=16, num_epochs=50, learning_rate=3e-4,
                early_stopping_patience=5, weight_decay=0.01, warmup_ratio=0.1,
                label_smoothing=0.1, save_total_limit=3, augmentation_strength="medium",
                image_size=224, scheduler="cosine", seed=42, resume_from_checkpoint=None,
                gradient_accumulation_steps=1, mixed_precision="no", push_to_hub=False):
    """
    Train a custom bird species classifier.
    
    Args:
        data_dir: Directory with train/val folders containing species subdirectories
        output_dir: Directory to save the trained model
        model_name: Hugging Face model name (default: google/efficientnet-b0)
        batch_size: Training batch size (default: 16)
        num_epochs: Number of training epochs (default: 50)
        learning_rate: Initial learning rate (default: 3e-4)
        early_stopping_patience: Early stopping patience in epochs (default: 5, 0 to disable)
        weight_decay: Weight decay for regularization (default: 0.01)
        warmup_ratio: Learning rate warmup ratio (default: 0.1)
        label_smoothing: Label smoothing factor (default: 0.1)
        save_total_limit: Maximum number of checkpoints to keep (default: 3)
        augmentation_strength: Data augmentation intensity: none, light, medium, heavy (default: medium)
        image_size: Input image size in pixels (default: 224)
        scheduler: Learning rate scheduler: cosine, linear, constant (default: cosine)
        seed: Random seed for reproducibility (default: 42)
        resume_from_checkpoint: Path to checkpoint to resume training from (default: None)
        gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        mixed_precision: Mixed precision training: no, fp16, bf16 (default: no)
        push_to_hub: Push trained model to HuggingFace Hub (default: False)
    
    Returns:
        str: Path to the final trained model
    """
    from pathlib import Path
    from datetime import datetime
    from vogel_model_trainer.i18n import _
    import random
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    
    print("="*60)
    print(_('train_header'))
    print("="*60)
    print(_('train_ctrl_c_hint'))
    print("="*60)
    
    # Detect species from directory structure
    print(_('train_detecting_species'))
    species = get_species_from_directory(data_dir)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = output_dir / f"bird-classifier-{timestamp}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(_('train_output_dir', path=model_output_dir))
    print(_('train_species', species=', '.join(species)))
    print(_('train_num_classes', count=len(species)))
    
    # Load dataset
    print(_('train_loading_dataset'))
    dataset = load_dataset("imagefolder", data_dir=str(data_dir))
    
    print(_('train_dataset_size', train=len(dataset['train'])))
    print(_('train_val_size', val=len(dataset['validation'])))
    
    # Verify species match dataset labels
    dataset_labels = dataset["train"].features["label"].names
    print(_('train_dataset_labels', labels=dataset_labels))
    print(_('train_detected_species', species=species))
    
    if sorted(dataset_labels) != sorted(species):
        print(_('train_species_mismatch_warning'))
        print(_('train_species_mismatch_details', dataset=sorted(dataset_labels), detected=sorted(species)))
        raise ValueError("Species mismatch - bitte Verzeichnisstruktur prüfen!")
    
    # Use dataset labels (they are already correctly mapped)
    species = dataset_labels
    print(_('train_using_dataset_labels', labels=species))
    
    # Prepare model and processor with correct species order
    model, processor = prepare_model_and_processor(species)
    
    # Apply transforms
    print(_('train_applying_transforms'))
    print(_('train_augmentation_strength', strength=augmentation_strength))
    print(_('train_image_size', size=image_size))
    
    dataset["train"] = dataset["train"].map(
        lambda x: transform_function(x, processor, is_training=True, 
                                    augmentation_strength=augmentation_strength,
                                    image_size=image_size),
        batched=True,
        remove_columns=["image"]
    )
    dataset["validation"] = dataset["validation"].map(
        lambda x: transform_function(x, processor, is_training=False,
                                    augmentation_strength="none",
                                    image_size=image_size),
        batched=True,
        remove_columns=["image"]
    )
    
    # Set format for PyTorch
    dataset["train"].set_format(type="torch", columns=["pixel_values", "label"])
    dataset["validation"].set_format(type="torch", columns=["pixel_values", "label"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir=str(model_output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        push_to_hub=push_to_hub,
        remove_unused_columns=False,
        label_smoothing_factor=label_smoothing,
        lr_scheduler_type=scheduler,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=(mixed_precision == "fp16"),
        bf16=(mixed_precision == "bf16"),
        seed=seed,
    )
    
    # Add early stopping callback if enabled
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(_('train_early_stopping', patience=early_stopping_patience))
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
        compute_metrics=create_compute_metrics(species),
        data_collator=collate_fn,
        callbacks=callbacks,
    )
    
    # Train
    print(_('train_starting'))
    print(_('train_batch_size', size=batch_size))
    print(_('train_learning_rate', rate=learning_rate))
    print(_('train_epochs', epochs=num_epochs))
    print(_('train_weight_decay', decay=weight_decay))
    print(_('train_warmup_ratio', ratio=warmup_ratio))
    print(_('train_scheduler', scheduler=scheduler))
    if gradient_accumulation_steps > 1:
        print(_('train_gradient_accumulation', steps=gradient_accumulation_steps))
    if mixed_precision != "no":
        print(_('train_mixed_precision', precision=mixed_precision))
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    final_model_path = model_output_dir / "final"
    print(_('train_saving_model', path=final_model_path))
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))
    
    # Save config
    import json
    config = {
        "model_name": model_name,
        "species": species,
        "num_classes": len(species),
        "timestamp": timestamp,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate
    }
    with open(final_model_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(_('train_complete'))
    print(_('train_model_saved', path=final_model_path))
    
    return str(final_model_path)


def main():
    """Main training function (for direct script execution)."""
    from vogel_model_trainer.i18n import _
    
    try:
        train_model(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            model_name="google/efficientnet-b0",
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE
        )
    except Exception as e:
        print(_('train_error', error=e))
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
