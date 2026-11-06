#!/usr/bin/env python3
"""
Training script for custom bird species classifier.
Fine-tunes an EfficientNet model on the extracted bird images.
"""

import inspect
import json
import logging
import os
import random
import signal
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*pin_memory.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# Reduce transformers logging verbosity
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import (  # noqa: E402
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from datasets import load_dataset  # noqa: E402
from torchvision.transforms import (  # noqa: E402
    ColorJitter,
    Compose,
    GaussianBlur,
    RandomAffine,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
)
import numpy as np  # noqa: E402

from vogel_model_trainer.i18n import _  # noqa: E402
from vogel_model_trainer.utils.image import INFERENCE_BACKGROUND, to_rgb  # noqa: E402

# Configuration (defaults for direct script execution)
DATA_DIR = Path.home() / "vogel-training-data" / "organized"
OUTPUT_DIR = Path.home() / "vogel-models"
MODEL_NAME = "google/efficientnet-b0"  # Base model for fine-tuning
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 2e-4
IMAGE_SIZE = 224

# Background colors for transparent training images (helps the model ignore background)
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


def get_species_from_directory(data_dir):
    """Automatically detect species from train directory structure."""
    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")

    # Get all subdirectories (each is a species)
    species = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    if not species:
        raise ValueError(f"No species directories found in {train_dir}")

    return species


def prepare_model_and_processor(species, model_name=MODEL_NAME):
    """Load base model and processor."""
    print(_('train_loading_model', model=model_name))

    # Create label mappings
    id2label = {i: sp for i, sp in enumerate(species)}
    label2id = {sp: i for i, sp in enumerate(species)}

    processor = AutoImageProcessor.from_pretrained(model_name, revision="main")  # nosec B615
    model = AutoModelForImageClassification.from_pretrained(  # nosec B615
        model_name,
        revision="main",
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


def make_transform(processor, is_training=True, augmentation_strength="medium", image_size=224):
    """
    Build an on-the-fly transform for ``Dataset.set_transform``.

    The transform runs every time a sample is loaded, so random augmentation
    differs from epoch to epoch (``Dataset.map`` would apply it only once and
    cache the result).

    Transparent images get a random background color during training and the
    fixed inference background (gray) during validation, matching what
    classify/evaluate/test use.
    """
    augmentation = None
    if is_training and augmentation_strength != "none":
        augmentation = get_augmentation_transforms(augmentation_strength, image_size)

    def transform(batch):
        pixel_values = []
        for img in batch["image"]:
            if is_training and img.mode in ("RGBA", "LA"):
                bg_color = random.choice(BACKGROUND_COLORS)  # nosec B311 - augmentation, not crypto
                img = to_rgb(img, bg_color)
            else:
                img = to_rgb(img, INFERENCE_BACKGROUND)

            if augmentation is not None:
                img = augmentation(img)

            # Processor does final resize + normalization (same as inference)
            processed = processor(img, return_tensors="pt")
            pixel_values.append(processed["pixel_values"][0])

        return {"pixel_values": pixel_values, "label": batch["label"]}

    return transform


def transform_function(examples, processor, is_training=True, augmentation_strength="medium", image_size=224):
    """Backwards-compatible batch transform (applies ``make_transform`` once)."""
    transform = make_transform(processor, is_training, augmentation_strength, image_size)
    examples.update(transform(examples))
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
    """Stack transformed samples into a batch."""
    pixel_values = torch.stack([torch.as_tensor(example["pixel_values"]) for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# Global flag for graceful shutdown
interrupted = False


def signal_handler(sig, frame):
    """Handle Ctrl+C: first press stops after the current step, second press exits."""
    global interrupted
    if not interrupted:
        interrupted = True
        print("\n\n" + "=" * 60)
        print(_('train_interrupted'))
        print(_('train_waiting_clean_exit'))
        print("=" * 60 + "\n")
        print(_('train_force_exit_hint'))
    else:
        print(_('train_force_exit'))
        sys.exit(1)


class GracefulStopCallback(TrainerCallback):
    """Stop training cleanly after the current step once Ctrl+C was pressed."""

    def on_step_end(self, args, state, control, **kwargs):
        if interrupted:
            control.should_training_stop = True
            control.should_save = True
        return control


def _trainer_processor_kwarg(processor):
    """Pass the image processor using the argument name the installed transformers expects."""
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": processor}
    return {"tokenizer": processor}


def train_model(data_dir, output_dir, model_name=MODEL_NAME,
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
        image_size: Crop size used by the augmentation (default: 224). The final
            model input size is defined by the base model's image processor.
        scheduler: Learning rate scheduler: cosine, linear, constant (default: cosine)
        seed: Random seed for reproducibility (default: 42)
        resume_from_checkpoint: Path to checkpoint to resume training from (default: None)
        gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        mixed_precision: Mixed precision training: no, fp16, bf16 (default: no)
        push_to_hub: Push trained model to HuggingFace Hub (default: False)

    Returns:
        str: Path to the final trained model
    """
    global interrupted
    interrupted = False

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    print("=" * 60)
    print(_('train_header'))
    print("=" * 60)
    print(_('train_ctrl_c_hint'))
    print("=" * 60)

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
    dataset = load_dataset("imagefolder", data_dir=str(data_dir))  # nosec B615 - local dataset loading

    if "validation" not in dataset:
        raise ValueError(_('train_no_validation_split', path=data_dir))

    print(_('train_dataset_size', train=len(dataset['train'])))
    print(_('train_val_size', val=len(dataset['validation'])))

    # Verify species match dataset labels
    dataset_labels = dataset["train"].features["label"].names
    print(_('train_dataset_labels', labels=dataset_labels))
    print(_('train_detected_species', species=species))

    if sorted(dataset_labels) != sorted(species):
        print(_('train_species_mismatch_warning'))
        print(_('train_species_mismatch_details', dataset=sorted(dataset_labels), detected=sorted(species)))
        raise ValueError(_('train_species_mismatch_error'))

    # Use dataset labels (they are already correctly mapped)
    species = dataset_labels
    print(_('train_using_dataset_labels', labels=species))

    # Prepare model and processor with correct species order
    model, processor = prepare_model_and_processor(species, model_name)

    # Apply transforms on the fly (augmentation is re-sampled every epoch)
    print(_('train_applying_transforms'))
    print(_('train_augmentation_strength', strength=augmentation_strength))
    print(_('train_image_size', size=image_size))

    dataset["train"].set_transform(make_transform(
        processor, is_training=True, augmentation_strength=augmentation_strength, image_size=image_size))
    dataset["validation"].set_transform(make_transform(
        processor, is_training=False, augmentation_strength="none", image_size=image_size))

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

    # Callbacks
    callbacks = [GracefulStopCallback()]
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(_('train_early_stopping', patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=create_compute_metrics(species),
        data_collator=collate_fn,
        callbacks=callbacks,
        **_trainer_processor_kwarg(processor),
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

    previous_handler = signal.signal(signal.SIGINT, signal_handler)
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    # Save final model
    final_model_path = model_output_dir / "final"
    print(_('train_saving_model', path=final_model_path))
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))

    config = {
        "model_name": model_name,
        "species": species,
        "num_classes": len(species),
        "timestamp": timestamp,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "label_smoothing": label_smoothing,
        "augmentation_strength": augmentation_strength,
        "image_size": image_size,
        "scheduler": scheduler,
        "seed": seed,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "interrupted": interrupted,
    }
    with open(final_model_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if interrupted:
        print(_('train_stopped_early'))
    print(_('train_complete'))
    print(_('train_model_saved', path=final_model_path))

    return str(final_model_path)


def main():
    """Main training function (for direct script execution)."""
    try:
        train_model(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            model_name=MODEL_NAME,
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
    sys.exit(main())
