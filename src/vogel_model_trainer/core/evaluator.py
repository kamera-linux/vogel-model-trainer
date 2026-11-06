#!/usr/bin/env python3
"""
Model evaluation and analytics module.

Provides comprehensive evaluation metrics for trained bird classifiers,
including confusion matrix, per-species metrics, and misclassification analysis.
"""

import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from vogel_model_trainer.i18n import _
from vogel_model_trainer.utils.image import find_images, load_rgb

# Suppress harmless Cholesky decomposition warnings from PyTorch optimizer
warnings.filterwarnings('ignore', message='.*Cholesky.*')
warnings.filterwarnings('ignore', message='.*positive-definiteness.*')

def load_model(model_path: str) -> Tuple[AutoModelForImageClassification, AutoImageProcessor, List[str]]:
    """
    Load trained model and processor.
    
    Args:
        model_path: Path to trained model or Hugging Face model ID
        
    Returns:
        Tuple of (model, processor, species_list)
    """
    # Check if it's a local path or Hugging Face model ID
    local_path = Path(model_path).expanduser()
    is_local = local_path.exists()
    
    model_source = str(local_path) if is_local else model_path
    
    print(_('evaluate_loading_model', path=model_source))
    
    try:
        model = AutoModelForImageClassification.from_pretrained(model_source, revision="main")  # nosec B615
        processor = AutoImageProcessor.from_pretrained(model_source, revision="main")  # nosec B615
    except Exception as e:
        raise FileNotFoundError(_('evaluate_model_not_found', path=model_source)) from e
    
    # Get species list
    species = list(model.config.id2label.values())
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(_('evaluate_model_loaded', 
           device="GPU" if torch.cuda.is_available() else "CPU",
           species=len(species)))
    
    return model, processor, species


def get_test_images(test_dir: str) -> Dict[str, List[Path]]:
    """
    Get test images organized by species (ground truth from folder structure).

    Args:
        test_dir: Directory with species subfolders

    Returns:
        Dict mapping species name to list of image paths
    """
    test_path = Path(test_dir).expanduser()

    if not test_path.exists():
        raise FileNotFoundError(_('evaluate_test_dir_not_found', path=str(test_path)))

    species_images = {}
    for species_dir in sorted(test_path.iterdir()):
        if species_dir.is_dir():
            images = find_images(species_dir, recursive=False)
            if images:
                species_images[species_dir.name] = images
    return species_images


def classify_image(image_path: Path, model, processor) -> Tuple[Optional[str], float]:
    """
    Classify a single image.

    Returns:
        Tuple of (predicted_species, confidence); (None, 0.0) on error
    """
    try:
        inputs = processor(images=load_rgb(image_path), return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        return model.config.id2label[predicted_idx.item()], confidence.item()
    except Exception as e:
        print(_('evaluate_image_error', path=str(image_path), error=str(e)))
        return None, 0.0


def classify_images_batched(image_paths: List[Path], model, processor,
                            batch_size: int = 32) -> List[Tuple[Optional[str], float]]:
    """Classify images in batches; returns (species, confidence) per input, (None, 0.0) on error."""
    device = next(model.parameters()).device
    batch_size = max(1, int(batch_size))
    predictions: List[Tuple[Optional[str], float]] = []

    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start:start + batch_size]
        images, ok_idx = [], []
        chunk_preds: List[Tuple[Optional[str], float]] = [(None, 0.0)] * len(chunk)
        for i, path in enumerate(chunk):
            try:
                images.append(load_rgb(path))
                ok_idx.append(i)
            except Exception as e:
                print(_('evaluate_image_error', path=str(path), error=str(e)))
        if images:
            try:
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
                    conf, idx = torch.max(probs, 1)
                for i, c, k in zip(ok_idx, conf.tolist(), idx.tolist()):
                    chunk_preds[i] = (model.config.id2label[k], c)
            except Exception:
                for i in ok_idx:
                    chunk_preds[i] = classify_image(chunk[i], model, processor)
        predictions.extend(chunk_preds)
    return predictions


def calculate_metrics(confusion_matrix: Dict[str, Dict[str, int]], 
                     species_list: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, F1-score for each species.
    
    Args:
        confusion_matrix: Confusion matrix data
        species_list: List of all species
        
    Returns:
        Dict with metrics per species
    """
    metrics = {}
    
    for species in species_list:
        # True positives
        tp = confusion_matrix.get(species, {}).get(species, 0)
        
        # False positives (predicted as species but was another)
        fp = sum(confusion_matrix.get(other, {}).get(species, 0) 
                for other in species_list if other != species)
        
        # False negatives (was species but predicted as another)
        fn = sum(confusion_matrix.get(species, {}).get(other, 0) 
                for other in species_list if other != species)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        total = tp + fn  # Total actual instances of this species
        
        metrics[species] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total': total
        }
    
    return metrics


def print_confusion_matrix(confusion_matrix: Dict[str, Dict[str, int]], 
                          species_list: List[str]):
    """
    Print confusion matrix in a formatted table.
    
    Args:
        confusion_matrix: Confusion matrix data
        species_list: List of all species
    """
    print("\n" + "="*80)
    print(_('evaluate_confusion_matrix_header'))
    print("="*80)
    
    # Header
    max_width = max(len(s) for s in species_list)
    header = f"{'Actual/Predicted':<{max_width+2}}"
    for species in species_list:
        header += f"{species[:8]:>10}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for actual in species_list:
        row = f"{actual:<{max_width+2}}"
        for predicted in species_list:
            count = confusion_matrix.get(actual, {}).get(predicted, 0)
            row += f"{count:>10}"
        print(row)
    print()


def print_metrics(metrics: Dict[str, Dict[str, float]], species_list: List[str]):
    """
    Print detailed metrics for each species.
    
    Args:
        metrics: Metrics data per species
        species_list: List of all species
    """
    print("="*80)
    print(_('evaluate_metrics_header'))
    print("="*80)
    print(f"{'Species':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Samples':>10}")
    print("-"*80)
    
    total_samples = 0
    weighted_f1 = 0.0
    
    for species in species_list:
        m = metrics[species]
        print(f"{species:<20} {m['precision']:>10.1%} {m['recall']:>10.1%} "
              f"{m['f1_score']:>10.1%} {m['total']:>10}")
        total_samples += m['total']
        weighted_f1 += m['f1_score'] * m['total']
    
    print("-"*80)
    macro_f1 = sum(m['f1_score'] for m in metrics.values()) / len(metrics)
    weighted_f1 = weighted_f1 / total_samples if total_samples > 0 else 0.0
    
    print(f"{'Macro Average':<20} {' ':>10} {' ':>10} {macro_f1:>10.1%} {total_samples:>10}")
    print(f"{'Weighted Average':<20} {' ':>10} {' ':>10} {weighted_f1:>10.1%} {' ':>10}")
    print()


def export_misclassifications(misclassifications: List[Dict], 
                              output_file: str):
    """
    Export misclassified images to CSV.
    
    Args:
        misclassifications: List of misclassification records
        output_file: Output CSV file path
    """
    if not misclassifications:
        print(_('evaluate_no_misclassifications'))
        return
    
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'actual', 'predicted', 'confidence', 'reason'])
        writer.writeheader()
        writer.writerows(misclassifications)
    
    print(_('evaluate_misclassifications_saved', 
           count=len(misclassifications), 
           path=str(output_path)))


def export_metrics_json(metrics: Dict, confusion_matrix: Dict, 
                       overall_accuracy: float, output_file: str):
    """
    Export all metrics to JSON file.
    
    Args:
        metrics: Per-species metrics
        confusion_matrix: Confusion matrix data
        overall_accuracy: Overall accuracy
        output_file: Output JSON file path
    """
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'overall_accuracy': overall_accuracy,
        'metrics': metrics,
        'confusion_matrix': confusion_matrix
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(_('evaluate_json_saved', path=str(output_path)))


def evaluate_model(model_path: str,
                   test_dir: str,
                   export_misclassified: Optional[str] = None,
                   export_json: Optional[str] = None,
                   min_confidence: float = 0.0,
                   batch_size: int = 32):
    """
    Evaluate model on test dataset.

    Predictions below ``min_confidence`` count as "uncertain": they are not
    correct, but they are reported separately from wrong-species errors.
    The confusion matrix always shows the raw top-1 predictions.

    Args:
        model_path: Path to trained model or Hugging Face model ID
        test_dir: Directory with test images (organized by species folders)
        export_misclassified: Optional CSV file for misclassified/uncertain images
        export_json: Optional JSON file for all metrics
        min_confidence: Minimum confidence for a prediction to count as correct
        batch_size: Inference batch size

    Returns:
        dict with overall accuracy, per-species metrics and counts (None if no test images)
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence}")

    print("=" * 80)
    print(_('evaluate_header'))
    print("=" * 80)
    print()

    model, processor, model_species = load_model(model_path)
    test_images = get_test_images(test_dir)

    if not test_images:
        print(_('evaluate_no_test_images', path=test_dir))
        return None

    test_species = sorted(test_images.keys())
    if set(model_species) != set(test_species):
        print(_('evaluate_species_mismatch'))
        print(f"  {_('evaluate_model_species')}: {sorted(set(model_species))}")
        print(f"  {_('evaluate_test_species')}: {test_species}")
        print()

    total_images = sum(len(images) for images in test_images.values())
    print(_('evaluate_found_images', count=total_images, species=len(test_species)))
    print(f"  {_('evaluate_species_list')}: {', '.join(test_species)}")
    print()

    confusion_matrix = defaultdict(lambda: defaultdict(int))
    misclassifications = []
    correct = 0
    total = 0
    wrong = 0
    uncertain = 0

    print(_('evaluate_processing'))
    for species_name in tqdm(test_species, desc=_('evaluate_species_progress')):
        image_paths = test_images[species_name]
        predictions = classify_images_batched(image_paths, model, processor, batch_size)
        for img_path, (predicted, confidence) in zip(image_paths, predictions):
            if predicted is None:
                continue

            confusion_matrix[species_name][predicted] += 1
            total += 1

            if predicted != species_name:
                wrong += 1
                reason = 'wrong_species'
            elif confidence < min_confidence:
                uncertain += 1
                reason = 'low_confidence'
            else:
                correct += 1
                continue

            misclassifications.append({
                'image': str(img_path),
                'actual': species_name,
                'predicted': predicted,
                'confidence': f"{confidence:.4f}",
                'reason': reason,
            })

    overall_accuracy = correct / total if total > 0 else 0.0

    # Include species the model predicted that have no test folder (they produce false positives)
    predicted_species = {p for row in confusion_matrix.values() for p in row}
    species_list = sorted(set(test_species) | predicted_species)
    metrics = calculate_metrics(confusion_matrix, species_list)

    print()
    print_confusion_matrix(confusion_matrix, species_list)
    print_metrics(metrics, species_list)

    print("=" * 80)
    print(_('evaluate_overall_accuracy', accuracy=overall_accuracy * 100))
    print(f"{_('evaluate_correct')}: {correct}/{total}")
    print(f"{_('evaluate_misclassified')}: {wrong}")
    if min_confidence > 0:
        print(_('evaluate_uncertain', count=uncertain, threshold=min_confidence))
    print("=" * 80)
    print()

    if export_misclassified:
        export_misclassifications(misclassifications, export_misclassified)

    if export_json:
        export_metrics_json(metrics, {k: dict(v) for k, v in confusion_matrix.items()},
                            overall_accuracy, export_json)

    print(_('evaluate_complete'))
    return {
        'overall_accuracy': overall_accuracy,
        'metrics': metrics,
        'correct': correct,
        'wrong': wrong,
        'uncertain': uncertain,
        'total': total,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <model_path> <test_dir> [--export-misclassified <file>] [--export-json <file>]")
        sys.exit(1)
    
    evaluate_model(
        model_path=sys.argv[1],
        test_dir=sys.argv[2]
    )
