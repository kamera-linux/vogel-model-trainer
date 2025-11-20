#!/usr/bin/env python3
"""
Batch classifier for bird species classification.

Provides bulk classification of bird images with trained models,
including CSV export, auto-sorting, and detailed statistics.
"""

import csv
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from vogel_model_trainer.i18n import _


def load_classifier(model_path: str) -> Tuple[AutoModelForImageClassification, AutoImageProcessor, List[str]]:
    """
    Load trained classifier model and processor.
    
    Args:
        model_path: Path to trained model directory or Hugging Face model ID
        
    Returns:
        Tuple of (model, processor, species_list)
    """
    # Check if it's a local path or Hugging Face model ID
    local_path = Path(model_path).expanduser()
    is_local = local_path.exists()
    
    if is_local:
        model_source = str(local_path)
    else:
        # Assume it's a Hugging Face model ID
        model_source = model_path
    
    print(_('classify_loading_model', path=model_source))
    
    try:
        # Load model and processor (works for both local and HF models)
        model = AutoModelForImageClassification.from_pretrained(model_source)
        processor = AutoImageProcessor.from_pretrained(model_source)
    except Exception as e:
        raise FileNotFoundError(_('classify_model_not_found', path=model_source)) from e
    
    # Get species list from model config
    species = list(model.config.id2label.values())
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(_('classify_model_loaded', 
           device="GPU" if torch.cuda.is_available() else "CPU",
           species=len(species)))
    
    return model, processor, species


def get_image_files(input_dir: str, recursive: bool = True) -> List[Path]:
    """
    Get all image files from directory.
    
    Args:
        input_dir: Input directory path
        recursive: Search recursively
        
    Returns:
        List of image file paths
    """
    input_path = Path(input_dir).expanduser()
    
    if not input_path.exists():
        raise FileNotFoundError(_('classify_input_not_found', path=str(input_path)))
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if recursive:
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    else:
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def classify_image(image_path: Path, model, processor, device, top_k: int = 1) -> List[Tuple[str, float]]:
    """
    Classify a single image.
    
    Args:
        image_path: Path to image file
        model: Trained model
        processor: Image processor
        device: Torch device
        top_k: Number of top predictions to return
        
    Returns:
        List of (species, confidence) tuples
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(probs[0])))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            species = model.config.id2label[idx.item()]
            confidence = prob.item()
            results.append((species, confidence))
        
        return results
    
    except Exception as e:
        print(_('classify_image_error', path=str(image_path), error=str(e)))
        return []


def classify_batch(image_files: List[Path], model, processor, top_k: int = 1, 
                   batch_size: int = 32) -> Dict[Path, List[Tuple[str, float]]]:
    """
    Classify batch of images.
    
    Args:
        image_files: List of image paths
        model: Trained model
        processor: Image processor
        top_k: Number of top predictions per image
        batch_size: Processing batch size
        
    Returns:
        Dictionary mapping image paths to prediction lists
    """
    device = next(model.parameters()).device
    results = {}
    
    print(_('classify_processing_header', total=len(image_files)))
    
    for image_path in tqdm(image_files, desc=_('classify_progress')):
        predictions = classify_image(image_path, model, processor, device, top_k)
        if predictions:
            results[image_path] = predictions
    
    return results


def export_csv(results: Dict[Path, List[Tuple[str, float]]], csv_path: str, top_k: int):
    """
    Export classification results to CSV.
    
    Args:
        results: Classification results dictionary
        csv_path: Output CSV file path
        top_k: Number of top predictions
    """
    csv_file = Path(csv_path).expanduser()
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(_('classify_exporting_csv', path=str(csv_file)))
    
    # Build CSV header
    header = ['filename', 'predicted_species', 'confidence']
    for i in range(2, top_k + 1):
        header.extend([f'top_{i}_species', f'top_{i}_confidence'])
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for image_path, predictions in sorted(results.items()):
            row = [image_path.name]
            
            # Add predictions
            for i, (species, confidence) in enumerate(predictions):
                if i == 0:
                    row.extend([species, f'{confidence:.4f}'])
                else:
                    row.extend([species, f'{confidence:.4f}'])
            
            # Fill empty columns if fewer predictions than top_k
            while len(row) < len(header):
                row.extend(['', ''])
            
            writer.writerow(row)
    
    print(_('classify_csv_saved', path=str(csv_file), count=len(results)))


def sort_images(results: Dict[Path, List[Tuple[str, float]]], output_dir: str,
                min_confidence: float = 0.0, move: bool = False, 
                unknown_folder: str = "unknown", dry_run: bool = False) -> Dict[str, int]:
    """
    Sort images into species folders based on classification results.
    
    Args:
        results: Classification results dictionary
        output_dir: Output directory for sorted images
        min_confidence: Minimum confidence threshold
        move: Move instead of copy
        unknown_folder: Folder name for low-confidence images
        dry_run: Only simulate, don't actually move/copy files
        
    Returns:
        Dictionary with statistics per species
    """
    output_path = Path(output_dir).expanduser()
    stats = {}
    unknown_count = 0
    
    action = _('classify_moving') if move else _('classify_copying')
    mode = _('classify_dry_run_mode') if dry_run else ''
    
    print(_('classify_sorting_header', action=action, mode=mode))
    
    for image_path, predictions in tqdm(results.items(), desc=_('classify_sort_progress')):
        if not predictions:
            continue
        
        top_species, top_confidence = predictions[0]
        
        # Check confidence threshold
        if top_confidence < min_confidence:
            target_dir = output_path / unknown_folder
            unknown_count += 1
        else:
            target_dir = output_path / top_species
            stats[top_species] = stats.get(top_species, 0) + 1
        
        # Create target directory
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / image_path.name
        
        # Handle filename conflicts
        counter = 1
        while target_path.exists() and not dry_run:
            stem = image_path.stem
            suffix = image_path.suffix
            target_path = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Move or copy file
        if not dry_run:
            try:
                if move:
                    shutil.move(str(image_path), str(target_path))
                else:
                    shutil.copy2(str(image_path), str(target_path))
            except Exception as e:
                print(_('classify_file_error', path=str(image_path), error=str(e)))
    
    # Add unknown count to stats
    if unknown_count > 0:
        stats[unknown_folder] = unknown_count
    
    return stats


def print_statistics(results: Dict[Path, List[Tuple[str, float]]], 
                     sort_stats: Optional[Dict[str, int]] = None):
    """
    Print classification statistics.
    
    Args:
        results: Classification results dictionary
        sort_stats: Optional sorting statistics
    """
    print("\n" + "="*60)
    print(_('classify_statistics_header'))
    print("="*60)
    
    # Overall stats
    total = len(results)
    print(_('classify_total_classified', count=total))
    
    # Species distribution
    species_count = {}
    confidence_sum = {}
    
    for predictions in results.values():
        if predictions:
            species, confidence = predictions[0]
            species_count[species] = species_count.get(species, 0) + 1
            confidence_sum[species] = confidence_sum.get(species, 0.0) + confidence
    
    if species_count:
        print(_('classify_species_distribution'))
        for species in sorted(species_count.keys()):
            count = species_count[species]
            avg_conf = confidence_sum[species] / count
            percentage = (count / total) * 100
            print(f"  {species}: {count} ({percentage:.1f}%) - Ø {avg_conf:.2%} confidence")
    
    # Sorting stats
    if sort_stats:
        print(_('classify_sorted_statistics'))
        for species, count in sorted(sort_stats.items()):
            print(f"  {species}: {count}")


def classify_images(model_path: str, input_dir: str, 
                   sort_output: Optional[str] = None,
                   min_confidence: float = 0.0,
                   csv_report: Optional[str] = None,
                   top_k: int = 1,
                   batch_size: int = 32,
                   move: bool = False,
                   delete_source: bool = False,
                   force: bool = False,
                   dry_run: bool = False,
                   recursive: bool = True):
    """
    Classify bird images in bulk.
    
    Args:
        model_path: Path to trained model
        input_dir: Input directory with images
        sort_output: Optional output directory for sorted images
        min_confidence: Minimum confidence for sorting (0.0-1.0)
        csv_report: Optional CSV report file path
        top_k: Number of top predictions (1-5)
        batch_size: Processing batch size
        move: Move instead of copy when sorting
        delete_source: Delete source directory after processing
        force: Skip confirmation prompts
        dry_run: Simulate without actual file operations
        recursive: Search for images recursively
    """
    print("="*60)
    print(_('classify_header'))
    print("="*60)
    
    # Validate parameters
    if top_k < 1 or top_k > 5:
        raise ValueError(_('classify_invalid_top_k', k=top_k))
    
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError(_('classify_invalid_confidence', conf=min_confidence))
    
    # Confirmation for delete_source
    if delete_source and not force and not dry_run:
        print("\n" + "="*60)
        print(_('classify_delete_warning'))
        print("="*60)
        response = input(_('classify_delete_confirm'))
        if response.lower() not in ['y', 'yes', 'j', 'ja']:
            print(_('classify_cancelled'))
            return
    
    # Load model
    model, processor, species = load_classifier(model_path)
    
    print(_('classify_detected_species', species=', '.join(species)))
    
    # Get image files
    image_files = get_image_files(input_dir, recursive)
    
    if not image_files:
        print(_('classify_no_images', path=input_dir))
        return
    
    print(_('classify_found_images', count=len(image_files)))
    
    # Classify images
    results = classify_batch(image_files, model, processor, top_k, batch_size)
    
    # Export CSV if requested
    if csv_report:
        export_csv(results, csv_report, top_k)
    
    # Sort images if requested
    sort_stats = None
    if sort_output:
        sort_stats = sort_images(results, sort_output, min_confidence, 
                                 move, "unknown", dry_run)
        
        if not dry_run:
            print(_('classify_sorting_complete', path=sort_output))
    
    # Delete source directory if requested
    if delete_source and not dry_run and sort_output:
        input_path = Path(input_dir).expanduser()
        print(_('classify_deleting_source', path=str(input_path)))
        try:
            shutil.rmtree(input_path)
            print(_('classify_source_deleted'))
        except Exception as e:
            print(_('classify_delete_error', error=str(e)))
    
    # Print statistics
    print_statistics(results, sort_stats)
    
    if dry_run:
        print("\n" + _('classify_dry_run_complete'))
    else:
        print("\n" + _('classify_complete'))


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python classifier.py <model_path> <input_dir> [--sort-output <dir>] [--csv-report <file>]")
        sys.exit(1)
    
    classify_images(
        model_path=sys.argv[1],
        input_dir=sys.argv[2],
        sort_output=sys.argv[3] if len(sys.argv) > 3 else None,
        top_k=3
    )
