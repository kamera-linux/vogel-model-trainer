#!/usr/bin/env python3
"""
Script to test the trained custom bird classifier on sample images.
"""

import sys
from pathlib import Path
from transformers import pipeline
from PIL import Image

def test_model(model_path: str, data_dir: str = None, image_path: str = None):
    """
    Test the model on validation dataset or single image.
    
    Args:
        model_path: Path to trained model directory
        data_dir: Path to organized dataset directory (with val/ folder)
        image_path: Path to single image file (alternative to data_dir)
    
    Returns:
        dict: Test results with accuracy metrics
    """
    from pathlib import Path
    import random
    from vogel_model_trainer.i18n import _
    
    model_path = str(Path(model_path).expanduser())
    
    print(_('test_loading_model', path=model_path))
    classifier = pipeline(
        "image-classification",
        model=model_path,
        device=-1  # CPU
    )
    
    # Single image test
    if image_path:
        image_path = str(Path(image_path).expanduser())
        print(_('test_classifying_image', path=image_path))
        img = Image.open(image_path)
        
        results = classifier(img, top_k=5)
        
        print(_('test_results'))
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['label']:15s} - {result['score']:.4f} ({result['score']*100:.1f}%)")
        
        return {"image": image_path, "predictions": results}
    
    # Validation set test
    if data_dir:
        data_dir = Path(data_dir).expanduser()
        val_dir = data_dir / "val"
        
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        print(_('test_on_validation', path=val_dir))
        print("=" * 70)
        
        species = [d.name for d in val_dir.iterdir() if d.is_dir()]
        correct = 0
        total = 0
        results_by_species = {}
        
        for sp in sorted(species):
            species_dir = val_dir / sp
            images = list(species_dir.glob("*.jpg"))
            
            if not images:
                print(_('test_no_images_found', species=sp))
                continue
            
            # Test random sample
            sample_size = min(5, len(images))
            sample_images = random.sample(images, sample_size)
            
            sp_correct = 0
            for img_path in sample_images:
                img = Image.open(img_path)
                predictions = classifier(img, top_k=1)
                predicted = predictions[0]['label']
                confidence = predictions[0]['score']
                
                is_correct = predicted == sp
                sp_correct += is_correct
                total += 1
                
                status = "✅" if is_correct else "❌"
                if not is_correct:  # Nur Fehler anzeigen
                    print(f"{status} {sp:12s} -> {predicted:12s} ({confidence:.4f})")
            
            correct += sp_correct
            sp_accuracy = sp_correct / sample_size
            results_by_species[sp] = sp_accuracy
            print(f"   {sp:12s}: {sp_correct}/{sample_size} = {sp_accuracy*100:.1f}%")
        
        overall_accuracy = correct / total if total > 0 else 0
        
        print("=" * 70)
        print(_('test_overall_accuracy', correct=correct, total=total, accuracy=overall_accuracy*100))
        
        return {
            "overall_accuracy": overall_accuracy,
            "by_species": results_by_species,
            "correct": correct,
            "total": total
        }
    
    raise ValueError("Entweder data_dir oder image_path muss angegeben werden")

if __name__ == "__main__":
    from vogel_model_trainer.i18n import _
    
    if len(sys.argv) < 2:
        print(_('test_usage'))
        print(_('test_usage_single'))
        print(_('test_usage_single_cmd'))
        print()
        print(_('test_usage_validation'))
        print(_('test_usage_validation_cmd'))
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if len(sys.argv) == 3 and not sys.argv[2].startswith("--"):
        image_path = sys.argv[2]
        test_model(model_path, image_path=image_path)
    elif "--data-dir" in sys.argv:
        data_dir_idx = sys.argv.index("--data-dir") + 1
        data_dir = sys.argv[data_dir_idx] if data_dir_idx < len(sys.argv) else None
        test_model(model_path, data_dir=data_dir)
    else:
        print(_('test_error_no_input'))
        sys.exit(1)
