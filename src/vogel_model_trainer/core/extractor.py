#!/usr/bin/env python3
"""
Extract bird crops from videos and images for training data collection.

Video, image and convert mode share one crop pipeline
(filter -> classify -> background removal -> resize -> deduplicate -> save),
so all modes treat padding, resizing and duplicates the same way.
"""

import argparse
import glob
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

from vogel_model_trainer.i18n import _
from vogel_model_trainer.utils.image import IMAGE_EXTENSIONS, INFERENCE_BACKGROUND, to_rgb, unique_path

# Suppress harmless Cholesky decomposition warnings from PyTorch optimizer
warnings.filterwarnings('ignore', message='.*Cholesky.*')
warnings.filterwarnings('ignore', message='.*positive-definiteness.*')

# Default configuration (can be overridden via command line)
DEFAULT_THRESHOLD = 0.5  # Higher threshold for better quality birds
DEFAULT_SAMPLE_RATE = 3  # Check more frames for better coverage
DEFAULT_MODEL = "yolov26n.pt"
TARGET_IMAGE_SIZE = 224  # Optimal size for EfficientNet-B0 training
COCO_BIRD_CLASS = 14

# Image processing constants
MIN_IMAGE_SIZE = 50  # Minimum image dimension for background removal
ALPHA_TRANSPARENCY_THRESHOLD = 10  # Alpha value below which pixels are considered transparent
BLACK_PIXEL_THRESHOLD = 20  # Grayscale value below which pixels are considered black
PROGRESS_LOG_INTERVAL = 100  # Log progress every N frames/images
LEGACY_BLUR_THRESHOLD = 100  # Laplacian variance threshold for --skip-blurry

# Background removal constants (rembg/alpha matting)
ALPHA_MATTING_FOREGROUND_THRESHOLD = 240  # Threshold for foreground in alpha matting
ALPHA_MATTING_BACKGROUND_THRESHOLD = 10  # Threshold for background in alpha matting
ALPHA_MATTING_ERODE_SIZE = 10  # Erosion kernel size for alpha matting
GRAY_BACKGROUND_DEFAULT = (128, 128, 128)  # Default gray background color (BGR)

VIDEO_EXTENSIONS = frozenset({'.mp4', '.avi', '.mov', '.mkv', '.webm', '.ogv', '.m4v'})

REMBG_MODELS = ['u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use']

# Named background colors (RGB)
BG_COLOR_NAMES = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'green-screen': (0, 255, 0),
    'blue-screen': (0, 0, 255),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_detection_model_path(detection_model: str) -> str:
    """Resolve detection model path from common locations while preserving model IDs."""
    model_input = str(detection_model).strip()
    if not model_input:
        return DEFAULT_MODEL

    expanded = Path(model_input).expanduser()

    # Existing explicit file path (absolute or relative)
    if expanded.is_file():
        return str(expanded)

    # Keep absolute path unchanged so downstream error remains clear if file is missing
    if expanded.is_absolute():
        return str(expanded)

    # Try common project/package-relative locations for local model files
    candidate_paths = [
        Path.cwd() / expanded,
        Path(__file__).resolve().parents[3] / expanded,
        Path(__file__).resolve().parent / expanded,
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return str(candidate)

    # Not a local file in known locations -> treat as model ID/name
    return model_input


def _load_yolo_model(detection_model: str) -> Tuple[YOLO, str]:
    """Load YOLO model with fallback aliases for yolov26n default naming."""
    resolved = _resolve_detection_model_path(detection_model)

    attempts = [resolved]
    if Path(str(resolved)).name.lower() == "yolov26n.pt":
        # Some environments resolve this family as yolo26n.pt
        attempts.append("yolo26n.pt")

    last_error = None
    seen = set()

    for model_ref in attempts:
        if model_ref in seen:
            continue
        seen.add(model_ref)

        try:
            print(_('loading_yolo') + f" {model_ref}")
            return YOLO(model_ref), model_ref
        except Exception as exc:  # try next alias, report the last error below
            last_error = exc

    raise FileNotFoundError(
        f"YOLO model could not be loaded: {detection_model}. "
        f"Tried: {', '.join(attempts)}. "
        "Use --detection-model with an absolute model path if needed."
    ) from last_error


@dataclass
class ExtractionModels:
    """Models shared across all files of one extraction run."""
    yolo: YOLO
    classifier: Optional[AutoModelForImageClassification] = None
    processor: Optional[AutoImageProcessor] = None
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))


def load_species_classifier(species_model: str):
    """Load a species classifier on the best available device."""
    print(_('loading_species') + f" {species_model}")
    model_ref = str(Path(species_model).expanduser()) if Path(species_model).expanduser().exists() else species_model
    processor = AutoImageProcessor.from_pretrained(model_ref, revision="main")  # nosec B615
    classifier = AutoModelForImageClassification.from_pretrained(model_ref, revision="main")  # nosec B615
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()
    print(_('loaded_species_classes', count=len(classifier.config.id2label)))
    return classifier, processor, device


def load_models(detection_model: Optional[str] = None, species_model: Optional[str] = None) -> ExtractionModels:
    """Load detection (and optional species) model once for a whole run."""
    detection_model = detection_model or DEFAULT_MODEL
    classifier = processor = None
    device = torch.device('cpu')
    if species_model:
        classifier, processor, device = load_species_classifier(species_model)

    yolo, loaded_ref = _load_yolo_model(detection_model)
    if loaded_ref != detection_model:
        print(f"ℹ️ Using YOLO model reference: {loaded_ref}")
    return ExtractionModels(yolo=yolo, classifier=classifier, processor=processor, device=device)


def classify_crop(bird_crop: np.ndarray, models: ExtractionModels) -> Tuple[str, float]:
    """Classify a BGR crop with the species classifier."""
    bird_image = Image.fromarray(cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB))
    inputs = models.processor(images=bird_image, return_tensors="pt")
    inputs = {k: v.to(models.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = models.classifier(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
    return models.classifier.config.id2label[idx], float(probs[idx].item())


# ---------------------------------------------------------------------------
# Image quality
# ---------------------------------------------------------------------------

def _to_bgr(image: np.ndarray, background_bgr: Sequence[int] = GRAY_BACKGROUND_DEFAULT) -> np.ndarray:
    """Return a 3-channel BGR image (grayscale expanded, BGRA composited onto background)."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bg = np.empty_like(image[:, :, :3])
        bg[:] = background_bgr
        return (image[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
    return image


def calculate_motion_quality(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate motion/blur quality metrics for an image.

    Args:
        image: BGR or BGRA image (numpy array)

    Returns:
        dict: Quality metrics including:
            - sharpness: Laplacian variance (higher = sharper)
            - edge_quality: Sobel gradient magnitude (higher = clearer edges)
            - overall: Combined quality score
    """
    gray = cv2.cvtColor(_to_bgr(image), cv2.COLOR_BGR2GRAY)

    # 1. Laplacian Variance (Sharpness measure)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Sobel Gradient Magnitude (Edge quality)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = float(np.sqrt(sobelx**2 + sobely**2).mean())

    # 3. Combined overall score (weighted average)
    overall_score = laplacian_var * 0.7 + gradient_mag * 0.3

    return {
        'sharpness': laplacian_var,
        'edge_quality': gradient_mag,
        'overall': overall_score
    }


def is_motion_acceptable(quality_metrics: Dict[str, float],
                         min_sharpness: Optional[float] = None,
                         min_edge_quality: Optional[float] = None) -> Tuple[bool, str]:
    """
    Check if motion quality metrics meet minimum thresholds.

    Args:
        quality_metrics: Dict from calculate_motion_quality()
        min_sharpness: Minimum sharpness score (default: None = no filter)
        min_edge_quality: Minimum edge quality score (default: None = no filter)

    Returns:
        tuple: (is_acceptable: bool, reason: str)
    """
    if min_sharpness is not None and quality_metrics['sharpness'] < min_sharpness:
        return False, 'low_sharpness'

    if min_edge_quality is not None and quality_metrics['edge_quality'] < min_edge_quality:
        return False, 'poor_edges'

    return True, 'accepted'


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

_rembg_remove = None
_rembg_new_session = None
_rembg_sessions: Dict[str, object] = {}
_rembg_unavailable = False


def _get_rembg_session(model_name: str):
    """Import rembg lazily and cache one session per model (loading a model is expensive)."""
    global _rembg_remove, _rembg_new_session, _rembg_unavailable
    if _rembg_unavailable:
        return None, None
    if _rembg_remove is None:
        try:
            with warnings.catch_warnings():
                # ONNX Runtime device discovery warnings on systems with discrete GPUs
                warnings.filterwarnings("ignore", category=UserWarning)
                from rembg import new_session, remove
        except ImportError:
            _rembg_unavailable = True
            print("⚠️ Warning: rembg not installed. Background removal disabled. Install with: pip install rembg")
            return None, None
        _rembg_remove, _rembg_new_session = remove, new_session

    if model_name not in _rembg_sessions:
        _rembg_sessions[model_name] = _rembg_new_session(model_name)
    return _rembg_remove, _rembg_sessions[model_name]


def remove_background(image, margin=10, iterations=10, bg_color=GRAY_BACKGROUND_DEFAULT, model_name='u2net',
                      transparent=False, fill_black_areas=False, expand_mask=0):
    """
    Remove background from bird image using rembg (AI-based segmentation).

    Args:
        image: BGR image (numpy array)
        margin: Not used, kept for API compatibility
        iterations: Not used, kept for API compatibility
        bg_color: Background color as BGR tuple (default: (128, 128, 128) = gray).
                  Ignored if transparent=True
        model_name: rembg model to use (default: 'u2net')
                    Options: 'u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use'
        transparent: If True, return BGRA image with transparent background
        fill_black_areas: If True, make black BACKGROUND/PADDING areas transparent.
                          Only affects areas already identified as background by rembg,
                          black feathers are preserved. Only used with transparent=True.
        expand_mask: Pixels to expand the foreground mask (keeps more background around
                     bird details like feet/beak). 0 = default rembg behavior

    Returns:
        numpy array: Image with replaced background (BGRA if transparent=True, BGR otherwise)
    """
    if image is None or image.size == 0:
        return image

    rembg_remove, session = _get_rembg_session(model_name)
    if rembg_remove is None:
        return image

    image = _to_bgr(image)
    height, width = image.shape[:2]

    # Minimum size check
    if height < MIN_IMAGE_SIZE or width < MIN_IMAGE_SIZE:
        return image

    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # alpha_matting improves edge quality
        output = rembg_remove(
            pil_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,
            alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD,
            alpha_matting_erode_size=ALPHA_MATTING_ERODE_SIZE,
            post_process_mask=True
        )

        output_np = np.array(output.convert('RGBA'))
        rgb = output_np[:, :, :3]
        alpha = output_np[:, :, 3]

        alpha_float = alpha.astype(np.float32) / 255.0

        # Expand mask if requested to keep more background around bird details
        if expand_mask > 0:
            kernel_size = expand_mask * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            alpha_binary = (alpha > ALPHA_TRANSPARENCY_THRESHOLD).astype(np.uint8) * 255
            alpha_float = cv2.dilate(alpha_binary, kernel, iterations=1).astype(np.float32) / 255.0
            # Dilated areas need the original pixels, not rembg's blanked-out background
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Smooth and clean the alpha channel
        alpha_smooth = cv2.GaussianBlur(alpha_float, (3, 3), 0)
        kernel = np.ones((3, 3), np.uint8)
        alpha_cleaned = cv2.morphologyEx((alpha_smooth * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        alpha_final = alpha_cleaned.astype(np.float32) / 255.0

        if transparent:
            if fill_black_areas:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                black_mask = gray < BLACK_PIXEL_THRESHOLD
                background_mask = alpha_final < (ALPHA_TRANSPARENCY_THRESHOLD / 255.0)
                alpha_final[black_mask & background_mask] = 0.0

            alpha_channel = (alpha_final * 255).astype(np.uint8)
            result_rgba = np.dstack((rgb, alpha_channel))
            return cv2.cvtColor(result_rgba, cv2.COLOR_RGBA2BGRA)

        bg_rgb = np.array([bg_color[2], bg_color[1], bg_color[0]], dtype=np.uint8)  # BGR to RGB
        background = np.full((height, width, 3), bg_rgb, dtype=np.uint8)
        alpha_3channel = alpha_final[:, :, np.newaxis]
        result_rgb = (rgb * alpha_3channel + background * (1 - alpha_3channel)).astype(np.uint8)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Warning: Background removal failed: {e}")
        return image


# ---------------------------------------------------------------------------
# Shared crop pipeline
# ---------------------------------------------------------------------------

def _cv_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image).convert('RGB')
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def finalize_crop(image: np.ndarray, target_image_size: int,
                  bg_color: Sequence[int] = GRAY_BACKGROUND_DEFAULT) -> Image.Image:
    """
    Convert an OpenCV crop to the PIL image that will be saved.

    With ``target_image_size > 0`` the crop is scaled to fit (aspect ratio kept)
    and padded to a square: transparent padding for BGRA crops, ``bg_color``
    (BGR) otherwise.
    """
    pil = _cv_to_pil(image)
    if not target_image_size or target_image_size <= 0:
        return pil

    pil.thumbnail((target_image_size, target_image_size), Image.Resampling.LANCZOS)
    if pil.mode == 'RGBA':
        canvas = Image.new('RGBA', (target_image_size, target_image_size), (0, 0, 0, 0))
    else:
        canvas = Image.new('RGB', (target_image_size, target_image_size),
                           (bg_color[2], bg_color[1], bg_color[0]))
    offset = ((target_image_size - pil.width) // 2, (target_image_size - pil.height) // 2)
    canvas.paste(pil, offset)
    return canvas


def save_pil_image(img: Image.Image, base_path: Path, quality: int = 95) -> Path:
    """Save as PNG when the image has transparency, otherwise as JPEG. Returns the path."""
    if img.mode == 'RGBA':
        path = base_path.with_suffix('.png')
        img.save(path, 'PNG', compress_level=6)
    else:
        path = base_path.with_suffix('.jpg')
        img.convert('RGB').save(path, 'JPEG', quality=quality)
    return path


def compute_hash(img: Image.Image):
    """Perceptual hash, computed on the inference background so PNG and JPEG compare fairly."""
    return imagehash.phash(to_rgb(img, INFERENCE_BACKGROUND))


def find_duplicate(img_hash, hash_cache: Dict[str, object], threshold: int) -> Optional[Tuple[str, int]]:
    """Return (path, distance) of the first cached image within ``threshold``, else None."""
    for existing_path, existing_hash in hash_cache.items():
        distance = img_hash - existing_hash
        if distance <= threshold:
            return existing_path, distance
    return None


def load_existing_hashes(output_path: Path) -> Dict[str, object]:
    """Hash all images already present in ``output_path`` (cross-session deduplication)."""
    print(_('dedup_loading_existing'))
    hash_cache = {}
    if output_path.exists():
        for img_path in output_path.rglob('*'):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(img_path) as img:
                    img.load()
                    hash_cache[str(img_path)] = compute_hash(img)
            except Exception:  # nosec B112 - skip corrupted images
                continue
    if hash_cache:
        print(_('dedup_loaded_existing', count=len(hash_cache)))
    return hash_cache


def expand_box(x1, y1, x2, y2, frame_w, frame_h, crop_padding):
    """
    Clamp a box to the frame and grow/shrink it by ``crop_padding`` percent.

    Positive values add context around the bird, negative values crop tighter.
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if crop_padding:
        pad_x = int((x2 - x1) * crop_padding / 100.0)
        pad_y = int((y2 - y1) * crop_padding / 100.0)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(frame_w, x2 + pad_x), min(frame_h, y2 + pad_y)
    return x1, y1, x2, y2


@dataclass
class ExtractionSettings:
    """Options shared by video and image extraction."""
    output_path: Path
    bird_species: Optional[str] = None
    threshold: float = DEFAULT_THRESHOLD
    target_image_size: int = TARGET_IMAGE_SIZE
    species_threshold: Optional[float] = None
    target_class: int = COCO_BIRD_CLASS
    max_detections: int = 10
    min_box_size: int = 50
    max_box_size: int = 800
    quality: int = 95
    skip_blurry: bool = False
    deduplicate: bool = False
    similarity_threshold: int = 5
    min_sharpness: Optional[float] = None
    min_edge_quality: Optional[float] = None
    save_quality_report: bool = False
    remove_bg: bool = False
    bg_color: Tuple[int, int, int] = GRAY_BACKGROUND_DEFAULT
    bg_model: str = 'u2net'
    bg_transparent: bool = False
    bg_fill_black: bool = False
    crop_padding: int = 0

    @property
    def needs_quality_metrics(self) -> bool:
        return (self.min_sharpness is not None or self.min_edge_quality is not None
                or self.save_quality_report)


@dataclass
class ExtractionStats:
    detected: int = 0          # Birds passing detection + box filters
    exported: int = 0          # Saved images
    skipped_species: int = 0   # Below species threshold
    duplicates: int = 0
    motion_rejected: int = 0
    blurry: int = 0            # --skip-blurry
    species_counts: Dict[str, int] = field(default_factory=dict)
    accepted_quality: List[float] = field(default_factory=list)
    rejected_blur: List[float] = field(default_factory=list)
    rejected_edges: List[float] = field(default_factory=list)


def process_detection(frame: np.ndarray, box: Tuple[int, int, int, int], det_conf: float,
                      settings: ExtractionSettings, stats: ExtractionStats, models: ExtractionModels,
                      hash_cache: Optional[Dict[str, object]], session_id: str,
                      frame_tag: Optional[str], label: str) -> Optional[Path]:
    """
    Run one detection through the shared pipeline and save it.

    Returns the saved path, or None if the crop was filtered out.
    """
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(*box, frame_w, frame_h, settings.crop_padding)
    bird_crop = frame[y1:y2, x1:x2]
    if bird_crop.size == 0:
        return None

    stats.detected += 1

    # Motion/blur quality
    if settings.needs_quality_metrics:
        metrics = calculate_motion_quality(bird_crop)
        ok, reason = is_motion_acceptable(metrics, settings.min_sharpness, settings.min_edge_quality)
        if not ok:
            stats.motion_rejected += 1
            if reason == 'low_sharpness':
                stats.rejected_blur.append(metrics['sharpness'])
            else:
                stats.rejected_edges.append(metrics['edge_quality'])
            return None

    if settings.skip_blurry:
        gray = cv2.cvtColor(bird_crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < LEGACY_BLUR_THRESHOLD:
            stats.blurry += 1
            return None

    if settings.needs_quality_metrics:
        stats.accepted_quality.append(metrics['overall'])

    # Species
    species_name = None
    species_conf = None
    if models.classifier is not None:
        species_name, species_conf = classify_crop(bird_crop, models)
        if settings.species_threshold is not None and species_conf < settings.species_threshold:
            stats.skipped_species += 1
            print(_('bird_skipped', species=species_name, conf=species_conf,
                    threshold=settings.species_threshold, frame=label))
            return None
    elif settings.bird_species:
        species_name = settings.bird_species

    # Background removal (crop_padding > 0 also expands the foreground mask by that many pixels)
    if settings.remove_bg:
        bird_crop = remove_background(
            bird_crop, bg_color=settings.bg_color, model_name=settings.bg_model,
            transparent=settings.bg_transparent, fill_black_areas=settings.bg_fill_black,
            expand_mask=max(0, settings.crop_padding))

    final_img = finalize_crop(bird_crop, settings.target_image_size, settings.bg_color)

    # Deduplicate on the final image, i.e. the same representation as files on disk
    img_hash = None
    if settings.deduplicate and hash_cache is not None:
        img_hash = compute_hash(final_img)
        duplicate = find_duplicate(img_hash, hash_cache, settings.similarity_threshold)
        if duplicate:
            stats.duplicates += 1
            print(_('dedup_skipped_duplicate', filename=Path(duplicate[0]).name, distance=duplicate[1]))
            return None

    save_dir = settings.output_path / species_name if species_name else settings.output_path
    save_dir.mkdir(parents=True, exist_ok=True)

    parts = [session_id, uuid.uuid4().hex[:8]]
    if frame_tag:
        parts.append(frame_tag)
    if species_conf is not None:
        parts += [f"det{det_conf:.2f}", f"cls{species_conf:.2f}"]
    else:
        parts.append(f"c{det_conf:.2f}")
    save_path = save_pil_image(final_img, save_dir / "_".join(parts), settings.quality)

    if img_hash is not None:
        hash_cache[str(save_path)] = img_hash

    stats.exported += 1
    if species_name:
        stats.species_counts[species_name] = stats.species_counts.get(species_name, 0) + 1
        print(_('bird_extracted', count=stats.exported, species=species_name,
                conf=species_conf if species_conf is not None else 1.0, frame=label))
    else:
        print(_('bird_extracted_simple', count=stats.exported, frame=label, conf=det_conf))
    return save_path


def _detections(models: ExtractionModels, frame: np.ndarray, settings: ExtractionSettings):
    """Yield (box, conf) for birds passing confidence, box-size and per-frame limits."""
    results = models.yolo(frame, conf=settings.threshold, classes=[settings.target_class], verbose=False)
    accepted = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != settings.target_class:
                continue
            conf = float(box.conf[0])
            if conf < settings.threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            box_size = max(x2 - x1, y2 - y1)
            if box_size < settings.min_box_size or box_size > settings.max_box_size:
                continue
            if accepted >= settings.max_detections:
                return
            accepted += 1
            yield (x1, y1, x2, y2), conf


def _print_settings(settings: ExtractionSettings, models: ExtractionModels):
    print(_('detection_threshold', threshold=settings.threshold))
    if settings.species_threshold is not None:
        if models.classifier is not None:
            print(_('species_threshold', threshold=settings.species_threshold))
        else:
            print(_('species_threshold_ignored'))
    if settings.target_image_size > 0:
        print(_('image_size', size=settings.target_image_size))
    else:
        print(_('image_size_original'))
    if settings.max_detections < 999:
        print(_('max_detections_per_frame', max=settings.max_detections))
    if settings.min_box_size > 0:
        print(_('min_box_size_filter', size=settings.min_box_size))
    if settings.max_box_size < 9999:
        print(_('max_box_size_filter', size=settings.max_box_size))
    if settings.quality < 95:
        print(_('jpeg_quality_filter', quality=settings.quality))
    if settings.skip_blurry:
        print(_('blur_detection_filter'))
    if settings.deduplicate:
        print(_('dedup_filter', threshold=settings.similarity_threshold))
    if settings.min_sharpness is not None:
        print(_('motion_sharpness_filter', threshold=settings.min_sharpness))
    if settings.min_edge_quality is not None:
        print(_('motion_edge_filter', threshold=settings.min_edge_quality))
    if settings.remove_bg:
        print(_('background_removal_enabled'))

    if models.classifier is not None:
        print(_('mode_autosorting'))
    elif settings.bird_species:
        print(_('mode_manual', species=settings.bird_species))
    else:
        print(_('mode_standard'))


def _print_summary(settings: ExtractionSettings, stats: ExtractionStats):
    print(_('extraction_complete'))
    print(_('output_directory', path=settings.output_path))
    print(_('detected_birds_total', count=stats.detected))
    print(_('exported_birds_total', count=stats.exported))

    if stats.skipped_species > 0:
        print(_('skipped_birds_total', count=stats.skipped_species, threshold=settings.species_threshold))

    if settings.deduplicate and stats.duplicates > 0:
        total_checked = stats.exported + stats.duplicates
        percent = (stats.duplicates / total_checked * 100) if total_checked > 0 else 0
        print(_('dedup_stats'))
        print(_('dedup_stats_checked', count=total_checked))
        print(_('dedup_stats_skipped', count=stats.duplicates, percent=percent))

    if stats.motion_rejected > 0:
        print(_('motion_rejected_stats', count=stats.motion_rejected))
    if stats.blurry > 0:
        print(_('blurry_rejected_stats', count=stats.blurry))

    if settings.save_quality_report:
        print("\n" + _('quality_report_title'))
        print("━" * 60)
        total = stats.detected
        accepted = len(stats.accepted_quality)
        rejected = stats.motion_rejected

        def pct(n):
            return n / total * 100 if total > 0 else 0

        print(_('quality_report_total', count=total))
        print(_('quality_report_accepted', count=accepted, percent=pct(accepted)))
        print(_('quality_report_rejected', count=rejected, percent=pct(rejected)))
        if stats.accepted_quality:
            print(_('quality_report_avg_accepted', score=np.mean(stats.accepted_quality)))
        if stats.rejected_blur:
            print(_('quality_report_avg_rejected_blur', score=np.mean(stats.rejected_blur)))
        if stats.rejected_edges:
            print(_('quality_report_avg_rejected_edges', score=np.mean(stats.rejected_edges)))
        print("━" * 60)

    if stats.species_counts:
        print(_('species_breakdown'))
        for species, count in sorted(stats.species_counts.items(), key=lambda x: x[1], reverse=True):
            print(_('species_count', species=species, count=count))


def _build_settings(output_dir, **kwargs) -> ExtractionSettings:
    settings = ExtractionSettings(output_path=Path(output_dir).expanduser())
    for key, value in kwargs.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    settings.target_image_size = max(0, int(settings.target_image_size or 0))
    return settings


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_birds_from_video(video_path, output_dir, bird_species=None,
                             detection_model=None, species_model=None,
                             threshold=None, sample_rate=None, target_image_size=TARGET_IMAGE_SIZE,
                             species_threshold=None, target_class=COCO_BIRD_CLASS,
                             max_detections=10, min_box_size=50, max_box_size=800,
                             quality=95, skip_blurry=False,
                             deduplicate=False, similarity_threshold=5,
                             min_sharpness=None, min_edge_quality=None,
                             save_quality_report=False, remove_bg=False,
                             bg_color=GRAY_BACKGROUND_DEFAULT, bg_model='u2net',
                             bg_transparent=False, bg_fill_black=False,
                             crop_padding=0, models: Optional[ExtractionModels] = None,
                             hash_cache: Optional[Dict[str, object]] = None):
    """
    Extract bird crops from video and save as images.

    Args:
        video_path: Path to video file
        output_dir: Base directory to save extracted bird images
        bird_species: Manually specified bird species (creates subdirectory)
        detection_model: YOLO model path for bird detection (default: yolov26n.pt)
        species_model: Custom species classifier model path for automatic sorting
        threshold: Detection confidence threshold (default: 0.5)
        sample_rate: Analyze every Nth frame (default: 3)
        target_image_size: Target image size in pixels (default: 224, 0 for original)
        species_threshold: Minimum species confidence (only used with species_model)
        target_class: COCO class for bird (14)
        max_detections: Maximum number of detections per frame (default: 10)
        min_box_size: Minimum bounding box size in pixels (default: 50)
        max_box_size: Maximum bounding box size in pixels (default: 800)
        quality: JPEG quality 1-100 (default: 95)
        skip_blurry: Skip blurry images (default: False)
        deduplicate: Skip duplicate/similar images (default: False)
        similarity_threshold: Hamming distance threshold for duplicates 0-64 (default: 5)
        min_sharpness: Minimum sharpness score (default: None = no filter)
        min_edge_quality: Minimum edge quality score (default: None = no filter)
        save_quality_report: Print detailed quality report (default: False)
        remove_bg, bg_color, bg_model, bg_transparent, bg_fill_black: Background removal options
        crop_padding: Percent to grow (positive) or shrink (negative) the detection box.
                      With remove_bg, positive values also expand the foreground mask by that many pixels.
        models: Preloaded models (avoids reloading for every file)
        hash_cache: Shared deduplication cache (avoids re-hashing the output folder)
    """
    sample_rate = sample_rate if sample_rate is not None else DEFAULT_SAMPLE_RATE
    if sample_rate < 1:
        raise ValueError("sample_rate must be >= 1")

    settings = _build_settings(
        output_dir, bird_species=bird_species, threshold=threshold, target_image_size=target_image_size,
        species_threshold=species_threshold, target_class=target_class, max_detections=max_detections,
        min_box_size=min_box_size, max_box_size=max_box_size, quality=quality, skip_blurry=skip_blurry,
        deduplicate=deduplicate, similarity_threshold=similarity_threshold, min_sharpness=min_sharpness,
        min_edge_quality=min_edge_quality, save_quality_report=save_quality_report, remove_bg=remove_bg,
        bg_color=bg_color, bg_model=bg_model, bg_transparent=bg_transparent, bg_fill_black=bg_fill_black,
        crop_padding=crop_padding)

    if models is None:
        models = load_models(detection_model, species_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(_('cannot_open_video', path=video_path))
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    print(_('video_info') + f" {Path(video_path).name}")
    print(_('total_frames', total=total_frames, fps=fps))
    print(_('analyzing_every_nth', n=sample_rate))
    _print_settings(settings, models)

    settings.output_path.mkdir(parents=True, exist_ok=True)
    if settings.deduplicate and hash_cache is None:
        hash_cache = load_existing_hashes(settings.output_path)

    session_id = f"{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stats = ExtractionStats()
    frame_num = 0

    try:
        while True:
            if frame_num % sample_rate != 0:
                # Skipped frames only need to be grabbed, not decoded
                if not cap.grab():
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                for box, conf in _detections(models, frame, settings):
                    process_detection(frame, box, conf, settings, stats, models, hash_cache,
                                      session_id, f"f{frame_num:06d}", str(frame_num))

            frame_num += 1
            if frame_num % (PROGRESS_LOG_INTERVAL * sample_rate) == 0:
                if total_frames > 0:
                    progress = min(100.0, frame_num / total_frames * 100)
                    print(_('progress', percent=progress, current=frame_num, total=total_frames))
                else:
                    print(_('progress_unknown_total', current=frame_num))

    except KeyboardInterrupt:
        print(_('extraction_interrupted'))
    finally:
        cap.release()

    _print_summary(settings, stats)
    print(_('session_id', id=session_id))
    if models.classifier is not None:
        print(_('filename_format', format=f"{session_id}_<id>_f<frame>_det<det-conf>_cls<species-conf>.jpg"))
    else:
        print(_('filename_format', format=f"{session_id}_<unique-id>_f<frame>_c<confidence>.jpg"))

    print(_('next_steps'))
    print(_('next_step_review', path=settings.output_path))
    if models.classifier is not None or settings.bird_species:
        print(_('next_step_verify'))
    else:
        print(_('next_step_manual_sort'))
    print(_('next_step_organize'))
    print(_('next_step_train'))
    return stats


def extract_birds_from_image(image_path, output_dir, bird_species=None,
                             detection_model=None, species_model=None,
                             threshold=None, target_image_size=TARGET_IMAGE_SIZE,
                             species_threshold=None, target_class=COCO_BIRD_CLASS,
                             max_detections=10, min_box_size=50, max_box_size=800,
                             quality=95, deduplicate=False, similarity_threshold=5,
                             min_sharpness=None, min_edge_quality=None,
                             save_quality_report=False, remove_bg=False,
                             bg_color=GRAY_BACKGROUND_DEFAULT, bg_model='u2net',
                             bg_transparent=False, bg_fill_black=False,
                             crop_padding=0, resize_to_target=True, skip_blurry=False,
                             models: Optional[ExtractionModels] = None,
                             hash_cache: Optional[Dict[str, object]] = None, verbose: bool = True):
    """
    Extract bird crops from a single image.
    Same options and behaviour as extract_birds_from_video().
    """
    if not resize_to_target:
        target_image_size = 0

    settings = _build_settings(
        output_dir, bird_species=bird_species, threshold=threshold, target_image_size=target_image_size,
        species_threshold=species_threshold, target_class=target_class, max_detections=max_detections,
        min_box_size=min_box_size, max_box_size=max_box_size, quality=quality, skip_blurry=skip_blurry,
        deduplicate=deduplicate, similarity_threshold=similarity_threshold, min_sharpness=min_sharpness,
        min_edge_quality=min_edge_quality, save_quality_report=save_quality_report, remove_bg=remove_bg,
        bg_color=bg_color, bg_model=bg_model, bg_transparent=bg_transparent, bg_fill_black=bg_fill_black,
        crop_padding=crop_padding)

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    if models is None:
        models = load_models(detection_model, species_model)
    if verbose:
        _print_settings(settings, models)

    settings.output_path.mkdir(parents=True, exist_ok=True)
    if settings.deduplicate and hash_cache is None:
        hash_cache = load_existing_hashes(settings.output_path)

    session_id = f"{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stats = ExtractionStats()
    for box, conf in _detections(models, frame, settings):
        process_detection(frame, box, conf, settings, stats, models, hash_cache,
                          session_id, None, Path(image_path).name)

    _print_summary(settings, stats)
    return stats


def convert_bird_images(source_dir, target_dir, remove_bg=True, bg_color=GRAY_BACKGROUND_DEFAULT,
                        bg_model='u2net', bg_transparent=True, bg_fill_black=False,
                        crop_padding=0, min_sharpness=None, min_edge_quality=None,
                        quality_report=False, quality=95, deduplicate=False,
                        similarity_threshold=5, resize_to_target=True, target_image_size=TARGET_IMAGE_SIZE):
    """
    Convert already-extracted bird images with consistent processing.
    Useful for normalizing existing training datasets.

    - Loads existing bird crop images (no detection), keeping an existing alpha channel
    - Applies quality filtering, background removal, resizing and deduplication
    - Mirrors the source folder structure (species subdirectories) in target_dir

    Args:
        crop_padding: Pixels of padding added around each image before processing
        (other args as in extract_birds_from_video)
    """
    source_path = Path(source_dir).expanduser()
    target_path = Path(target_dir).expanduser()

    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return None

    target_resolved = target_path.resolve()
    image_files = sorted(
        p for p in source_path.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and target_resolved not in p.resolve().parents
    )

    if not image_files:
        print(f"❌ No images found in: {source_dir}")
        return None

    print(_("convert_images_found", count=len(image_files)))
    print(_("convert_source_dir", path=source_path))
    print(_("convert_target_dir", path=target_path))
    print()

    size = target_image_size if resize_to_target else 0
    converted_count = 0
    skipped_quality = 0
    skipped_duplicate = 0
    failed = 0
    hash_cache: Dict[str, object] = {}
    written = set()
    quality_stats = {'accepted': [], 'rejected_blur': [], 'rejected_edges': []}
    needs_metrics = min_sharpness is not None or min_edge_quality is not None or quality_report

    for idx, img_file in enumerate(image_files, 1):
        rel_path = img_file.relative_to(source_path)
        target_file = target_path / rel_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if idx % PROGRESS_LOG_INTERVAL == 0 or idx == 1:
            print(f"Processing {idx}/{len(image_files)}: {rel_path}")

        image = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"⚠️  Failed to load: {rel_path}")
            failed += 1
            continue
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image, alpha=255.0 / max(1, int(image.max())))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Padding canvas around the image
        if crop_padding > 0:
            image = cv2.copyMakeBorder(
                image, crop_padding, crop_padding, crop_padding, crop_padding, cv2.BORDER_CONSTANT,
                value=(0, 0, 0, 0) if image.shape[2] == 4 else tuple(int(c) for c in bg_color))

        if needs_metrics:
            metrics = calculate_motion_quality(image)
            ok, reason = is_motion_acceptable(metrics, min_sharpness, min_edge_quality)
            if not ok:
                skipped_quality += 1
                key = 'rejected_blur' if reason == 'low_sharpness' else 'rejected_edges'
                quality_stats[key].append(metrics['overall'])
                continue
            quality_stats['accepted'].append(metrics['overall'])

        if remove_bg:
            image = remove_background(
                _to_bgr(image), bg_color=bg_color, model_name=bg_model,
                transparent=bg_transparent, fill_black_areas=bg_fill_black, expand_mask=0)

        final_img = finalize_crop(image, size, bg_color)

        if deduplicate:
            img_hash = compute_hash(final_img)
            if find_duplicate(img_hash, hash_cache, similarity_threshold):
                skipped_duplicate += 1
                continue

        base = target_file.with_suffix('')
        ext = '.png' if final_img.mode == 'RGBA' else '.jpg'
        if base.with_suffix(ext) in written:
            # e.g. "bird.jpg" and "bird.png" in the same folder
            base = unique_path(base.with_suffix(ext)).with_suffix('')
        save_path = save_pil_image(final_img, base, quality)
        written.add(save_path)
        if deduplicate:
            hash_cache[str(save_path)] = img_hash

        converted_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("=" * 60)
    print(_("convert_target_directory", path=target_path))
    print(_("convert_total_processed", count=len(image_files)))
    print(f"✅ Converted: {converted_count}")
    if failed:
        print(f"⚠️  Failed to load: {failed}")
    if skipped_quality > 0:
        print(f"⏭️  Skipped (quality): {skipped_quality}")
    if skipped_duplicate > 0:
        print(f"⏭️  Skipped (duplicate): {skipped_duplicate}")

    if quality_report:
        print("\n" + "=" * 60)
        print(_("convert_quality_report"))
        print("=" * 60)

        total = len(image_files)
        accepted = len(quality_stats['accepted'])
        print(f"Total images: {total}")
        print(f"Accepted: {accepted} ({accepted / total * 100:.1f}%)")
        print(f"Rejected: {skipped_quality} ({skipped_quality / total * 100:.1f}%)")
        if quality_stats['accepted']:
            print(f"Average quality (accepted): {np.mean(quality_stats['accepted']):.2f}")
        if quality_stats['rejected_blur']:
            print(f"Average quality (rejected blur): {np.mean(quality_stats['rejected_blur']):.2f}")
        if quality_stats['rejected_edges']:
            print(f"Average quality (rejected edges): {np.mean(quality_stats['rejected_edges']):.2f}")

    return {
        'total': len(image_files),
        'converted': converted_count,
        'skipped_quality': skipped_quality,
        'skipped_duplicate': skipped_duplicate,
        'failed': failed,
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_bg_color(value: str) -> Tuple[int, int, int]:
    """Parse a color name or 'R,G,B' into an RGB tuple (argparse type)."""
    key = value.strip().lower()
    if key in BG_COLOR_NAMES:
        return BG_COLOR_NAMES[key]
    try:
        parts = tuple(int(p) for p in key.split(','))
    except ValueError:
        parts = ()
    if len(parts) != 3 or not all(0 <= p <= 255 for p in parts):
        raise argparse.ArgumentTypeError(
            f"invalid color '{value}': use R,G,B (0-255) or one of {', '.join(sorted(BG_COLOR_NAMES))}")
    return parts


def add_extract_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all extraction options (shared by `vogel-trainer extract` and this module's main)."""
    # Mode selection
    parser.add_argument('--convert', action='store_true',
                        help='Convert mode: process existing bird images (no detection). '
                             'Requires --source and --target.')
    parser.add_argument('--source', help='Source directory with existing bird images (--convert mode)')
    parser.add_argument('--target', help='Target directory for converted images (--convert mode)')

    # Input/output
    parser.add_argument('input', nargs='?',
                        help='Video/image file, directory, or glob pattern (e.g. "*.mp4", "~/Media/**/*")')
    parser.add_argument('--folder', help='Base directory for extracted bird images (required for extraction)')
    parser.add_argument('-o', '--output', dest='folder_alias', help=argparse.SUPPRESS)
    parser.add_argument('--bird', help='Manual bird species name (e.g. rotkehlchen). Creates subdirectory.')
    parser.add_argument('--species-model', help='Path to custom species classifier for automatic sorting')
    parser.add_argument('--recursive', '-r', action='store_true', help='Search directories recursively')

    # Detection
    parser.add_argument('--detection-model', default=None, help=f'YOLO detection model (default: {DEFAULT_MODEL})')
    parser.add_argument('--threshold', type=float, default=None,
                        help=f'Detection confidence threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--species-threshold', type=float, default=None,
                        help='Minimum species confidence, e.g. 0.85 (only with --species-model)')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help=f'Analyze every Nth video frame (default: {DEFAULT_SAMPLE_RATE})')
    parser.add_argument('--max-detections', type=int, default=10,
                        help='Maximum number of bird detections per frame (default: 10)')
    parser.add_argument('--min-box-size', type=int, default=50,
                        help='Minimum bounding box size in pixels (default: 50)')
    parser.add_argument('--max-box-size', type=int, default=800,
                        help='Maximum bounding box size in pixels (default: 800)')

    # Output image
    parser.add_argument('--image-size', type=int, default=TARGET_IMAGE_SIZE,
                        help=f'Target image size in pixels (default: {TARGET_IMAGE_SIZE}, 0 = original size)')
    parser.add_argument('--no-resize', action='store_true', help='Keep original size (same as --image-size 0)')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality 1-100 (default: 95)')
    parser.add_argument('--crop-padding', type=int, default=0,
                        help='Grow (positive) or shrink (negative) the detection box by this percentage. '
                             'With background removal, positive values also expand the mask by N pixels '
                             '(keeps feet/beak). In --convert mode: pixels of padding around each image.')

    # Quality and deduplication
    parser.add_argument('--skip-blurry', action='store_true', help='Skip blurry crops (simple Laplacian check)')
    parser.add_argument('--deduplicate', action='store_true',
                        help='Skip duplicate/similar images using perceptual hashing')
    parser.add_argument('--similarity-threshold', type=int, default=5,
                        help='Hamming distance threshold for duplicates (0-64, default: 5)')
    parser.add_argument('--min-sharpness', type=float, default=None,
                        help='Minimum sharpness score (Laplacian variance), typical 100-300')
    parser.add_argument('--min-edge-quality', type=float, default=None,
                        help='Minimum edge quality score (Sobel gradient), typical 50-150')
    parser.add_argument('--quality-report', '--save-quality-report', dest='quality_report',
                        action='store_true', help='Print detailed quality statistics')

    # Background removal
    parser.add_argument('--bg-remove', '--remove-background', dest='bg_remove', action='store_true',
                        help='Remove background using rembg AI model')
    parser.add_argument('--bg-color', type=parse_bg_color, default=BG_COLOR_NAMES['gray'],
                        help='Background color: white, black, gray, green-screen, blue-screen or R,G,B '
                             '(default: gray). Ignored with --bg-transparent.')
    parser.add_argument('--bg-model', default='u2net', choices=REMBG_MODELS,
                        help='Background removal model (default: u2net)')
    parser.add_argument('--bg-transparent', dest='bg_transparent', action='store_true', default=False,
                        help='Transparent background (PNG with alpha channel)')
    parser.add_argument('--no-bg-transparent', dest='bg_transparent', action='store_false',
                        help='Use colored background (default)')
    parser.add_argument('--bg-fill-black', dest='bg_fill_black', action='store_true', default=False,
                        help='Make black padding/background areas transparent (with --bg-transparent)')
    parser.add_argument('--no-bg-fill-black', dest='bg_fill_black', action='store_false',
                        help='Keep black padding areas (default)')

    parser.add_argument('--log', action='store_true',
                        help='Also write console output to a log file '
                             '(/var/log/vogel-kamera-linux/ or ~/.local/state/vogel-model-trainer/logs/)')


def _validate_args(args, parser):
    if args.quality is not None and not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    if args.sample_rate is not None and args.sample_rate < 1:
        parser.error("--sample-rate must be >= 1")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.species_threshold is not None and not 0 <= args.species_threshold <= 1:
        parser.error("--species-threshold must be between 0 and 1")
    if not 0 <= args.similarity_threshold <= 64:
        parser.error("--similarity-threshold must be between 0 and 64")
    if args.image_size < 0:
        parser.error("--image-size must be >= 0")


def _collect_input_files(input_arg: str, recursive: bool) -> List[Path]:
    input_path = Path(input_arg).expanduser()
    supported = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

    if any(ch in input_arg for ch in '*?['):
        candidates = [Path(p) for p in glob.glob(str(input_path), recursive=recursive)]
    elif input_path.is_dir():
        candidates = list(input_path.glob('**/*' if recursive else '*'))
    elif input_path.is_file():
        candidates = [input_path]
    else:
        raise FileNotFoundError(f"File/directory not found: {input_arg}")

    return sorted({p for p in candidates if p.is_file() and p.suffix.lower() in supported})


def run(args, parser: argparse.ArgumentParser) -> int:
    """Run extraction/conversion from parsed arguments. Returns an exit code."""
    from vogel_model_trainer.utils.logging import capture_output

    _validate_args(args, parser)
    bg_color_rgb = args.bg_color
    bg_color_bgr = (bg_color_rgb[2], bg_color_rgb[1], bg_color_rgb[0])
    image_size = 0 if args.no_resize else args.image_size

    with capture_output('extract', enabled=args.log):
        if args.convert:
            if not args.source or not args.target:
                parser.error("--convert mode requires both --source and --target")
            print("=" * 70)
            print("🔄 CONVERT MODE: Processing existing bird images")
            print("=" * 70)
            print()
            result = convert_bird_images(
                source_dir=args.source,
                target_dir=args.target,
                remove_bg=args.bg_remove,
                bg_color=bg_color_bgr,
                bg_model=args.bg_model,
                bg_transparent=args.bg_transparent,
                bg_fill_black=args.bg_fill_black,
                crop_padding=args.crop_padding,
                min_sharpness=args.min_sharpness,
                min_edge_quality=args.min_edge_quality,
                quality_report=args.quality_report,
                quality=args.quality,
                deduplicate=args.deduplicate,
                similarity_threshold=args.similarity_threshold,
                resize_to_target=image_size > 0,
                target_image_size=image_size,
            )
            return 0 if result is not None else 1

        if not args.input:
            parser.error("input is required for extraction mode (or use --convert)")
        output_dir = args.folder or args.folder_alias
        if not output_dir:
            parser.error("--folder is required for extraction mode")

        try:
            input_files = _collect_input_files(args.input, args.recursive)
        except FileNotFoundError as exc:
            print(f"❌ {exc}")
            return 1
        if not input_files:
            print(f"❌ No video or image files found matching: {args.input}")
            return 1

        video_files = [f for f in input_files if f.suffix.lower() in VIDEO_EXTENSIONS]
        image_files = [f for f in input_files if f.suffix.lower() in IMAGE_EXTENSIONS]

        print(_("files_found_header", total=len(input_files)))
        if video_files:
            print(_("files_videos_count", count=len(video_files)))
        if image_files:
            print(_("files_images_count", count=len(image_files)))
        all_files = video_files + image_files
        for i, f in enumerate(all_files[:10], 1):
            file_type = "🎬" if f in video_files else "🖼️"
            print(f"   {i}. {file_type} {f.name}")
        if len(all_files) > 10:
            print(_("files_and_more", count=len(all_files) - 10))
        print()

        if args.bird and args.species_model:
            print("⚠️  Warning: Both --bird and --species-model specified. Using auto-classification.")
        if args.species_threshold is not None and not args.species_model:
            print(_('species_threshold_ignored'))

        # Load models and the deduplication cache once for all files
        models = load_models(args.detection_model, args.species_model)
        hash_cache = load_existing_hashes(Path(output_dir).expanduser()) if args.deduplicate else None

        extract_params = {
            'output_dir': output_dir,
            'bird_species': args.bird,
            'threshold': args.threshold,
            'target_image_size': image_size,
            'species_threshold': args.species_threshold,
            'max_detections': args.max_detections,
            'min_box_size': args.min_box_size,
            'max_box_size': args.max_box_size,
            'quality': args.quality,
            'skip_blurry': args.skip_blurry,
            'deduplicate': args.deduplicate,
            'similarity_threshold': args.similarity_threshold,
            'min_sharpness': args.min_sharpness,
            'min_edge_quality': args.min_edge_quality,
            'save_quality_report': args.quality_report,
            'remove_bg': args.bg_remove,
            'bg_color': bg_color_bgr,
            'bg_model': args.bg_model,
            'bg_transparent': args.bg_transparent,
            'bg_fill_black': args.bg_fill_black,
            'crop_padding': args.crop_padding,
            'models': models,
            'hash_cache': hash_cache,
        }

        errors = 0
        if video_files:
            print(f"\n{'=' * 70}")
            print(_("processing_videos_header", count=len(video_files)))
            print(f"{'=' * 70}\n")
            for idx, video_file in enumerate(video_files, 1):
                print(f"\n{'=' * 70}")
                print(_("processing_video_item", idx=idx, total=len(video_files), name=video_file.name))
                print(f"{'=' * 70}")
                try:
                    extract_birds_from_video(video_path=str(video_file), sample_rate=args.sample_rate,
                                             **extract_params)
                except Exception as e:
                    errors += 1
                    print(_('error_processing', name=video_file.name, error=e))
                    print(_('continuing'))

        if image_files:
            print(f"\n{'=' * 70}")
            print(_("processing_images_header", count=len(image_files)))
            print(f"{'=' * 70}\n")
            for idx, image_file in enumerate(image_files, 1):
                print(f"\n{'=' * 70}")
                print(_("processing_image_item", idx=idx, total=len(image_files), name=image_file.name))
                print(f"{'=' * 70}")
                try:
                    extract_birds_from_image(image_path=str(image_file), verbose=(idx == 1), **extract_params)
                except Exception as e:
                    errors += 1
                    print(_('error_processing', name=image_file.name, error=e))
                    print(_('continuing'))

        print(f"\n{'=' * 70}")
        print(_('all_videos_processed', path=output_dir))
        print(f"{'=' * 70}")
        return 1 if errors and errors == len(all_files) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Extract bird crops from videos or images for training data collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vogel-trainer extract video.mp4 --folder data/ --bird rotkehlchen
  vogel-trainer extract "~/Videos/*.mp4" --folder data/ --species-model ~/models/classifier/final/
  vogel-trainer extract ~/Videos/ --folder data/ --bird kohlmeise --recursive
  vogel-trainer extract photo.jpg --folder data/ --image-size 0
  vogel-trainer extract --convert --source ~/data --target ~/data-transparent \\
      --bg-remove --bg-transparent --crop-padding 10 --min-sharpness 80
        """
    )
    add_extract_arguments(parser)
    args = parser.parse_args(argv)
    return run(args, parser)


if __name__ == '__main__':
    sys.exit(main())
