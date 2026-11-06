"""
Shared image helpers.

All inference paths (validation during training, test, classify, evaluate,
extract auto-sorting) must convert images the same way, otherwise transparent
PNGs end up on different backgrounds during training and inference.
"""

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

from PIL import Image

# Neutral gray background used for transparent images at inference time.
INFERENCE_BACKGROUND: Tuple[int, int, int] = (128, 128, 128)

# Image file extensions (lowercase) recognized throughout the package.
IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'})

# Folders created by the cleanup commands. They are skipped when scanning a
# dataset recursively so that a second run does not process its own output.
RESERVED_OUTPUT_DIRS = frozenset({'duplicates', 'low_quality', 'invalid_gray', 'invalid_transparent'})


def to_rgb(img: Image.Image, background: Sequence[int] = INFERENCE_BACKGROUND) -> Image.Image:
    """
    Convert a PIL image to RGB.

    Images with transparency are composited onto ``background`` instead of
    simply dropping the alpha channel (which would yield a black background).
    """
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        canvas = Image.new('RGB', rgba.size, tuple(background))
        canvas.paste(rgba, mask=rgba.split()[3])
        return canvas
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img


def load_rgb(path: Union[str, Path], background: Sequence[int] = INFERENCE_BACKGROUND) -> Image.Image:
    """Open an image file and return a fully loaded RGB image (file handle closed)."""
    with Image.open(path) as img:
        img.load()
        return to_rgb(img, background)


def is_image_file(path: Path) -> bool:
    """Return True if ``path`` has a supported image extension (case-insensitive)."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_images(directory: Union[str, Path], recursive: bool = True,
                exclude_dirs: Iterable[Union[str, Path]] = (),
                skip_reserved: bool = True) -> List[Path]:
    """
    Find image files in ``directory``.

    Args:
        directory: Directory to scan
        recursive: Descend into subdirectories
        exclude_dirs: Absolute directories whose contents are skipped
        skip_reserved: Skip the cleanup output folders (duplicates/, low_quality/, ...)

    Returns:
        Sorted list of unique image paths
    """
    directory = Path(directory).expanduser()
    excluded = [Path(d).expanduser().resolve() for d in exclude_dirs]
    pattern = '**/*' if recursive else '*'

    result = set()
    for path in directory.glob(pattern):
        if not path.is_file() or not is_image_file(path):
            continue
        rel_parts = path.relative_to(directory).parts[:-1]
        if skip_reserved and any(part in RESERVED_OUTPUT_DIRS for part in rel_parts):
            continue
        if excluded:
            resolved = path.resolve()
            if any(resolved.is_relative_to(ex) if hasattr(resolved, 'is_relative_to')
                   else str(resolved).startswith(str(ex) + '/') for ex in excluded):
                continue
        result.add(path)
    return sorted(result)


def unique_path(path: Path) -> Path:
    """Return ``path`` or ``path`` with a numeric suffix if it already exists."""
    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate
