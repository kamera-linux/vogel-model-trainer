"""Shared test helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class FakeBox:
    def __init__(self, xyxy, conf, cls=14):
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([cls])


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """Stands in for ultralytics.YOLO: returns the same boxes for every frame."""

    def __init__(self, boxes):
        self.boxes = [FakeBox(*b) for b in boxes]
        self.calls = 0

    def __call__(self, frame, **kwargs):
        self.calls += 1
        return [FakeResult(self.boxes)]


def textured_image(width, height, seed=0):
    """Random, sharp RGB image (sharp enough for blur filters, unique per seed)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def write_image(path, array=None, mode="RGB", size=(64, 64), seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    if array is None:
        array = textured_image(*size, seed=seed)
    Image.fromarray(array, mode=None).convert(mode).save(path)
    return path


@pytest.fixture
def fake_yolo():
    return FakeYOLO
