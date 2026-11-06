"""Tests for the shared crop pipeline (no real YOLO/classifier needed)."""
import argparse

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import FakeYOLO, textured_image
from vogel_model_trainer.core import extractor
from vogel_model_trainer.core.extractor import ExtractionModels


def _frame(path, w=640, h=480, seed=1):
    frame = textured_image(w, h, seed=seed)
    cv2.imwrite(str(path), frame)
    return path


def test_image_extraction_keeps_aspect_ratio(tmp_path):
    img = _frame(tmp_path / "in.jpg")
    models = ExtractionModels(yolo=FakeYOLO([((100, 100, 300, 200), 0.9)]))  # 200x100 box

    stats = extractor.extract_birds_from_image(img, tmp_path / "out", models=models, bg_color=(10, 20, 30))

    assert stats.exported == 1
    saved = next((tmp_path / "out").glob("*.jpg"))
    with Image.open(saved) as out:
        assert out.size == (224, 224)
        # Top padding row uses bg_color (BGR 10,20,30 -> RGB 30,20,10), bird is centered
        top = np.asarray(out)[0, 112]
        assert np.allclose(top, (30, 20, 10), atol=8)


def test_original_size_with_image_size_zero(tmp_path):
    img = _frame(tmp_path / "in.jpg")
    models = ExtractionModels(yolo=FakeYOLO([((100, 100, 300, 200), 0.9)]))
    extractor.extract_birds_from_image(img, tmp_path / "out", models=models, target_image_size=0)
    with Image.open(next((tmp_path / "out").glob("*.jpg"))) as out:
        assert out.size == (200, 100)


def test_crop_padding_is_percentage_in_image_mode(tmp_path):
    img = _frame(tmp_path / "in.jpg")
    models = ExtractionModels(yolo=FakeYOLO([((100, 100, 300, 200), 0.9)]))
    extractor.extract_birds_from_image(img, tmp_path / "out", models=models, target_image_size=0, crop_padding=50)
    with Image.open(next((tmp_path / "out").glob("*.jpg"))) as out:
        assert out.size == (400, 200)  # 50 % on each side


def test_species_threshold_without_classifier_keeps_birds(tmp_path):
    img = _frame(tmp_path / "in.jpg")
    models = ExtractionModels(yolo=FakeYOLO([((100, 100, 300, 200), 0.9)]))
    stats = extractor.extract_birds_from_image(img, tmp_path / "out", models=models, species_threshold=0.9)
    assert stats.exported == 1


def test_manual_species_creates_subfolder(tmp_path):
    img = _frame(tmp_path / "in.jpg")
    models = ExtractionModels(yolo=FakeYOLO([((100, 100, 300, 200), 0.9)]))
    extractor.extract_birds_from_image(img, tmp_path / "out", models=models, bird_species="amsel")
    assert len(list((tmp_path / "out" / "amsel").glob("*.jpg"))) == 1


def test_box_filters_and_max_detections(tmp_path):
    img = _frame(tmp_path / "in.jpg")
    boxes = [
        ((0, 0, 20, 20), 0.9),        # too small
        ((10, 10, 110, 110), 0.9),    # ok
        ((200, 200, 300, 300), 0.9),  # ok but over max_detections
        ((50, 50, 150, 150), 0.2),    # below threshold
    ]
    models = ExtractionModels(yolo=FakeYOLO(boxes))
    stats = extractor.extract_birds_from_image(img, tmp_path / "out", models=models, max_detections=1)
    assert stats.detected == 1
    assert stats.exported == 1


def test_deduplication_uses_existing_and_new_images(tmp_path):
    img = _frame(tmp_path / "in.jpg")
    models = ExtractionModels(yolo=FakeYOLO([((100, 100, 300, 200), 0.9)]))
    first = extractor.extract_birds_from_image(img, tmp_path / "out", models=models, deduplicate=True)
    # Second run: same crop must be detected as duplicate of the file on disk
    second = extractor.extract_birds_from_image(img, tmp_path / "out", models=models, deduplicate=True)
    assert first.exported == 1
    assert second.exported == 0
    assert second.duplicates == 1


def test_quality_report_counts_rejected_as_detected(tmp_path):
    flat = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "flat.jpg"), flat)
    models = ExtractionModels(yolo=FakeYOLO([((100, 100, 300, 200), 0.9)]))
    stats = extractor.extract_birds_from_image(tmp_path / "flat.jpg", tmp_path / "out", models=models,
                                               min_sharpness=50, save_quality_report=True)
    assert stats.detected == 1
    assert stats.motion_rejected == 1
    assert stats.exported == 0


def test_video_extraction_reuses_model_and_handles_sampling(tmp_path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(12):
        writer.write(textured_image(320, 240, seed=i))
    writer.release()

    yolo = FakeYOLO([((50, 50, 150, 150), 0.9)])
    models = ExtractionModels(yolo=yolo)
    stats = extractor.extract_birds_from_video(video, tmp_path / "out", models=models, sample_rate=4)
    assert yolo.calls == 3  # frames 0, 4, 8
    assert stats.exported == 3
    names = sorted(p.name for p in (tmp_path / "out").glob("*.jpg"))
    assert all(n.startswith("clip_") and "_f0000" in n for n in names)


def test_convert_keeps_alpha_and_avoids_name_collisions(tmp_path):
    src = tmp_path / "src" / "amsel"
    src.mkdir(parents=True)
    rgba = np.zeros((60, 80, 4), dtype=np.uint8)
    rgba[10:50, 10:70] = (0, 0, 255, 255)
    cv2.imwrite(str(src / "bird.png"), rgba)
    cv2.imwrite(str(src / "bird.jpg"), textured_image(80, 60))

    result = extractor.convert_bird_images(tmp_path / "src", tmp_path / "dst", remove_bg=False,
                                           bg_transparent=False)
    assert result["converted"] == 2
    files = sorted(p.name for p in (tmp_path / "dst" / "amsel").iterdir())
    assert files == ["bird.jpg", "bird.png"]
    with Image.open(tmp_path / "dst" / "amsel" / "bird.png") as out:
        assert out.mode == "RGBA" and out.size == (224, 224)


def test_motion_quality_accepts_bgra():
    bgra = np.dstack([textured_image(50, 50), np.full((50, 50), 255, np.uint8)])
    metrics = extractor.calculate_motion_quality(bgra)
    assert metrics["sharpness"] > 0


def test_rembg_session_is_cached_per_model(monkeypatch):
    created = []
    monkeypatch.setattr(extractor, "_rembg_remove", lambda *a, **k: None)
    monkeypatch.setattr(extractor, "_rembg_new_session", lambda name: created.append(name) or name)
    monkeypatch.setattr(extractor, "_rembg_sessions", {})
    monkeypatch.setattr(extractor, "_rembg_unavailable", False)

    for _ in range(3):
        extractor._get_rembg_session("u2netp")
    extractor._get_rembg_session("isnet-general-use")
    assert created == ["u2netp", "isnet-general-use"]


@pytest.mark.parametrize("value,expected", [
    ("gray", (128, 128, 128)),
    ("green-screen", (0, 255, 0)),
    ("1,2,3", (1, 2, 3)),
])
def test_parse_bg_color(value, expected):
    assert extractor.parse_bg_color(value) == expected


def test_parse_bg_color_rejects_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        extractor.parse_bg_color("300,0,0")


def test_extract_parser_accepts_all_documented_options():
    parser = argparse.ArgumentParser()
    extractor.add_extract_arguments(parser)
    args = parser.parse_args([
        "video.mp4", "--folder", "out", "--image-size", "0", "--max-detections", "3",
        "--min-box-size", "10", "--max-box-size", "500", "--skip-blurry", "--log",
        "--remove-background", "--save-quality-report", "--bg-color", "white", "--bg-model", "u2netp",
    ])
    assert args.image_size == 0
    assert args.max_detections == 3
    assert args.min_box_size == 10 and args.max_box_size == 500
    assert args.skip_blurry and args.log
    assert args.bg_remove and args.quality_report
    assert args.bg_color == (255, 255, 255)
