"""Tests for deduplicate / quality-check / clean-* and shared helpers."""
import io
import subprocess
import sys

import numpy as np
from PIL import Image

from conftest import textured_image, write_image
from vogel_model_trainer.core import deduplicator
from vogel_model_trainer.utils.image import to_rgb
from vogel_model_trainer.utils.logging import Tee


def test_find_duplicates_skips_previous_output_folder(tmp_path):
    img = textured_image(64, 64, seed=5)
    write_image(tmp_path / "a" / "1.jpg", img)
    write_image(tmp_path / "a" / "2.jpg", img)
    write_image(tmp_path / "duplicates" / "a" / "3.jpg", img)  # from an earlier --mode move
    write_image(tmp_path / "b" / "other.jpg", seed=9)

    dups, scanned = deduplicator.find_duplicates(tmp_path, recursive=True, return_count=True)
    assert scanned == 3
    assert sum(len(v) for v in dups.values()) == 1


def test_deduplicate_move_reports_total(tmp_path):
    img = textured_image(64, 64, seed=5)
    write_image(tmp_path / "1.jpg", img)
    write_image(tmp_path / "2.jpg", img)
    stats = deduplicator.deduplicate_dataset(tmp_path, mode="move", recursive=True)
    assert stats["total_images"] == 2
    assert stats["moved"] == 1
    # Second run must not see the moved file again
    assert deduplicator.deduplicate_dataset(tmp_path, mode="report", recursive=True)["duplicates_found"] == 0


def test_quality_check_transparent_png_not_too_dark(tmp_path):
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[25:75, 25:75, :3] = textured_image(50, 50, seed=3)
    rgba[25:75, 25:75, 3] = 255
    path = tmp_path / "bird.png"
    Image.fromarray(rgba, "RGBA").save(path)
    result = deduplicator.check_image_quality(path, blur_threshold=0, min_filesize=0, check_brightness=True)
    assert "too_dark" not in result["issue_codes"]


def test_quality_check_counts_all_images(tmp_path):
    write_image(tmp_path / "ok.jpg", size=(200, 200))
    Image.new("RGB", (10, 10), (128, 128, 128)).save(tmp_path / "tiny.jpg")
    stats = deduplicator.quality_check_dataset(tmp_path, min_filesize=0, mode="report")
    assert stats["total_checked"] == 2
    assert stats["low_quality_found"] == 1


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "vogel_model_trainer.cli.main", *args],
                          capture_output=True, text=True)


def test_clean_gray_actually_checks_images(tmp_path):
    Image.new("RGB", (50, 50), (128, 128, 128)).save(tmp_path / "all_gray.jpg")
    write_image(tmp_path / "sub" / "colorful.jpg", np.full((50, 50, 3), (200, 30, 30), np.uint8))
    result = _cli("clean-gray", str(tmp_path), "--mode", "move", "--recursive")
    assert result.returncode == 0, result.stderr
    assert "Total images checked: 2" in result.stdout
    assert (tmp_path / "invalid_gray" / "all_gray.jpg").exists()
    assert (tmp_path / "invalid_gray" / "sub" / "colorful.jpg").exists()


def test_cli_errors_return_nonzero(tmp_path):
    result = _cli("organize", str(tmp_path / "missing"), "-o", str(tmp_path / "out"))
    assert result.returncode == 1


def test_to_rgb_uses_gray_background():
    rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    assert to_rgb(rgba).getpixel((0, 0)) == (128, 128, 128)


def test_tee_supports_isatty_and_writes_everywhere():
    a, b = io.StringIO(), io.StringIO()
    tee = Tee(a, b)
    tee.write("hello")
    assert a.getvalue() == b.getvalue() == "hello"
    assert tee.isatty() is False
