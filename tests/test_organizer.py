"""Tests for train/val organization."""
import pytest

from conftest import write_image
from vogel_model_trainer.core import organizer


def _make_species(root, species, sessions, per_session):
    for s in range(sessions):
        session = f"video{s}_20250101_12000{s}"
        for i in range(per_session):
            write_image(root / species / f"{session}_{i:08x}_f{i:06d}_c0.90.jpg", seed=s * 100 + i)


def test_session_key():
    assert organizer.session_key("rec_20250101_120000_20261106_101010_deadbeef_f000001_c0.90.jpg") == \
        "rec_20250101_120000_20261106_101010"
    assert organizer.session_key("some_photo.jpg") == "some_photo"


def test_session_split_has_no_leakage(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_species(src, "amsel", sessions=5, per_session=4)
    _make_species(src, "meise", sessions=5, per_session=4)

    stats = organizer.organize_dataset(src, out, split_by="session")

    for species in ("amsel", "meise"):
        train = {organizer.session_key(p) for p in (out / "train" / species).iterdir()}
        val = {organizer.session_key(p) for p in (out / "val" / species).iterdir()}
        assert train and val
        assert not train & val
        assert stats[species]["split_by"] == "session"


def test_split_is_reproducible(tmp_path):
    src = tmp_path / "src"
    _make_species(src, "amsel", sessions=6, per_session=3)
    _make_species(src, "meise", sessions=6, per_session=3)
    organizer.organize_dataset(src, tmp_path / "a", seed=7)
    organizer.organize_dataset(src, tmp_path / "b", seed=7)
    a = sorted(p.name for p in (tmp_path / "a" / "val").rglob("*.jpg"))
    b = sorted(p.name for p in (tmp_path / "b" / "val").rglob("*.jpg"))
    assert a == b


def test_single_session_falls_back_to_image_split(tmp_path):
    src = tmp_path / "src"
    _make_species(src, "amsel", sessions=1, per_session=10)
    stats = organizer.organize_dataset(src, tmp_path / "out")
    assert stats["amsel"]["split_by"] == "image"
    assert stats["amsel"]["val"] == 2


def test_max_images_does_not_delete_sources(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_species(src, "amsel", sessions=4, per_session=5)  # 20
    _make_species(src, "meise", sessions=4, per_session=3)  # 12
    organizer.organize_dataset(src, out, max_images_per_class=12)

    assert len(list((src / "amsel").iterdir())) == 20  # untouched
    copied = len(list((out / "train" / "amsel").iterdir())) + len(list((out / "val" / "amsel").iterdir()))
    assert copied == 12


def test_imbalance_error_happens_before_anything_is_written(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_species(src, "amsel", sessions=4, per_session=10)
    _make_species(src, "meise", sessions=2, per_session=2)
    with pytest.raises(ValueError):
        organizer.organize_dataset(src, out)
    assert not out.exists()
    assert len(list((src / "amsel").iterdir())) == 40


def test_legacy_video_folders_and_output_inside_source(tmp_path):
    src = tmp_path / "src"
    write_image(src / "kohlmeise_video20241108" / "a.jpg", seed=1)
    write_image(src / "kohlmeise_video20241109" / "b.png", seed=2)
    write_image(src / "amsel" / "c.jpeg", seed=3)
    write_image(src / "amsel" / "d.jpg", seed=4)
    out = src / "organized"
    organizer.organize_dataset(src, out, tolerance_percent=100)
    # Second run must not pick up organized/ as a species
    images = organizer.collect_images_by_species(src, exclude_dirs=[out])
    assert set(images) == {"kohlmeise", "amsel"}
    assert len(images["kohlmeise"]) == 2


def test_clean_removes_old_split(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_species(src, "amsel", sessions=3, per_session=2)
    write_image(out / "train" / "stale" / "old.jpg")
    organizer.organize_dataset(src, out, clean=True)
    assert not (out / "train" / "stale").exists()
