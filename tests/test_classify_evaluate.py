"""Tests for classify (file handling) and evaluate (metrics)."""
from pathlib import Path

from conftest import write_image
from vogel_model_trainer.core import classifier, evaluator


def test_get_image_files_excludes_sort_output(tmp_path):
    write_image(tmp_path / "in" / "a.jpg")
    write_image(tmp_path / "in" / "sorted" / "amsel" / "b.jpg")
    files = classifier.get_image_files(tmp_path / "in", recursive=True, exclude_dirs=[tmp_path / "in" / "sorted"])
    assert [p.name for p in files] == ["a.jpg"]


def test_sort_and_delete_keeps_unprocessed_and_output(tmp_path):
    inp = tmp_path / "in"
    a = write_image(inp / "a.jpg", seed=1)
    b = write_image(inp / "sub" / "b.jpg", seed=2)
    broken = inp / "broken.jpg"
    broken.write_bytes(b"not an image")
    sort_out = inp / "sorted"  # output INSIDE the input folder

    results = {a: [("amsel", 0.9)], b: [("meise", 0.4)]}
    stats, processed = classifier.sort_images(results, sort_out, min_confidence=0.5, return_processed=True)
    assert stats == {"amsel": 1, "unknown": 1}

    remaining = classifier.delete_processed_sources(processed, inp, sort_out, moved=False)

    assert (sort_out / "amsel" / "a.jpg").exists()
    assert (sort_out / "unknown" / "b.jpg").exists()
    assert broken.exists()           # never classified -> kept
    assert not a.exists() and not b.exists()
    assert not (inp / "sub").exists()  # empty folder removed
    assert remaining == 1


def test_sort_handles_name_conflicts(tmp_path):
    a = write_image(tmp_path / "x" / "bird.jpg", seed=1)
    b = write_image(tmp_path / "y" / "bird.jpg", seed=2)
    classifier.sort_images({a: [("amsel", 1.0)], b: [("amsel", 1.0)]}, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out" / "amsel").iterdir()) == ["bird.jpg", "bird_1.jpg"]


def test_calculate_metrics():
    cm = {"a": {"a": 8, "b": 2}, "b": {"b": 5, "c": 1}}
    metrics = evaluator.calculate_metrics(cm, ["a", "b", "c"])
    assert metrics["a"]["recall"] == 0.8
    assert metrics["b"]["precision"] == 5 / 7
    assert metrics["c"]["false_positives"] == 1


def test_evaluate_separates_low_confidence(tmp_path, monkeypatch):
    for sp in ("amsel", "meise"):
        for i in range(2):
            write_image(tmp_path / "test" / sp / f"{i}.jpg", seed=i)

    class FakeConfig:
        id2label = {0: "amsel", 1: "meise"}

    class FakeModel:
        config = FakeConfig()

    monkeypatch.setattr(evaluator, "load_model", lambda path: (FakeModel(), None, ["amsel", "meise"]))
    preds = {
        "amsel": [("amsel", 0.95), ("amsel", 0.40)],  # correct, uncertain
        "meise": [("amsel", 0.90), ("unbekannt", 0.80)],  # wrong, wrong (species without test folder)
    }
    monkeypatch.setattr(evaluator, "classify_images_batched",
                        lambda paths, *a, **k: preds[Path(paths[0]).parent.name])

    out_csv = tmp_path / "mis.csv"
    result = evaluator.evaluate_model("m", tmp_path / "test", export_misclassified=str(out_csv), min_confidence=0.5)

    assert result["correct"] == 1
    assert result["uncertain"] == 1
    assert result["wrong"] == 2
    assert "unbekannt" in result["metrics"]
    assert "low_confidence" in out_csv.read_text()
