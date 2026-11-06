"""Tests for trainer wiring and translations."""
import ast
import re
from pathlib import Path

import torch
from PIL import Image

from vogel_model_trainer import i18n
from vogel_model_trainer.core import trainer

SRC = Path(__file__).parent.parent / "src" / "vogel_model_trainer"


def test_prepare_model_uses_requested_model(monkeypatch):
    loaded = []

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loaded.append(name)
            return object()

    monkeypatch.setattr(trainer, "AutoImageProcessor", FakeAuto)
    monkeypatch.setattr(trainer, "AutoModelForImageClassification", FakeAuto)
    trainer.prepare_model_and_processor(["a", "b"], "google/efficientnet-b3")
    assert loaded == ["google/efficientnet-b3", "google/efficientnet-b3"]


def test_trainer_processor_kwarg_matches_installed_transformers():
    import inspect
    kwarg = trainer._trainer_processor_kwarg("proc")
    assert list(kwarg)[0] in inspect.signature(trainer.Trainer.__init__).parameters


class _FakeProcessor:
    def __call__(self, img, return_tensors="pt"):
        arr = torch.tensor(list(img.resize((8, 8)).getdata()), dtype=torch.float32)
        return {"pixel_values": arr.view(1, 8, 8, 3).permute(0, 3, 1, 2)}


def test_validation_transform_uses_gray_background():
    transform = trainer.make_transform(_FakeProcessor(), is_training=False)
    out = transform({"image": [Image.new("RGBA", (8, 8), (255, 0, 0, 0))], "label": [0]})
    assert torch.all(out["pixel_values"][0] == 128)
    assert out["label"] == [0]


def test_training_transform_is_random_per_call():
    torch.manual_seed(0)
    transform = trainer.make_transform(_FakeProcessor(), is_training=True, augmentation_strength="heavy",
                                       image_size=8)
    img = Image.effect_noise((32, 32), 64).convert("RGB")
    results = {tuple(transform({"image": [img], "label": [0]})["pixel_values"][0].flatten().tolist())
               for _ in range(5)}
    assert len(results) > 1


def test_graceful_stop_callback(monkeypatch):
    class Control:
        should_training_stop = False
        should_save = False

    monkeypatch.setattr(trainer, "interrupted", True)
    control = trainer.GracefulStopCallback().on_step_end(None, None, Control())
    assert control.should_training_stop and control.should_save


def _used_keys():
    keys = set()
    for path in SRC.rglob("*.py"):
        if path.name == "i18n.py":
            continue
        keys |= set(re.findall(r"""\b_\(\s*['"]([A-Za-z0-9_]+)['"]""", path.read_text(encoding="utf-8")))
    return keys


def test_all_used_translation_keys_exist():
    for lang, table in i18n.TRANSLATIONS.items():
        missing = _used_keys() - table.keys()
        assert not missing, f"{lang}: {sorted(missing)}"


def test_languages_have_same_keys_and_placeholders():
    en = i18n.TRANSLATIONS["en"]
    for lang in ("de", "ja"):
        table = i18n.TRANSLATIONS[lang]
        assert set(table) == set(en), lang
        for key, text in en.items():
            assert set(re.findall(r"{(\w+)", text)) == set(re.findall(r"{(\w+)", table[key])), (lang, key)


def test_no_duplicate_keys_in_translation_source():
    tree = ast.parse((SRC / "i18n.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            assert len(keys) == len(set(keys))
