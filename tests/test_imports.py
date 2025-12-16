"""Test basic imports and module structure."""
import pytest


def test_version_import():
    """Test that version can be imported."""
    from vogel_model_trainer import __version__
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__.split('.')) >= 2


def test_cli_imports():
    """Test that CLI modules can be imported."""
    from vogel_model_trainer.cli.main import main
    assert callable(main)


def test_core_imports():
    """Test that core modules can be imported."""
    from vogel_model_trainer.core.trainer import train_model
    from vogel_model_trainer.core.extractor import extract_birds
    from vogel_model_trainer.core.organizer import organize_dataset
    from vogel_model_trainer.core.deduplicator import deduplicate_images
    from vogel_model_trainer.core.tester import test_model
    
    assert callable(train_model)
    assert callable(extract_birds)
    assert callable(organize_dataset)
    assert callable(deduplicate_images)
    assert callable(test_model)


def test_i18n_import():
    """Test that i18n module can be imported."""
    from vogel_model_trainer.i18n import _
    assert callable(_)
    
    # Test basic translation
    test_key = 'train_starting'
    result = _(test_key)
    assert isinstance(result, str)
