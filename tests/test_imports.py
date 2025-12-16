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
    from vogel_model_trainer.core import trainer
    from vogel_model_trainer.core import extractor
    from vogel_model_trainer.core import organizer
    from vogel_model_trainer.core import deduplicator
    from vogel_model_trainer.core import tester
    from vogel_model_trainer.core import evaluator
    
    # Verify modules loaded successfully
    assert trainer is not None
    assert extractor is not None
    assert organizer is not None
    assert deduplicator is not None
    assert tester is not None
    assert evaluator is not None


def test_i18n_import():
    """Test that i18n module can be imported."""
    from vogel_model_trainer.i18n import _
    assert callable(_)
    
    # Test basic translation
    test_key = 'train_starting'
    result = _(test_key)
    assert isinstance(result, str)
