"""Test CLI functionality."""
import subprocess
import pytest


def test_vogel_trainer_command_exists():
    """Test that vogel-trainer command is installed."""
    result = subprocess.run(
        ['vogel-trainer', '--version'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'vogel-trainer' in result.stdout.lower() or 'vogel-trainer' in result.stderr.lower()


def test_vogel_trainer_help():
    """Test that help command works."""
    result = subprocess.run(
        ['vogel-trainer', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    output = result.stdout.lower()
    assert 'extract' in output
    assert 'train' in output
    assert 'organize' in output


def test_extract_help():
    """Test extract command help."""
    result = subprocess.run(
        ['vogel-trainer', 'extract', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert '--folder' in result.stdout.lower()


def test_train_help():
    """Test train command help."""
    result = subprocess.run(
        ['vogel-trainer', 'train', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'data' in result.stdout.lower()
    assert '--output' in result.stdout.lower()


def test_organize_help():
    """Test organize command help."""
    result = subprocess.run(
        ['vogel-trainer', 'organize', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'source' in result.stdout.lower()
    assert '--output' in result.stdout.lower()


def test_evaluate_help():
    """Test evaluate command help."""
    result = subprocess.run(
        ['vogel-trainer', 'evaluate', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert '--species-model' in result.stdout.lower()
