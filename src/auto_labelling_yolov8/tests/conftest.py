"""Pytest configuration file."""
import os
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_image_path(test_data_dir):
    """Return the path to a sample test image."""
    return test_data_dir / "sample_image.jpg"

@pytest.fixture
def raw_output_dir(tmp_path):
    """Create and return a temporary directory for raw output images."""
    output_dir = tmp_path / "raw"
    output_dir.mkdir(exist_ok=True)
    return output_dir

@pytest.fixture
def dataset_output_dir(tmp_path):
    """Create and return a temporary directory for dataset output."""
    output_dir = tmp_path / "dataset"
    output_dir.mkdir(exist_ok=True)
    return output_dir 