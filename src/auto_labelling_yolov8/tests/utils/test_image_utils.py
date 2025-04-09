"""Tests for image utilities."""
import cv2
import numpy as np
import pytest
from pathlib import Path
from utils.image_utils import (
    load_image,
    preprocess_image,
    resize_image,
    normalize_image,
)

@pytest.fixture
def create_sample_image(test_data_dir):
    """Create a sample image for testing."""
    img_path = test_data_dir / "sample_image.jpg"
    if not img_path.exists():
        # Create a simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (70, 70), (255, 255, 255), -1)
        cv2.imwrite(str(img_path), img)
    return img_path

def test_load_image(create_sample_image):
    """Test loading an image."""
    img = load_image(create_sample_image)
    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 100, 3)
    assert img.dtype == np.uint8

def test_load_image_not_found():
    """Test loading a non-existent image."""
    with pytest.raises(FileNotFoundError):
        load_image(Path("non_existent.jpg"))

def test_preprocess_image(create_sample_image):
    """Test image preprocessing."""
    img = load_image(create_sample_image)
    processed_img = preprocess_image(img)
    assert isinstance(processed_img, np.ndarray)
    assert processed_img.dtype == np.float32
    assert processed_img.max() <= 1.0
    assert processed_img.min() >= 0.0

def test_resize_image(create_sample_image):
    """Test image resizing."""
    img = load_image(create_sample_image)
    target_size = (224, 224)
    resized_img = resize_image(img, target_size)
    assert resized_img.shape[:2] == target_size

def test_normalize_image(create_sample_image):
    """Test image normalization."""
    img = load_image(create_sample_image)
    normalized_img = normalize_image(img)
    assert normalized_img.dtype == np.float32
    assert normalized_img.max() <= 1.0
    assert normalized_img.min() >= 0.0 