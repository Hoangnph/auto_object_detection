"""Tests for image cropping utilities."""
import pytest
import numpy as np
from pathlib import Path
from utils.cropping_utils import (
    crop_detection,
    save_cropped_image,
    process_detections
)
from utils.image_utils import load_image

def test_crop_detection():
    """Test cropping detection from image."""
    # Create test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = 255  # White square
    
    # Test bbox
    bbox = [20, 20, 80, 80]
    
    # Crop detection
    cropped = crop_detection(image, bbox)
    
    # Check crop size and content
    assert cropped.shape == (60, 60, 3)
    assert np.all(cropped[10:50, 10:50] == 255)  # White square should be centered

def test_save_cropped_image(tmp_path):
    """Test saving cropped image."""
    # Create test image
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    
    # Save image
    image_path = save_cropped_image(
        image,
        tmp_path,
        "001",
        "test_class"
    )
    
    # Check saved image
    assert image_path.exists()
    assert image_path.name == "001_test_class.jpg"
    
    # Load and verify image
    loaded = load_image(image_path)
    assert loaded.shape == image.shape
    assert np.all(loaded == image)

def test_process_detections(tmp_path, create_sample_image):
    """Test processing multiple detections."""
    # Load test image
    image = load_image(create_sample_image)
    
    # Create test detections
    detections = [
        {
            "bbox": [10, 10, 50, 50],
            "class_name": "class1",
            "confidence": 0.9
        },
        {
            "bbox": [60, 60, 90, 90],
            "class_name": "class2",
            "confidence": 0.8
        }
    ]
    
    # Process detections
    results = process_detections(
        image,
        detections,
        tmp_path,
        min_confidence=0.7
    )
    
    # Check results
    assert len(results) == 2
    for result in results:
        assert "image_path" in result
        assert "metadata_path" in result
        assert Path(result["image_path"]).exists()
        assert Path(result["metadata_path"]).exists()

def test_invalid_bbox():
    """Test cropping with invalid bbox."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test cases that should raise ValueError
    invalid_bboxes = [
        [-10, 10, 50, 50],  # Negative coordinates
        [10, 10, 150, 50],  # Out of bounds
        [50, 50, 30, 30],   # Invalid dimensions
    ]
    
    for bbox in invalid_bboxes:
        with pytest.raises(ValueError):
            crop_detection(image, bbox)

def test_confidence_filtering(tmp_path, create_sample_image):
    """Test filtering detections by confidence."""
    image = load_image(create_sample_image)
    
    detections = [
        {
            "bbox": [10, 10, 50, 50],
            "class_name": "class1",
            "confidence": 0.5
        },
        {
            "bbox": [60, 60, 90, 90],
            "class_name": "class2",
            "confidence": 0.9
        }
    ]
    
    # Process with high confidence threshold
    results = process_detections(
        image,
        detections,
        tmp_path,
        min_confidence=0.8
    )
    
    # Should only include the second detection
    assert len(results) == 1
    assert "class2" in Path(results[0]["image_path"]).stem 