"""Tests for YOLO detector."""
import pytest
import numpy as np
from pathlib import Path
from models.yolo_detector import YOLODetector
from utils.image_utils import load_image

@pytest.fixture
def yolo_detector():
    """Create a YOLO detector instance."""
    return YOLODetector()

def test_model_initialization(yolo_detector):
    """Test YOLO model initialization."""
    assert yolo_detector.model is not None
    assert yolo_detector.confidence_threshold == 0.25  # default value

def test_model_prediction(yolo_detector, create_sample_image):
    """Test object detection on a sample image."""
    # Load test image
    image = load_image(create_sample_image)
    
    # Perform detection
    detections = yolo_detector.detect(image)
    
    # Check detection format
    assert isinstance(detections, list)
    if len(detections) > 0:
        for detection in detections:
            assert "bbox" in detection
            assert "confidence" in detection
            assert "class_id" in detection
            assert "class_name" in detection
            
            # Check bbox format
            assert len(detection["bbox"]) == 4
            assert all(isinstance(coord, (int, float)) for coord in detection["bbox"])
            
            # Check confidence
            assert 0 <= detection["confidence"] <= 1
            
            # Check class info
            assert isinstance(detection["class_id"], int)
            assert isinstance(detection["class_name"], str)

def test_confidence_threshold(yolo_detector, create_sample_image):
    """Test confidence threshold filtering."""
    # Set high confidence threshold
    yolo_detector.confidence_threshold = 0.9
    
    # Load test image
    image = load_image(create_sample_image)
    
    # Perform detection
    detections = yolo_detector.detect(image)
    
    # Check that all detections meet threshold
    for detection in detections:
        assert detection["confidence"] >= 0.9

def test_invalid_image(yolo_detector):
    """Test detection with invalid image."""
    with pytest.raises(ValueError):
        yolo_detector.detect(np.zeros((10, 10)))  # Invalid image size

def test_visualization(yolo_detector, create_sample_image, tmp_path):
    """Test detection visualization."""
    # Load test image
    image = load_image(create_sample_image)
    
    # Perform detection
    detections = yolo_detector.detect(image)
    
    # Visualize detections
    output_path = tmp_path / "output.jpg"
    yolo_detector.visualize_detections(image, detections, output_path)
    
    # Check if output image was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0 