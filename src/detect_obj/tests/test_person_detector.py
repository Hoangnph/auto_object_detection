import pytest
import cv2
import numpy as np
from pathlib import Path
from src.detect_obj.person_detector import PersonDetector

def test_person_detector_initialization():
    detector = PersonDetector()
    assert detector is not None
    assert hasattr(detector, 'model')

def test_detect_and_count_people():
    detector = PersonDetector()
    # Load test image from the imgs directory
    image_path = Path(__file__).parent.parent / 'imgs' / 'kleomenis-spyroglou-ElX8Lc-FYjw-unsplash.jpg'
    image = cv2.imread(str(image_path))
    assert image is not None, f"Test image not found at {image_path}"
    
    # Test detection
    count, annotated_image = detector.detect_and_count_people(image)
    
    # Basic assertions
    assert isinstance(count, int)
    assert count > 0, "Should detect at least one person in the mall image"
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape == image.shape 