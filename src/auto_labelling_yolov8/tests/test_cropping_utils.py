"""Tests for image cropping utilities."""
import pytest
import cv2
import numpy as np
from pathlib import Path
from utils.cropping_utils import crop_and_save_objects
from models.yolo_detector import YOLODetector
from utils.image_utils import load_image

def test_crop_and_save_objects():
    """Test cropping and saving detected objects."""
    # Initialize detector
    detector = YOLODetector(confidence_threshold=0.3)
    
    # Load test image
    image_path = Path("input/cars.png")
    image = load_image(image_path)
    
    # Perform detection
    detections = detector.detect(image)
    
    # Create output directory
    output_dir = Path("raw/images")
    output_dir.mkdir(exist_ok=True)
    
    # Crop and save objects
    saved_paths = crop_and_save_objects(
        image=image,
        detections=detections,
        output_dir=output_dir,
        prefix="car"
    )
    
    # Verify results
    assert len(saved_paths) > 0, "No objects were cropped and saved"
    for path in saved_paths:
        assert path.exists(), f"Cropped image not found: {path}"
        assert path.stat().st_size > 0, f"Cropped image is empty: {path}"
        
    # Print saved paths
    print("\nSaved cropped images:")
    for path in saved_paths:
        print(f"- {path}")

if __name__ == "__main__":
    test_crop_and_save_objects() 