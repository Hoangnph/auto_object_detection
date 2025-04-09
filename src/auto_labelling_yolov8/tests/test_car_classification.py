"""Test car classification using YOLOv8."""
import cv2
import numpy as np
from pathlib import Path
from models.yolo_detector import YOLODetector
from utils.image_utils import load_image

def test_car_classification():
    """Test car detection and classification."""
    # Initialize detector
    detector = YOLODetector(confidence_threshold=0.3)
    
    # Load test image
    image_path = Path("input/cars.png")
    image = load_image(image_path)
    
    # Perform detection
    detections = detector.detect(image)
    
    # Print detections
    print("\nDetected objects:")
    for detection in detections:
        confidence = detection["confidence"]
        class_name = detection["class_name"]
        print(f"- {class_name} (confidence: {confidence:.2f})")
    
    # Visualize detections
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "car_detection_result.jpg"
    detector.visualize_detections(image, detections, output_path)
    
    # Verify results
    assert len(detections) > 0, "No objects detected"
    assert any(d["class_name"] == "car" for d in detections), "No cars detected"
    
    # Check visualization output
    assert output_path.exists(), "Visualization not saved"
    assert output_path.stat().st_size > 0, "Visualization file is empty"

if __name__ == "__main__":
    test_car_classification() 