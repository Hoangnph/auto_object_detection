import cv2
from pathlib import Path
from datetime import datetime
import json
from ..person_detector import PersonDetector
from ..config.detection_params import DetectionConfig

def run_detection_test(config: DetectionConfig, image_name: str) -> dict:
    """Run detection with given config and return results"""
    detector = PersonDetector(config)
    
    # Load image
    image_path = config.input_dir / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Run detection
    count, annotated_image = detector.detect_and_count_people(image)
    
    # Generate timestamp-based output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"detection_{timestamp}.jpg"
    output_path = config.output_dir / output_name
    
    # Save annotated image
    cv2.imwrite(str(output_path), annotated_image)
    
    # Return results
    return {
        "timestamp": timestamp,
        "config": {
            "model_name": config.model_name,
            "conf_threshold": config.conf_threshold,
            "iou_threshold": config.iou_threshold,
            "max_det": config.max_det
        },
        "results": {
            "person_count": count,
            "output_image": str(output_path)
        }
    }

def main():
    # Define different configurations to test
    configs = [
        # Test different models
        DetectionConfig(model_name="yolov8n.pt", conf_threshold=0.25),
        DetectionConfig(model_name="yolov8n.pt", conf_threshold=0.15),  # Lower confidence threshold
        DetectionConfig(model_name="yolov8n.pt", conf_threshold=0.25, max_det=500),  # More detections
        
        # You can add more configurations here, for example:
        # DetectionConfig(model_name="yolov8m.pt", conf_threshold=0.25),  # Medium model
        # DetectionConfig(model_name="yolov8l.pt", conf_threshold=0.25),  # Large model
    ]
    
    image_name = "kleomenis-spyroglou-ElX8Lc-FYjw-unsplash.jpg"
    results = []
    
    # Run tests with each configuration
    for config in configs:
        try:
            result = run_detection_test(config, image_name)
            results.append(result)
            print(f"\nTest completed for config:")
            print(f"Model: {config.model_name}")
            print(f"Confidence threshold: {config.conf_threshold}")
            print(f"People detected: {result['results']['person_count']}")
            print(f"Output saved to: {result['results']['output_image']}")
        except Exception as e:
            print(f"Error with config {config.model_name}: {str(e)}")
    
    # Save results to JSON
    output_file = Path("outputs") / "detection_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    main() 