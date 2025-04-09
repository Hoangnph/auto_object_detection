import cv2
from pathlib import Path
import sys
from src.detect_obj.person_detector import PersonDetector
from src.detect_obj.config.detection_params import DetectionConfig

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "src" / "auto_labelling_yolov8" / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"

    # Create directories if they don't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Parse command line arguments
    if len(sys.argv) > 1:
        image_name = sys.argv[1]
    else:
        image_name = 'cars.png'  # Default image
    
    # Initialize detector with custom config
    config = DetectionConfig(
        conf_threshold=0.25,  # Confidence threshold
        max_det=300,         # Maximum number of detections
    )
    detector = PersonDetector(config)
    
    # Load image
    image_path = raw_data_dir / image_name
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print(f"Please place your image in the {raw_data_dir} directory")
        return
        
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Detect and count vehicles
    count, annotated_image = detector.detect_and_count_people(image)
    
    # Print results
    print(f"Number of vehicles detected: {count}")
    
    # Save the annotated image
    output_path = processed_data_dir / 'latest_detection.jpg'
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    main()
