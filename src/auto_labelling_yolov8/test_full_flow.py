"""Test full auto-labeling flow."""
import cv2
import os
import sys
import asyncio
from pathlib import Path
import json
import psutil
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.yolo_detector import YOLODetector
from models.llm_labeler import LLMLabeler
from utils.cropping_utils import crop_and_save_objects
from loguru import logger
from load_env import load_environment
from datetime import datetime

# Configure logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="DEBUG")  # Add handler with DEBUG level

# Load environment variables
load_environment()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run auto-labeling on an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model', type=str, default='models/yolov8x.pt',
                      help='Path to YOLO model (default: models/yolov8x.pt)')
    parser.add_argument('--output', type=str, default='output',
                      help='Output directory (default: output)')
    return parser.parse_args()

async def test_full_flow(image_path: str, model_path: str, output_dir: str):
    """Test the full auto-labeling flow."""
    try:
        # Setup paths
        input_path = Path(image_path)
        
        # Create a single output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{output_dir}/run_{timestamp}")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        images_dir = output_dir / "images"
        metadata_dir = output_dir / "metadata"
        logs_dir = output_dir / "logs"
        config_dir = output_dir / "config"
        
        for dir_path in [images_dir, metadata_dir, logs_dir, config_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Create README.md
        readme_content = f"""# Auto-labeling Run Results

## Run Information
- Timestamp: {timestamp}
- Input Image: {input_path}
- Output Directory: {output_dir}
- Model Used: {model_path}

## Directory Structure
- `images/`: Contains cropped images of detected objects
- `metadata/`: Contains JSON files with detection and labeling information
- `logs/`: Contains detailed logs of the process
- `config/`: Contains configuration files used

## Process Overview
1. Object Detection: YOLOv8 model detects vehicles in the input image
2. Image Cropping: Detected objects are cropped and saved
3. LLM Labeling: Each cropped image is labeled using LLM
4. Results: Final labels and metadata are saved in JSON format

## File Descriptions
- `environment.json`: System and runtime information
- `detections.json`: YOLO detection results
- `labeling_results.json`: Final labeling results
"""
        
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)
            
        # Save environment info
        env_info = {
            "timestamp": timestamp,
            "python_version": sys.version,
            "platform": sys.platform,
            "input_image": str(input_path),
            "model_path": str(model_path),
            "start_time": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": os.cpu_count(),
                "memory_usage": psutil.Process().memory_info().rss,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }
        
        with open(metadata_dir / "environment.json", "w") as f:
            json.dump(env_info, f, indent=2)

        # Load and preprocess image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")

        # Initialize YOLO detector with specified model
        detector = YOLODetector(model_name=model_path)
        logger.info(f"YOLO detector initialized with model: {model_path}")

        # Detect objects
        detections = detector.detect(image)
        logger.info(f"Detected {len(detections)} objects")

        # Crop and save detected objects
        cropped_paths = []
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det["confidence"]
            class_name = det["class_name"]
            
            # Crop image
            cropped = image[y1:y2, x1:x2]
            
            # Save cropped image to output/images directory
            crop_path = images_dir / f"{class_name}_{class_name}_{i:03d}_{confidence:.2f}.jpg"
            cv2.imwrite(str(crop_path), cropped)
            cropped_paths.append(crop_path)
            logger.info(f"Saved cropped image to {crop_path}")

        # Save detection results
        detection_info = {
            "total_detections": len(detections),
            "detections": [
                {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "class_name": det["class_name"],
                    "cropped_path": str(cropped_paths[i]),
                    "image_size": {
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                }
                for i, det in enumerate(detections)
            ],
            "detection_stats": {
                "average_confidence": sum(d["confidence"] for d in detections) / len(detections),
                "class_distribution": {
                    cls: sum(1 for d in detections if d["class_name"] == cls)
                    for cls in set(d["class_name"] for d in detections)
                }
            }
        }
        
        with open(metadata_dir / "detections.json", "w") as f:
            json.dump(detection_info, f, indent=2)

        # Initialize LLM labeler with output directory
        llm_labeler = LLMLabeler(output_dir=output_dir)
        logger.info("LLM labeler initialized")

        # Test connection with first cropped image
        test_image = cv2.imread(str(cropped_paths[0]))
        if test_image is None:
            raise ValueError(f"Failed to load test image: {cropped_paths[0]}")
        test_result = await llm_labeler.test_connection(test_image)
        if not test_result:
            raise ValueError("LLM test connection failed")
        logger.info("LLM test successful")

        # Label all cropped images
        results = await llm_labeler.label_batch(cropped_paths, str(input_path))
        logger.info(f"Labeled {len(results)} images")
        
        # Print results
        for result in results:
            if result["status"] == "success":
                logger.info(f"Image {result['image_path']}: {result['label_info']}")
            else:
                logger.error(f"Image {result['image_path']}: {result['error']}")

        # Update environment info with end time and stats
        env_info["end_time"] = datetime.now().isoformat()
        env_info["duration_seconds"] = (datetime.fromisoformat(env_info["end_time"]) - 
                                      datetime.fromisoformat(env_info["start_time"])).total_seconds()
        env_info["labeling_stats"] = {
            "total_images": len(results),
            "successful_labels": len([r for r in results if r["status"] == "success"]),
            "failed_labels": len([r for r in results if r["status"] == "error"]),
            "success_rate": len([r for r in results if r["status"] == "success"]) / len(results) * 100
        }
        
        with open(metadata_dir / "environment.json", "w") as f:
            json.dump(env_info, f, indent=2)

        return {
            "detections": len(detections),
            "labeled": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "output_dir": str(output_dir)
        }

    except Exception as e:
        logger.error(f"Error in test_full_flow: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_arguments()
    results = asyncio.run(test_full_flow(args.image_path, args.model, args.output))
    print(f"\nTest completed successfully! Results saved to: {results['output_dir']}") 