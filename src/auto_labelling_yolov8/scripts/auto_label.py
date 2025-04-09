#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from ..config.config_loader import load_config, Config
from ..utils.visualization import draw_detections
from ..utils.format_converter import convert_to_format
from ..utils.logger import setup_logger

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Auto Label Images using YOLOv8')
    parser.add_argument('--config', type=str, default='config/labelling_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Input directory with images')
    parser.add_argument('--output', type=str, help='Output directory for labels')
    parser.add_argument('--format', type=str, choices=['yolo', 'coco', 'voc'],
                      help='Output format for labels')
    return parser.parse_args()

def load_images(image_dir: Path) -> List[Path]:
    """Load all images from directory."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [f for f in image_dir.iterdir() 
            if f.suffix.lower() in valid_extensions]

def process_image(model: YOLO, 
                 image_path: Path,
                 output_dir: Path,
                 conf_threshold: float = 0.25) -> Tuple[int, List]:
    """
    Process a single image with YOLOv8 model
    
    Args:
        model: YOLOv8 model
        image_path: Path to input image
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Tuple containing:
            - Number of detections
            - List of detection results
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Run inference
    results = model(image)[0]
    
    # Get detection results
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    
    # Draw detections on image
    annotated_image = draw_detections(
        image, boxes, scores, class_ids, conf_threshold
    )
    
    # Save annotated image
    output_image_path = output_dir / "images" / image_path.name
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), annotated_image)
    
    # Save YOLO format labels
    label_path = output_dir / "labels" / f"{image_path.stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to YOLO format and save
    height, width = image.shape[:2]
    with open(label_path, 'w') as f:
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score < conf_threshold:
                continue
            # Convert to YOLO format (normalized coordinates)
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / (2 * width)
            y_center = (y1 + y2) / (2 * height)
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            # Write to file
            f.write(f"{int(class_id)} {x_center} {y_center} {w} {h}\n")
    
    return len(boxes), results

def main():
    """Main function for auto labelling."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.input:
        config.paths.raw_data = Path(args.input)
    if args.output:
        config.paths.processed_data = Path(args.output)
    if args.format:
        config.output.format = args.format

    # Setup logging
    setup_logger(config.logging)
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info(f"Loading model: {config.model.name}")
    model = YOLO(str(config.paths.models_dir / config.model.name))
    model.to(config.model.device)
    
    # Get list of images
    image_files = load_images(config.paths.raw_data)
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process images
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            num_detections, results = process_image(
                model, 
                image_path, 
                config.paths.processed_data,
                conf_threshold=config.model.confidence_threshold
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            continue
    
    logger.info("Auto labelling completed!")

if __name__ == '__main__':
    main() 