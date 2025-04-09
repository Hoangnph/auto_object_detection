"""Image cropping utilities."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Tuple
from loguru import logger

from utils.directory_utils import save_metadata, get_next_image_number

def crop_detection(
    image: np.ndarray,
    bbox: List[float]
) -> np.ndarray:
    """Crop detection from image.
    
    Args:
        image: Input image
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Cropped image
        
    Raises:
        ValueError: If bbox is invalid
    """
    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, bbox)
    
    # Validate coordinates
    if x1 < 0 or y1 < 0:
        raise ValueError("Negative coordinates in bbox")
        
    if x2 > image.shape[1] or y2 > image.shape[0]:
        raise ValueError("Bbox coordinates out of image bounds")
        
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox dimensions")
        
    # Crop image
    return image[y1:y2, x1:x2].copy()

def save_cropped_image(
    image: np.ndarray,
    output_dir: Path,
    image_id: str,
    class_name: str
) -> Path:
    """Save cropped image.
    
    Args:
        image: Image to save
        output_dir: Output directory
        image_id: Image identifier
        class_name: Class name for filename
        
    Returns:
        Path to saved image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"{image_id}_{class_name}.jpg"
    output_path = output_dir / filename
    
    # Save image
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logger.debug(f"Saved cropped image to {output_path}")
    
    return output_path

def process_detections(
    image: np.ndarray,
    detections: List[Dict],
    output_dir: Path,
    min_confidence: float = 0.5
) -> List[Dict]:
    """Process multiple detections from an image.
    
    Args:
        image: Input image
        detections: List of detections from YOLO model
        output_dir: Base output directory
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of results, each containing:
            - image_path: Path to saved cropped image
            - metadata_path: Path to saved metadata
    """
    output_dir = Path(output_dir)
    
    # Create directory structure
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for detection in detections:
        # Skip if below confidence threshold
        if detection["confidence"] < min_confidence:
            continue
            
        try:
            # Get next image number
            image_id = get_next_image_number(images_dir)
            
            # Crop and save image
            cropped = crop_detection(image, detection["bbox"])
            image_path = save_cropped_image(
                cropped,
                images_dir,
                image_id,
                detection["class_name"]
            )
            
            # Save metadata
            metadata = {
                "image_id": image_id,
                "original_bbox": detection["bbox"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "cropped_size": cropped.shape[:2]
            }
            metadata_path = save_metadata(metadata_dir, image_id, metadata)
            
            results.append({
                "image_path": str(image_path),
                "metadata_path": str(metadata_path)
            })
            
        except Exception as e:
            logger.error(f"Error processing detection: {e}")
            continue
            
    return results

def crop_and_save_objects(
    image: np.ndarray,
    detections: List[Dict],
    output_dir: Path,
    prefix: str = "object"
) -> List[Path]:
    """Crop and save detected objects from an image.
    
    Args:
        image: Input image as numpy array
        detections: List of detections from YOLO detector
        output_dir: Directory to save cropped images
        prefix: Prefix for saved image filenames
        
    Returns:
        List of paths to saved cropped images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for i, detection in enumerate(detections):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, detection["bbox"])
        
        # Crop object from image
        cropped = image[y1:y2, x1:x2]
        
        # Generate filename
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        filename = f"{prefix}_{class_name}_{i:03d}_{confidence:.2f}.jpg"
        output_path = output_dir / filename
        
        # Save cropped image
        cv2.imwrite(str(output_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        saved_paths.append(output_path)
        
        logger.debug(f"Saved cropped object to {output_path}")
    
    return saved_paths 