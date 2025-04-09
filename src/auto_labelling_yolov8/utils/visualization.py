import cv2
import numpy as np
from typing import List, Tuple

from ..config.config_loader import VisualizationConfig

# COCO class names
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def get_color(class_id: int) -> Tuple[int, int, int]:
    """Generate a unique color for each class"""
    np.random.seed(class_id)
    color = tuple(map(int, np.random.randint(100, 255, size=3)))
    return color

def draw_detections(image: np.ndarray, 
                   boxes: List[List[float]], 
                   scores: List[float], 
                   class_ids: List[int],
                   conf_threshold: float = 0.25) -> np.ndarray:
    """
    Draw detection boxes and labels on the image
    
    Args:
        image: Input image
        boxes: List of bounding boxes [x1, y1, x2, y2]
        scores: List of confidence scores
        class_ids: List of class IDs
        conf_threshold: Confidence threshold for displaying detections
        
    Returns:
        Annotated image
    """
    image_with_boxes = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Dictionary to count objects of each class
    class_counts = {}
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score < conf_threshold:
            continue
            
        # Convert class_id to int
        class_id = int(class_id)
        
        # Update class count
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Get class name and color
        class_name = COCO_CLASSES.get(class_id, f'class_{class_id}')
        color = get_color(class_id)
        
        # Draw box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f'{class_name} {score:.2f}'
        
        # Get label size and background position
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image_with_boxes, 
                     (x1, y1 - label_h - baseline - 5),
                     (x1 + label_w, y1),
                     color, -1)
        
        # Draw label text
        cv2.putText(image_with_boxes, label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add summary text at the top of the image
    summary_text = "Detected: " + ", ".join([f"{COCO_CLASSES[k]} ({v})" 
                                           for k, v in class_counts.items()])
    
    # Create semi-transparent overlay for summary text
    overlay = image_with_boxes.copy()
    cv2.rectangle(overlay, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image_with_boxes, 0.4, 0, image_with_boxes)
    
    # Add summary text
    cv2.putText(image_with_boxes, summary_text,
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image_with_boxes

def draw_boxes(image: np.ndarray,
               boxes: List,
               config: VisualizationConfig) -> np.ndarray:
    """Draw bounding boxes on image with labels and confidence scores.
    
    Args:
        image: Input image as numpy array (BGR format)
        boxes: List of detection boxes from YOLO
        config: Visualization configuration
        
    Returns:
        Annotated image with boxes and labels
    """
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        
        # Get confidence score
        conf = float(box.conf[0])
        
        # Get class name
        cls = int(box.cls[0])
        
        # Draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            config.colors['box'],
            config.box_thickness
        )
        
        # Prepare label text
        label = f'Class {cls}: {conf:.2f}'
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            config.text_scale,
            config.text_thickness
        )
        
        # Draw text background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            config.colors['background'],
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.text_scale,
            config.colors['text'],
            config.text_thickness
        )
    
    return image 