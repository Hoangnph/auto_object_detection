from typing import List, Tuple
import numpy as np

def convert_to_yolo(boxes: List,
                   image_shape: Tuple[int, int, int],
                   include_confidence: bool = True) -> str:
    """Convert detection boxes to YOLO format.
    
    YOLO format: <class> <x_center> <y_center> <width> <height> [confidence]
    All values are normalized to [0, 1]
    """
    height, width = image_shape[:2]
    lines = []
    
    for box in boxes:
        # Get normalized coordinates
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        
        # Convert to center format
        x_center = (x1 + x2) / (2 * width)
        y_center = (y1 + y2) / (2 * height)
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        
        # Get class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Format line
        if include_confidence:
            line = f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {conf:.6f}"
        else:
            line = f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        lines.append(line)
    
    return '\n'.join(lines)

def convert_to_coco(boxes: List,
                   image_shape: Tuple[int, int, int],
                   include_confidence: bool = True) -> str:
    """Convert detection boxes to COCO format.
    
    COCO format: [x_min, y_min, width, height]
    Values are in absolute pixels
    """
    lines = []
    
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        width = x2 - x1
        height = y2 - y1
        
        # Get class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Format line
        if include_confidence:
            line = f"{cls} {x1:.1f} {y1:.1f} {width:.1f} {height:.1f} {conf:.6f}"
        else:
            line = f"{cls} {x1:.1f} {y1:.1f} {width:.1f} {height:.1f}"
        lines.append(line)
    
    return '\n'.join(lines)

def convert_to_voc(boxes: List,
                  image_shape: Tuple[int, int, int],
                  include_confidence: bool = True) -> str:
    """Convert detection boxes to Pascal VOC format.
    
    VOC format: <class> <x_min> <y_min> <x_max> <y_max> [confidence]
    Values are in absolute pixels
    """
    lines = []
    
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        
        # Get class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Format line
        if include_confidence:
            line = f"{cls} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {conf:.6f}"
        else:
            line = f"{cls} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
        lines.append(line)
    
    return '\n'.join(lines)

def convert_to_format(boxes: List,
                     image_shape: Tuple[int, int, int],
                     format: str,
                     include_confidence: bool = True) -> str:
    """Convert detection boxes to specified format."""
    format_converters = {
        'yolo': convert_to_yolo,
        'coco': convert_to_coco,
        'voc': convert_to_voc
    }
    
    if format not in format_converters:
        raise ValueError(f"Unsupported format: {format}")
    
    return format_converters[format](boxes, image_shape, include_confidence) 