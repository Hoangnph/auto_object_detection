"""YOLO object detector module."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Tuple
from ultralytics import YOLO
from loguru import logger

class YOLODetector:
    """YOLOv8 object detector class."""
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25
    ):
        """Initialize YOLO detector.
        
        Args:
            model_name: Name or path of the YOLO model
            confidence_threshold: Confidence threshold for detections
        """
        logger.info(f"Initializing YOLOv8 detector with model {model_name}")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in an image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
                
        Raises:
            ValueError: If image is invalid
        """
        # Validate image
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Invalid image format")
            
        if image.shape[0] < 32 or image.shape[1] < 32:
            raise ValueError("Image too small, minimum size is 32x32")
            
        logger.debug("Running object detection")
        results = self.model(image)[0]
        
        detections = []
        for box in results.boxes:
            confidence = float(box.conf)
            
            # Filter by confidence
            if confidence < self.confidence_threshold:
                continue
                
            # Get detection info
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            class_id = int(box.cls)
            class_name = results.names[class_id]
            
            detections.append({
                "bbox": bbox.tolist(),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name
            })
            
        logger.debug(f"Found {len(detections)} objects")
        return detections
        
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_path: Union[str, Path]
    ) -> None:
        """Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detections from detect()
            output_path: Path to save visualization
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make copy of image for visualization
        vis_image = image.copy()
        
        # Draw each detection
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            # Convert bbox to integers
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
        # Save visualization
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        logger.debug(f"Saved visualization to {output_path}") 