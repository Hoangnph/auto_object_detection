import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from src.detect_obj.config.detection_params import DetectionConfig

class PersonDetector:
    def __init__(self, config: DetectionConfig):
        """
        Initialize the YOLO model for person detection.
        
        Args:
            config: Configuration object containing detection parameters
        """
        # Setup paths
        self.base_dir = Path(__file__).parent.parent.parent
        self.model_dir = self.base_dir / "src" / "auto_labelling_yolov8" / "models"
        self.output_dir = self.base_dir / "src" / "auto_labelling_yolov8" / "outputs"
        self.log_dir = self.base_dir / "src" / "auto_labelling_yolov8" / "logs"
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.log_dir / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log(f"Using device: {self.device}")
        
        # Load model
        model_path = self.model_dir / "yolov8x.pt"
        if not model_path.exists():
            self._log(f"Downloading model to {model_path}")
            self.model = YOLO("yolov8x.pt")
            self.model.save(str(model_path))
        else:
            self._log(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
        
        self.model.to(self.device)
        
        # Store configuration
        self.config = config
        
        # Define vehicle classes (COCO dataset indices)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def _log(self, message: str):
        """Log message to file and print to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
        
    def detect_and_count_people(self, image: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Detect and count vehicles in the image
        Args:
            image: Input image in BGR format
        Returns:
            tuple containing:
                - count: Number of vehicles detected
                - annotated_image: Image with detection boxes drawn
        """
        # Run inference
        self._log("Running inference on image")
        results = self.model(image)[0]
        
        # Filter for vehicle classes
        vehicle_detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            if int(cls) in self.vehicle_classes and conf > self.config.conf_threshold:
                vehicle_detections.append(r)
        
        self._log(f"Detected {len(vehicle_detections)} vehicles")
        
        # Draw boxes for vehicles
        annotated_image = image.copy()
        for det in vehicle_detections:
            x1, y1, x2, y2, conf, cls = det
            # Draw rectangle
            cv2.rectangle(annotated_image, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 2)
            # Add label
            label = f"Vehicle: {conf:.2f}"
            cv2.putText(annotated_image, 
                       label, 
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       2)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(output_path), annotated_image)
        self._log(f"Saved annotated image to {output_path}")
        
        return len(vehicle_detections), annotated_image 