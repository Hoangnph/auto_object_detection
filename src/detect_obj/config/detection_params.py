from dataclasses import dataclass
from pathlib import Path

@dataclass
class DetectionConfig:
    # Model parameters
    model_name: str = "yolov8n.pt"  # Có thể thay đổi sang m, l, x để tăng độ chính xác
    conf_threshold: float = 0.9    # Ngưỡng tin cậy cho việc phát hiện
    iou_threshold: float = 0.45      # Ngưỡng IoU cho NMS
    
    # Detection parameters
    target_class: int = 0            # 0 = person trong COCO dataset
    max_det: int = 300              # Số lượng đối tượng tối đa có thể phát hiện
    
    # Visualization parameters
    box_thickness: int = 2           # Độ dày của bounding box
    text_thickness: int = 2          # Độ dày của text
    text_scale: float = 0.5          # Kích thước của text
    box_color: tuple = (0, 255, 0)   # Màu của bounding box (BGR)
    text_color: tuple = (0, 255, 0)  # Màu của text (BGR)
    
    # Path configurations
    base_dir: Path = Path(__file__).parent.parent
    model_dir: Path = base_dir / "models"
    input_dir: Path = base_dir / "data/images"
    output_dir: Path = base_dir / "outputs"
    
    def __post_init__(self):
        """Ensure all required directories exist"""
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.input_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Update model path to use model directory
        self.model_path = str(self.model_dir / self.model_name)

# Default configuration
default_config = DetectionConfig() 