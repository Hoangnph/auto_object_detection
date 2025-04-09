from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import yaml

@dataclass
class ModelConfig:
    name: str
    confidence_threshold: float
    iou_threshold: float
    device: str
    max_detections: int

@dataclass
class DataConfig:
    input_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    augmentation: bool

@dataclass
class ClassesConfig:
    use_all: bool
    selected: Optional[List[str]]

@dataclass
class OutputConfig:
    format: str
    save_images: bool
    save_crops: bool
    include_confidence: bool

@dataclass
class VisualizationConfig:
    box_thickness: int
    text_thickness: int
    text_scale: float
    colors: Dict[str, List[int]]

@dataclass
class PathsConfig:
    models_dir: Path
    raw_data: Path
    processed_data: Path
    output_dir: Path

@dataclass
class LoggingConfig:
    level: str
    save_to_file: bool
    log_file: Path

@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    classes: ClassesConfig
    output: OutputConfig
    visualization: VisualizationConfig
    paths: PathsConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            data=DataConfig(**config_dict['data']),
            classes=ClassesConfig(**config_dict['classes']),
            output=OutputConfig(**config_dict['output']),
            visualization=VisualizationConfig(**config_dict['visualization']),
            paths=PathsConfig(
                models_dir=Path(config_dict['paths']['models_dir']),
                raw_data=Path(config_dict['paths']['raw_data']),
                processed_data=Path(config_dict['paths']['processed_data']),
                output_dir=Path(config_dict['paths']['output_dir'])
            ),
            logging=LoggingConfig(
                level=config_dict['logging']['level'],
                save_to_file=config_dict['logging']['save_to_file'],
                log_file=Path(config_dict['logging']['log_file'])
            )
        )

    def validate(self) -> None:
        """Validate configuration values."""
        # Model validation
        if self.model.confidence_threshold < 0 or self.model.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if self.model.iou_threshold < 0 or self.model.iou_threshold > 1:
            raise ValueError("IoU threshold must be between 0 and 1")
        if self.model.device not in ['cuda', 'cpu']:
            raise ValueError("Device must be either 'cuda' or 'cpu'")

        # Data validation
        if any(size <= 0 for size in self.data.input_size):
            raise ValueError("Input size dimensions must be positive")
        if self.data.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.data.num_workers < 0:
            raise ValueError("Number of workers must be non-negative")

        # Output validation
        if self.output.format not in ['yolo', 'coco', 'voc']:
            raise ValueError("Output format must be one of: yolo, coco, voc")

        # Classes validation
        if not self.classes.use_all and not self.classes.selected:
            raise ValueError("Must either use all classes or specify selected classes")

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for path in [self.paths.models_dir, self.paths.raw_data,
                    self.paths.processed_data, self.paths.output_dir]:
            path.mkdir(parents=True, exist_ok=True)

def load_config(config_path: str) -> Config:
    """Load and validate configuration."""
    config = Config.from_yaml(config_path)
    config.validate()
    config.create_directories()
    return config 