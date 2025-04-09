"""Dataset management utilities."""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from loguru import logger

def create_dataset_structure(base_dir: Path) -> Dict[str, Path]:
    """Create YOLO dataset directory structure.
    
    Args:
        base_dir: Base directory for dataset
        
    Returns:
        Dictionary containing paths to created directories
    """
    base_dir = Path(base_dir)
    
    # Create directory structure
    dirs = {}
    for data_type in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            dir_path = base_dir / data_type / split
            dir_path.mkdir(parents=True, exist_ok=True)
            dirs[f"{data_type}_{split}"] = dir_path
            
    logger.info(f"Created dataset structure in {base_dir}")
    return dirs

def convert_to_yolo_format(
    bbox: List[float],
    image_size: Tuple[int, int]
) -> List[float]:
    """Convert bounding box to YOLO format.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        image_size: Image dimensions (width, height)
        
    Returns:
        YOLO format bbox [x_center, y_center, width, height]
    """
    # Extract coordinates
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_size
    
    # Convert to center coordinates
    x_center = (x1 + x2) / (2 * img_w)
    y_center = (y1 + y2) / (2 * img_h)
    
    # Convert width and height
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    
    # Ensure values are between 0 and 1
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)
    
    return [x_center, y_center, width, height]

def create_data_yaml(
    dataset_dir: Path,
    class_names: List[str]
) -> Path:
    """Create data.yaml file for YOLO training.
    
    Args:
        dataset_dir: Dataset directory
        class_names: List of class names
        
    Returns:
        Path to created yaml file
    """
    dataset_dir = Path(dataset_dir)
    yaml_path = dataset_dir / "data.yaml"
    
    # Create yaml content
    data = {
        "path": str(dataset_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names)
    }
    
    # Write yaml file
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
        
    logger.info(f"Created data.yaml at {yaml_path}")
    return yaml_path

def split_dataset(
    image_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = None
) -> Dict[str, List[Path]]:
    """Split dataset into train/val/test sets.
    
    Args:
        image_dir: Directory containing images
        train_ratio: Ratio of training set
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing file lists for each split
        
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
        
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        
    # Get all image files
    image_files = list(image_dir.glob("*.jpg"))
    np.random.shuffle(image_files)
    
    # Calculate split indices
    n_files = len(image_files)
    train_idx = int(n_files * train_ratio)
    val_idx = train_idx + int(n_files * val_ratio)
    
    # Split files
    splits = {
        "train": image_files[:train_idx],
        "val": image_files[train_idx:val_idx],
        "test": image_files[val_idx:]
    }
    
    # Log split sizes
    for split, files in splits.items():
        logger.info(f"{split} set: {len(files)} files")
        
    return splits 