"""Script to create YOLO dataset from labeled images."""
import os
import sys
import json
import shutil
from pathlib import Path
import yaml
from loguru import logger
import random
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_env import load_environment

def create_dataset_structure(base_dir: Path) -> dict:
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

def split_dataset(image_files: list, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """Split dataset into train/val/test sets.
    
    Args:
        image_files: List of image files
        train_ratio: Ratio of training set
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        seed: Random seed
        
    Returns:
        Dictionary containing split indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    n = len(image_files)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

def create_data_yaml(dataset_dir: Path, class_names: list) -> Path:
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

def main():
    """Main function to create YOLO dataset."""
    # Load environment variables
    load_environment()
    
    # Get script directory
    script_dir = Path(__file__).parent.parent
    
    # Load config
    with open(script_dir / "config/labelling_config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Set paths
    raw_dir = script_dir / "raw"
    dataset_dir = script_dir / "data"
    
    # Create dataset structure
    dirs = create_dataset_structure(dataset_dir)
    
    # Get all labeled images and their metadata
    image_files = []
    for img_path in (raw_dir / "images").glob("*.jpg"):
        metadata_path = raw_dir / "metadata" / f"{img_path.stem}.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                # Only include images with valid labels
                if "labels" in metadata and metadata["labels"]:
                    image_files.append((img_path, metadata))
    
    if not image_files:
        raise ValueError(f"No labeled images found in {raw_dir}")
    
    # Split dataset
    splits = split_dataset(image_files)
    
    # Copy images and create labels
    class_names = config["classes"]["selected"]
    for split, files in splits.items():
        logger.info(f"Processing {split} split ({len(files)} images)")
        
        for img_path, metadata in files:
            # Get class from metadata
            labels = metadata.get("labels", [])
            if not labels:
                logger.warning(f"No labels found in metadata for {img_path}")
                continue
                
            # Use first label (since we're dealing with cropped single-object images)
            label = labels[0]
            class_name = label.get("class")
            
            if class_name not in class_names:
                logger.warning(f"Unknown class {class_name} in {img_path}")
                continue
            
            # Copy image
            shutil.copy2(img_path, dirs[f"images_{split}"] / img_path.name)
            
            # Create label file
            class_id = class_names.index(class_name)
            label_path = dirs[f"labels_{split}"] / img_path.with_suffix(".txt").name
            
            # For cropped objects, use full image coordinates (centered)
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            
            # YOLO format: <class_id> <x_center> <y_center> <width> <height>
            # For cropped objects, use centered coordinates
            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    # Create data.yaml
    create_data_yaml(dataset_dir, class_names)
    
    # Print statistics
    for split in ["train", "val", "test"]:
        n_images = len(list(dirs[f"images_{split}"].glob("*.jpg")))
        n_labels = len(list(dirs[f"labels_{split}"].glob("*.txt")))
        logger.info(f"{split}: {n_images} images, {n_labels} labels")

if __name__ == "__main__":
    main() 