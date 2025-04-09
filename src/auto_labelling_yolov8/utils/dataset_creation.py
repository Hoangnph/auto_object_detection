"""Dataset creation pipeline."""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Union
from loguru import logger

from utils.dataset_utils import (
    create_dataset_structure,
    convert_to_yolo_format,
    create_data_yaml,
    split_dataset
)

def create_yolo_dataset(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = None
) -> Dict:
    """Create YOLO dataset from raw data.
    
    Args:
        raw_dir: Directory containing raw data
        output_dir: Output directory for dataset
        train_ratio: Ratio of training set
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing dataset information
        
    Raises:
        ValueError: If raw directory does not exist or is empty
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    
    # Validate input directory
    if not raw_dir.exists():
        raise ValueError(f"Raw directory not found: {raw_dir}")
        
    images_dir = raw_dir / "images"
    metadata_dir = raw_dir / "metadata"
    
    if not images_dir.exists() or not metadata_dir.exists():
        raise ValueError("Invalid raw data directory structure")
        
    # Create dataset structure
    dirs = create_dataset_structure(output_dir)
    
    # Split dataset
    splits = split_dataset(
        images_dir,
        train_ratio,
        val_ratio,
        test_ratio,
        seed
    )
    
    # Copy and organize images
    copy_info = copy_and_organize_images(images_dir, output_dir, splits)
    
    # Get unique class names and create class map
    class_names = set()
    for metadata_file in metadata_dir.glob("*.json"):
        with open(metadata_file) as f:
            metadata = json.load(f)
            class_names.add(metadata["class_name"])
            
    class_map = {name: i for i, name in enumerate(sorted(class_names))}
    
    # Generate label files
    label_info = generate_label_files(metadata_dir, output_dir, class_map)
    
    # Create data.yaml
    yaml_path = create_data_yaml(output_dir, list(class_names))
    
    # Validate dataset
    validation_result = validate_dataset(output_dir)
    if not validation_result["is_valid"]:
        logger.warning("Dataset validation failed:")
        for error in validation_result["errors"]:
            logger.warning(f"  - {error}")
            
    return {
        "class_map": class_map,
        "yaml_path": str(yaml_path),
        "statistics": {
            "total_images": len(copy_info["copied_files"]),
            "train_images": len(splits["train"]),
            "val_images": len(splits["val"]),
            "test_images": len(splits["test"]),
            "classes": len(class_names)
        },
        "validation": validation_result
    }

def copy_and_organize_images(
    source_dir: Path,
    output_dir: Path,
    splits: Dict[str, List[Path]]
) -> Dict:
    """Copy and organize images into train/val/test splits.
    
    Args:
        source_dir: Source directory containing images
        output_dir: Output directory for dataset
        splits: Dictionary containing file lists for each split
        
    Returns:
        Dictionary containing copy information
    """
    copied_files = {}
    
    for split_name, files in splits.items():
        split_dir = output_dir / "images" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            source_path = source_dir / file.name
            target_path = split_dir / file.name
            
            shutil.copy2(source_path, target_path)
            copied_files[str(source_path)] = str(target_path)
            
    return {"copied_files": copied_files}

def generate_label_files(
    metadata_dir: Path,
    output_dir: Path,
    class_map: Dict[str, int]
) -> Dict:
    """Generate YOLO format label files.
    
    Args:
        metadata_dir: Directory containing metadata files
        output_dir: Output directory for dataset
        class_map: Mapping from class names to indices
        
    Returns:
        Dictionary containing label generation information
    """
    generated_files = {}
    
    for metadata_file in metadata_dir.glob("*.json"):
        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
            
        # Get image ID and class
        image_id = metadata_file.stem
        class_name = metadata["class_name"]
        class_id = class_map[class_name]
        
        # Convert bbox to YOLO format
        yolo_bbox = convert_to_yolo_format(
            metadata["bbox"],
            metadata.get("image_size", (640, 640))  # Default size if not specified
        )
        
        # Create label file
        for split in ["train", "val", "test"]:
            split_dir = output_dir / "labels" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if corresponding image exists
            image_path = output_dir / "images" / split / f"{image_id}_{class_name}.jpg"
            if image_path.exists():
                label_path = split_dir / f"{image_id}_{class_name}.txt"
                
                # Write label file
                with open(label_path, "w") as f:
                    coords = " ".join(map(str, yolo_bbox))
                    f.write(f"{class_id} {coords}\n")
                    
                generated_files[str(image_path)] = str(label_path)
                
    return {"generated_files": generated_files}

def validate_dataset(dataset_dir: Path) -> Dict:
    """Validate YOLO dataset structure and contents.
    
    Args:
        dataset_dir: Dataset directory
        
    Returns:
        Dictionary containing validation results
    """
    errors = []
    
    # Check directory structure
    required_dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for dir_path in required_dirs:
        if not (dataset_dir / dir_path).exists():
            errors.append(f"Missing directory: {dir_path}")
            
    # Check data.yaml
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        errors.append("Missing data.yaml file")
        
    # Check image-label pairs
    for split in ["train", "val", "test"]:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
            
        # Check each image has corresponding label
        for image_path in images_dir.glob("*.jpg"):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"Missing label file for {image_path.name}")
                
    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    } 