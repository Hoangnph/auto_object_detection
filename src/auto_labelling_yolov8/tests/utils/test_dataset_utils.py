"""Tests for dataset utilities."""
import pytest
import numpy as np
from pathlib import Path
from utils.dataset_utils import (
    create_dataset_structure,
    convert_to_yolo_format,
    create_data_yaml,
    split_dataset
)

def test_create_dataset_structure(tmp_path):
    """Test creating dataset directory structure."""
    dirs = create_dataset_structure(tmp_path)
    
    # Check required directories exist
    assert (tmp_path / "images" / "train").exists()
    assert (tmp_path / "images" / "val").exists()
    assert (tmp_path / "images" / "test").exists()
    assert (tmp_path / "labels" / "train").exists()
    assert (tmp_path / "labels" / "val").exists()
    assert (tmp_path / "labels" / "test").exists()
    
    # Check returned paths
    assert all(path.exists() for path in dirs.values())

def test_convert_to_yolo_format():
    """Test converting bounding box to YOLO format."""
    # Test case 1: Simple conversion
    image_size = (100, 100)  # width, height
    bbox = [20, 30, 60, 80]  # x1, y1, x2, y2
    
    yolo_bbox = convert_to_yolo_format(bbox, image_size)
    
    # Check format: [x_center, y_center, width, height]
    assert len(yolo_bbox) == 4
    assert all(0 <= coord <= 1 for coord in yolo_bbox)
    
    # Test case 2: Edge case - full image
    bbox = [0, 0, 100, 100]
    yolo_bbox = convert_to_yolo_format(bbox, image_size)
    assert np.allclose(yolo_bbox, [0.5, 0.5, 1.0, 1.0])

def test_create_data_yaml(tmp_path):
    """Test creating data.yaml file."""
    # Test inputs
    dataset_dir = tmp_path
    class_names = ["class1", "class2", "class3"]
    
    # Create yaml file
    yaml_path = create_data_yaml(dataset_dir, class_names)
    
    # Check file exists
    assert yaml_path.exists()
    
    # Check content
    with open(yaml_path) as f:
        content = f.read()
        assert "path:" in content
        assert "train:" in content
        assert "val:" in content
        assert "test:" in content
        assert "names:" in content
        for class_name in class_names:
            assert class_name in content

def test_split_dataset(tmp_path):
    """Test dataset splitting."""
    # Create test files
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(10):
        (image_dir / f"image_{i}.jpg").touch()
    
    # Split dataset
    splits = split_dataset(
        image_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    # Check split sizes
    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 1
    
    # Check no overlap between splits
    all_files = set()
    for files in splits.values():
        for file in files:
            assert file not in all_files
            all_files.add(file)

def test_invalid_split_ratios():
    """Test invalid split ratios."""
    with pytest.raises(ValueError):
        split_dataset(
            Path("dummy"),
            train_ratio=0.8,
            val_ratio=0.3,  # Total > 1.0
            test_ratio=0.2
        ) 