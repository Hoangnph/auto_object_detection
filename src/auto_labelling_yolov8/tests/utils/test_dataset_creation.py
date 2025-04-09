"""Tests for dataset creation pipeline."""
import pytest
import shutil
from pathlib import Path
from utils.dataset_creation import (
    create_yolo_dataset,
    copy_and_organize_images,
    generate_label_files,
    validate_dataset
)

@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample dataset structure."""
    # Create source directories
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "images").mkdir()
    (raw_dir / "metadata").mkdir()
    
    # Create sample images and metadata
    for i in range(5):
        # Create image
        image_path = raw_dir / "images" / f"{i:03d}_car.jpg"
        image_path.touch()
        
        # Create metadata
        metadata_path = raw_dir / "metadata" / f"{i:03d}.json"
        with open(metadata_path, "w") as f:
            f.write(f'{{"class_name": "car", "bbox": [10, 10, 50, 50]}}')
            
    return raw_dir

def test_create_yolo_dataset(tmp_path, sample_dataset):
    """Test complete dataset creation pipeline."""
    # Create dataset
    dataset_info = create_yolo_dataset(
        sample_dataset,
        tmp_path / "dataset",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # Check dataset structure
    assert (tmp_path / "dataset" / "images" / "train").exists()
    assert (tmp_path / "dataset" / "labels" / "train").exists()
    assert (tmp_path / "dataset" / "data.yaml").exists()
    
    # Check dataset info
    assert "class_map" in dataset_info
    assert "statistics" in dataset_info
    assert dataset_info["statistics"]["total_images"] == 5

def test_copy_and_organize_images(tmp_path, sample_dataset):
    """Test image copying and organization."""
    # Create splits
    splits = {
        "train": [Path("000_car.jpg"), Path("001_car.jpg")],
        "val": [Path("002_car.jpg")],
        "test": [Path("003_car.jpg"), Path("004_car.jpg")]
    }
    
    # Copy images
    copy_info = copy_and_organize_images(
        sample_dataset / "images",
        tmp_path,
        splits
    )
    
    # Check copied files
    assert len(list((tmp_path / "train").glob("*.jpg"))) == 2
    assert len(list((tmp_path / "val").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "test").glob("*.jpg"))) == 2
    
    # Check copy info
    assert len(copy_info["copied_files"]) == 5
    assert all(path.exists() for path in copy_info["copied_files"].values())

def test_generate_label_files(tmp_path, sample_dataset):
    """Test label file generation."""
    # Generate labels
    label_info = generate_label_files(
        sample_dataset / "metadata",
        tmp_path,
        {"car": 0}
    )
    
    # Check label files
    label_files = list(tmp_path.glob("*.txt"))
    assert len(label_files) == 5
    
    # Check label content
    with open(label_files[0]) as f:
        content = f.read().strip()
        assert len(content.split()) == 5  # class_id + 4 bbox coords

def test_validate_dataset(tmp_path, sample_dataset):
    """Test dataset validation."""
    # Create invalid dataset
    dataset_dir = tmp_path / "invalid_dataset"
    shutil.copytree(sample_dataset, dataset_dir)
    
    # Remove some files to make it invalid
    (dataset_dir / "images" / "000_car.jpg").unlink()
    
    # Validate dataset
    validation_result = validate_dataset(dataset_dir)
    
    assert not validation_result["is_valid"]
    assert len(validation_result["errors"]) > 0

def test_empty_dataset():
    """Test handling empty dataset."""
    with pytest.raises(ValueError):
        create_yolo_dataset(
            Path("non_existent"),
            Path("output"),
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        ) 