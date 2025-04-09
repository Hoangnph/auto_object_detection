"""Tests for LLM-based labeling system."""
import pytest
from pathlib import Path
from models.llm_labeler import LLMLabeler
from utils.image_utils import load_image

def test_llm_labeling():
    """Test LLM-based image labeling."""
    # Initialize labeler
    labeler = LLMLabeler()
    
    # Load a cropped image
    image_dir = Path("raw/images")
    test_images = list(image_dir.glob("car_*.jpg"))
    assert len(test_images) > 0, "No test images found"
    
    image = load_image(test_images[0])
    
    # Get detailed label
    label_info = labeler.get_detailed_label(image)
    
    # Verify label structure
    assert "make" in label_info, "Car make not found in label"
    assert "model" in label_info, "Car model not found in label"
    assert "color" in label_info, "Car color not found in label"
    assert "type" in label_info, "Car type not found in label"
    
    # Print label info
    print("\nDetailed label information:")
    for key, value in label_info.items():
        print(f"- {key}: {value}")

def test_batch_labeling():
    """Test batch labeling of multiple images."""
    # Initialize labeler
    labeler = LLMLabeler()
    
    # Get test images
    image_dir = Path("raw/images")
    test_images = list(image_dir.glob("car_*.jpg"))[:3]  # Test with first 3 images
    
    # Perform batch labeling
    batch_results = labeler.label_batch(test_images)
    
    # Verify results
    assert len(batch_results) == len(test_images)
    for result in batch_results:
        assert "image_path" in result
        assert "label_info" in result
        assert "status" in result
        assert result["status"] == "success"
    
    # Print batch results
    print("\nBatch labeling results:")
    for result in batch_results:
        print(f"\nImage: {result['image_path']}")
        for key, value in result["label_info"].items():
            print(f"- {key}: {value}")

if __name__ == "__main__":
    test_llm_labeling()
    test_batch_labeling() 