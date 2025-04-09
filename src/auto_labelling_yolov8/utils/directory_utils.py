"""Directory management utilities."""
import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger

def create_directory_structure(base_dir: Path) -> Dict[str, Path]:
    """Create directory structure for raw images and metadata.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Dictionary containing paths to created directories
    """
    base_dir = Path(base_dir)
    
    # Create raw directory structure
    raw_dir = base_dir / "raw"
    metadata_dir = raw_dir / "metadata"
    images_dir = raw_dir / "images"
    
    # Create directories
    for dir_path in [raw_dir, metadata_dir, images_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")
    
    return {
        "raw_dir": raw_dir,
        "metadata_dir": metadata_dir,
        "images_dir": images_dir
    }

def save_metadata(
    metadata_dir: Path,
    image_id: str,
    metadata: Dict[str, Any]
) -> Path:
    """Save metadata for an image.
    
    Args:
        metadata_dir: Directory to save metadata
        image_id: Image identifier
        metadata: Metadata dictionary
        
    Returns:
        Path to saved metadata file
    """
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = metadata_dir / f"{image_id}.json"
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.debug(f"Saved metadata to {metadata_path}")
    return metadata_path

def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load metadata from file.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Loaded metadata dictionary
        
    Raises:
        FileNotFoundError: If metadata file does not exist
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    with open(metadata_path) as f:
        metadata = json.load(f)
        
    logger.debug(f"Loaded metadata from {metadata_path}")
    return metadata

def get_next_image_number(images_dir: Path) -> str:
    """Get next available image number.
    
    Args:
        images_dir: Directory containing images
        
    Returns:
        Next available image number as string (zero-padded to 3 digits)
    """
    images_dir = Path(images_dir)
    
    # Get existing image numbers
    existing_numbers = []
    for image_path in images_dir.glob("*.jpg"):
        try:
            number = int(image_path.stem.split("_")[0])
            existing_numbers.append(number)
        except (ValueError, IndexError):
            continue
            
    # Get next number
    next_number = max(existing_numbers, default=0) + 1
    
    return f"{next_number:03d}" 