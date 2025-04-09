"""Image processing utilities."""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from loguru import logger

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        FileNotFoundError: If the image file does not exist
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    logger.debug(f"Loading image from {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True
) -> np.ndarray:
    """Preprocess image for model input.
    
    Args:
        image: Input image
        target_size: Target size for resizing
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image
    """
    logger.debug(f"Preprocessing image to size {target_size}")
    
    # Resize image
    image = resize_image(image, target_size)
    
    # Normalize if requested
    if normalize:
        image = normalize_image(image)
        
    return image

def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=interpolation)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1].
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0 