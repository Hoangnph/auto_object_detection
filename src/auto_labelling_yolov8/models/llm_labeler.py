import os
import cv2
import json
import yaml
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from loguru import logger
import numpy as np

class LLMLabeler:
    """LLM-based image labeling class."""
    
    def __init__(self, config_path: str = "config/labelling_config.yaml", output_dir: Optional[Path] = None) -> None:
        """Initialize LLM labeler with configuration.
        
        Args:
            config_path: Path to the configuration file
            output_dir: Optional output directory for saving results
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = output_dir
        
        # Initialize API-related attributes
        self.api_keys = {}
        self.api_urls = {}
        self.available_apis = []
        
        # Load API keys from environment
        self._load_api_keys()
        
        # Validate configuration
        self._validate_config()

        logger.info("LLM labeler initialized")

    def _load_api_keys(self) -> None:
        """Load API keys from environment variables.
        
        Checks for OpenAI, Google Gemini, and Anthropic Claude API keys.
        Stores keys and URLs in instance variables.
        Logs success/failure of loading attempts.
        """
        # Initialize API storage
        self.api_keys = {}
        self.api_urls = {}
        self.available_apis = []
        
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.api_keys["openai"] = openai_key
            self.api_urls["openai"] = os.getenv("OPENAI_URL", "https://api.openai.com/v1")
            self.available_apis.append("openai")
            logger.info("OpenAI API key loaded successfully")
        else:
            logger.warning("OpenAI API key not found in environment")
            
        # Google Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.api_keys["gemini"] = gemini_key
            self.api_urls["gemini"] = os.getenv("GEMINI_URL", "https://generativelanguage.googleapis.com/v1")
            self.available_apis.append("gemini")
            logger.info("Google Gemini API key loaded successfully")
        else:
            logger.warning("Google Gemini API key not found in environment")
            
        # Anthropic Claude
        claude_key = os.getenv("CLAUDE_API_KEY")
        if claude_key:
            self.api_keys["claude"] = claude_key
            self.api_urls["claude"] = os.getenv("CLAUDE_URL", "https://api.anthropic.com/v1")
            self.available_apis.append("claude")
            logger.info("Anthropic Claude API key loaded successfully")
        else:
            logger.warning("Anthropic Claude API key not found in environment")
            
        if not self.available_apis:
            logger.warning("No API keys found in environment variables")

    async def test_connection(self, image: np.ndarray) -> bool:
        """Test connection to LLM API with a sample image.
        
        Args:
            image: Image to test with
            
        Returns:
            bool: True if test was successful, False otherwise
        """
        try:
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Get classes from config
            classes = self.config["classes"]["selected"]
            
            # Format prompt
            prompt = self.config["llm"]["prompt_template"].format(
                classes="\n".join(f"- {cls}" for cls in classes),
                example_images=""  # Empty string for example_images placeholder
            )
            
            # Try each API in order
            for api in self.config["llm"]["api_order"]:
                try:
                    logger.info(f"Testing {api} API...")
                    response = None
                    result = None
                    
                    if api == "openai":
                        response = await self._call_openai(image_base64, prompt)
                        if response and "choices" in response:
                            result = response["choices"][0]["message"]["content"]
                    elif api == "gemini":
                        response = await self._call_gemini(image_base64, prompt)
                        if response and "candidates" in response:
                            result = response["candidates"][0]["content"]["parts"][0]["text"]
                    else:  # claude
                        response = await self._call_claude(image_base64, prompt)
                        if response and isinstance(response, dict):
                            # Handle Claude 3 response format
                            if 'content' in response and isinstance(response['content'], list):
                                for content_item in response['content']:
                                    if content_item.get('type') == 'text':
                                        result = content_item.get('text', '')
                                        break
                            elif 'content' in response and isinstance(response['content'], str):
                                result = response['content']
                    
                    if not result:
                        logger.error(f"No valid response content from {api} API")
                        continue
                        
                    # Parse response
                    try:
                        # First try to parse as JSON
                        data = json.loads(result)
                        if "class_name" in data and "confidence" in data:
                            class_name = data["class_name"]
                            confidence = float(data["confidence"])
                            
                            # Validate class name and confidence
                            if class_name in classes and 0 <= confidence <= 1:
                                logger.info(f"Test successful with {api} API: {class_name} {confidence}")
                                return True
                            else:
                                logger.error(f"Invalid response format from {api} API: {result}")
                    except json.JSONDecodeError:
                        # If not JSON, try to parse as direct string format
                        try:
                            parts = result.strip().split()
                            if len(parts) >= 2:
                                class_name = parts[0]
                                confidence = float(parts[1])
                                
                                # Validate class name and confidence
                                if class_name in classes and 0 <= confidence <= 1:
                                    logger.info(f"Test successful with {api} API: {class_name} {confidence}")
                                    return True
                                else:
                                    logger.error(f"Invalid response format from {api} API: {result}")
                        except (ValueError, IndexError):
                            logger.error(f"Failed to parse direct string response from {api} API: {result}")
                    except (KeyError, ValueError) as e:
                        logger.error(f"Invalid response structure from {api} API: {e}")
                        
                except Exception as e:
                    logger.warning(f"Failed to test {api} API: {str(e)}")
                    continue
            
            logger.error("All API tests failed")
            return False
            
        except Exception as e:
            logger.error(f"Error in test_connection: {str(e)}")
            return False

    async def get_detailed_label(self, image_path: str) -> Optional[Tuple[str, float]]:
        """Get detailed label for an image using configured LLM API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Optional[Tuple[str, float]]: Tuple of (class_name, confidence) if successful,
                                       None otherwise
        """
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Get classes from config
            classes = self.config["classes"]["selected"]
            
            # Format prompt
            prompt = self.config["llm"]["prompt_template"].format(
                classes="\n".join(f"- {cls}" for cls in classes),
                example_images=""  # Empty string for example_images placeholder
            )
            
            # Try each API in order
            for api in self.config["llm"]["api_order"]:
                try:
                    logger.info(f"Trying {api} API for {image_path}...")
                    response = None
                    result = None
                    
                    if api == "openai":
                        response = await self._call_openai(image_base64, prompt)
                        if response and "choices" in response:
                            result = response["choices"][0]["message"]["content"]
                    elif api == "gemini":
                        response = await self._call_gemini(image_base64, prompt)
                        if response and "candidates" in response:
                            result = response["candidates"][0]["content"]["parts"][0]["text"]
                    else:  # claude
                        response = await self._call_claude(image_base64, prompt)
                        if response and isinstance(response, dict):
                            # Handle Claude 3 response format
                            if 'content' in response and isinstance(response['content'], list):
                                for content_item in response['content']:
                                    if content_item.get('type') == 'text':
                                        result = content_item.get('text', '')
                                        break
                            elif 'content' in response and isinstance(response['content'], str):
                                result = response['content']
                    
                    if not result:
                        logger.error(f"No valid response content from {api} API for {image_path}")
                        continue
                        
                    # Parse response
                    try:
                        # First try to parse as JSON
                        data = json.loads(result)
                        if "class_name" in data and "confidence" in data:
                            class_name = data["class_name"]
                            confidence = float(data["confidence"])
                            
                            # Validate class name and confidence
                            if class_name in classes and 0 <= confidence <= 1:
                                logger.info(f"Successfully labeled {image_path}: {class_name} {confidence}")
                                return class_name, confidence
                            else:
                                logger.error(f"Invalid response format from {api} API: {result}")
                    except json.JSONDecodeError:
                        # If not JSON, try to parse as direct string format
                        try:
                            parts = result.strip().split()
                            if len(parts) >= 2:
                                class_name = parts[0]
                                confidence = float(parts[1])
                                
                                # Validate class name and confidence
                                if class_name in classes and 0 <= confidence <= 1:
                                    logger.info(f"Successfully labeled {image_path}: {class_name} {confidence}")
                                    return class_name, confidence
                                else:
                                    logger.error(f"Invalid response format from {api} API: {result}")
                        except (ValueError, IndexError):
                            logger.error(f"Failed to parse direct string response from {api} API: {result}")
                    except (KeyError, ValueError) as e:
                        logger.error(f"Invalid response structure from {api} API: {e}")
                        
                except Exception as e:
                    logger.warning(f"Failed to get label from {api} API for {image_path}: {str(e)}")
                    continue
            
            logger.error(f"All API attempts failed for {image_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting label for {image_path}: {str(e)}")
            return None

    def save_results(self, results: List[Dict], original_image_path: str) -> None:
        """Save labeling results in the specified format.
        
        Args:
            results: List of labeling results
            original_image_path: Path to the original image
        """
        try:
            if not self.output_dir:
                raise ValueError("Output directory not specified")
                
            # Get output format from config
            output_format = self.config["output"]["format"].lower()
            
            # Save results based on format
            if output_format == "yolo":
                self._save_yolo_format(results, original_image_path)
            elif output_format == "coco":
                self._save_coco_format(results, original_image_path)
            elif output_format == "voc":
                self._save_voc_format(results, original_image_path)
            else:
                logger.error(f"Unsupported output format: {output_format}")
                
            # Save summary JSON
            summary_path = self.output_dir / "metadata" / "labeling_results.json"
            with open(summary_path, "w") as f:
                json.dump({
                    "original_image": str(original_image_path),
                    "timestamp": datetime.now().isoformat(),
                    "format": output_format,
                    "results": results
                }, f, indent=2)
                
            logger.info(f"Saved labeling results to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def _save_yolo_format(self, results: List[Dict], original_image_path: str) -> None:
        """Save results in YOLO format."""
        # Create labels directory
        labels_dir = self.output_dir / "metadata" / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        # Create class mapping
        classes = self.config["classes"]["selected"]
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Save class mapping
        with open(self.output_dir / "metadata" / "classes.txt", "w") as f:
            for cls in classes:
                f.write(f"{cls}\n")
        
        # Save results
        label_file = labels_dir / "results.txt"
        with open(label_file, "w") as f:
            for result in results:
                if result["status"] == "success" and result["label_info"]:
                    class_name, confidence = result["label_info"]
                    class_idx = class_to_idx[class_name]
                    
                    # YOLO format: <class_index> <confidence>
                    if self.config["output"]["include_confidence"]:
                        f.write(f"{class_idx} {confidence:.4f}\n")
                    else:
                        f.write(f"{class_idx}\n")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dict containing configuration parameters
        
        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If config file is invalid YAML
        """
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration file: {e}")
            raise 

    def _validate_config(self) -> None:
        """Validate the loaded configuration.
        
        Raises:
            ValueError: If required configuration fields are missing or invalid.
        """
        if not self.config:
            raise ValueError("Configuration is empty")
            
        # Check required top-level sections
        required_sections = ["model", "data", "classes", "llm", "output", "paths"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing '{section}' section in configuration")
        
        # Validate model config
        model_config = self.config["model"]
        required_model_fields = ["name", "confidence_threshold", "iou_threshold", "device"]
        for field in required_model_fields:
            if field not in model_config:
                raise ValueError(f"Missing '{field}' in model configuration")
                
        # Validate classes config
        classes_config = self.config["classes"]
        if not isinstance(classes_config.get("selected", []), list):
            raise ValueError("'selected' classes must be a list")
        if not classes_config["selected"]:
            raise ValueError("No classes selected for detection")
            
        # Validate LLM config
        llm_config = self.config["llm"]
        if "api_order" not in llm_config:
            raise ValueError("Missing 'api_order' in llm configuration")
        if "prompt_template" not in llm_config:
            raise ValueError("Missing 'prompt_template' in llm configuration")
            
        # Validate output config
        output_config = self.config["output"]
        if "format" not in output_config:
            raise ValueError("Missing 'format' in output configuration")
        if output_config["format"] not in ["yolo", "coco", "voc"]:
            raise ValueError(f"Invalid output format: {output_config['format']}")
            
        logger.info("Configuration validated successfully")

    async def _call_openai(self, image_base64: str, prompt: str) -> Dict:
        """Call OpenAI API with image and prompt.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Prompt text
            
        Returns:
            Dict: API response
        """
        try:
            import aiohttp
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_keys['openai']}"
            }
            
            data = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_urls['openai']}/chat/completions", 
                                      headers=headers, 
                                      json=data) as response:
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {}

    async def _call_gemini(self, image_base64: str, prompt: str) -> Dict:
        """Call Google Gemini API with image and prompt.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Prompt text
            
        Returns:
            Dict: API response
        """
        try:
            import aiohttp
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            url = f"{self.api_urls['gemini']}/models/gemini-pro-vision:generateContent?key={self.api_keys['gemini']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return {}

    async def _call_claude(self, image_base64: str, prompt: str) -> Dict:
        """Call Anthropic Claude API with image and prompt.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Prompt text
            
        Returns:
            Dict: API response
        """
        try:
            import aiohttp
            import json
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_keys["claude"],
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_urls['claude'], 
                                      headers=headers, 
                                      json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Claude API error: {error_text}")
                        return {}
                    response_json = await response.json()
                    logger.debug(f"Claude API response: {response_json}")
                    return response_json
                    
        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            return {}

    async def label_batch(self, image_paths: List[str], original_image_path: str) -> List[Dict]:
        """Label a batch of images using configured LLM API.
        
        Args:
            image_paths: List of paths to cropped images
            original_image_path: Path to the original image
            
        Returns:
            List[Dict]: List of labeling results for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                # Get label for image
                label_info = await self.get_detailed_label(image_path)
                
                if label_info:
                    class_name, confidence = label_info
                    results.append({
                        "image_path": str(image_path),
                        "status": "success",
                        "label_info": label_info,
                        "timestamp": datetime.now().isoformat()
                    })
                    logger.info(f"Successfully labeled {image_path}: {class_name} {confidence}")
                else:
                    results.append({
                        "image_path": str(image_path),
                        "status": "error",
                        "error": "Failed to get label",
                        "timestamp": datetime.now().isoformat()
                    })
                    logger.error(f"Failed to get label for {image_path}")
                    
            except Exception as e:
                results.append({
                    "image_path": str(image_path),
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                logger.error(f"Error labeling {image_path}: {str(e)}")
                
        # Save results
        if self.output_dir:
            self.save_results(results, original_image_path)
            
        return results 