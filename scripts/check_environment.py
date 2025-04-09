#!/usr/bin/env python3
"""Check environment setup for auto-labeling system."""
import os
import sys
from pathlib import Path
import pkg_resources
import importlib
from dotenv import load_dotenv

# Get the project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_environment_variables():
    """Load environment variables from .env.local file."""
    env_file = PROJECT_ROOT / ".env.local"
    if env_file.exists():
        print(f"Loading environment variables from {env_file}")
        load_dotenv(env_file)
    else:
        print(f"WARNING: Environment file {env_file} not found")

def check_python_version():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("WARNING: Python version should be >= 3.8")
        return False
    return True

def check_dependencies():
    """Check required dependencies."""
    required = {
        'numpy': '1.21.0',
        'opencv-python': '4.5.0',
        'PyYAML': '6.0',
        'loguru': '0.7.0',
        'requests': '2.31.0',
        'ultralytics': '8.0.0',
        'python-dotenv': '1.0.0'
    }
    
    all_installed = True
    for package, min_version in required.items():
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}: {version} (required >= {min_version})")
            if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                print(f"WARNING: {package} version should be >= {min_version}")
                all_installed = False
        except pkg_resources.DistributionNotFound:
            print(f"ERROR: {package} is not installed")
            all_installed = False
    return all_installed

def check_environment_variables():
    """Check required environment variables."""
    required_vars = ['CLAUDE_API_KEY']  # Add other API keys if needed
    all_set = True
    for var in required_vars:
        if not os.getenv(var):
            print(f"WARNING: Environment variable {var} is not set")
            all_set = False
        else:
            print(f"{var} is set")
    return all_set

def check_yolo_model():
    """Check YOLOv8 model."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # This will download the model if not present
        print("YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load YOLOv8 model: {e}")
        return False

def main():
    """Run all checks."""
    print("\n=== Checking Environment Setup ===\n")
    
    # Load environment variables first
    load_environment_variables()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment Variables", check_environment_variables),
        ("YOLOv8 Model", check_yolo_model)
    ]
    
    all_passed = True
    for name, check in checks:
        print(f"\n--- Checking {name} ---")
        if not check():
            all_passed = False
            print(f"WARNING: {name} check failed")
    
    print("\n=== Environment Check Complete ===")
    if not all_passed:
        print("\nWARNING: Some checks failed. Please fix the issues above.")
        sys.exit(1)
    else:
        print("\nAll checks passed successfully!")

if __name__ == "__main__":
    main() 