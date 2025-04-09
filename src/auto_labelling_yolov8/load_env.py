"""Load environment variables from .venv.local file."""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    env_path = Path(__file__).parent / '.venv.local'
    if env_path.exists():
        load_dotenv(env_path)
        print("Loaded environment variables from .venv.local")
    else:
        print("Warning: .venv.local file not found")

if __name__ == "__main__":
    load_environment() 