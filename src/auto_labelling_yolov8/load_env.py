"""Load environment variables from .env.local file."""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    # Get project root directory (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env.local'
    
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"Loaded environment variables from {env_path}")
        
        # Print loaded variables (excluding sensitive data)
        env_vars = {
            "MODEL_PATH": os.getenv("MODEL_PATH"),
            "DEVICE": os.getenv("DEVICE"),
            "ANTHROPIC_URL": os.getenv("ANTHROPIC_URL"),
            "OPENAI_URL": os.getenv("OPENAI_URL"),
            "GEMINI_URL": os.getenv("GEMINI_URL")
        }
        print("Environment variables loaded:")
        for key, value in env_vars.items():
            if value:
                print(f"  {key}: {value}")
    else:
        print(f"Warning: {env_path} file not found")

if __name__ == "__main__":
    load_environment() 