#!/bin/bash

# Exit on error
set -e

# Get the absolute path of the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env.local" | xargs)
else
    echo "Error: .env.local file not found"
    exit 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] <image_path>"
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -m, --model       Specify YOLO model path (default: models/yolov8x.pt)"
    echo "  -o, --output      Specify output directory (default: output)"
    echo ""
    echo "Example:"
    echo "  $0 input/cars.png"
    echo "  $0 -m models/yolov8x.pt input/cars.png"
}

# Default values
MODEL_PATH="models/yolov8x.pt"
OUTPUT_DIR="output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            if [[ -z "$IMAGE_PATH" ]]; then
                IMAGE_PATH="$1"
            else
                echo "Error: Multiple image paths provided"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if image path is provided
if [[ -z "$IMAGE_PATH" ]]; then
    echo "Error: Image path is required"
    usage
    exit 1
fi

# Convert relative paths to absolute paths
if [[ ! "$IMAGE_PATH" = /* ]]; then
    IMAGE_PATH="$PROJECT_ROOT/$IMAGE_PATH"
fi
if [[ ! "$MODEL_PATH" = /* ]]; then
    MODEL_PATH="$PROJECT_ROOT/$MODEL_PATH"
fi
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi

# Check if image exists
if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Check if model exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please download the model and place it in the models directory"
    exit 1
fi

# Check if virtual environment exists
VENV_DIR="$PROJECT_ROOT/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Function to activate virtual environment based on OS
activate_venv() {
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # macOS or Linux
        source "$VENV_DIR/bin/activate"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows
        source "$VENV_DIR/Scripts/activate"
    else
        echo "Error: Unsupported operating system"
        exit 1
    fi
}

# Activate virtual environment
echo "Activating virtual environment..."
activate_venv

# Install requirements if not already installed
if [[ ! -f "$VENV_DIR/requirements_installed" ]]; then
    echo "Installing requirements..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
    touch "$VENV_DIR/requirements_installed"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the auto labeling script
echo "Running auto labeling on $IMAGE_PATH..."
cd src/auto_labelling_yolov8
python test_full_flow.py --model "$MODEL_PATH" --output "$OUTPUT_DIR" "$IMAGE_PATH"

# Deactivate virtual environment
deactivate

echo "Auto labeling completed successfully!"
echo "Results saved to: $OUTPUT_DIR" 