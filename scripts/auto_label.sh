#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Error checking function
check_error() {
    if [ $? -ne 0 ] && [ "$2" != "ignore" ]; then
        echo -e "${RED}[ERROR] $1${NC}"
        exit 1
    fi
}

# Logging functions
log_message() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

debug_message() {
    echo -e "${YELLOW}[DEBUG] $1${NC}"
}

# Check input arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 <input_dir> [output_dir] [format]${NC}"
    echo -e "  format: yolo (default), coco, voc"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-data/processed}"  # Default to data/processed if not provided
FORMAT="${3:-yolo}"  # Default to YOLO format if not provided

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}[ERROR] Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Check and activate virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    log_message "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    check_error "Failed to create virtual environment"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate
check_error "Failed to activate virtual environment"
log_message "Virtual environment activated"

# Check and install dependencies
if [ ! -f "src/auto_labelling_yolov8/requirements.txt" ]; then
    echo -e "${RED}[ERROR] requirements.txt not found${NC}"
    exit 1
fi

log_message "Installing dependencies..."
pip install -r src/auto_labelling_yolov8/requirements.txt
check_error "Failed to install dependencies"

# Create necessary directories
mkdir -p src/auto_labelling_yolov8/models
check_error "Failed to create models directory"

# Check and manage YOLO model
MODEL_NAME="yolov8x.pt"  # Using YOLOv8x for better accuracy
MODEL_PATH="src/auto_labelling_yolov8/models/$MODEL_NAME"
if [ ! -f "$MODEL_PATH" ]; then
    log_message "Model not found in project directory. Downloading..."
    if [ -f "$MODEL_NAME" ]; then
        log_message "Found model in current directory, moving to project models directory..."
        mv "$MODEL_NAME" "$MODEL_PATH"
        check_error "Failed to move model file"
    else
        # Download directly to the models directory using curl
        cd src/auto_labelling_yolov8/models
        curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt -o yolov8x.pt
        check_error "Failed to download model"
        cd ../../..
    fi
    log_message "Model setup completed"
else
    log_message "Using existing model from: $MODEL_PATH"
fi

# Run auto labelling
log_message "Running auto labelling..."
PYTHONPATH=. python -m src.auto_labelling_yolov8.scripts.auto_label \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --format "$FORMAT" \
    2>&1 | tee auto_label.log

check_error "Auto labelling failed"

# Check results
if [ -d "$OUTPUT_DIR/labels" ]; then
    LABEL_COUNT=$(ls "$OUTPUT_DIR/labels" | wc -l)
    log_message "Auto labelling completed successfully!"
    log_message "Processed labels: $LABEL_COUNT"
    log_message "Results saved to: $OUTPUT_DIR"
else
    echo -e "${RED}[ERROR] No labels generated${NC}"
    exit 1
fi

# Clean up
rm auto_label.log

log_message "Process completed successfully!" 