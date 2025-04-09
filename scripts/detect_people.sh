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
if [ $# -ne 1 ]; then
    echo -e "${RED}Usage: $0 <path_to_image>${NC}"
    exit 1
fi

IMAGE_PATH="$1"

# Check if image file exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}[ERROR] Image file not found: $IMAGE_PATH${NC}"
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
if [ ! -f "src/detect_obj/requirements.txt" ]; then
    echo -e "${RED}[ERROR] requirements.txt not found${NC}"
    exit 1
fi

log_message "Installing dependencies..."
pip install -r src/detect_obj/requirements.txt
check_error "Failed to install dependencies"

# Create necessary directories
mkdir -p src/detect_obj/data/images
check_error "Failed to create data directory"
mkdir -p src/detect_obj/models
check_error "Failed to create models directory"

# Check and manage YOLO model
MODEL_NAME="yolov8n.pt"
MODEL_PATH="src/detect_obj/models/$MODEL_NAME"
if [ ! -f "$MODEL_PATH" ]; then
    log_message "Model not found in project directory. Downloading..."
    if [ -f "$MODEL_NAME" ]; then
        log_message "Found model in current directory, moving to project models directory..."
        mv "$MODEL_NAME" "$MODEL_PATH"
        check_error "Failed to move model file"
    else
        # Download directly to the models directory
        cd src/detect_obj/models
        wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
        check_error "Failed to download model"
        cd ../../..
    fi
    log_message "Model setup completed"
else
    log_message "Using existing model from: $MODEL_PATH"
fi

# Copy image to data/images if needed
IMAGE_FILENAME=$(basename "$IMAGE_PATH")
TARGET_PATH="src/detect_obj/data/images/$IMAGE_FILENAME"

if [ "$IMAGE_PATH" != "$TARGET_PATH" ]; then
    debug_message "Copying image to data/images directory..."
    cp "$IMAGE_PATH" "$TARGET_PATH"
    check_error "Failed to copy image"
else
    debug_message "Image already in correct location, skipping copy..."
fi

debug_message "Image filename: $IMAGE_FILENAME"

# Run detection
log_message "Running person detection..."
PYTHONPATH=. python -m src.detect_obj.main "$IMAGE_FILENAME" 2>&1 | tee detection.log
check_error "Detection failed"

# Check results
if [ -f "src/detect_obj/outputs/latest_detection.jpg" ]; then
    log_message "Detection completed successfully!"
    log_message "Results saved to: src/detect_obj/outputs/latest_detection.jpg"
    
    # Display detected person count from log
    PERSON_COUNT=$(grep "Number of people detected:" detection.log | awk '{print $NF}')
    log_message "Number of people detected: $PERSON_COUNT"
else
    echo -e "${RED}[ERROR] Detection output not found${NC}"
    exit 1
fi

# Clean up
rm detection.log

log_message "Process completed successfully!" 