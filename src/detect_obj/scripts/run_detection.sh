#!/bin/bash

# Màu sắc cho output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Hàm kiểm tra lỗi
check_error() {
    if [ $? -ne 0 ] && [ "$2" != "ignore" ]; then
        echo -e "${RED}[ERROR] $1${NC}"
        exit 1
    fi
}

# Hàm hiển thị thông báo
log_message() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

debug_message() {
    echo -e "${YELLOW}[DEBUG] $1${NC}"
}

# Kiểm tra đối số đầu vào
if [ $# -ne 1 ]; then
    echo -e "${RED}Usage: $0 <path_to_image>${NC}"
    exit 1
fi

IMAGE_PATH="$1"

# Kiểm tra file ảnh tồn tại
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}[ERROR] Image file not found: $IMAGE_PATH${NC}"
    exit 1
fi

# Kiểm tra và activate virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    log_message "Creating virtual environment..."
    python3.10 -m venv $VENV_DIR
    check_error "Failed to create virtual environment"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate
check_error "Failed to activate virtual environment"
log_message "Virtual environment activated"

# Kiểm tra và cài đặt dependencies
if [ ! -f "src/detect_obj/requirements.txt" ]; then
    echo -e "${RED}[ERROR] requirements.txt not found${NC}"
    exit 1
fi

log_message "Installing dependencies..."
pip install -r src/detect_obj/requirements.txt
check_error "Failed to install dependencies"

# Tạo thư mục data/images nếu chưa tồn tại
mkdir -p src/detect_obj/data/images
check_error "Failed to create data directory"

# Copy ảnh vào thư mục data/images nếu cần
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

# Chạy detection
log_message "Running person detection..."
PYTHONPATH=. python -m src.detect_obj.main "$IMAGE_FILENAME" 2>&1 | tee detection.log
check_error "Detection failed"

# Kiểm tra kết quả
if [ -f "src/detect_obj/outputs/latest_detection.jpg" ]; then
    log_message "Detection completed successfully!"
    log_message "Results saved to: src/detect_obj/outputs/latest_detection.jpg"
    
    # Hiển thị số người được phát hiện từ log
    PERSON_COUNT=$(grep "Number of people detected:" detection.log | awk '{print $NF}')
    log_message "Number of people detected: $PERSON_COUNT"
else
    echo -e "${RED}[ERROR] Detection output not found${NC}"
    exit 1
fi

# Clean up
rm detection.log

log_message "Process completed successfully!" 