#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/../.venv"

# Print header
echo "============================================="
echo "Batch Auto-labeling System - Run Script"
echo "============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "Please run 'python -m venv .venv' first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install/upgrade required packages
echo "📦 Installing/upgrading required packages..."
pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"
pip install psutil  # Required for system monitoring

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p "$PROJECT_ROOT/input"
mkdir -p "$PROJECT_ROOT/output"

# Check if input directory exists and has images
if [ ! -d "$PROJECT_ROOT/input" ]; then
    echo "❌ Input directory not found"
    echo "Please create an 'input' directory and add your images"
    exit 1
fi

# Get list of image files
IMAGE_FILES=($(find "$PROJECT_ROOT/input" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \)))

if [ ${#IMAGE_FILES[@]} -eq 0 ]; then
    echo "❌ No image files found in input directory"
    echo "Please add some images (jpg, jpeg, png) to the input directory"
    exit 1
fi

echo "📸 Found ${#IMAGE_FILES[@]} images to process"
echo ""

# Process each image
for image_path in "${IMAGE_FILES[@]}"; do
    echo "🔄 Processing: $(basename "$image_path")"
    
    # Create a temporary symlink to the current image
    ln -sf "$image_path" "$PROJECT_ROOT/input/cars.png"
    
    # Run the program
    cd "$PROJECT_ROOT"
    python test_full_flow.py
    
    # Check if the program ran successfully
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed $(basename "$image_path")"
    else
        echo "❌ Failed to process $(basename "$image_path")"
        echo "Skipping to next image..."
    fi
    
    echo ""
done

# Remove temporary symlink
rm -f "$PROJECT_ROOT/input/cars.png"

# Deactivate virtual environment
deactivate

echo "============================================="
echo "Batch processing completed"
echo "============================================="
echo ""
echo "Results have been saved to the output directory:"
echo "  $PROJECT_ROOT/output/run_*"
echo ""
echo "Each run creates a new directory with timestamp"
echo "containing images, metadata, and documentation" 