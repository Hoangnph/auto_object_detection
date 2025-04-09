#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$(dirname "$PROJECT_ROOT")")"
VENV_DIR="$WORKSPACE_ROOT/.venv"

# Print header
echo "============================================="
echo "Auto-labeling System - Run Script"
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

# Check if input image exists
if [ ! -f "$PROJECT_ROOT/input/cars.png" ]; then
    echo "⚠️ Warning: input/cars.png not found"
    echo "Please place your input image in the input directory"
    exit 1
fi

# Run the program
echo "🚀 Starting auto-labeling process..."
cd "$PROJECT_ROOT"
python test_full_flow.py

# Check if the program ran successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Auto-labeling completed successfully!"
    echo "Results have been saved to the output directory"
    echo ""
    echo "To view the results, check the output directory:"
    echo "  $PROJECT_ROOT/output/run_*"
    echo ""
    echo "Each run creates a new directory with timestamp"
    echo "containing images, metadata, and documentation"
else
    echo ""
    echo "❌ Auto-labeling failed!"
    echo "Please check the error messages above"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
echo "============================================="
echo "Script completed"
echo "=============================================" 