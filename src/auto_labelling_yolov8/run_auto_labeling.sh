#!/bin/bash

# Exit on error
set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with timestamp
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    log "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check for required environment variables
required_vars=("OPENAI_API_KEY" "GOOGLE_API_KEY" "ANTHROPIC_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    log "${YELLOW}Warning: The following environment variables are not set:${NC}"
    printf '%s\n' "${missing_vars[@]}"
    log "${YELLOW}Some features may not be available${NC}"
fi

# Check for input image
if [[ ! -f "input/cars.png" ]]; then
    log "${RED}Error: Input image 'input/cars.png' not found${NC}"
    log "Please place your input image in the 'input' directory"
    exit 1
fi

# Create required directories
log "${GREEN}Creating required directories...${NC}"
mkdir -p input output

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
    log "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Run the Python script
log "${GREEN}Starting auto-labeling process...${NC}"
cd "$SCRIPT_DIR"

# Run the Python script and capture its output
if python3 test_full_flow.py; then
    log "${GREEN}Auto-labeling completed successfully!${NC}"
    log "Check the output directory for results"
else
    log "${RED}Error: Auto-labeling process failed${NC}"
    exit 1
fi

# Deactivate virtual environment if it was activated
if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
fi 