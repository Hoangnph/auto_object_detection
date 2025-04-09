#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_message "Git is not installed. Please install git first." "$RED"
    exit 1
fi

# Initialize git if not already initialized
if [ ! -d .git ]; then
    print_message "Initializing git repository..." "$YELLOW"
    git init
fi

# Security checks
print_message "Running security checks..." "$YELLOW"

# Check for .env files
if [ -f .env ]; then
    print_message "WARNING: .env file detected. Make sure it's in .gitignore" "$RED"
    read -p "Press Enter to continue or Ctrl+C to abort"
fi

# Check for API keys in code
print_message "Checking for potential API keys in code..." "$YELLOW"
if grep -r "api_key\|API_KEY\|secret\|SECRET" --exclude-dir={.git,venv,.venv,env} .; then
    print_message "WARNING: Potential API keys found in code. Please review above lines." "$RED"
    read -p "Press Enter to continue or Ctrl+C to abort"
fi

# Add files to git
print_message "Adding files to git..." "$YELLOW"
git add .

# Show what will be committed
print_message "Files to be committed:" "$YELLOW"
git status

# Confirm with user
read -p "Do you want to continue with the commit? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_message "Deployment aborted." "$RED"
    exit 1
fi

# Commit
print_message "Committing changes..." "$YELLOW"
git commit -m "Initial commit: Auto Object Detection project"

# Add remote if not exists
if ! git remote | grep -q "origin"; then
    print_message "Adding remote origin..." "$YELLOW"
    git remote add origin https://github.com/Hoangnph/auto_object_detection.git
fi

# Push to GitHub
print_message "Pushing to GitHub..." "$YELLOW"
git push -u origin main

print_message "Deployment completed successfully!" "$GREEN"
print_message "Repository: https://github.com/Hoangnph/auto_object_detection" "$GREEN" 