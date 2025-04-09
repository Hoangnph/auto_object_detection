# Auto Object Detection with YOLOv8 and LLM

An advanced auto-labeling system that combines YOLOv8 object detection with Large Language Models (LLMs) for intelligent image classification and labeling.

## 🌟 Features

- **Multi-Model Object Detection**: Supports multiple YOLOv8 models (yolov8n.pt, yolov8x.pt)
- **Intelligent LLM Labeling**: Integration with multiple LLM providers:
  - Claude API
  - OpenAI API
  - Google Gemini API
- **Automated Pipeline**: Complete workflow from detection to labeling
- **Cross-Platform Support**: Works on Windows, Linux, and macOS
- **Organized Output Structure**: Timestamp-based directories with clear organization
- **Secure API Management**: Environment-based API key handling

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Hoangnph/auto_object_detection.git
cd auto_object_detection
```

2. Create and activate virtual environment:
```bash
# The script will handle this automatically
./scripts/run_auto_labeling.sh
```

## 🔧 Configuration

1. Create a `.env` file in the project root:
```env
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

2. Configure model settings in `src/auto_labelling_yolov8/config/labelling_config.yaml`

## 🚀 Usage

Run the auto-labeling script:
```bash
./scripts/run_auto_labeling.sh -m models/yolov8x.pt input/your_image.jpg
```

Options:
- `-m, --model`: Path to YOLO model (default: models/yolov8x.pt)
- `-o, --output`: Custom output directory (optional)

## 📁 Project Structure

```
auto_object_detection/
├── scripts/
│   └── run_auto_labeling.sh       # Main execution script
├── src/
│   └── auto_labelling_yolov8/     # Core package
│       ├── config/                 # Configuration files
│       ├── models/                # Model implementations
│       └── docs/                  # Documentation
├── tests/                         # Test suite
└── requirements.txt              # Python dependencies
```

## 🔒 Security Notes

- API keys are managed through environment variables
- Sensitive data is not committed to the repository
- Model weights should be downloaded separately

## 📋 Requirements

- Python 3.8+
- OpenCV
- YOLOv8
- LLM API access (Claude/OpenAI/Gemini)
- Sufficient storage for images and models

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Completed Features

- [x] YOLOv8 integration with multiple model support
- [x] Multi-provider LLM integration
- [x] Automated image processing pipeline
- [x] Cross-platform compatibility
- [x] Secure API key management
- [x] Organized output structure
- [x] Comprehensive documentation
- [x] Shell script automation
- [x] Virtual environment management 