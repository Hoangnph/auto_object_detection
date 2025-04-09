# Auto Labelling YOLOv8

Hệ thống tự động gán nhãn ảnh sử dụng YOLOv8 và các mô hình LLM (Claude, OpenAI, Gemini) cho các dự án Computer Vision.

## Mô tả dự án

Dự án này cung cấp một pipeline tự động để:
1. Phát hiện đối tượng trong ảnh sử dụng YOLOv8
2. Tự động gán nhãn chi tiết cho các đối tượng được phát hiện bằng LLM
3. Lưu trữ và quản lý kết quả gán nhãn
4. Cung cấp giao diện để xem và điều chỉnh nhãn

## Cài đặt

1. Clone repository:
```bash
git clone <repository_url>
cd auto_labelling_yolov8
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate  # Windows
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

4. Cấu hình API keys:
- Tạo file `.venv.local` trong thư mục gốc
- Thêm các API keys cần thiết:
```bash
OPENAI_API_KEY=your_openai_key
GOOGLE_GEMINI_API_KEY=your_gemini_key
ANTHROPIC_CLAUDE_API_KEY=your_claude_key
```

## Cấu trúc dự án

```
src/auto_labelling_yolov8/
├── config/               # Cấu hình và tham số
│   └── labelling_config.yaml  # Cấu hình gán nhãn
├── models/              # Model files
│   ├── yolo_detector.py  # YOLOv8 detector
│   └── llm_labeler.py   # LLM labeler
├── output/             # Kết quả đầu ra
│   └── run_YYYYMMDD_HHMMSS/  # Thư mục kết quả cho mỗi lần chạy
│       ├── images/     # Ảnh đã cắt
│       └── metadata/   # Metadata và kết quả gán nhãn
├── utils/              # Tiện ích
│   ├── image_utils.py  # Xử lý ảnh
│   └── logging_utils.py # Logging
├── docs/              # Tài liệu
└── test_full_flow.py  # Script test toàn bộ pipeline
```

## Sử dụng

1. Chuẩn bị dữ liệu:
   - Đặt ảnh cần gán nhãn vào thư mục `input/`
   - Cấu hình các tham số trong `config/labelling_config.yaml`

2. Chạy auto labelling:
```bash
python test_full_flow.py
```

3. Kết quả sẽ được lưu trong thư mục `output/run_YYYYMMDD_HHMMSS/` với cấu trúc:
   - `images/`: Chứa các ảnh đã cắt
   - `metadata/`: Chứa kết quả gán nhãn và metadata

## Cấu hình

File `labelling_config.yaml` cho phép cấu hình:
- Các loại phương tiện được phép
- Ngưỡng tin cậy cho phát hiện và gán nhãn
- Các tham số cho YOLOv8 và LLM

## Testing

Chạy test toàn bộ pipeline:
```bash
python test_full_flow.py
```

## Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## Contact

- Author: [Your Name]
- Email: [your.email@example.com] 