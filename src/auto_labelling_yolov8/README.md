# Auto-Labelling System with YOLOv8 and LLM

Hệ thống tự động gán nhãn sử dụng YOLOv8 và Large Language Models (LLM) để phát hiện, cắt và gán nhãn chi tiết cho đối tượng trong ảnh.

## Yêu cầu hệ thống

### Phần cứng
- CPU: Intel/AMD x64 hoặc ARM64
- RAM: Tối thiểu 8GB
- GPU: Không bắt buộc, nhưng khuyến nghị NVIDIA với CUDA để tăng tốc độ xử lý

### Phần mềm
- Python 3.8 trở lên
- OpenCV
- YOLOv8
- OpenAI API (hoặc các LLM API khác được hỗ trợ)

## Cài đặt

1. Clone repository:
```bash
git clone <repository_url>
cd auto_labelling_yolov8
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Thiết lập biến môi trường:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Cấu trúc thư mục

```
auto_labelling_yolov8/
├── input/                  # Thư mục chứa ảnh đầu vào
├── raw/                    # Thư mục chứa ảnh đã cắt và metadata
│   ├── images/            # Ảnh đã cắt
│   └── metadata/          # Thông tin metadata
├── output/                # Kết quả cuối cùng
├── models/               
│   ├── yolo_detector.py   # YOLOv8 detector
│   └── llm_labeler.py     # LLM labeling system
└── utils/
    ├── image_utils.py     # Tiện ích xử lý ảnh
    ├── cropping_utils.py  # Tiện ích cắt ảnh
    └── directory_utils.py # Tiện ích quản lý thư mục
```

## Thông tin cần chuẩn bị

1. **API Key**:
   - OpenAI API key (bắt buộc)
   - Hoặc API key của các LLM khác (tùy chọn)

2. **Dữ liệu đầu vào**:
   - Ảnh cần gán nhãn (định dạng: jpg, png)
   - Đặt trong thư mục `input/`

3. **Cấu hình YOLOv8**:
   - Model: Mặc định sử dụng YOLOv8n
   - Confidence threshold: Mặc định 0.3
   - Có thể điều chỉnh trong file config

4. **Cấu hình LLM**:
   - Model: Mặc định GPT-4-Vision
   - Prompt template có thể tùy chỉnh
   - Số lần retry khi gặp lỗi

## Sử dụng

1. **Phát hiện và cắt đối tượng**:
```python
from models.yolo_detector import YOLODetector
from utils.cropping_utils import crop_and_save_objects

# Khởi tạo detector
detector = YOLODetector(confidence_threshold=0.3)

# Phát hiện và cắt đối tượng
detections = detector.detect(image)
saved_paths = crop_and_save_objects(image, detections, "raw/images")
```

2. **Gán nhãn chi tiết**:
```python
from models.llm_labeler import LLMLabeler

# Khởi tạo labeler
labeler = LLMLabeler()

# Gán nhãn cho một ảnh
label_info = labeler.get_detailed_label(image)

# Gán nhãn hàng loạt
batch_results = labeler.label_batch(image_paths)
```

## Xử lý lỗi

1. **API Rate Limits**:
   - Hệ thống tự động retry khi gặp lỗi API
   - Có thể điều chỉnh số lần retry và thời gian chờ

2. **Lỗi phát hiện**:
   - Kiểm tra confidence threshold
   - Đảm bảo ảnh đầu vào rõ ràng và đúng định dạng

3. **Lỗi gán nhãn**:
   - Kiểm tra API key và quyền truy cập
   - Xem log để biết chi tiết lỗi

## Tối ưu hiệu suất

1. **Batch Processing**:
   - Sử dụng `label_batch()` thay vì xử lý từng ảnh
   - Điều chỉnh batch size phù hợp với rate limit

2. **Caching**:
   - Kết quả được lưu trong metadata
   - Tránh gọi API lặp lại cho cùng một ảnh

3. **Parallel Processing**:
   - Có thể chạy nhiều instance cho các batch khác nhau
   - Chú ý đến rate limit của API

## Logging và Monitoring

- Log được lưu trong thư mục `logs/`
- Sử dụng loguru để theo dõi quá trình xử lý
- Kiểm tra metadata để theo dõi kết quả

## Hỗ trợ và Đóng góp

- Tạo issue trên GitHub
- Pull requests được chào đón
- Tuân thủ coding style và test coverage 