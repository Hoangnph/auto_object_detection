# Person Detection with YOLOv8

Dự án sử dụng YOLOv8 để phát hiện và đếm số người trong ảnh. Được thiết kế đặc biệt để phân tích ảnh đám đông và cung cấp số lượng người chính xác.

## Cài đặt

1. Tạo môi trường ảo Python 3.10:
```bash
python3.10 -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

2. Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

## Sử dụng

### Chạy phát hiện cơ bản
1. Đặt ảnh của bạn vào thư mục `data/images/`
2. Chạy script phát hiện:
```bash
python -m src.detect_obj.main
```

### Điều chỉnh tham số và thử nghiệm
1. Chỉnh sửa các tham số trong `config/detection_params.py`
2. Chạy script tự động thử nghiệm:
```bash
python -m src.detect_obj.scripts.auto_tune
```

Script sẽ:
- Phát hiện và đếm người trong ảnh với các cấu hình khác nhau
- Lưu ảnh kết quả vào thư mục `outputs/`
- Lưu kết quả chi tiết vào `outputs/detection_results.json`

## Cấu trúc dự án

```
src/detect_obj/
├── config/
│   ├── detection_params.py    # Cấu hình tham số
│   └── __init__.py
├── data/
│   ├── raw/                  # Dữ liệu thô
│   └── processed/            # Dữ liệu đã xử lý
├── models/                   # Model weights
│   └── yolov8n.pt
├── outputs/                  # Kết quả đầu ra
├── tests/                    # Unit tests
│   ├── test_person_detector.py
│   └── __init__.py
├── utils/                    # Tiện ích
│   └── __init__.py
├── scripts/                  # Scripts thực thi
│   ├── auto_tune.py
│   └── __init__.py
├── docs/                     # Tài liệu
│   └── README.md
├── main.py                   # Script chính
├── person_detector.py        # Lớp detector
├── requirements.txt          # Dependencies
└── __init__.py
```

## Tham số có thể điều chỉnh

Trong `config/detection_params.py`:
- `model_name`: Tên model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `conf_threshold`: Ngưỡng tin cậy (0.0 - 1.0)
- `iou_threshold`: Ngưỡng IoU cho NMS (0.0 - 1.0)
- `max_det`: Số lượng đối tượng tối đa có thể phát hiện

## Testing

Chạy tests bằng pytest:
```bash
python -m pytest src/detect_obj/tests/
```
