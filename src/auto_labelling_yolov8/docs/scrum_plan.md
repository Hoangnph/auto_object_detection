# Kế hoạch Phát triển Dự án Auto-Labelling YOLOv8

## Sprint 1: Thiết lập Cơ sở và Phát hiện Đối tượng (2 tuần)

### User Story 1: Thiết lập Môi trường Phát triển
- **Task 1.1**: Cấu hình môi trường phát triển
  - [ ] Tạo/ hoặc kiểm tra virtual environment đã có từ thư mục root chưa
  - [ ] Cài đặt dependencies từ requirements.txt
  - [ ] Cấu hình logging system
  - [ ] Tạo cấu trúc thư mục tests/
  - [ ] Viết test cases cho các utility functions

### User Story 2: Phát hiện Đối tượng với OpenCV và YOLOv8
- **Task 2.1**: Tích hợp OpenCV
  - [ ] Viết unit tests cho image loading và preprocessing
  - [ ] Implement image loading utilities
  - [ ] Implement image preprocessing functions
  - [ ] Viết documentation cho các functions

- **Task 2.2**: Tích hợp YOLOv8
  - [ ] Viết unit tests cho YOLO model initialization
  - [ ] Implement YOLO model loading
  - [ ] Implement object detection pipeline
  - [ ] Viết documentation cho detection pipeline

## Sprint 2: Quản lý Ảnh và Cắt Ảnh (2 tuần)

### User Story 3: Quản lý Thư mục Ảnh
- **Task 3.1**: Tạo cấu trúc thư mục
  - [ ] Viết unit tests cho directory management
  - [ ] Implement raw/ directory creation
  - [ ] Implement directory management utilities
  - [ ] Viết documentation cho directory structure

### User Story 4: Hệ thống Cắt Ảnh
- **Task 4.1**: Phát triển logic cắt ảnh
  - [ ] Viết unit tests cho cropping functions
  - [ ] Implement cropping logic
  - [ ] Implement image saving with metadata
  - [ ] Viết documentation cho cropping system

## Sprint 3: Hệ thống Gán nhãn thông minh (3 tuần)

### User Story 5: Tích hợp LLM
- **Task 5.1**: Cấu hình API
  - [ ] Viết unit tests cho API configuration
  - [ ] Implement API key management
  - [ ] Implement error handling
  - [ ] Viết documentation cho API integration

### User Story 6: Hệ thống Gán nhãn dựa trên mẫu
- **Task 6.1**: Phát triển hệ thống gán nhãn
  - [ ] Viết unit tests cho labeling system
  - [ ] Implement sample image processing
  - [ ] Implement label matching algorithm
  - [ ] Viết documentation cho labeling system

## Sprint 4: Tạo Dataset YOLOv8 (2 tuần)

### User Story 7: Cấu trúc Dataset
- **Task 7.1**: Tạo cấu trúc dataset
  - [ ] Viết unit tests cho dataset structure
  - [ ] Implement YOLO format conversion
  - [ ] Implement directory organization
  - [ ] Viết documentation cho dataset structure

### User Story 8: Pipeline tạo dataset
- **Task 8.1**: Phát triển pipeline
  - [ ] Viết unit tests cho dataset pipeline
  - [ ] Implement image copying system
  - [ ] Implement label file generation
  - [ ] Implement dataset splitting
  - [ ] Viết documentation cho dataset pipeline

## Quy trình Kiểm thử (TDD)

### Quy tắc Kiểm thử
1. Viết test case trước khi implement code
2. Mỗi function phải có ít nhất:
   - Test case cho happy path
   - Test case cho error cases
   - Test case cho edge cases

### Cấu trúc Test
```python
def test_function_name():
    # Arrange
    # Act
    # Assert
```

### Công cụ Kiểm thử
- pytest cho unit tests
- pytest-cov cho coverage reports
- pytest-mock cho mocking

## Quy trình Scrum

### Sprint Planning
- Mỗi sprint kéo dài 2-3 tuần
- Planning meeting vào đầu sprint
- Review meeting vào cuối sprint
- Daily standup meetings

### Definition of Done
Mỗi task được coi là hoàn thành khi:
1. Code đã được viết và review
2. Unit tests đã được viết và pass
3. Documentation đã được cập nhật
4. Code coverage đạt ít nhất 80%
5. Không còn lỗi linting

### Cập nhật Documentation
- Cập nhật task_list.md sau mỗi task
- Cập nhật API documentation
- Cập nhật README.md khi cần
- Ghi lại các quyết định kỹ thuật quan trọng

## Theo dõi Tiến độ
- Sử dụng task_list.md để theo dõi tiến độ
- Đánh dấu [x] cho các task đã hoàn thành
- Cập nhật documentation sau mỗi task
- Review code trước khi merge 