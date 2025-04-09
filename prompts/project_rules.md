# Quy tắc phát triển dự án

## 1. Cấu trúc thư mục chuẩn
Mỗi dự án phải tuân thủ cấu trúc thư mục sau:
```
src/
└── project_name/              # Thư mục gốc của dự án
    ├── config/               # Cấu hình và tham số
    │   └── __init__.py
    ├── data/                # Dữ liệu
    │   ├── raw/            # Dữ liệu thô
    │   └── processed/      # Dữ liệu đã xử lý
    ├── models/             # Model files (weights, checkpoints)
    ├── outputs/            # Kết quả đầu ra
    ├── tests/              # Unit tests và integration tests
    │   └── __init__.py
    ├── utils/              # Tiện ích và helper functions
    │   └── __init__.py
    ├── scripts/            # Scripts thực thi
    │   └── __init__.py
    ├── docs/               # Tài liệu
    │   ├── README.md      # Tài liệu chính
    │   └── api/           # Tài liệu API
    ├── requirements.txt    # Dependencies
    └── __init__.py        # Package initialization
```

## 2. Quy tắc phát triển
1. Sử dụng nguyên tắc TDD (Test-Driven Development)
2. Tài liệu phải được viết ở định dạng Markdown (.md)
3. Code phải được tối ưu, không có code thừa
4. Tuân thủ PEP 8 cho Python code
5. Mỗi module phải có docstring và type hints
6. Sử dụng relative imports trong package

## 3. Quy trình làm việc
1. Lập kế hoạch chi tiết trước khi bắt đầu
2. Tuân thủ quy trình Scrum
3. Test phải pass trước khi chuyển sang task mới
4. Code review bắt buộc trước khi merge

## 4. Quản lý dependencies
1. Sử dụng virtual environment
2. Liệt kê đầy đủ dependencies trong requirements.txt
3. Chỉ định phiên bản cụ thể cho mỗi package
4. Tách biệt dev dependencies và production dependencies

## 5. Testing
1. Unit tests cho mọi chức năng
2. Integration tests cho workflows chính
3. Test coverage tối thiểu 80%
4. Sử dụng pytest làm testing framework

## 6. Documentation
1. README.md trong thư mục docs phải bao gồm:
   - Mô tả dự án
   - Hướng dẫn cài đặt
   - Hướng dẫn sử dụng
   - Cấu trúc dự án
   - API documentation (nếu có)
2. Mỗi module phải có docstring đầy đủ
3. Complex workflows phải có flow diagram

## 7. Version Control
1. Sử dụng semantic versioning
2. Mỗi commit phải có message rõ ràng
3. Branch cho mỗi feature/bugfix
4. Squash commits trước khi merge

## 8. Continuous Integration
1. Automated testing cho mọi pull request
2. Code quality checks (linting, type checking)
3. Security vulnerability scanning
4. Build và test tự động

## 9. Monitoring và Logging
1. Structured logging
2. Error tracking
3. Performance monitoring
4. Usage analytics

## 10. Security
1. Không commit credentials
2. Sử dụng environment variables cho sensitive data
3. Regular security updates
4. Input validation và sanitization