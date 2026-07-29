# Sidewalk Encroachment Detection

## Chạy bằng Docker

### Yêu cầu
- Docker
- Docker Compose

### Build và chạy
```bash
docker compose up --build
```

Sau đó mở trình duyệt tại:
```text
http://localhost:8501
```

### Dừng container
```bash
docker compose down
```

### Lưu ý
- Ứng dụng dùng Streamlit và tải các model YOLO từ thư mục models/.
- Nếu muốn bind mount dữ liệu đầu vào/đầu ra, có thể chỉnh sửa file docker-compose.yml.
