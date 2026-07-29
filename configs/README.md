# Cấu hình cho pipeline phát hiện lấn chiếm vỉa hè

Thư mục này chứa các file YAML dùng để quản lý tham số cho ứng dụng.

## Các file cấu hình

- `contact_thresholds.yaml`
  - Chứa các ngưỡng về mức độ vùng đáy bbox chạm vào sidewalk.
  - Dùng để quyết định một vật thể có thật sự nằm trên vỉa hè hay không.

- `duplicate_detection.yaml`
  - Chứa các tham số dùng để loại bỏ các bbox bị phát hiện trùng lặp cho cùng một vật thể.
  - Giúp giảm nhiễu khi model phát hiện một đối tượng nhiều lần.

- `scoring.yaml`
  - Chứa trọng số và ngưỡng confidence cho từng loại đối tượng.
  - Ảnh hưởng trực tiếp đến điểm số lấn chiếm cuối cùng.

- `sidewalk_postprocess.yaml`
  - Chứa các tham số hậu xử lý mask vỉa hè.
  - Giúp làm sạch mask và khôi phục vùng sidewalk bị che bởi vật cản.

## Cách chỉnh sửa

- Mở file YAML tương ứng.
- Chỉnh giá trị theo ý muốn.
- Khởi động lại ứng dụng để áp dụng thay đổi.

## Ghi chú

- Các file này được app.py đọc để lấy cấu hình.
- Nếu thêm một tham số mới, nên cập nhật cả file YAML và phần đọc cấu hình trong app.py nếu cần.
