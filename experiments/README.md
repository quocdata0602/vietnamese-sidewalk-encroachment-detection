# Thư mục experiments

Thư mục này dùng để chạy các thí nghiệm so sánh và kiểm tra hiệu quả của các quy tắc trong pipeline phát hiện lấn chiếm vỉa hè.

## Mục đích

- Kiểm tra các trường hợp cụ thể về:
  - overlap với sidewalk
  - contact ở vùng đáy bbox
  - duplicate detection
  - mức độ đánh giá lấn chiếm theo score
- So sánh kết quả trước/sau khi thay đổi logic hoặc tham số.
- Tạo các ảnh và bảng kết quả để dễ đối chiếu.

## Cấu trúc thư mục

- `input/`
  - `experiment_cases.csv`: danh sách các case thí nghiệm và kỳ vọng kết quả.
  - `experiment_images/`: ảnh dùng cho các thí nghiệm.
  - `threshold_images/`: ảnh dùng để thử các ngưỡng khác nhau.

- `output/`
  - Chứa kết quả chạy thí nghiệm như ảnh so sánh, bảng CSV và thư mục kết quả theo từng nhóm thí nghiệm.

- `experiments.ipynb`
  - Notebook thực hiện các thí nghiệm và tạo output.

## Các nhóm thí nghiệm

- `C`: kiểm tra ảnh hưởng của threshold overlap ở vùng tiếp xúc.
- `D`: kiểm tra xe trên road nhưng gần sidewalk, dùng bottom-contact và road-check.
- `E`: kiểm tra duplicate detection suppression.
- `F`: kiểm tra phân loại mức độ lấn chiếm theo score.

## Các file đầu vào quan trọng

- `input/experiment_cases.csv`: định nghĩa các case, kỳ vọng và mục tiêu quan sát.
- `experiments.ipynb`: notebook chính để chạy toàn bộ thí nghiệm.

## Kết quả đầu ra

Sau khi chạy notebook, thư mục `output/` sẽ chứa:
- ảnh so sánh trước/sau
- file CSV tóm tắt
- thư mục kết quả theo từng nhóm thí nghiệm

## Ghi chú

- Notebook này được thiết kế để đồng bộ logic với app chính.
- Nếu chỉnh sửa logic trong app hoặc config, nên chạy lại notebook để kiểm tra lại các case thí nghiệm.
