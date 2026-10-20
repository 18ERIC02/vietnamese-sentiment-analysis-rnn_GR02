# HCMUTE Vietnamese RNN Sentiment Analysis - Desktop Application

Ứng dụng Desktop phân tích cảm xúc bình luận tiếng Việt sử dụng mạng nơ-ron hồi quy **RNN** (Recurrent Neural Network), được xây dựng bằng **Python 3.11** và **PySide6 (Qt for Python)**.

Giao diện được thiết kế hiện đại theo chuẩn Dashboard học thuật của Trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh (HCMUTE).

---

## 1. Thông tin đồ án

- **Trường**: Trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh (HCMUTE)
- **Học phần**: Deep Learning
- **Giảng viên hướng dẫn**: PGS. TS. Hoàng Văn Dũng
- **Nhóm thực hiện (Nhóm 02)**:
  1. **Trịnh Nguyễn Anh Hào** - MSSV: `2611307`
  2. **Lê Huy Huân** - MSSV: `2611308`
  3. **Nguyễn Nam Triều Tiên** - MSSV: `2611323`
- **Bộ dữ liệu thực nghiệm**: UIT-VSFC (Vietnamese Students' Feedback Corpus)

---

## 2. Đặc điểm kỹ thuật

- **Mô hình**: Mạng nơ-ron hồi quy RNN (triển khai kỹ thuật qua `tf.keras.layers.SimpleRNN`, hiển thị trên giao diện theo chuẩn ngắn gọn **RNN**).
- **Trọng số & Checkpoint**: Nạp trực tiếp từ `models/best_model.keras`.
- **Text Vectorization**: Tích hợp sẵn trong checkpoint (chuỗi token hóa độ dài 60, từ vựng 2.487 token, embedding dimension 64, RNN units 64).
- **Hỗ trợ tiếng Việt Windows**: Tự động vá bộ nạp `vocabulary.txt` dưới dạng `utf-8` tránh lỗi bảng mã cp1252 của Keras trên hệ điều hành Windows.
- **Xử lý đa luồng (Multi-threading)**: Nạp mô hình ngầm qua `ModelLoadWorker` và inference bất đồng bộ qua `InferenceWorker` (dùng `QThread`), giúp giao diện luôn mượt mà và không bao giờ bị đơ (Not Responding).
- **Bố cục giao diện**:
  - **Header**: Logo HCMUTE, tên trường chính thức bằng tiếng Việt & tiếng Anh, kèm ảnh toàn cảnh khuôn viên tòa nhà trung tâm HCMUTE.
  - **Sidebar**: Khối thông tin môn học, giảng viên, sinh viên thực hiện, giới thiệu tóm tắt và các huy hiệu công nghệ (Python, TensorFlow, RNN, PySide6, Vietnamese NLP).
  - **Khu vực nhập liệu**: Ô `QTextEdit` bo góc, bộ đếm 500 ký tự thời gian thực, 3 nút ví dụ thực tế về phản hồi sinh viên ("Thầy giảng rất dễ hiểu.", "Bài giảng quá khó hiểu.", "Buổi học bắt đầu lúc 7 giờ.").
  - **Nút phân tích**: Nút bấm màu xanh dương nổi bật với trạng thái hiệu ứng hover/pressed/disabled.
  - **Khối kết quả**:
    - *Bên trái*: Thẻ Hero hiển thị biểu tượng cảm xúc lớn, tên nhãn dự đoán (Tích cực / Trung tính / Tiêu cực) và xác suất dự đoán phần trăm cao nhất. Nền tự động đổi màu tương ứng (Xanh lá / Xanh dương / Đỏ).
    - *Bên phải*: Biểu đồ 3 thanh xác suất Softmax độc lập cho từng lớp cảm xúc, căn chỉnh hoàn hảo theo trục dọc.
    - *Bên dưới*: Khối nhận xét tự động theo ngữ cảnh dự đoán của mô hình.

---

## 3. Cấu trúc thư mục

```text
desktop_app/
├── assets/
│   ├── banner.png           # Banner tòa nhà trung tâm HCMUTE
│   ├── banner_feathered.png # Banner hiệu ứng làm mờ alpha biên trái liền mạch
│   ├── chat_decor.png       # Hình minh họa tin nhắn trang trí trong suốt RGBA
│   └── hcmute_logo.png      # Logo chính thức HCMUTE
├── inference/
│   ├── __init__.py
│   └── predictor.py         # Module nạp checkpoint và suy luận cảm xúc
├── models/
│   └── best_model.keras     # Checkpoint Keras nguyên vẹn đã huấn luyện
├── ui/
│   ├── __init__.py
│   ├── main_window.py       # Cửa sổ chính của ứng dụng
│   ├── styles.py            # Bảng màu và Qt Style Sheet (QSS)
│   └── widgets.py           # Custom Widgets (thanh xác suất, hero card, workers)
├── build.ps1                # Script PowerShell tự động đóng gói PyInstaller
├── main.py                  # Điểm khởi chạy ứng dụng
├── requirements.txt         # Danh sách thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn
```

---

## 4. Hướng dẫn cài đặt & Khởi chạy

### Cách 1: Chạy trực tiếp từ mã nguồn Python

1. Mở PowerShell hoặc Command Prompt tại thư mục `desktop_app`:
   ```powershell
   cd D:\Master_CS\Deep_Learning\rnn_project_Gr02\desktop_app
   ```

2. Kích hoạt môi trường ảo (hoặc sử dụng Python có cài PySide6, TensorFlow, Keras):
   ```powershell
   & "..\rnn_demo\.venv\Scripts\Activate.ps1"
   ```

3. Chạy ứng dụng:
   ```powershell
   python main.py
   ```

---

## 5. Hướng dẫn đóng gói thành file `.exe`

Ứng dụng có thể được đóng gói thành file thực thi chạy độc lập trên Windows mà không cần cài đặt Python:

1. Chạy script đóng gói:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\build.ps1
   ```

2. Nếu muốn giữ cửa sổ console để xem log debug:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\build.ps1 -Console
   ```

3. File thực thi sau khi hoàn tất sẽ nằm tại:
   ```text
   desktop_app\dist\RNN_Sentiment_HCMUTE\RNN_Sentiment_HCMUTE.exe
   ```

Chỉ cần double-click file `.exe` để mở ứng dụng Desktop.
