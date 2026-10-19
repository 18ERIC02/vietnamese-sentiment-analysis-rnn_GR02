# Demo Streamlit — phân tích cảm xúc tiếng Việt bằng SimpleRNN

Đây là giao diện suy luận (inference) cho mô hình đã huấn luyện trên Google Colab. Không train lại.

## 1. Checkpoint

Checkpoint đã huấn luyện phải nằm tại `models/best_model.keras`, cạnh `app.py` theo cấu trúc:

```text
rnn_demo/
  app.py
  requirements.txt
  models/
    best_model.keras
```

Tên file phải là `best_model.keras`, không có dấu cách trước `.keras`.

## 2. Chạy trên Windows (PowerShell)

Sau khi clone repository, tạo môi trường Python 3.11 64-bit trong `rnn_demo` và cài dependencies:

```powershell
Set-Location .\rnn_demo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_app.ps1
```

Launcher luôn dùng `.venv\Scripts\python.exe`, không phụ thuộc `python` trên PATH.
Nếu đã có `.venv` phù hợp thì bỏ qua bước tạo môi trường và cài lại thư viện.
Nếu PowerShell chặn file `.ps1`, có thể chạy trực tiếp bằng lệnh:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Các phiên bản TensorFlow và Keras trong `requirements.txt` khớp với checkpoint.

Mở liên kết localhost Streamlit hiện trên màn hình, thường là http://localhost:8501.

## 3. Lưu ý về phiên bản TensorFlow

Demo dùng TensorFlow 2.21.0 và Keras 3.13.2 để khớp checkpoint. Trên Windows,
gói TensorFlow trong cấu hình này chạy inference bằng CPU; không cần CUDA để demo.
Ứng dụng cũng nạp vocabulary tiếng Việt của `TextVectorization` bằng UTF-8 để
không phụ thuộc encoding mặc định của Windows.

## 4. Lưu ý khi thuyết trình

Ứng dụng dự đoán 3 nhãn 0=Tiêu cực, 1=Trung lập, 2=Tích cực. Dữ liệu UIT-VSFC là phản hồi học tập, nên tránh khẳng định mô hình hoạt động tốt ở mọi lĩnh vực. Các xác suất là output Softmax, không nhất thiết là confidence đã hiệu chuẩn.

## 5. Giao diện demo

Trang mở sẵn câu ví dụ đầu tiên và dự đoán bằng checkpoint thật. Có thể nhập câu mới
rồi bấm **Phân tích cảm xúc**, hoặc bấm một trong ba câu ví dụ để cập nhật ô nhập
và kết quả. Câu rỗng sẽ hiện cảnh báo.

Dashboard có ba card thông tin, hai card kết quả, bốn thống kê và timeline tám
bước. Kích thước từ vựng và độ dài chuỗi được đọc từ TextVectorization trong
checkpoint; thời gian inference được đo khi chạy mô hình. Token và ID trong
timeline cũng được lấy từ lớp xử lý đã lưu. Nếu một phiên bản Keras không cho
truy xuất phần chi tiết này, giao diện hiển thị "Không khả dụng" thay vì số giả.

Logo HCMUTE tại assets/hcmute_logo.png được trích từ báo cáo Word của nhóm trong
project. Giao diện được thiết kế theo ảnh tham chiếu ở giai đoạn phát triển;
ứng dụng không cần ảnh đó để chạy.
