# Vietnamese Sentiment Analysis Using Recurrent Neural Networks

Đồ án học phần Deep Learning: phân loại cảm xúc bình luận ngắn tiếng Việt trong ngữ cảnh phản hồi sinh viên bằng mạng SimpleRNN. Ứng dụng Streamlit sử dụng checkpoint đã huấn luyện, không huấn luyện lại khi dự đoán.

## Dataset

Mô hình được huấn luyện trên [UIT-VSFC (Vietnamese Students' Feedback Corpus)](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback), gồm các tập train, validation và test với nhãn cảm xúc 0 = Negative, 1 = Neutral, 2 = Positive. Notebook tải dữ liệu trực tiếp từ nguồn này. Trang dữ liệu hiện ghi giấy phép là “unknown”, nên repository không phân phối lại dữ liệu gốc hoặc CSV chứa toàn bộ câu test. Vui lòng xem điều kiện sử dụng tại nguồn trước khi tải và sử dụng dữ liệu.

Nếu sử dụng UIT-VSFC, hãy trích dẫn công trình của Nguyen và cộng sự, *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis*, KSE 2018, DOI: [10.1109/KSE.2018.8573337](https://doi.org/10.1109/KSE.2018.8573337).

## Model Architecture

`TextVectorization → Embedding(64) → SimpleRNN(64) → Dropout(0.3) → Dense(3) + Softmax`

`TextVectorization` nằm trong checkpoint. Ứng dụng đưa vào tensor string có shape `(batch, 1)` và trả về ba xác suất Softmax.

## Training and Evaluation

[Notebook/RNN_Sentiment_GR02.ipynb](Notebook/RNN_Sentiment_GR02.ipynb) chứa quy trình tải dữ liệu, huấn luyện và đánh giá. Thư mục [Result/](Result/) lưu history, biểu đồ accuracy/loss, báo cáo validation và kết quả test. Các chỉ số test dưới đây được lấy từ [test_metrics.json](Result/test_evaluation/test_metrics.json), không tính lại trong ứng dụng:

| Chỉ số test | Giá trị |
| --- | ---: |
| Accuracy | 88.53% |
| Precision macro | 80.26% |
| Recall macro | 65.73% |
| F1 macro | 66.92% |

Checkpoint dùng cho demo là [rnn_demo/models/best_model.keras](rnn_demo/models/best_model.keras). Bản checkpoint giống hệt trong `Result/` được giữ trên máy nhóm nhưng không đưa lên Git để tránh lưu trùng.

## Project Structure

```text
Notebook/                         Notebook huấn luyện và đánh giá
Result/                           History, biểu đồ, validation và test metrics
desktop_app/                      Ứng dụng Desktop Native (PySide6)
  assets/                         Logo HCMUTE, banner trường và icon trang trí
  inference/                      Module nạp checkpoint và suy luận cảm xúc
  models/best_model.keras         Checkpoint phục vụ ứng dụng Desktop
  ui/                             Mã nguồn giao diện PySide6 (MainWindow, widgets, styles)
  build.ps1                       Script tự động đóng gói Windows EXE (PyInstaller)
  main.py                         Điểm khởi chạy ứng dụng Desktop
  requirements.txt                Danh sách thư viện phụ thuộc cho Desktop App
  RNN_Sentiment_HCMUTE.spec       File cấu hình đóng gói PyInstaller
  README.md                       Hướng dẫn chi tiết về Desktop App
rnn_demo/
  app.py                           Ứng dụng Streamlit
  assets/                          CSS và logo HCMUTE
  models/best_model.keras           Checkpoint dùng cho inference
  requirements.txt                 Dependencies cho ứng dụng
  run_app.ps1                      Launcher Windows
  README.md                        Hướng dẫn chi tiết cho demo
Bao_cao_RNN_tieng_Viet.docx         Báo cáo đồ án
Báo_cáo_Tìm_hiểu_về_Nơ-ron_Hồi_Quy_Deep_Learning_Group02.pptx
Báo_cáo_Tìm_hiểu_về_Nơ-ron_Hồi_Quy_Deep_Learning_Group02.pdf
```

File `Result/test_evaluation/test_predictions.csv` không có trong repository vì chứa các câu từ tập test. Các chỉ số tổng hợp và biểu đồ vẫn có trong `Result/`.

## Installation on Windows

Cài Python 3.11 64-bit. Trong PowerShell, tại thư mục repository đã clone:

```powershell
Set-Location .\rnn_demo
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`python --version` cần hiển thị 3.11.x. Trên máy đã có `.venv` phù hợp, bỏ qua bước tạo môi trường và cài đặt lại.

## Run Streamlit Application

Tại `rnn_demo`:

```powershell
.\run_app.ps1
```

Hoặc chạy trực tiếp bằng `.\.venv\Scripts\python.exe -m streamlit run app.py`. Mở URL localhost do Streamlit in ra. Xem thêm [hướng dẫn demo](rnn_demo/README.md).

## Desktop Application (PySide6)

Ứng dụng Desktop Native chạy trực tiếp trên Windows (sử dụng PySide6/Qt và TensorFlow/Keras) với giao diện Dashboard học thuật HCMUTE hiện đại, không cần mở trình duyệt web:

### 1. Khởi chạy từ mã nguồn Python
```powershell
Set-Location .\desktop_app
..\rnn_demo\.venv\Scripts\python.exe main.py
```
*(Nếu sử dụng môi trường Python riêng, cài đặt phụ thuộc qua `pip install -r requirements.txt` rồi chạy `python main.py`)*.

### 2. Đóng gói thành file thực thi `.exe` độc lập trên Windows
Ứng dụng có thể được đóng gói thành file thực thi chạy độc lập trên Windows mà không cần cài đặt Python:
```powershell
Set-Location .\desktop_app
powershell -ExecutionPolicy Bypass -File .\build.ps1
```
File thực thi sau khi hoàn tất sẽ nằm tại:
```text
desktop_app\dist\RNN_Sentiment_HCMUTE\RNN_Sentiment_HCMUTE.exe
```
Xem thêm tài liệu chi tiết tại [desktop_app/README.md](desktop_app/README.md).

## Usage

Nhập bình luận tiếng Việt và bấm **Phân tích cảm xúc**, hoặc chọn một trong ba câu ví dụ. Giao diện hiển thị nhãn dự đoán, xác suất của ba lớp và timeline xử lý văn bản. Xác suất Softmax không phải độ tin cậy đã hiệu chuẩn.

## Limitations

Mô hình học từ phản hồi sinh viên, nên có thể dự đoán kém hơn với văn bản ngoài lĩnh vực. Cần kiểm tra thêm nguy cơ overfitting khi áp dụng trên dữ liệu mới. Lớp Neutral có recall thấp trong báo cáo test, vì vậy không nên diễn giải một dự đoán đơn lẻ như kết luận chắc chắn.
