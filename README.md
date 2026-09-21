# IntentGuard

IntentGuard là prototype phân loại tin nhắn hỗ trợ ngân hàng tiếng Việt theo 77 intent và phát hiện câu hỏi ngoài phạm vi (UNKNOWN). Hệ thống trả về intent chính, confidence đã hiệu chỉnh, ba nhãn có khả năng cao nhất và ba câu gần nhất trong dữ liệu train.

Ví dụ:

~~~text
Input:  Tôi bị tính phí hai lần cho cùng một giao dịch.
Intent: transaction_charged_twice
~~~

## Chức năng

- Pipeline làm sạch và chia GreenNode/banking77-vn.
- Baseline TF-IDF word/character n-gram + Logistic Regression.
- Mô hình DL dùng sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 + classifier.
- Temperature scaling để hiệu chỉnh confidence.
- Threshold được chọn trên split in-domain riêng để trả về UNKNOWN.
- Đánh giá OOD trên MASSIVE tiếng Việt và 100 FAQ ngân hàng tự thu thập.
- Error analysis cho nhầm intent, false rejection và false acceptance.
- Demo Streamlit hỗ trợ baseline và DL khi có đủ artifact.

## Dữ liệu

### Banking77-VN

Nguồn in-domain là GreenNode/banking77-vn. Train gốc được khử trùng lặp và chia theo seed 42:

| Split | Số câu | Vai trò |
|---|---:|---|
| Train | 6.459 | Huấn luyện model và retrieval |
| Model selection | 923 | Chọn tham số/checkpoint |
| Calibration | 923 | Fit temperature |
| Threshold | 923 | Chọn ngưỡng UNKNOWN |
| Test gốc | 2.378 | Đánh giá cuối |

Test gốc không tham gia chọn model, calibration hoặc threshold.

### Dữ liệu OOD

Dự án dùng hai nguồn OOD:

1. **MASSIVE vi-VN**: các scenario alarm, weather, music, calendar và IoT.
2. **100 FAQ ngân hàng tự thu thập**: nằm tại [data/raw/vietnam_banks_faq_100.csv](data/raw/vietnam_banks_faq_100.csv), gồm bốn topic LOAN, SAVING, INTEREST_RATE, PROMOTION, mỗi topic 25 câu và có URL nguồn.

Pipeline FAQ:

- Kiểm tra đúng 100 câu, URL HTTPS, topic, độ dài và trùng với Banking77-VN.
- Giữ text_original, URL và topic để truy vết.
- Chuẩn hóa tên ngân hàng thành “ngân hàng” ở input mô hình.
- Thu gọn hai câu nhiều vế về một nhu cầu có sẵn trong câu gốc.
- Chia 50 validation / 50 test theo URL, giữ cùng URL và câu gần trùng trong một split.
- Không dùng FAQ để huấn luyện, chọn checkpoint hoặc chọn threshold.

Bản rà nhãn nằm tại [reports/faq_ood_label_audit.md](reports/faq_ood_label_audit.md). URL đã được người dùng xác nhận; nhãn OOD đã được Codex AI rà với 77 intent và chưa phải bộ nhãn vàng do hai người gán nhãn độc lập.

## Kết quả

| Backend | Macro-F1 Banking77 test | Top-3 | MASSIVE test UNKNOWN recall | FAQ test UNKNOWN recall |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | **0,8939** | **0,9697** | **0,8999** | **0,7400** |
| Multilingual MiniLM + classifier | 0,8832 | 0,9617 | 0,7180 | 0,6400 |

Baseline là backend mặc định vì có kết quả tổng thể tốt hơn. DL đa ngôn ngữ cải thiện đáng kể so với encoder tiếng Anh ban đầu và đạt macro-F1 0,8960 trên model-selection.

Báo cáo:

- [So sánh model và OOD](reports/comparison.md)
- [Phân tích lỗi](reports/error_analysis.md)
- [Chẩn đoán chi tiết 100 FAQ](reports/faq_ood_diagnostic.md)
- [Audit nhãn FAQ](reports/faq_ood_label_audit.md)

## Cài đặt

Yêu cầu Python 3.11–3.13 và [uv](https://docs.astral.sh/uv/).

~~~powershell
uv sync
~~~

Trên Windows AMD64, pyproject.toml ưu tiên wheel PyTorch CUDA 12.8. Có thể điều chỉnh nguồn PyTorch nếu môi trường không phù hợp.

## Chạy pipeline

### 1. Chuẩn bị Banking77-VN và MASSIVE

~~~powershell
uv run python scripts/prepare_data.py
~~~

### 2. Kiểm tra và chia 100 FAQ

~~~powershell
uv run python scripts/prepare_real_faq_ood.py
~~~

### 3. Huấn luyện baseline

~~~powershell
uv run python scripts/train_baseline.py
~~~

### 4. Huấn luyện DL hai giai đoạn

~~~powershell
uv run python scripts/train_deep.py --artifact-dir data/experiments/deep_stage1
uv run python scripts/train_deep.py --resume-from data/experiments/deep_stage1 --artifact-dir data/artifacts --learning-rate 0.00001
~~~

Cả hai giai đoạn chỉ chọn checkpoint bằng model-selection. Test Banking77-VN, MASSIVE và FAQ chỉ được đọc sau khi chốt model.

### 5. Calibration và đánh giá

~~~powershell
uv run python scripts/calibrate_and_evaluate.py
uv run python scripts/evaluate_faq_ood.py
~~~

Temperature được fit trên calibration split. Threshold là ngưỡng confidence cao nhất vẫn giữ tỷ lệ từ chối nhầm in-domain không quá 5% trên threshold split.

## Demo Streamlit

Chạy local:

~~~powershell
uv run streamlit run app/streamlit_app.py
~~~

Giao diện cho phép:

- Chọn baseline hoặc DL nếu artifact tương ứng tồn tại.
- Nhập tin nhắn hoặc dùng câu thử nhanh.
- Xem intent/UNKNOWN, confidence và threshold.
- Xem top-3 intent và ba câu tương tự từ train.

## Deploy Streamlit Community Cloud

Repository chứa artifact baseline cần cho demo cloud. Artifact DL khoảng 520 MB không được commit; giao diện tự ẩn lựa chọn DL khi không có artifact.

1. Mở [Streamlit Community Cloud](https://share.streamlit.io/).
2. Chọn repository wimaniac/IntentGuard, branch main.
3. Đặt **Main file path** là app/streamlit_app.py.
4. Trong **Advanced settings**, chọn Python 3.12.
5. Không cần khai báo secrets.
6. Bấm **Deploy**.

app/requirements.txt nằm cạnh entry point và chỉ chứa dependency cần cho baseline, giúp thời gian build và mức dùng RAM nhỏ hơn môi trường huấn luyện đầy đủ.

## Kiểm thử

~~~powershell
uv run python -m pytest
uv run ruff check src scripts app tests
~~~

Các test kiểm tra split không rò rỉ, calibration, threshold, retrieval, FAQ có URL, audit nhãn và UI Streamlit headless.

## Cấu trúc repository

~~~text
app/                 Demo Streamlit và dependency cloud
configs/             Cấu hình dữ liệu, model và calibration
data/raw/            100 FAQ nguồn được đưa vào Git
data/artifacts/      Artifact baseline phục vụ demo cloud
scripts/             Entry point chuẩn bị dữ liệu, train và đánh giá
src/intentguard/     Logic dữ liệu, model, calibration, metrics và inference
tests/               Kiểm thử tự động
reports/             Báo cáo kết quả cuối
~~~

## Giới hạn

- FAQ website có văn phong sạch hơn tin nhắn hỗ trợ thực tế.
- Bộ FAQ test có 50 câu nên khoảng tin cậy của recall còn rộng.
- Nhãn FAQ chưa được hai người độc lập xác minh.
- Câu lãi suất có thể bị nhận nhầm thành exchange_rate với confidence cao; threshold đơn lẻ chưa giải quyết hết semantic OOD gần miền.
- GreenNode/banking77-vn được công bố với giấy phép CC BY-NC-SA 4.0. Prototype này chỉ dùng cho mục đích nghiên cứu phi thương mại.
