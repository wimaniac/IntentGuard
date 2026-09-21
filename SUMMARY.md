# Tóm tắt trạng thái IntentGuard

## 1. Mục tiêu hoặc giai đoạn hiện tại

IntentGuard là prototype phi thương mại phân loại tin nhắn hỗ trợ ngân hàng tiếng Việt theo 77 intent, trả top-3, confidence, câu tương tự và `UNKNOWN` cho câu ngoài phạm vi.

Dự án đã hoàn thành pipeline dữ liệu, baseline ML, một mô hình DL, OOD detection, báo cáo lỗi và demo Streamlit. Repository đang ở giai đoạn hoàn thiện tài liệu và phát hành bản demo baseline trên Streamlit Community Cloud.

## 2. Công việc đã hoàn thành

- Chuẩn bị `GreenNode/banking77-vn` với seed 42: 6.459 train, 923 model-selection, 923 calibration, 923 threshold và 2.378 test.
- Huấn luyện baseline TF-IDF word/character n-gram + Logistic Regression.
- Fine-tune `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + classifier theo hai giai đoạn, chỉ chọn checkpoint bằng model-selection.
- Fit temperature trên calibration split và chọn threshold trên threshold split với tỷ lệ từ chối nhầm in-domain tối đa 5%.
- Tích hợp retrieval ba câu tương tự từ train.
- Đánh giá OOD bằng MASSIVE `vi-VN` và 100 FAQ ngân hàng tự thu thập.
- Kiểm tra 100 FAQ: bốn topic, mỗi topic 25 câu, 45 URL, không trùng chính xác Banking77-VN.
- Chuẩn hóa tên ngân hàng ở 67 câu; giữ `text_original`, URL và topic để truy vết.
- Chia FAQ thành 50 validation và 50 test theo URL; giữ các câu gần trùng trong cùng split.
- Rà nhãn FAQ với 77 intent và lập bảng quyết định cho từng câu.
- Hoàn thiện báo cáo so sánh, chẩn đoán FAQ, audit nhãn và phân tích lỗi.
- Hoàn thiện Streamlit: baseline/DL khi có artifact, top-3, confidence, threshold và ba câu tương tự.
- Đưa artifact baseline vào Git để Streamlit Community Cloud có thể chạy trực tiếp.
- Rút gọn repository còn pipeline và báo cáo cuối; xóa ứng dụng, script, test, template, dữ liệu trung gian và báo cáo thử nghiệm không còn dùng.
- Viết lại README theo trạng thái hiện hành của dự án.

## 3. Công việc đang chờ

- Nếu cần benchmark có độ tin cậy công bố, tổ chức hai người gán nhãn độc lập cho 100 FAQ và xử lý bất đồng.
- Nếu cần ước lượng hiệu năng vận hành, bổ sung tập test từ tin nhắn hỗ trợ thực tế.
- Nếu muốn bật DL trên Streamlit Cloud, lưu model trong model registry hoặc Git LFS rồi kiểm tra RAM và thời gian khởi động.

## 4. File và module chính

- `README.md`: mục tiêu, dữ liệu, kết quả, cách chạy và hướng dẫn deploy.
- `configs/default.yaml`: cấu hình Banking77-VN, MASSIVE, model và calibration.
- `data/raw/vietnam_banks_faq_100.csv`: 100 FAQ nguồn có URL.
- `data/faq_ood_quality_status.json`: trạng thái và provenance bộ FAQ hiện hành.
- `data/artifacts/`: artifact baseline phục vụ demo cloud.
- `src/intentguard/data.py`: tải, làm sạch và chia Banking77-VN/MASSIVE.
- `src/intentguard/models/`: baseline và DL.
- `src/intentguard/calibration.py`: temperature scaling và threshold.
- `src/intentguard/inference.py`: API suy luận dùng chung.
- `scripts/prepare_real_faq_ood.py`: kiểm tra, chuẩn hóa và chia 100 FAQ.
- `scripts/calibrate_and_evaluate.py`: calibration và báo cáo tổng hợp.
- `scripts/evaluate_faq_ood.py`: chẩn đoán chi tiết FAQ.
- `app/streamlit_app.py`: giao diện kiểm thử.
- `reports/comparison.md`: kết quả chính.
- `reports/error_analysis.md`: phân tích lỗi.
- `reports/faq_ood_diagnostic.md`: metric FAQ theo split và topic.
- `reports/faq_ood_label_audit.md`: audit nhãn FAQ.

## 5. Quyết định kỹ thuật và lý do

- Baseline là backend mặc định vì đạt kết quả tổng thể tốt hơn DL trên test và cả hai nguồn OOD.
- Encoder DL đa ngôn ngữ phù hợp dữ liệu tiếng Việt và tốt hơn encoder tiếng Anh được thử ban đầu.
- Test Banking77-VN, MASSIVE và FAQ không tham gia chọn model, temperature hoặc threshold.
- FAQ không dùng để huấn luyện; đây là tập OOD gần miền nhằm đo khả năng trả `UNKNOWN`.
- Tên ngân hàng được chuẩn hóa để giảm shortcut theo thương hiệu, còn câu gốc và URL vẫn được giữ để audit.
- Repository chỉ giữ báo cáo cuối và artifact baseline cần cho demo cloud; artifact DL khoảng 520 MB không đưa vào Git.

## 6. Lệnh và kết quả gần nhất

- `uv lock --offline`: PASS; loại dependency không còn dùng và cập nhật `uv.lock`.
- `uv run --no-sync ruff check src scripts app tests`: PASS.
- `uv run --no-sync python -m pytest`: PASS, 16 test.
- `uv run --no-sync python scripts/prepare_real_faq_ood.py`: PASS; 100 câu, 45 URL, validation/test 50/50.
- `uv run --no-sync python scripts/calibrate_and_evaluate.py`: PASS; tạo lại `comparison.md/json`.
- `uv run --no-sync python scripts/evaluate_faq_ood.py`: PASS; tạo lại chẩn đoán FAQ.
- Banking77-VN test: baseline macro-F1 0,8939, top-3 0,9697; DL macro-F1 0,8832, top-3 0,9617.
- MASSIVE test recall `UNKNOWN`: baseline 0,8999; DL 0,7180.
- FAQ test recall `UNKNOWN`: baseline 0,7400; DL 0,6400.

## 7. Lỗi, rủi ro, giả định và câu hỏi mở

- 100 FAQ có URL do người dùng xác nhận; Codex AI rà nhãn nhưng chưa có hai người gán nhãn độc lập.
- FAQ website có văn phong sạch hơn tin nhắn khách hàng thực tế.
- FAQ test chỉ có 50 câu và nhiều câu cùng URL, nên khoảng tin cậy của recall còn rộng.
- Câu về lãi suất có thể bị nhận thành `exchange_rate` với confidence cao; threshold confidence đơn lẻ chưa xử lý hết semantic OOD gần miền.
- DL hiện thấp hơn baseline: chênh macro-F1 test 0,0107 và FAQ recall 0,10.
- `GreenNode/banking77-vn` công bố giấy phép CC BY-NC-SA 4.0; dự án hiện được mô tả là prototype nghiên cứu phi thương mại.

## 8. Bước tiếp theo cụ thể

1. Kiểm tra lần deploy mới nhất trên Streamlit Community Cloud sau khi nhánh `main` được cập nhật.
2. Dùng các câu known/UNKNOWN mẫu để kiểm tra UI và log lỗi nếu có.
3. Chỉ mở vòng cải thiện model mới khi có tập model-selection bổ sung; giữ các test cuối tách khỏi vòng tối ưu.
