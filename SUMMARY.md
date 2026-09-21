# Tóm tắt trạng thái IntentGuard

## 1. Mục tiêu/giai đoạn hiện tại

Prototype phi thương mại phân loại 77 intent Banking77-VN, trả top-3, confidence, câu tương tự và `UNKNOWN`. Giai đoạn cải thiện DL và hoàn thiện giao diện Streamlit kiểm thử đã hoàn thành.

Bộ OOD ngân hàng hiện hành là 100 FAQ có URL do người dùng thu thập; người dùng xác nhận URL và Codex AI rà từng nhãn với 77 intent. Đây chưa phải bộ nhãn vàng do hai người gán nhãn độc lập.

## 2. Công việc đã hoàn thành

- Pipeline Banking77-VN từ `GreenNode/banking77-vn` chia theo seed 42: 6.459 train, 923 model-selection, 923 calibration, 923 threshold và 2.378 test.
- Baseline TF-IDF + Logistic Regression, DL + classifier, temperature scaling, threshold và retrieval câu tương tự đều có artifact hoạt động.
- DL mới dùng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, max length 128. Phân bố token train có trung vị 17, p95 45, p99 65; chỉ 1/6.459 câu dài hơn 128 token.
- Fine-tune DL theo hai giai đoạn: 6 epoch learning rate 2e-5, sau đó tiếp tục 6 epoch learning rate 1e-5. Checkpoint được chọn bằng model-selection macro-F1 0,8960; DL cũ đạt 0,7019.
- Test Banking77-VN: baseline macro-F1 0,8939, top-3 0,9697; DL mới macro-F1 0,8832, top-3 0,9617. DL cũ có macro-F1 0,7030.
- MASSIVE test recall `UNKNOWN`: baseline 0,8999; DL mới 0,7180; DL cũ 0,3843. Tỷ lệ từ chối nhầm in-domain test tương ứng 0,0378 và 0,0416.
- FAQ test recall `UNKNOWN`: baseline 0,7400 (37/50); DL mới 0,6400 (32/50); DL cũ 0,4000 (20/50). Khoảng bootstrap 95% theo URL của DL mới là [0,4821; 0,8205].
- DL mới đã được đưa vào `data/artifacts`. DL và báo cáo cũ được sao lưu tại `data/experiments/deep_before_multilingual`.
- Báo cáo chính và chẩn đoán FAQ đã được tạo lại. Demo tiếp tục dùng baseline mặc định vì baseline vẫn nhỉnh hơn trên test và OOD.
- Suy luận tích hợp DL trả đúng `transaction_charged_twice` với confidence 0,9889, top-3 và ba câu tương tự cho ví dụ phí hai lần.
- Giao diện Streamlit có lựa chọn baseline/DL, ba câu thử nhanh, form nhập liệu, thẻ quyết định/confidence/threshold, bảng top-3 và ba câu tương tự. Model chỉ nạp sau khi người dùng bấm phân tích.
- Luồng UI đã được kiểm tra headless: ví dụ phí hai lần trả `transaction_charged_twice`; ví dụ vay mua nhà trả `UNKNOWN` với confidence 0,2205 dưới threshold 0,3879.
- Bộ FAQ mới có 100 câu, bốn topic LOAN/SAVING/INTEREST_RATE/PROMOTION mỗi topic 25 câu, 45 URL. Input mô hình ẩn tên ngân hàng ở 67 câu nhưng giữ câu gốc và URL.
- Ba cặp gần trùng ngữ nghĩa được giữ cùng split. Validation/test có 50 câu mỗi phần và không dùng confidence để chia hoặc chọn threshold.
- Hàng đợi UTS2017_Bank cũ và bộ 25 FAQ + 75 câu DeepSeek lịch sử chỉ được giữ để truy vết, không dùng trong đánh giá hiện hành.

## 3. Việc đang chờ

- Nếu cần chuẩn công bố nghiêm ngặt, hai người độc lập gán lại nhãn FAQ và giải quyết bất đồng.
- Nếu cần ước lượng hiệu năng vận hành, bổ sung tin nhắn hỗ trợ thực tế vì FAQ website có văn phong sạch hơn.
- Nếu tiếp tục tối ưu DL, chỉ chọn bằng train/model-selection; hạn chế số vòng đọc test để tránh điều chỉnh gián tiếp theo test.

## 4. File/module chính

- `configs/default.yaml`: encoder đa ngôn ngữ, max length 128 và cấu hình giai đoạn fine-tune đầu.
- `src/intentguard/models/deep.py`: train, tiếp tục fine-tune, lưu/nạp encoder và embedding index.
- `scripts/train_deep.py`: CLI có model, artifact, max length, epoch, learning rate và resume checkpoint.
- `scripts/train_multilingual_candidate.py`: thí nghiệm classifier tuyến tính trên embedding đóng băng.
- `scripts/promote_deep_candidate.py`: kiểm tra score, sao lưu và đưa candidate tốt hơn vào artifact chính.
- `scripts/calibrate_and_evaluate.py`: hỗ trợ đánh giá artifact/report thử nghiệm riêng và tạo báo cáo chính.
- `data/artifacts/`: model DL đa ngôn ngữ, calibration và retrieval index hiện hành.
- `reports/comparison.md`/`.json`: kết quả Banking77-VN, MASSIVE và FAQ mới nhất.
- `reports/faq_ood_diagnostic.md`/`.json`: recall theo topic, bootstrap theo URL và lỗi FAQ.
- `reports/error_analysis.md`: phân tích lỗi đã cập nhật theo model mới.
- `data/raw/vietnam_banks_faq_100.csv`, `data/processed/faq_ood_real/`: nguồn và split FAQ hiện hành.
- `app/streamlit_app.py`: giao diện kiểm thử Streamlit đơn giản, dùng widget native và cache model theo backend.
- `src/intentguard/inference.py`: API suy luận chung cho UI và kiểm thử.

## 5. Quyết định kỹ thuật

- Dùng encoder đa ngôn ngữ vì dữ liệu đích là tiếng Việt. Candidate chỉ được thay vào artifact chính sau khi vượt DL cũ trên model-selection.
- Không dùng Banking77-VN test, MASSIVE hoặc FAQ để chọn encoder/checkpoint/siêu tham số. Các test chỉ được chạy sau khi chốt candidate 0,8960 trên model-selection.
- Temperature fit trên calibration in-domain. Threshold chọn trên split threshold in-domain với tỷ lệ từ chối tối đa 5%; không tối ưu threshold bằng OOD.
- Không huấn luyện bằng FAQ. Tên ngân hàng được chuẩn hóa thành `ngân hàng` để giảm shortcut, còn câu gốc và URL được giữ để audit.
- Baseline vẫn là backend Streamlit mặc định vì đạt kết quả tổng thể cao hơn DL mới.
- Không sinh thêm FAQ bằng LLM. Audit FAQ có provenance `codex_ai`, không được gọi là nhãn vàng hai người.

## 6. Lệnh và kết quả gần nhất

- `uv run --no-sync python scripts/calibrate_and_evaluate.py`: PASS; tạo lại báo cáo baseline + DL đa ngôn ngữ.
- `uv run --no-sync python scripts/evaluate_faq_ood.py`: PASS; FAQ test baseline 0,7400, DL 0,6400.
- `uv run --no-sync python -m pytest`: PASS, 28 passed trong 8,57 giây.
- `uv run --no-sync ruff check src scripts app tests`: PASS, all checks passed.
- Kiểm tra `IntentGuardPredictor` backend deep: PASS; trả intent, confidence, top-3 và 3 similar cases.
- Streamlit `AppTest`: PASS cho màn hình ban đầu, intent known và nhánh `UNKNOWN`; không có exception, hai bảng kết quả đều có ba dòng.
- Streamlit đang chạy tại `http://localhost:8502/`; health endpoint trả HTTP 200 và `ok`.
- `git status --short`: không chạy được vì workspace không phải repository Git.
- Trên máy hiện tại cần đặt `UV_CACHE_DIR` vào `.uv-cache` trong workspace; chạy model offline bằng `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1` khi artifact/cache đã có.

## 7. Lỗi, rủi ro và giả định

- DL mới cải thiện lớn nhưng vẫn thấp hơn baseline: chênh macro-F1 test 0,0106; MASSIVE recall 0,1819; FAQ recall 0,1000.
- Câu lãi suất vẫn thường bị nhận thành `exchange_rate`. Ví dụ “Lãi suất tiết kiệm có kỳ hạn là gì?” có confidence 0,9504, nên confidence threshold đơn lẻ chưa xử lý hết semantic OOD gần miền.
- FAQ website không đại diện đầy đủ cho tin nhắn khách hàng thật. Khoảng tin cậy rộng do chỉ có 50 câu test và nhiều câu cùng URL.
- Audit nhãn do Codex AI thực hiện; chưa có hai người rà độc lập. Một số chương trình ưu đãi có thể hết hiệu lực.
- Banking77-VN ghi giấy phép CC BY-NC-SA 4.0; prototype chỉ dùng phi thương mại.
- File `.env` và API key không được in hoặc lưu trong báo cáo. Luồng cải thiện DL không gọi DeepSeek.

## 8. Bước tiếp theo

1. Thử giao diện tại `http://localhost:8502/` với cả baseline và DL.
2. Khi cần benchmark nghiêm ngặt, tổ chức hai người gán nhãn độc lập 100 FAQ rồi adjudicate bất đồng.
3. Khi có dữ liệu thực tế, tạo thêm OOD test từ tin nhắn khách hàng và giữ tách biệt khỏi mọi vòng chọn model.
