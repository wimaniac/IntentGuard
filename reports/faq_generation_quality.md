# Rà soát bộ FAQ sinh bằng DeepSeek

Nguồn: `data/raw/faq_ngan_hang.csv` do người dùng cung cấp, gồm 25 câu hỏi với cột `text`, `topic`. File nguồn chưa có URL FAQ hay tên ngân hàng cho từng câu, nên hiện chỉ truy vết được về câu trong CSV.

Kết quả tại `data/interim/faq_ood_100_candidates.csv`: 100 dòng, gồm 25 câu nguồn và 75 câu DeepSeek sinh (3 câu trên mỗi nguồn). Phân bố `LOAN` 32, `SAVING` 28, `INTEREST_RATE` 20, `PROMOTION` 20. Mọi dòng giữ ID nhóm `seed_id`; phản hồi API thô nằm tại `data/interim/faq_generation_deepseek.jsonl`.

Kiểm tra cơ học sau sinh: không thiếu `text`/`topic`, không có câu ngoài 20–300 ký tự, không có câu trùng chính xác theo khóa chuẩn hóa. Script cũng kiểm tra trùng chính xác với 11.437 câu Banking77-VN đã xử lý và lọc email, chuỗi số dài, URL trước khi chọn câu. Bộ `duplicate_groups` hiện tại không tìm được cặp gần trùng chữ ở ngưỡng cosine 0,92; điều đó **không** có nghĩa là các biến thể độc lập về ngữ nghĩa.

Rà nội dung cho thấy nhiều câu là cách diễn đạt khác nhau của cùng một nhu cầu FAQ, đúng mục đích sinh biến thể. Câu nguồn `faq:0010` trộn “mở tài khoản” với “gửi tiết kiệm”; các biến thể chỉ hỏi về tiết kiệm. Một số câu nguồn khác chứa hai nhánh, chẳng hạn `faq:0015` (cách nhận lãi/người nhận lãi) và `faq:0021` (ưu đãi tiết kiệm/vay), nên biến thể có thể tập trung vào một nhánh. Những trường hợp này cần xét khi xác nhận nhãn OOD.

Sau rà cục bộ, năm câu mơ hồ hoặc lệch ý được thay bằng các biến thể có sẵn trong phản hồi thô: `faq:0007:g3`, `faq:0010`, `faq:0010:g2`, `faq:0022:g3`, `faq:0025:g3`. Audit DeepSeek tiếp theo đánh dấu tám câu nhiều ý thuộc ba nhóm nguồn; tám vị trí này được DeepSeek viết lại thành câu một ý. Câu cũ, câu thay và lý do có trong `reports/faq_ood_local_audit.csv`; [báo cáo audit API](faq_ood_api_audit.md) ghi hai lượt kiểm tra. Bộ curated cuối còn 100 dòng: 21 FAQ nguyên văn và 79 câu sinh/sửa. Kiểm tra lại không có câu trùng chính xác với Banking77-VN.

Đã chia theo 25 `seed_id`, mỗi nhóm bốn câu nằm cùng split: 48 validation (12 nhóm) và 52 test (13 nhóm). Bộ FAQ/LLM có [báo cáo chẩn đoán](faq_ood_diagnostic.md) riêng, dùng threshold được chọn trước trên in-domain; **không dùng để chọn threshold hoặc thay metric OOD chính**. Sau khi người dùng cho phép rõ phạm vi gửi API, DeepSeek audit 100/100 câu cuối đạt tiêu chí của prompt. Việc này vẫn không tạo nhãn vàng độc lập vì cùng model tham gia sinh và audit, và nguồn FAQ chưa có URL. Cổng `data/ood_bank_quality_status.json` vẫn `pending_audit`.
