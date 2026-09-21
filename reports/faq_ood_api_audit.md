# Audit DeepSeek cho bộ FAQ/LLM

Người dùng đã cho phép gửi 100 câu FAQ/LLM và danh sách 77 intent tới DeepSeek. Script `scripts/audit_faq_deepseek.py` gửi theo 25 nhóm `seed_id`, mỗi nhóm bốn câu, lưu phản hồi từng nhóm tại `data/interim/faq_audit_deepseek.jsonl` và bảng từng câu tại `data/interim/faq_ood_api_audit.csv`. Khóa API chỉ được đọc từ môi trường hoặc `.env`, không lưu trong các file kết quả.

## Kết quả

- Lượt đầu: 92/100 câu đạt toàn bộ năm cờ; 8 câu bị đánh dấu `single_issue=false`. Cả 100 câu đều được đánh giá là yêu cầu ngân hàng đúng topic, ngoài 77 intent và trung thành với câu nguồn.
- Tám câu nhiều ý thuộc ba nhóm `faq:0004` (hạn mức và thời hạn vay), `faq:0007` (tất toán trước hạn và phí), `faq:0021` (khuyến mãi vay và tiết kiệm). DeepSeek viết lại tám vị trí thành câu một ý; bản sửa lưu tại `data/interim/faq_single_issue_rewrites.json`.
- Audit lại **chỉ ba nhóm đã sửa**: bộ curated cuối cùng có 100/100 câu `provisional_ood`, không còn `needs_review`. Bảng API audit đã được đối chiếu đúng `source_case_id` và `text` với `data/processed/faq_ood_curated/all.csv`.
- Bộ cuối gồm 21 FAQ nguyên văn và 79 câu DeepSeek sinh hoặc sửa. Hai split vẫn là 48 validation / 52 test theo 25 nhóm nguồn, không rò rỉ nhóm.

## Cách diễn giải

Đây là **nhãn tạm duyệt bởi DeepSeek**, không phải nhãn vàng độc lập: cùng nhà cung cấp model tham gia cả sinh và audit câu. Bộ chỉ có 25 nhu cầu nguồn và file FAQ chưa ghi URL hay ngân hàng xuất xứ. Vì vậy `data/ood_bank_quality_status.json` vẫn `pending_audit`, báo cáo chính `reports/comparison.md` vẫn chỉ tính MASSIVE. Metric FAQ/LLM được công bố riêng trong `reports/faq_ood_diagnostic.md` với giới hạn này.
