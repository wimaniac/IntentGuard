# Chẩn đoán OOD trên 100 FAQ ngân hàng có URL

Nguồn hiện hành: `data/raw/vietnam_banks_faq_100.csv` do người dùng thu thập, 100 câu và 45 URL. Người dùng xác nhận URL; Codex rà lại từng nhãn OOD so với 77 intent. Đây là audit bằng AI, không phải bộ nhãn vàng do hai người gán nhãn độc lập.
Metric FAQ cũng nằm trong `comparison.md` với nguồn gốc nhãn được ghi rõ; bảng dưới đây là phân tích chi tiết.

Input mô hình thay tên ngân hàng bằng `ngân hàng` ở 67 câu; câu gốc và URL được giữ trong artifact. Hai câu nhiều vế được thu gọn về một nhu cầu có sẵn trong câu gốc; 3 câu gần trùng ngữ nghĩa được giữ cùng split với câu tương ứng. Threshold và temperature đọc từ artifact đã chọn trên dữ liệu in-domain, không chọn lại bằng FAQ. Validation/test chia 50/50, mỗi topic 12–13 câu; các câu cùng URL nằm trọn trong một split. Khoảng 95% cho recall bootstrap theo URL vì nhiều câu cùng lấy từ một trang.

| Backend | Split | Số câu | Số URL | Recall UNKNOWN | CI 95% theo URL | AUROC | AUPRC | Từ chối nhầm ID |
|---|---|---:|---:|---:|---|---:|---:|---:|
| baseline | validation | 50 | 23 | 0.7400 | [0.6190, 0.8718] | 0.9425 | 0.5325 | 0.0498 |
| baseline | test | 50 | 22 | 0.7400 | [0.6216, 0.8611] | 0.9603 | 0.5593 | 0.0378 |
| deep | validation | 50 | 23 | 0.6800 | [0.5471, 0.8333] | 0.9478 | 0.4653 | 0.0498 |
| deep | test | 50 | 22 | 0.6400 | [0.4821, 0.8205] | 0.9546 | 0.3748 | 0.0416 |

## Recall UNKNOWN trên test theo topic

- **baseline:** INTEREST_RATE 0.615 (13 câu); LOAN 0.750 (12 câu); PROMOTION 0.667 (12 câu); SAVING 0.923 (13 câu)
- **deep:** INTEREST_RATE 0.308 (13 câu); LOAN 0.750 (12 câu); PROMOTION 1.000 (12 câu); SAVING 0.538 (13 câu)

## Giới hạn

- Người dùng đã xác nhận URL là đúng. Codex chưa tự đối chiếu nguyên văn từng câu trên trang hoặc hiệu lực của từng chương trình ưu đãi.
- Nhãn OOD và topic được Codex rà theo 77 intent; đây là lượt rà bằng AI, chưa có kiểm tra của hai người độc lập.
- FAQ trên website có văn phong sạch hơn tin nhắn khách hàng thật. Không suy rộng recall ở đây thành hiệu năng vận hành.
- AUPRC phụ thuộc tỷ lệ OOD/ID; không so trực tiếp AUPRC FAQ với MASSIVE.
- Ví dụ được mô hình nhận thành intent đã học nằm trong file JSON cùng tên để phân tích lỗi.
