# Chẩn đoán OOD trên FAQ/LLM ngân hàng

Bộ này gồm FAQ người dùng cung cấp và biến thể do DeepSeek sinh. Đã rà cục bộ và DeepSeek audit 100/100 câu sau sửa; chưa có nhãn vàng độc lập.
Đây là phép kiểm tra bổ sung, không thay metric OOD chính trong `comparison.md`.

Threshold và temperature đọc từ artifact đã chọn trên dữ liệu in-domain; không chọn lại bằng FAQ/LLM. Câu cùng `seed_id` được giữ trong một split: 48 validation (12 nguồn) và 52 test (13 nguồn). Khoảng 95% cho recall được bootstrap theo nhóm nguồn, nên phản ánh sự phụ thuộc giữa các biến thể.

| Backend | Split | Số câu | Số nguồn | Recall UNKNOWN | CI 95% theo nguồn | AUROC | AUPRC | Từ chối nhầm ID |
|---|---|---:|---:|---:|---|---:|---:|---:|
| baseline | validation | 48 | 12 | 0.7500 | [0.5417, 0.9167] | 0.9404 | 0.5911 | 0.0498 |
| baseline | test | 52 | 13 | 0.8462 | [0.7500, 0.9231] | 0.9773 | 0.6655 | 0.0378 |
| deep | validation | 48 | 12 | 0.3333 | [0.1875, 0.5000] | 0.7386 | 0.1871 | 0.0498 |
| deep | test | 52 | 13 | 0.2308 | [0.0769, 0.4038] | 0.6989 | 0.0995 | 0.0492 |

## Recall UNKNOWN trên test theo topic

- **baseline:** INTEREST_RATE 0.833 (12 câu); LOAN 0.812 (16 câu); PROMOTION 1.000 (8 câu); SAVING 0.812 (16 câu)
- **deep:** INTEREST_RATE 0.500 (12 câu); LOAN 0.000 (16 câu); PROMOTION 0.500 (8 câu); SAVING 0.125 (16 câu)

## Giới hạn

- Chỉ 25 nhu cầu FAQ gốc; số câu sinh thêm không tạo ra tình huống độc lập tương ứng.
- Câu nguồn chưa có URL/tên ngân hàng để xác minh xuất xứ. Audit bằng cùng nhà cung cấp model đã sinh câu và nhãn topic không tương đương nhãn vàng độc lập.
- Tập FAQ/LLM có văn phong sạch hơn tin nhắn khách hàng thật. Không suy rộng recall ở đây thành hiệu năng vận hành.
- AUPRC phụ thuộc tỷ lệ OOD/ID; tập FAQ nhỏ hơn nhiều so với in-domain reference, nên không so trực tiếp AUPRC FAQ với MASSIVE.
- Ví dụ được mô hình nhận thành intent đã học nằm trong file JSON cùng tên để phân tích lỗi.
