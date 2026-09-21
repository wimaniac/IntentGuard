# Rà nhãn OOD cho 100 FAQ ngân hàng

Người dùng đã xác nhận 45 URL nguồn trong `data/raw/vietnam_banks_faq_100.csv`. Codex AI đọc 100 câu, đối chiếu nội dung với danh sách 77 intent và một số ví dụ in-domain của các nhãn dễ nhầm. Bảng quyết định từng câu nằm trong `reports/faq_ood_label_audit.csv`. Đây là lượt rà nhãn bằng AI, không phải bộ nhãn vàng do hai người gán nhãn độc lập.

## Kết quả

- **100/100** câu là yêu cầu thông tin ngân hàng nằm ngoài 77 intent theo lượt rà này; bốn topic nguồn đều chấp nhận được. Không phát hiện câu trùng chính xác với Banking77-VN.
- **2 câu gốc nhiều vế** được thu gọn trong `text_reviewed`, giữ nguyên `text_original` và URL: `faq_real:091` chỉ hỏi cách dùng tiền hoàn; `faq_real:097` chỉ hỏi điều kiện nhận quà T1. Cả hai ý đã có trong câu gốc, không sinh câu hỏi mới.
- **3 câu gần trùng ngữ nghĩa** được ghi trong cột `semantic_duplicate_of`: `048→035`, `065→064`, `077→066`. Mỗi cặp nằm cùng split; các câu vẫn được tính riêng trong metric 100 mẫu, nên mức độc lập của 100 câu thấp hơn 100 tình huống hoàn toàn khác nhau.
- Câu `066` về chính sách cộng thêm lãi suất có thể xem là `INTEREST_RATE` hoặc `PROMOTION`; câu `077` cùng nội dung được gắn `PROMOTION`. Cả hai đều OOD, nhưng metric theo topic phụ thuộc cách phân nhóm này.

## Các ranh giới dễ nhầm với intent đã học

| ID | Intent gần nghĩa | Lý do vẫn là OOD |
|---|---|---|
| 026 | `fiat_currency_support` | Hỏi **tiền gửi tiết kiệm bằng ngoại tệ**, không hỏi khả năng giữ/đổi tiền tệ trong tài khoản thông thường. |
| 027 | `balance_not_updated_after_cheque_or_cash_deposit` | Hỏi **cách xem** số dư tiết kiệm; không báo số dư chưa cập nhật sau nộp tiền. |
| 073–075 | `card_payment_fee_charged` | Hỏi lãi suất thẻ tín dụng, không hỏi phí của một giao dịch thanh toán. |
| 082 | `card_acceptance` | Hỏi phương thức thanh toán đủ điều kiện ưu đãi, không hỏi nơi chấp nhận thẻ. |
| 090–091, 095–096 | `request_refund`, `Refund_not_showing_up` | Hỏi quyền lợi hoặc lịch nhận **cashback khuyến mãi**, không yêu cầu hoàn trả giao dịch từ người bán. |

## Giới hạn của kết luận

Xác nhận của người dùng về URL được giữ làm provenance. Codex không đối chiếu nguyên văn từng câu trên trang hoặc kiểm tra chương trình ưu đãi còn hiệu lực. Bản audit là phán đoán ngữ nghĩa của AI; nếu cần benchmark nhãn vàng, cần hai người gán nhãn độc lập và giải quyết bất đồng. FAQ có văn phong khác tin nhắn khách hàng thực, nên recall trên bộ này không suy rộng trực tiếp thành hiệu năng vận hành.
