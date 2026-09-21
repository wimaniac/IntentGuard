# Kiểm tra chất lượng nhãn OOD ngân hàng

Ngày kiểm tra: 2026-09-20. Tập 100 câu trước đây gồm 94 nhãn DeepSeek và 6 nhãn thủ công. Đây là kiểm tra bảo thủ: chỉ rút các câu sai rõ ràng, chưa xác nhận các câu còn lại là đúng.

## Kết quả

| Kết quả | Số câu |
|---|---:|
| Rút nhãn `approved_ood=1` vì là lời khen, chúc mừng hoặc chia sẻ trải nghiệm | 41 |
| Rút nhãn vì ngoài bốn chủ đề mục tiêu hoặc quá chung | 13 |
| Rút nhãn vì thuộc intent cũ hoặc trộn nhiều vấn đề | 6 |
| Rút nhãn vì nhận xét mơ hồ, không nêu nhu cầu cụ thể | 9 |
| Chuyển sang `needs_rewrite` vì lỗi chính tả/diễn đạt | 21 |
| Còn `approved_ood=1`, **chưa xác minh độc lập** | 10 |

Các ví dụ sai rõ ràng: `train:156` (“Tôi thấy có nhiều ưu đãi tuyệt vời.”) là lời khen; `train:763` (“Tôi muốn phản ánh chất lượng dịch vụ khách hàng của ngân hàng rất kém.”) là phàn nàn dịch vụ chung; `train:139` hỏi phí chuyển khoản, gần intent cũ. Phản hồi `rewrite` của DeepSeek từng được chấp nhận dù lý do của chính nó nói câu ngoài phạm vi. Luồng duyệt tự động đã được bổ sung cờ kiểm tra phạm vi, yêu cầu hỗ trợ và tính trung thành của bản viết lại.

`approved_ood=0` nghĩa là đã loại; `approved_ood=1` hiện chỉ là tạm duyệt; ô trống nghĩa là chưa có quyết định duyệt/loại hoặc cần viết lại, **không phải** nhãn OOD. Câu sai chính tả chỉ có thể được giữ nếu nội dung là yêu cầu cụ thể trong bốn chủ đề, rồi sửa cách viết mà không thêm sự kiện mới. Các câu như `train:46`, `train:47`, `train:110` đã chuyển sang `needs_rewrite`, chưa được dùng trong benchmark.

Tập OOD ngân hàng và các metric cũ đã bị tạm rút khỏi báo cáo hiện hành. File review, hai danh sách split và parquet cũ được giữ để truy vết nhưng bị chặn bởi `data/ood_bank_quality_status.json`. Bản sao trước kiểm tra: `data/raw/ood_bank_review_before_quality_audit.csv`; báo cáo metric cũ: `reports/comparison_before_quality_audit.md`.
