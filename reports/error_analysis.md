# Phân tích lỗi phân loại và phát hiện UNKNOWN

Nguồn số liệu: `reports/comparison.json` trên test Banking77-VN, MASSIVE vi-VN và 100 FAQ ngân hàng có URL; `reports/faq_ood_diagnostic.json` bổ sung phân tích FAQ theo topic và URL. Threshold chọn từ validation in-domain với giới hạn từ chối nhầm 5%; không chọn lại bằng OOD. Input FAQ cho mô hình đã ẩn tên ngân hàng, bản gốc được giữ để đối chiếu.

## Kết quả tổng quát

| Backend | Macro-F1 Banking77 test | Top-3 | MASSIVE test recall UNKNOWN | FAQ nguồn test recall UNKNOWN |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0,8939 | 0,9697 | 0,8999 | 0,7400 |
| Multilingual MiniLM + classifier | 0,8832 | 0,9617 | 0,7180 | 0,6400 |

FAQ test có 50 câu từ 22 URL nguồn; toàn bộ 100 câu gồm 45 URL. Người dùng xác nhận URL, Codex AI rà cả 100 nhãn OOD theo 77 intent, thu gọn hai câu nhiều vế và ghi ba cặp gần trùng ngữ nghĩa. Khoảng bootstrap theo URL của recall test là [0,6216; 0,8611] cho baseline và [0,4821; 0,8205] cho DL. Đây chưa phải bộ nhãn vàng do hai người độc lập; FAQ vẫn có văn phong khác tin nhắn khách hàng thật.

## Các nhóm lỗi phân loại thường gặp

- Hai backend vẫn nhầm giữa các intent gần nghĩa. Baseline nhầm `why_verify_identity` thành `verify_my_identity` 6 lần; DL nhầm chiều ngược lại 7 lần. Cặp này khác nhau ở mục đích hỏi lý do xác minh so với cách thực hiện xác minh.
- Các lỗi DL thường gặp tiếp theo là `topping_up_by_card` thành `top_up_reverted` 7 lần, `transfer_not_received_by_recipient` thành `pending_transfer` 5 lần và `card_swallowed` thành `declined_cash_withdrawal` 5 lần. Mô hình cần phân biệt trạng thái, kênh và vai trò giao dịch thay vì chỉ dựa vào chủ đề chung.
- Có câu in-domain bị trả UNKNOWN. Ví dụ DL từ chối “Ứng dụng không hiển thị thẻ tôi đã nhận được.” (`card_linking`, confidence 0,430) và câu ngắn “Đó có phải là thời điểm tốt để trao đổi không?” (`exchange_rate`, confidence 0,251). Threshold bảo vệ OOD tạo ra đánh đổi này; không nên hạ threshold dựa trên vài ví dụ.

## Lỗi phát hiện OOD

- Với MASSIVE test, baseline có recall UNKNOWN 0,8999 và từ chối nhầm in-domain 0,0378; DL lần lượt 0,7180 và 0,0416. DL vẫn nhận nhầm một số câu thời tiết hoặc báo thức, chẳng hạn “ngày mai tôi có nên mang theo dù” có confidence 0,898.
- Trên FAQ mới, baseline nhận nhầm câu “Lãi suất sau ưu đãi của khoản vay được xác định như thế nào?” thành `exchange_rate` (confidence 0,896). DL cũng kéo các câu “Lãi suất tiết kiệm có kỳ hạn là gì?” và “Lãi suất tiết kiệm không kỳ hạn được hiểu như thế nào?” về `exchange_rate`, với confidence lần lượt 0,950 và 0,904. Các ví dụ này cho thấy confidence cao chưa đủ bảo đảm câu thuộc phạm vi.
- Recall FAQ test của DL theo topic: `LOAN` 9/12, `SAVING` 7/13, `INTEREST_RATE` 4/13, `PROMOTION` 12/12. Baseline tương ứng 9/12, 12/13, 8/13 và 8/12. Encoder đa ngôn ngữ cải thiện mạnh câu khuyến mãi và vay, nhưng câu lãi suất vẫn gần `exchange_rate` trong không gian biểu diễn.

## Quyết định hiện tại

Giữ baseline làm backend demo mặc định vì vẫn cao hơn DL về macro-F1 và OOD recall trên các tập đang có. DL đa ngôn ngữ thay DL tiếng Anh vì được chọn trước bằng model-selection (0,8960 so với 0,7019), sau đó mới đánh giá test một lần. Không dùng FAQ để fine-tune hoặc chọn checkpoint. Nếu tiếp tục cải thiện, cần giữ nguyên test Banking77 và FAQ test, đồng thời giới hạn số vòng thử để tránh điều chỉnh gián tiếp theo test.
