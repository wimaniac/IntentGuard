# Tìm nguồn thay thế cho OOD ngân hàng

Ngày kiểm tra: 2026-09-20. Mục tiêu là tìm **100 câu hỏi tiếng Việt** dạng yêu cầu hỗ trợ ngân hàng, thuộc vay, tiết kiệm, lãi suất hoặc khuyến mãi, và nằm ngoài 77 intent của Banking77-VN. Không lấy câu chỉ vì có từ khóa hoặc nhãn chủ đề phù hợp.

## Kết quả

Chưa tìm được **một bộ dữ liệu có thể lấy trực tiếp 100 câu đạt tiêu chí**. Vì vậy, báo cáo hiện tại chỉ tính OOD trên MASSIVE; không tính metric OOD ngân hàng và không tái sử dụng split 50/50 cũ. 21 câu từ UTS2017_Bank vẫn chỉ là ví dụ tạm duyệt để xem lỗi định tính.

| Nguồn | Kiểm tra | Kết luận |
| --- | --- | --- |
| [MFAQ, cấu hình tiếng Việt](https://huggingface.co/datasets/clips/mfaq) | 27.157 cặp FAQ, có tên miền nguồn. Đã tải hai split `vi_flat`, thấy 458 tên miền nhưng không có miền ngân hàng. Các câu trùng từ khóa chủ yếu từ trang cho vay, casino hoặc khuyến mãi ngành khác. Giấy phép CC0 chỉ áp dụng cho cách đóng gói, không khẳng định quyền với văn bản gốc. | Không phù hợp để lấy 100 câu hỗ trợ ngân hàng. |
| [HVU_QA / GeneratingQuestions](https://huggingface.co/datasets/DANGDOCAO/GeneratingQuestions) | 43.913 mục QCA; 1.350 mục có từ khóa rộng về vay/tiết kiệm/lãi suất/ưu đãi. Sau lọc hình thức, còn 433 câu khác nhau, nhưng kiểm tra mẫu cho thấy nhiều câu là tin tức, quy định pháp luật, chính sách nhà nước hoặc câu hỏi dành cho tổ chức tín dụng. “Ưu đãi” chủ yếu là tín dụng chính sách, không phải khuyến mãi bán lẻ. | Không thể lấy ngẫu nhiên 100 câu mà vẫn giữ đúng phân bố yêu cầu hỗ trợ và bốn chủ đề. |
| [ViLQA](https://huggingface.co/datasets/huyhuy123/ViLQA) | 43.588 câu hỏi về văn bản pháp luật; có ví dụ liên quan khoản vay và lãi suất nhưng mục đích là legal QA. Dataset card ghi chỉ dùng cho nghiên cứu. | Khác nhiệm vụ hỗ trợ khách hàng ngân hàng. |
| [thuvienphapluat-question-query](https://huggingface.co/datasets/thangvip/thuvienphapluat-question-query) | 19.861 câu hỏi pháp luật; có vài câu tiết kiệm/vay giống yêu cầu khách hàng, nhưng đa số kèm nội dung pháp lý và dataset card không nêu giấy phép sử dụng. | Không đủ căn cứ để dùng làm bộ đánh giá 100 câu. |
| [Bitext retail banking](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset), [banking customer service query intent](https://huggingface.co/datasets/atulgupta002/banking_customer_service_query_intent) | Đúng dạng hỗ trợ ngân hàng nhưng chỉ có tiếng Anh. Dịch 100 câu sang tiếng Việt sẽ tạo tập dịch/biên soạn, không còn là 100 câu tiếng Việt lấy trực tiếp từ nguồn. | Không dùng như benchmark câu hỏi tiếng Việt tự nhiên. |
| [VNPT Money customer service](https://www.kaggle.com/datasets/tuantotti/customer-service-dataset) | Có hỏi đáp tiếng Việt về mobile money và khuyến mãi, nhưng là dịch vụ viễn thông/tài chính di động, không phải ngân hàng bán lẻ. | Không phù hợp với OOD ngân hàng gần miền. |

## Giới hạn của kết luận

Đây là cuộc tìm kiếm nguồn mở và kiểm tra mẫu theo tiêu chí của dự án, không chứng minh rằng không tồn tại nguồn khác. Với HVU_QA, số 433 chỉ là **ứng viên sau lọc hình thức**, không phải 433 nhãn OOD đã xác minh. Nếu sau này có dữ liệu tin nhắn hỗ trợ ngân hàng được phép sử dụng, cần kiểm tra thủ công hoặc kiểm tra độc lập từng nhãn, loại câu trùng Banking77-VN, rồi mới mở cổng chất lượng và tính metric.
