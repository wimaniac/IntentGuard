# IntentGuard

IntentGuard là prototype phi thương mại cho hai bài toán: phân loại tin nhắn tiếng Việt theo intent ngân hàng và phát hiện câu hỏi ngoài phạm vi (`UNKNOWN`). Dự án dùng dữ liệu `GreenNode/banking77-vn`, baseline TF-IDF word/character n-gram + Logistic Regression, một classifier tuyến tính trên `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, temperature scaling, threshold từ chối và retrieval câu tương tự.

> Đây là prototype nghiên cứu, không dùng cho quyết định vận hành hoặc tư vấn tài chính. Dataset Banking77-VN có giấy phép CC BY-NC-SA 4.0; hãy kiểm tra điều khoản nguồn trước khi tái sử dụng.

## Cài đặt

Yêu cầu Python 3.11–3.13 và `uv`.

```powershell
uv sync
```

Mặc định `pyproject.toml` ưu tiên wheel PyTorch CUDA 12.8 trên Windows AMD64. Nếu môi trường không phù hợp, điều chỉnh nguồn PyTorch trong `pyproject.toml` trước khi `uv lock`/`uv sync`.

## Pipeline dữ liệu

```powershell
uv run python scripts/prepare_data.py
```

Pipeline tải schema theo cấu hình, chỉ dùng `text` và `label`/`label_text`, loại duplicate train trùng test hoặc trùng nội bộ, ghi số lượng bị loại và chia train gốc theo seed 42 thành 70/10/10/10. Test gốc được giữ riêng, không dùng để chọn model hoặc threshold.

Các nguồn OOD được lưu tách khỏi classifier:

1. MASSIVE `vi-VN` được lọc theo topics trong `configs/default.yaml`.
2. UTS2017_Bank chỉ tạo candidate, **chưa phải nhãn OOD**. Chạy bước triage để xếp 216 ứng viên thuộc vay, tiết kiệm, lãi suất và khuyến mãi lên trước; các chủ đề khác vẫn có thể xem. Baseline confidence chỉ là tín hiệu tham khảo, không tự duyệt câu.

```powershell
uv run python scripts/prepare_review_queue.py
uv run streamlit run app/review_app.py
```

Hàng đợi tại `data/interim/ood_bank_review_queue.csv` có mức ưu tiên, dấu hiệu câu yêu cầu, điểm baseline và nhóm câu gần trùng. Trang review lưu từng quyết định vào `data/raw/ood_bank_review.csv`; người review có thể sửa câu nguồn thành câu hỏi hỗ trợ rõ nghĩa, đánh dấu duyệt/loại/cần viết lại và ghi lý do. Chỉ duyệt câu ngoài 77 intent, không chứa thông tin cá nhân và dài 20–300 ký tự. Sau khi có đúng 100 câu đã duyệt, nút **Kiểm tra và xuất 50 validation / 50 test** tạo hai danh sách `source_case_id`, giữ câu gần trùng cùng một split và kiểm tra không trùng dữ liệu Banking77-VN. Nếu nguồn nhận xét không đủ 100 câu hỏi hợp lệ, cần biên soạn thêm từ chủ đề nguồn và ghi rõ tập này là dữ liệu được biên soạn; không coi candidate chưa duyệt là nhãn vàng.

### Duyệt tự động bằng DeepSeek

Đặt `DEEPSEEK_API_KEY` trong `.env` ở thư mục dự án (đã được loại khỏi Git), hoặc biến môi trường. Chạy:

```powershell
uv run python scripts/auto_review_deepseek.py
```

Lệnh bỏ qua mọi câu đã có quyết định, ưu tiên bốn chủ đề mục tiêu, gửi từng câu cùng danh sách 77 intent và tự lưu kết quả sau mỗi câu. DeepSeek có thể duyệt, loại hoặc viết lại câu nguồn nếu giữ nguyên ý. Lệnh dừng khi đủ 100 câu, kiểm tra độ dài, câu trùng, nguồn và tách 50 validation / 50 test; nếu API gián đoạn, chạy lại để tiếp tục. Trước khi gửi, câu có chuỗi số dài hoặc email được loại cục bộ. File `data/raw/ood_bank_review_before_deepseek.csv` lưu bản sao quyết định ban đầu.

Nếu bốn chủ đề nguồn chưa đủ 100 câu, có thể xét thêm ứng viên được gắn nhãn nguồn khác. DeepSeek vẫn chỉ được duyệt câu có nội dung thực sự thuộc vay, tiết kiệm, lãi suất hoặc khuyến mãi:

```powershell
uv run python scripts/auto_review_deepseek.py --cross-topic-keywords
uv run python scripts/auto_review_deepseek.py --secondary
uv run python scripts/auto_review_deepseek.py --similar-pool 200
```

Các câu do DeepSeek tự duyệt là **nhãn do LLM tạo**, không tương đương tập nhãn vàng được người kiểm tra độc lập. Báo cáo ghi số nhãn DeepSeek/thủ công và phải được hiểu theo giới hạn đó. API có thể tính phí theo số câu được gọi. Trang review thủ công vẫn hỗ trợ nút **Gợi ý bằng DeepSeek** và cũng đọc key từ `.env`.

**Trạng thái hàng đợi UTS cũ:** Kiểm tra ngày 2026-09-20 đã rút 69/100 nhãn OOD sai rõ ràng và chuyển 21 câu sang `needs_rewrite`. Sau đó, 12 câu được người dùng đánh dấu `rewrite=1` đã được xử lý: 11 câu được biên tập trung thành với nguồn rồi tạm duyệt, 1 câu trộn phí chuyển tiền và bảo hiểm bị loại. Còn 21 câu tạm duyệt và 10 câu `needs_rewrite`; hàng đợi này không dùng để đánh giá. Bản sao trước sửa ở `data/raw/ood_bank_review_before_marked_rewrites.csv`; bảng câu gốc, câu sửa và lý do ở `reports/ood_rewrite_decisions.csv`. `approved_ood=0` là đã loại, `1` là tạm duyệt, ô trống là chưa quyết định hoặc cần viết lại. Cột `rewrite=1` chỉ là cờ cần xử lý, không tự tạo câu mới hoặc tự duyệt. Topic nguồn như `TRADEMARK` hay `OTHER` không tự quyết định câu là OOD. Xem [báo cáo kiểm tra](reports/ood_quality_audit.md). Split và parquet UTS cũ được giữ để truy vết nhưng không được pipeline sử dụng; lệnh DeepSeek cũ cũng dừng tại cổng chất lượng. Báo cáo OOD ngân hàng hiện hành dùng FAQ nguồn mới dưới đây.

Đã [tìm nguồn OOD ngân hàng thay thế](reports/ood_bank_source_search.md). Tập FAQ **hiện hành** là `data/raw/vietnam_banks_faq_100.csv` do người dùng thu thập: 100 câu, mỗi chủ đề `LOAN`, `SAVING`, `INTEREST_RATE`, `PROMOTION` có 25 câu, kèm 45 URL trang ngân hàng. Tập 25 FAQ + 75 biến thể DeepSeek trước đó chỉ giữ để truy vết; không còn dùng trong báo cáo FAQ hiện hành.

### Chuẩn bị và đánh giá FAQ nguồn mới

```powershell
uv run python scripts/prepare_real_faq_ood.py
uv run python scripts/evaluate_faq_ood.py
```

Bước chuẩn bị kiểm tra schema, URL HTTPS, topic, trùng lặp và trùng chính xác với Banking77-VN. File `data/processed/faq_ood_real/all.csv` giữ `text_original`, `text_reviewed`, `source_url`, `topic`, `source_case_id` và `text` đã ẩn tên ngân hàng. Hai câu gốc hỏi nhiều vế được thu gọn còn một nhu cầu có sẵn trong câu nguồn; 67 câu có tên ngân hàng được ẩn nhưng tên sản phẩm như UnionPay FreeWays và GameON vẫn giữ. Câu cùng URL được đặt trong cùng split; validation và test có 50 câu mỗi phần, mỗi topic 12–13 câu. Việc chia và threshold không dựa trên điểm OOD.

Người dùng đã xác nhận URL. Codex AI rà từng câu so với 77 intent: [tóm tắt audit](reports/faq_ood_label_audit.md) và [bảng 100 quyết định](reports/faq_ood_label_audit.csv) ghi nhãn OOD, topic, câu sửa, intent gần nhất của trường hợp dễ nhầm và ba câu gần trùng ngữ nghĩa. [Báo cáo chính](reports/comparison.md) hiện có MASSIVE và FAQ ngân hàng với provenance `codex_ai`; [báo cáo FAQ chi tiết](reports/faq_ood_diagnostic.md) dùng cùng temperature/threshold chọn bằng dữ liệu in-domain. Đây là đánh giá có lượt rà bằng AI, chưa phải bộ nhãn vàng do hai người gán nhãn độc lập. FAQ website cũng không đại diện hoàn toàn cho tin nhắn hỗ trợ thực tế. Báo cáo tổng hợp cũ đã sao lưu ở `reports/faq_ood_synthetic_legacy.md`; các script sinh/audit DeepSeek và `data/processed/faq_ood_curated/` là dữ liệu lịch sử, không cần chạy lại.

Để tạo lại báo cáo chính từ artifact in-domain và MASSIVE hiện có:

```powershell
uv run python scripts/calibrate_and_evaluate.py
```

Chế độ `prepare_data.py --review-only` vẫn dành cho hàng đợi UTS cũ và sẽ từ chối dùng split UTS khi cổng chất lượng chỉ cho phép FAQ đã rà bằng AI; bước này không cần cho FAQ mới.

## Huấn luyện và đánh giá

```powershell
uv run python scripts/train_baseline.py
uv run python scripts/calibrate_and_evaluate.py
```

`train_baseline.py` chọn tham số trên split model-selection. DL hiện hành dùng encoder đa ngôn ngữ và được fine-tune hai giai đoạn. Cả hai giai đoạn chỉ chọn checkpoint theo model-selection; test Banking77-VN và FAQ không tham gia lựa chọn:

```powershell
uv run python scripts/train_deep.py `
  --artifact-dir data/experiments/deep_multilingual_finetuned

uv run python scripts/train_deep.py `
  --resume-from data/experiments/deep_multilingual_finetuned `
  --artifact-dir data/experiments/deep_multilingual_continued `
  --learning-rate 0.00001
```

Checkpoint giai đoạn hai tốt nhất đạt macro-F1 0,8960 trên model-selection. Script lưu encoder fine-tune và tạo embedding index chỉ từ split train. Artifact cũ được sao lưu trước khi đưa candidate tốt hơn vào `data/artifacts` bằng `scripts/promote_deep_candidate.py`. Bước đánh giá fit temperature trên calibration, chọn threshold cao nhất để tỷ lệ từ chối in-domain không quá 5%, rồi mới đọc test gốc. OOD validation được so với in-domain threshold validation; OOD test được so với in-domain test.

Kết quả test hiện hành: baseline đạt macro-F1 0,8939 và DL đa ngôn ngữ đạt 0,8832. Recall `UNKNOWN` của DL tăng từ 0,3843 lên 0,7180 trên MASSIVE test và từ 0,4000 lên 0,6400 trên FAQ test. Baseline vẫn là backend mặc định vì đạt macro-F1 0,8939, MASSIVE recall 0,8999 và FAQ recall 0,7400.

Báo cáo sinh tại `reports/comparison.md` và `reports/comparison.json`, gồm accuracy, macro-F1, top-3 accuracy, NLL/ECE trước-sau calibration, AUROC/AUPRC, recall UNKNOWN, false rejection, cặp intent dễ nhầm và ví dụ lỗi OOD.

## Demo Streamlit

```powershell
uv run streamlit run app/streamlit_app.py
```

Demo dùng cùng `IntentGuardPredictor` với pipeline đánh giá. Baseline hiện là lựa chọn mặc định vì kết quả test tốt hơn DL. Khi confidence dưới threshold, intent chính là `UNKNOWN`; top-3 vẫn được hiển thị nhưng được đánh dấu là tham khảo. Cả hai backend đều trả ba câu tương tự từ split train: baseline dùng TF-IDF, DL dùng embedding encoder đã fine-tune. Model và retrieval index được cache trong phiên chạy app.

Giao diện có ba câu thử nhanh, lựa chọn `baseline`/`deep`, thẻ kết quả với confidence và threshold, bảng top-3 và ba câu train tương tự. Model chỉ được nạp sau khi bấm **Phân tích** để màn hình ban đầu hiển thị nhanh hơn.

### Triển khai trên Streamlit Community Cloud

Repository đã chuẩn bị bản cloud dùng baseline. Bốn artifact cần thiết của baseline được lưu trong GitHub; artifact DL khoảng 520 MB được loại khỏi Git để tránh vượt giới hạn file và tài nguyên triển khai. Khi không có artifact DL, giao diện tự ẩn lựa chọn `deep`. File `app/requirements.txt` chỉ cài các dependency cần cho baseline, giúp bản cloud nhẹ hơn môi trường huấn luyện đầy đủ trong `pyproject.toml`.

1. Đăng nhập [Streamlit Community Cloud](https://share.streamlit.io/) bằng GitHub.
2. Chọn **Create app** → **Yup, I have an app**.
3. Chọn repository `wimaniac/IntentGuard`, branch `main`.
4. Đặt **Main file path** là `app/streamlit_app.py`.
5. Trong **Advanced settings**, chọn Python `3.12`. Demo không cần secrets hoặc `DEEPSEEK_API_KEY`.
6. Bấm **Deploy**. Các lần push tiếp theo lên `main` sẽ cập nhật ứng dụng.

Streamlit Community Cloud tìm dependency cạnh entry point trước, vì vậy `app/requirements.txt` được dùng thay cho `uv.lock` ở root. Nếu muốn triển khai DL sau này, nên lưu fine-tuned model ở một model registry rồi tải có cache lúc khởi động, hoặc dùng Git LFS và theo dõi giới hạn RAM/thời gian khởi động của Community Cloud.

## Kiểm thử và lint

```powershell
uv run python -m pytest
uv run ruff check .
```

Các test đơn vị kiểm tra chia stratified không rò rỉ, thứ tự top-k, calibration, quy tắc đúng tại biên threshold, retrieval chỉ từ train, ghép OOD validation/test và bảo vệ file review. Huấn luyện đầy đủ cần network để tải dataset/model và GPU sẽ giúp rút ngắn thời gian DL; test đơn vị không yêu cầu tải artifact bên ngoài.

## Cấu trúc

- `src/intentguard/`: data, model, calibration, metrics, retrieval và inference.
- `scripts/`: các entry point tái lập pipeline.
- `app/`: demo suy luận và trang review OOD Streamlit.
- `configs/`: cấu hình versioned, không chứa dữ liệu câu hỏi.
- `data/`: dữ liệu tải/sinh ra, bị loại khỏi Git.
- `reports/`: báo cáo sinh ra, bị loại khỏi Git.
