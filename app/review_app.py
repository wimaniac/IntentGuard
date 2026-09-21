"""Giao diện Streamlit duyệt thủ công danh sách OOD ngân hàng đã xếp ưu tiên."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.auto_review import load_deepseek_key
from intentguard.config import load_config, resolve_project_paths
from intentguard.data import load_processed_bundle, load_reviewed_bank_ood
from intentguard.llm_review import suggest_review
from intentguard.review import save_review_decision, split_review_cases, validate_approved_rows


def _load_review(path: Path, queue: pd.DataFrame) -> pd.DataFrame:
    """Đọc tiến độ review; tạo CSV ban đầu khi dự án chưa có file quyết định."""
    if not path.exists():
        pd.DataFrame(
            {
                "text": queue["text"],
                "topic": queue["source_topic"],
                "source_case_id": queue["source_case_id"],
                "approved_ood": "",
                "reviewer_note": "",
                "review_status": "",
            }
        ).to_csv(path, index=False, encoding="utf-8")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _deepseek_api_key() -> str:
    """Đọc khoá DeepSeek từ môi trường, .env hoặc Streamlit secrets nếu có."""
    if key := os.getenv("DEEPSEEK_API_KEY", ""):
        return key
    if key := load_deepseek_key(ROOT / ".env"):
        return key
    try:
        return str(st.secrets.get("DEEPSEEK_API_KEY", ""))
    except FileNotFoundError:
        return ""


def _export_splits(config: dict, paths: dict[str, Path], review: pd.DataFrame, queue: pd.DataFrame) -> None:
    """Kiểm tra 100 câu đã duyệt rồi ghi danh sách case 50/50."""
    ood = config["data"]["ood"]
    bundle = load_processed_bundle(paths["processed"] / "in_domain")
    domain_texts = pd.concat(
        [bundle.train, bundle.model_selection, bundle.calibration, bundle.threshold, bundle.test],
        ignore_index=True,
    )["text"].astype(str).tolist()
    candidates = queue.rename(columns={"topic": "source_topic"})
    approved, source_texts = validate_approved_rows(
        review,
        candidates,
        domain_texts,
        int(ood["expected_total"]),
        int(ood["review"]["min_length"]),
        int(ood["review"]["max_length"]),
    )
    validation, test = split_review_cases(
        approved,
        source_texts,
        int(ood["expected_per_split"]),
        int(config["project"]["seed"]),
    )
    (ROOT / ood["bank_validation_cases"]).write_text("\n".join(validation) + "\n", encoding="utf-8")
    (ROOT / ood["bank_test_cases"]).write_text("\n".join(test) + "\n", encoding="utf-8")
    load_reviewed_bank_ood(config, paths)


def main() -> None:
    """Hiển thị hàng đợi, lưu quyết định và xuất tập OOD đã được duyệt."""
    st.set_page_config(page_title="Review OOD ngân hàng", page_icon=":material/rate_review:", layout="wide")
    st.title("Review OOD ngân hàng")
    st.caption("Tín hiệu model chỉ để tham khảo. Chỉ duyệt câu hỏi hỗ trợ hoàn chỉnh nằm ngoài 77 intent đã học.")
    config = load_config(ROOT / "configs/default.yaml")
    paths = resolve_project_paths(config, ROOT)
    ood = config["data"]["ood"]
    queue_path = paths["interim"] / "ood_bank_review_queue.csv"
    if not queue_path.exists():
        st.error("Chưa có hàng đợi. Chạy `uv run python scripts/prepare_review_queue.py` trước.")
        return
    queue = pd.read_csv(queue_path, dtype={"source_case_id": str})
    review_path = ROOT / ood["bank_review_file"]
    review = _load_review(review_path, queue)
    status = review.set_index("source_case_id")["review_status"] if "review_status" in review else None
    if status is None:
        status = review.set_index("source_case_id")["approved_ood"].map(
            {"1": "approved", "0": "rejected"}
        ).fillna("")
    queue["review_status"] = queue["source_case_id"].map(status).fillna("")
    approved_count = int((review["approved_ood"].astype(str) == "1").sum())
    quality_path = ROOT / ood.get("bank_quality_file", "data/ood_bank_quality_status.json")
    quality_pending = False
    quality: dict = {}
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality_pending = quality.get("status") != "verified"
    st.metric("Tạm duyệt OOD" if quality_pending else "Đã duyệt OOD", f"{approved_count}/{ood['expected_total']}")
    if quality_pending:
        st.error(
            "Tập OOD ngân hàng đang chờ kiểm tra chất lượng. "
            f"{quality.get('clear_rejections', 0)} nhãn sai rõ ràng đã được rút, "
            f"{quality.get('needs_rewrite', 0)} câu cần viết lại; "
            "các câu còn tạm duyệt chưa phải nhãn vàng và split cũ không được dùng để đánh giá."
        )
    st.info("Không dùng confidence thấp để tự gán UNKNOWN hoặc chỉ chọn các mẫu dễ cho tập test.")

    left, right = st.columns(2)
    with left:
        priority = st.selectbox("Nhóm ưu tiên", ["ưu tiên", "mở rộng", "khác", "tất cả"])
    with right:
        review_filter = st.selectbox("Trạng thái", ["chưa review", "cần viết lại", "đã duyệt", "đã loại", "tất cả"])
    visible = queue if priority == "tất cả" else queue[queue["priority"] == priority]
    status_values = {"chưa review": "", "cần viết lại": "needs_rewrite", "đã duyệt": "approved", "đã loại": "rejected"}
    if review_filter != "tất cả":
        visible = visible[visible["review_status"] == status_values[review_filter]]
    if visible.empty:
        st.info("Không có ứng viên theo bộ lọc này.")
    else:
        page_size = 30
        pages = max(1, (len(visible) + page_size - 1) // page_size)
        page = st.number_input("Trang", min_value=1, max_value=pages, value=1, step=1)
        page_rows = visible.iloc[(page - 1) * page_size : page * page_size]
        st.dataframe(
            page_rows[["source_case_id", "source_topic", "text", "quality_hint", "baseline_confidence"]],
            width="stretch",
            hide_index=True,
        )
        labels = {
            row.source_case_id: f"{row.source_case_id} · {row.source_topic} · {str(row.text)[:75]}"
            for row in page_rows.itertuples()
        }
        case_id = st.selectbox("Ứng viên cần xử lý", list(labels), format_func=lambda value: labels[value])
        selected = queue.loc[queue["source_case_id"] == case_id].iloc[0]
        current = review.loc[review["source_case_id"] == case_id].iloc[0]
        st.caption(f"Chủ đề nguồn: {selected['source_topic']} · Gợi ý: {selected['quality_hint']}")
        if int(selected["group_size"]) > 1:
            related = queue[
                (queue["duplicate_group"] == selected["duplicate_group"]) & (queue["source_case_id"] != case_id)
            ]
            st.caption(f"Có {len(related)} câu gần trùng trong nhóm nguồn; chỉ chọn khi nội dung thật sự khác nhau.")
            st.dataframe(related[["source_case_id", "text"]].head(5), width="stretch", hide_index=True)
        st.caption(
            f"Baseline: {selected['baseline_top1']} · confidence {selected['baseline_confidence']:.3f} "
            f"· UNKNOWN dự đoán: {bool(selected['baseline_unknown'])}"
        )
        st.text_area("Câu nguồn", value=str(selected["text"]), disabled=True)
        api_key = _deepseek_api_key()
        if st.button("Gợi ý bằng DeepSeek", disabled=not bool(api_key), key=f"deepseek_{case_id}"):
            try:
                metadata = json.loads((paths["processed"] / "in_domain" / "metadata.json").read_text(encoding="utf-8"))
                with st.spinner("DeepSeek đang phân tích câu đang xem..."):
                    st.session_state["review_suggestion"] = (
                        case_id,
                        suggest_review(
                            str(current["text"]),
                            str(selected["source_topic"]),
                            metadata["label_names"],
                            api_key,
                        ),
                    )
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                st.error(f"Không lấy được gợi ý: {error}")
        if not api_key:
            st.caption("Đặt DEEPSEEK_API_KEY trong môi trường hoặc Streamlit secrets để bật gợi ý.")
        else:
            st.caption(
                "Nút này gửi câu đang lưu và chủ đề nguồn đến DeepSeek; hãy kiểm tra thông tin cá nhân trước khi bấm."
            )
        suggestion_state = st.session_state.get("review_suggestion")
        if suggestion_state and suggestion_state[0] == case_id:
            suggestion = suggestion_state[1]
            decisions = {"approve": "Duyệt OOD", "rewrite": "Cần viết lại", "reject": "Loại"}
            with st.container(border=True):
                st.markdown(f"**DeepSeek gợi ý:** {decisions[suggestion.decision]}")
                st.write(suggestion.reason)
                if suggestion.closest_intent:
                    st.caption(f"Intent gần nhất: {suggestion.closest_intent}")
                if suggestion.rewritten_text:
                    st.code(suggestion.rewritten_text, language=None)
                st.caption(
                    "Gợi ý chỉ để tham khảo; quyết định và nội dung CSV chỉ thay đổi khi bạn bấm nút review bên dưới."
                )
        with st.form(f"review_{case_id}"):
            edited = st.text_area("Câu hỏi hỗ trợ sau khi rà soát", value=str(current["text"]), height=130)
            note = st.text_input("Ghi chú lý do", value=str(current.get("reviewer_note", "")))
            approve = st.form_submit_button("Duyệt OOD", type="primary")
            reject = st.form_submit_button("Loại")
            rewrite = st.form_submit_button("Cần viết lại")
        if approve or reject or rewrite:
            decision = "approved" if approve else "rejected" if reject else "needs_rewrite"
            try:
                save_review_decision(review_path, case_id, edited, decision, note)
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state.pop("review_suggestion", None)
                st.rerun()

    if approved_count == int(ood["expected_total"]) and not quality_pending:
        if st.button("Kiểm tra và xuất 50 validation / 50 test"):
            try:
                _export_splits(config, paths, review, queue)
            except (ValueError, FileNotFoundError, KeyError) as error:
                st.error(f"Chưa thể xuất: {error}")
            else:
                st.success("Đã xuất hai tập OOD ngân hàng sau khi kiểm tra nguồn và câu trùng.")


main()
