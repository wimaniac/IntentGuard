"""Giao diện Streamlit đơn giản để thử phân loại intent và phát hiện UNKNOWN."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.config import load_config, resolve_project_paths
from intentguard.inference import IntentGuardPredictor
from intentguard.schemas import InferenceResult

EXAMPLES = {
    "Phí hai lần": "Tôi bị tính phí hai lần cho cùng một giao dịch.",
    "Thẻ bị nuốt": "ATM đã nuốt thẻ của tôi, tôi phải làm gì?",
    "Ngoài phạm vi": "Tôi muốn vay mua nhà thì cần điều kiện gì?",
}


@st.cache_resource(show_spinner=False)
def load_predictor(artifact_dir: str, backend: str) -> IntentGuardPredictor:
    """Nạp và cache model cho từng backend trong vòng đời ứng dụng.

    Args:
        artifact_dir: Đường dẫn đến thư mục artifact.
        backend: Backend ``baseline`` hoặc ``deep``.

    Returns:
        Predictor đã nạp model, calibration và retrieval index.
    """
    return IntentGuardPredictor(artifact_dir, backend)


def _select_example(text: str) -> None:
    st.session_state["message_text"] = text


def _available_backends(artifact_dir: Path) -> list[str]:
    """Liệt kê backend có đủ artifact tối thiểu để chạy trong môi trường hiện tại."""
    available: list[str] = []
    baseline_files = ("baseline.json", "baseline.joblib", "calibration_baseline.json")
    if all((artifact_dir / name).exists() for name in baseline_files):
        available.append("baseline")
    if all(
        (artifact_dir / name).exists()
        for name in ("deep.json", "deep_classifier.pt", "calibration_deep.json", "encoder", "tokenizer")
    ):
        available.append("deep")
    return available


def _render_top_predictions(result: InferenceResult) -> None:
    predictions = pd.DataFrame(
        [
            {
                "Hạng": rank,
                "Intent": item.intent,
                "Confidence": item.probability,
            }
            for rank, item in enumerate(result.top_3, start=1)
        ]
    )
    st.subheader("Ba nhãn có khả năng cao nhất", icon=":material/leaderboard:")
    st.dataframe(
        predictions,
        hide_index=True,
        width="stretch",
        column_config={
            "Hạng": st.column_config.NumberColumn(width="small"),
            "Intent": st.column_config.TextColumn(width="large"),
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
        },
    )


def _render_similar_cases(result: InferenceResult) -> None:
    st.subheader("Câu tương tự trong dữ liệu train", icon=":material/find_in_page:")
    st.caption("Các câu này giúp giải thích vì sao mô hình đưa ra dự đoán.")
    if not result.similar_cases:
        st.info("Artifact hiện tại chưa có retrieval index.", icon=":material/info:")
        return
    similar = pd.DataFrame(
        [
            {
                "Câu tương tự": item.text,
                "Intent": item.intent,
                "Độ tương tự": item.similarity,
            }
            for item in result.similar_cases
        ]
    )
    st.dataframe(
        similar,
        hide_index=True,
        width="stretch",
        column_config={
            "Câu tương tự": st.column_config.TextColumn(width="large"),
            "Intent": st.column_config.TextColumn(width="medium"),
            "Độ tương tự": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
        },
    )


def _render_result(result: InferenceResult, threshold: float, backend: str) -> None:
    """Hiển thị quyết định, top-3 và các câu tương tự.

    Args:
        result: Kết quả suy luận đã được hiệu chỉnh confidence.
        threshold: Ngưỡng dùng để quyết định UNKNOWN.
        backend: Tên backend tạo ra kết quả.
    """
    with st.container(border=True):
        if result.is_unknown:
            st.warning(
                "Câu hỏi nằm ngoài phạm vi 77 intent đã học. Top-3 bên dưới chỉ dùng để tham khảo.",
                icon=":material/help:",
            )
        else:
            st.success("Mô hình nhận diện được intent trong phạm vi.", icon=":material/check_circle:")

        decision, confidence, cutoff = st.columns(3)
        decision.metric("Kết quả", result.intent)
        confidence.metric("Confidence", f"{result.confidence:.2%}")
        cutoff.metric("Ngưỡng UNKNOWN", f"{threshold:.2%}")
        st.caption(f"Backend: `{backend}` · Confidence đã qua temperature scaling")

    _render_top_predictions(result)
    _render_similar_cases(result)


def main() -> None:
    """Khởi chạy ứng dụng kiểm thử IntentGuard."""
    st.set_page_config(page_title="IntentGuard", page_icon=":material/shield:", layout="centered")
    st.title("Kiểm thử IntentGuard", icon=":material/shield:")
    st.caption("Phân loại yêu cầu ngân hàng, trả top-3 và cảnh báo câu hỏi ngoài phạm vi.")

    config = load_config(ROOT / "configs/default.yaml")
    paths = resolve_project_paths(config, ROOT)
    artifact_dir = paths["artifacts"]
    available_backends = _available_backends(artifact_dir)

    if not available_backends:
        st.error("Chưa có artifact model để chạy demo.", icon=":material/error:")
        return

    with st.sidebar:
        st.header("Cấu hình", icon=":material/tune:")
        default_backend = config["inference"]["backend"]
        if default_backend not in available_backends:
            default_backend = available_backends[0]
        backend = st.segmented_control(
            "Backend",
            options=available_backends,
            default=default_backend,
            key="backend",
        ) or "baseline"
        if backend == "baseline":
            st.badge("Khuyến nghị", icon=":material/check:", color="green")
            st.caption("TF-IDF + Logistic Regression · macro-F1 test 0,8939")
        else:
            st.badge("DL đa ngôn ngữ", icon=":material/language:", color="blue")
            st.caption("Multilingual MiniLM + classifier · macro-F1 test 0,8832")
        if "deep" not in available_backends:
            st.caption("Bản cloud dùng baseline để giữ kích thước triển khai gọn.")
        st.caption("Nếu confidence thấp hơn threshold, hệ thống trả về `UNKNOWN`.")

    st.subheader("Thử nhanh", icon=":material/bolt:")
    with st.container(horizontal=True):
        for label, example in EXAMPLES.items():
            st.button(
                label,
                key=f"example_{label}",
                on_click=_select_example,
                args=(example,),
                icon=":material/chat:",
            )

    with st.form("inference_form", border=True):
        text = st.text_area(
            "Tin nhắn khách hàng",
            key="message_text",
            height=130,
            placeholder="Ví dụ: Tôi bị tính phí hai lần cho cùng một giao dịch.",
        )
        submitted = st.form_submit_button(
            "Phân tích",
            type="primary",
            icon=":material/search:",
            width="stretch",
        )

    result_slot = st.container()
    if not submitted:
        st.caption("Chọn một ví dụ hoặc nhập câu hỏi, sau đó bấm **Phân tích**.")
        return
    if not text.strip():
        st.warning("Hãy nhập một câu hỏi trước khi phân tích.", icon=":material/edit_note:")
        return
    if not (artifact_dir / f"{backend}.json").exists():
        st.error(f"Chưa có artifact cho backend `{backend}`.", icon=":material/error:")
        return

    try:
        with result_slot.skeleton(height=220):
            predictor = load_predictor(str(artifact_dir), backend)
            result = predictor.predict(text.strip())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(f"Không thể chạy suy luận: {error}", icon=":material/error:")
        return
    _render_result(result, float(predictor.calibration["threshold"]), backend)


main()
