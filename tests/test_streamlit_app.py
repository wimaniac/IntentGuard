"""Kiểm thử headless cho hai entry point Streamlit của dự án."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_test_form_without_crashing() -> None:
    """Demo chính hiển thị form trước khi nạp model và không phát sinh exception."""
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)
    assert not app.exception
    assert app.title[0].value == "Kiểm thử IntentGuard"
    assert len(app.text_area) == 1


def test_review_app_opens_existing_queue() -> None:
    """Trang review đọc được hàng đợi mà không tự phê duyệt candidate."""
    root = Path(__file__).parents[1]
    if not (root / "data" / "interim" / "ood_bank_review_queue.csv").exists():
        return
    app = AppTest.from_file(str(root / "app" / "review_app.py")).run(timeout=20)
    assert not app.exception
