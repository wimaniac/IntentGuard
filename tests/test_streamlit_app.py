"""Kiểm thử headless cho ứng dụng Streamlit của dự án."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_test_form_without_crashing() -> None:
    """Demo chính hiển thị form trước khi nạp model và không phát sinh exception."""
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)
    assert not app.exception
    assert app.title[0].value == "Kiểm thử IntentGuard"
    assert len(app.text_area) == 1
