"""Kiểm thử duyệt tự động không ghi đè quyết định cũ và dừng đúng mục tiêu."""

import pandas as pd

from intentguard.auto_review import run_auto_review
from intentguard.llm_review import ReviewSuggestion


def test_auto_review_preserves_existing_and_stops_at_target(tmp_path) -> None:
    """Chỉ câu chưa review trong chủ đề mục tiêu mới được DeepSeek xử lý."""
    path = tmp_path / "review.csv"
    pd.DataFrame(
        {
            "source_case_id": ["a", "b", "c", "d"],
            "text": ["Câu đã duyệt về khoản vay", "Câu mới về tiết kiệm", "Câu ngoài chủ đề", "Câu chưa đến lượt"],
            "topic": ["LOAN", "SAVING", "CARD", "LOAN"],
            "approved_ood": ["1", "", "", ""],
            "review_status": ["approved", "", "", ""],
            "reviewer_note": ["người dùng", "", "", ""],
        }
    ).to_csv(path, index=False)
    queue = pd.DataFrame(
        {
            "source_case_id": ["a", "b", "c", "d"],
            "source_topic": ["LOAN", "SAVING", "CARD", "LOAN"],
            "text": ["Câu đã duyệt về khoản vay", "Câu mới về tiết kiệm", "Câu ngoài chủ đề", "Câu chưa đến lượt"],
            "duplicate_group": [0, 1, 2, 3],
        }
    )
    calls = []

    def judge(text, topic, labels):
        """Trả gợi ý giả và ghi nhận câu đã được gọi."""
        calls.append(text)
        return ReviewSuggestion("approve", "Ngoài phạm vi", None, None, True, True, False)

    result = run_auto_review(
        queue, path, ["known"], judge, 2, 20, 300, {"LOAN", "SAVING"}, [], "fake", lambda _message: None
    )
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert result["approved"] == 2
    assert calls == ["Câu mới về tiết kiệm"]
    assert saved.loc[0, "reviewer_note"] == "người dùng"
    assert saved.loc[3, "review_status"] == ""


def test_auto_review_rejects_rewrite_without_scope_and_request(tmp_path) -> None:
    """Câu khen không được duyệt chỉ vì DeepSeek trả một bản viết lại."""
    path = tmp_path / "review.csv"
    pd.DataFrame(
        {
            "source_case_id": ["a"],
            "text": ["Ưu đãi tuyệt quá, tôi rất thích."],
            "topic": ["PROMOTION"],
            "approved_ood": [""],
            "review_status": [""],
            "reviewer_note": [""],
        }
    ).to_csv(path, index=False)
    queue = pd.DataFrame(
        {
            "source_case_id": ["a"],
            "source_topic": ["PROMOTION"],
            "text": ["Ưu đãi tuyệt quá, tôi rất thích."],
            "duplicate_group": [0],
        }
    )
    result = run_auto_review(
        queue,
        path,
        ["known"],
        lambda *_args: ReviewSuggestion(
            "rewrite", "Câu khen không có yêu cầu", None, "Tôi muốn hỏi về ưu đãi ngân hàng.", False, False, False
        ),
        1,
        20,
        300,
        {"PROMOTION"},
        [],
        "fake",
        lambda _message: None,
    )
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert result["approved"] == 0
    assert saved.loc[0, "approved_ood"] == "0"
