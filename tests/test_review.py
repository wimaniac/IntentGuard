"""Kiểm thử triage và bảo vệ tập OOD do người review xác nhận."""

import pandas as pd
import pytest

from intentguard.review import (
    make_review_queue,
    save_review_decision,
    split_review_cases,
    validate_approved_rows,
)


def test_triage_prioritizes_topic_without_auto_approval() -> None:
    """Tín hiệu chủ đề xếp hàng đợi nhưng không tạo nhãn OOD."""
    candidates = pd.DataFrame(
        {
            "text": ["Dịch vụ tốt", "Tôi muốn hỏi điều kiện vay vốn?"],
            "source_split": ["train", "train"],
            "source_row": [0, 1],
            "source_topic": ["CUSTOMER_SUPPORT", "LOAN"],
        }
    )
    config = {
        "focus_topics": ["LOAN"],
        "secondary_topics": ["OTHER"],
        "request_cues": ["muốn hỏi"],
        "min_length": 20,
        "max_length": 300,
    }
    queue = make_review_queue(candidates, config)
    assert queue.iloc[0]["source_case_id"] == "train:1"
    assert queue.iloc[0]["priority"] == "ưu tiên"
    assert "approved_ood" not in queue.columns


def test_approved_rows_reject_overlap_with_in_domain() -> None:
    """Câu được duyệt không được trùng Banking77 sau chuẩn hoá."""
    candidates = pd.DataFrame(
        {"text": ["Câu gốc"], "source_split": ["train"], "source_row": [0], "source_topic": ["LOAN"]}
    )
    review = pd.DataFrame(
        {
            "text": ["Tôi muốn hỏi về khoản vay mới."],
            "topic": ["LOAN"],
            "source_case_id": ["train:0"],
            "approved_ood": ["1"],
        }
    )
    with pytest.raises(ValueError, match="trùng câu Banking77"):
        validate_approved_rows(review, candidates, ["tôi muốn hỏi về khoản vay mới!"], 1, 20, 300)


def test_split_keeps_near_duplicates_together() -> None:
    """Hai cách viết gần giống của cùng vấn đề nằm chung một split."""
    approved = pd.DataFrame(
        {
            "source_case_id": ["train:0", "train:1", "train:2", "train:3"],
            "text": [
                "Tôi cần hỏi điều kiện vay mua nhà hiện nay.",
                "Tôi cần hỏi điều kiện vay mua nhà hiện nay!",
                "Lãi suất tiết kiệm kỳ hạn một năm là bao nhiêu?",
                "Ngân hàng có chương trình ưu đãi cho khoản vay mới không?",
            ],
        }
    )
    originals = dict(zip(approved["source_case_id"], approved["text"], strict=True))
    validation, test = split_review_cases(approved, originals, per_split=2, seed=42)
    assert len(validation) == len(test) == 2
    assert ({"train:0", "train:1"} <= set(validation)) or ({"train:0", "train:1"} <= set(test))


def test_save_decision_only_updates_selected_case(tmp_path) -> None:
    """Lưu review giữ các dòng khác và không tự duyệt chúng."""
    path = tmp_path / "review.csv"
    pd.DataFrame(
        {
            "text": ["câu A", "câu B"],
            "topic": ["LOAN", "SAVING"],
            "source_case_id": ["train:0", "train:1"],
            "approved_ood": ["", ""],
            "reviewer_note": ["", ""],
        }
    ).to_csv(path, index=False)
    save_review_decision(path, "train:0", "Tôi muốn hỏi điều kiện vay mua nhà?", "approved", "Đã kiểm tra")
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert saved.loc[0, "approved_ood"] == "1"
    assert saved.loc[1, "approved_ood"] == ""
