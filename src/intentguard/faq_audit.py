"""Kiểm tra bản rà nhãn FAQ với đúng câu gốc và câu dùng để đánh giá."""

from __future__ import annotations

import pandas as pd


def validate_faq_label_audit(faq: pd.DataFrame, audit: pd.DataFrame) -> dict[str, int]:
    """Xác nhận đủ 100 quyết định OOD và bản audit khớp dữ liệu hiện hành.

    Args:
        faq: FAQ đã chuẩn bị, gồm hai split với cột `split`.
        audit: Bảng quyết định rà nhãn theo từng `source_case_id`.

    Returns:
        Số câu đã duyệt, câu được viết gọn và câu gần trùng ngữ nghĩa.

    Raises:
        ValueError: Khi bản audit thiếu câu, sai nội dung, sai cờ hoặc tách nhóm gần trùng.
    """
    required_faq = {"source_case_id", "text_original", "text_reviewed", "topic", "split"}
    required_audit = required_faq - {"split"} | {
        "ood_label", "topic_valid", "bank_request_valid", "reviewed_single_issue",
        "original_single_issue", "review_action", "semantic_duplicate_of", "reviewer_type",
    }
    if not required_faq.issubset(faq.columns) or not required_audit.issubset(audit.columns):
        raise ValueError("FAQ hoặc bảng audit thiếu cột bắt buộc")
    if len(faq) != 100 or len(audit) != 100:
        raise ValueError("Bản audit phải khớp đúng 100 FAQ hiện hành")
    if faq.source_case_id.duplicated().any() or audit.source_case_id.duplicated().any():
        raise ValueError("FAQ hoặc bản audit có source_case_id trùng")
    joined = faq.merge(audit, on="source_case_id", how="outer", suffixes=("_faq", "_audit"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Bản audit thiếu hoặc thừa ID FAQ")
    for column in ("text_original", "text_reviewed", "topic"):
        if not joined[f"{column}_faq"].eq(joined[f"{column}_audit"]).all():
            raise ValueError(f"Bản audit không khớp {column} hiện hành")
    for column in ("ood_label", "topic_valid", "bank_request_valid", "reviewed_single_issue"):
        if not joined[column].astype(str).eq("1").all():
            raise ValueError(f"Bản audit còn câu chưa đạt {column}")
    if not joined.reviewer_type.eq("codex_ai").all():
        raise ValueError("Bản audit thiếu nguồn gốc người rà nhãn AI")
    rewritten = joined.review_action.eq("focus_one_issue")
    if not joined.loc[rewritten, "original_single_issue"].astype(str).eq("0").all():
        raise ValueError("Câu viết gọn không được đánh dấu nhiều vấn đề trong bản gốc")
    if not joined.loc[~rewritten, "original_single_issue"].astype(str).eq("1").all():
        raise ValueError("Câu giữ nguyên bị đánh dấu nhiều vấn đề")
    by_id = faq.set_index("source_case_id")
    duplicates = audit[audit.semantic_duplicate_of.ne("")]
    for row in duplicates.itertuples():
        if row.semantic_duplicate_of not in by_id.index:
            raise ValueError("Bản audit tham chiếu nhóm gần trùng không tồn tại")
        if by_id.at[row.source_case_id, "split"] != by_id.at[row.semantic_duplicate_of, "split"]:
            raise ValueError("Hai câu gần trùng ngữ nghĩa bị tách validation/test")
    return {
        "approved_ood": int(len(joined)),
        "focused_rewrites": int(rewritten.sum()),
        "semantic_duplicate_followups": int(len(duplicates)),
    }
