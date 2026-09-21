"""Tự động duyệt ứng viên OOD bằng DeepSeek và giữ các quyết định đã có."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from intentguard.llm_review import ReviewSuggestion
from intentguard.review import review_key, save_review_decision


def load_deepseek_key(path: Path) -> str:
    """Đọc riêng DEEPSEEK_API_KEY từ file .env mà không in hoặc lưu khoá.

    Args:
        path: File .env của dự án.

    Returns:
        Giá trị API key, hoặc chuỗi rỗng nếu không tìm thấy.
    """
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        item = line.strip()
        if item.startswith("export "):
            item = item[7:].strip()
        if not item.startswith("DEEPSEEK_API_KEY="):
            continue
        return item.split("=", 1)[1].strip().strip('"\'')
    return ""


def rank_similar_candidates(queue: pd.DataFrame, review: pd.DataFrame, limit: int) -> list[str]:
    """Xếp ứng viên chưa xử lý theo độ gần từ vựng với các câu OOD đã duyệt.

    Args:
        queue: Hàng đợi có ID và câu nguồn.
        review: Các quyết định hiện có.
        limit: Số ứng viên tối đa cần trả.

    Returns:
        Danh sách ID theo độ tương tự giảm dần, chỉ dùng để ưu tiên gọi LLM.
    """
    approved = review.loc[review["approved_ood"].eq("1"), "text"].astype(str).tolist()
    decided = set(review.loc[review["review_status"].ne(""), "source_case_id"])
    remaining = queue.loc[~queue["source_case_id"].isin(decided)].copy()
    if not approved or remaining.empty:
        return []
    texts = approved + remaining["text"].astype(str).tolist()
    vectors = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1).fit_transform(texts)
    scores = (vectors[len(approved) :] @ vectors[: len(approved)].T).max(axis=1).toarray().ravel()
    remaining["_score"] = scores
    ordered = remaining.sort_values("_score", ascending=False, kind="stable")
    return ordered["source_case_id"].astype(str).head(limit).tolist()


def run_auto_review(
    queue: pd.DataFrame,
    review_path: Path,
    label_names: list[str],
    judge: Callable[[str, str, list[str]], ReviewSuggestion],
    expected_total: int,
    min_length: int,
    max_length: int,
    focus_topics: set[str],
    in_domain_texts: list[str],
    model: str,
    progress: Callable[[str], None] = print,
    candidate_ids: set[str] | None = None,
) -> dict[str, int]:
    """Duyệt hàng loạt nhóm mục tiêu và dừng khi đủ số câu hợp lệ.

    Args:
        queue: Hàng đợi ứng viên đã xếp ưu tiên.
        review_path: CSV quyết định hiện có, được lưu sau từng câu.
        label_names: Danh sách intent đã học.
        judge: Hàm gọi LLM cho một câu và trả gợi ý.
        expected_total: Số câu OOD cần đạt.
        min_length: Độ dài câu tối thiểu.
        max_length: Độ dài câu tối đa.
        focus_topics: Bốn chủ đề được phép đưa vào tập OOD.
        in_domain_texts: Các câu Banking77-VN để chặn trùng chính xác.
        model: Model DeepSeek dùng cho ghi nhận nguồn nhãn.
        progress: Hàm ghi tiến độ không chứa nội dung câu hay API key.
        candidate_ids: Nếu có, chỉ xét các ID được sàng lọc cục bộ.

    Returns:
        Thống kê số câu được xử lý, duyệt, loại và còn thiếu.

    Raises:
        RuntimeError: Khi API lỗi; các quyết định đã xử lý vẫn được giữ để chạy tiếp.
    """
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    if review["source_case_id"].duplicated().any():
        raise ValueError("File review có source_case_id trùng")
    approved = int(review["approved_ood"].eq("1").sum())
    if approved > expected_total:
        raise ValueError("Số câu đã duyệt vượt mục tiêu")
    decided = set(review.loc[review["review_status"].ne(""), "source_case_id"])
    existing_keys = {review_key(text) for text in review.loc[review["approved_ood"].eq("1"), "text"]}
    domain_keys = {review_key(text) for text in in_domain_texts}
    approved_groups = set(queue.loc[queue["source_case_id"].isin(
        review.loc[review["approved_ood"].eq("1"), "source_case_id"]
    ), "duplicate_group"])
    counts = {"processed": 0, "approved": approved, "rejected": 0, "remaining": expected_total - approved}
    for row in queue.itertuples():
        if counts["approved"] >= expected_total:
            break
        if (
            row.source_case_id in decided
            or row.source_topic not in focus_topics
            or (candidate_ids is not None and row.source_case_id not in candidate_ids)
        ):
            continue
        source_text = str(row.text).strip()
        contains_private_data = bool(
            re.search(r"\b\d{8,}\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", source_text)
        )
        if not source_text or len(source_text) > 1000 or contains_private_data:
            save_review_decision(
                review_path, row.source_case_id, source_text, "rejected", "auto: nguồn rỗng/quá dài/có định danh"
            )
            counts["rejected"] += 1
            counts["processed"] += 1
            continue
        try:
            suggestion = judge(source_text, str(row.source_topic), label_names)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"Dừng tại {row.source_case_id} sau {counts['processed']} câu; có thể chạy lại: {error}"
            ) from error
        candidate = suggestion.rewritten_text if suggestion.decision == "rewrite" else source_text
        candidate = (candidate or "").strip()
        key = review_key(candidate)
        eligible = (
            suggestion.decision in {"approve", "rewrite"}
            and suggestion.in_scope
            and suggestion.support_request
            and (suggestion.decision != "rewrite" or suggestion.faithful_rewrite)
            and min_length <= len(candidate) <= max_length
            and key not in existing_keys
            and key not in domain_keys
            and row.duplicate_group not in approved_groups
        )
        status = "approved" if eligible else "rejected"
        reason = suggestion.reason.replace("\n", " ").strip()[:180]
        note = f"auto:{model}:{suggestion.decision}; {reason}"
        if suggestion.decision in {"approve", "rewrite"} and not eligible:
            note += "; loại bởi kiểm tra phạm vi/yêu cầu/độ dài/trùng lặp"
        save_review_decision(review_path, row.source_case_id, candidate if eligible else source_text, status, note)
        if eligible:
            counts["approved"] += 1
            existing_keys.add(key)
            approved_groups.add(row.duplicate_group)
        else:
            counts["rejected"] += 1
        counts["processed"] += 1
        counts["remaining"] = expected_total - counts["approved"]
        progress(
            f"Đã xử lý {counts['processed']} | OOD {counts['approved']}/{expected_total} | loại {counts['rejected']}"
        )
    return counts
