"""Sàng lọc ứng viên OOD và giữ các quyết định review có thể tái lập."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def review_key(value: Any) -> str:
    """Tạo khóa so trùng Unicode từ một câu.

    Args:
        value: Câu hoặc giá trị cần chuẩn hoá.

    Returns:
        Khóa chữ thường không có dấu câu, hoặc chuỗi rỗng nếu câu rỗng.
    """
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).strip()


def make_review_queue(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    probabilities: np.ndarray | None = None,
    label_names: list[str] | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Xếp ưu tiên review theo chủ đề và dạng câu, không tự gán nhãn OOD.

    Args:
        candidates: Bảng ứng viên có text, source_split, source_row và source_topic.
        config: Cấu hình ``data.ood.review`` chứa nhóm chủ đề và giới hạn độ dài.
        probabilities: Xác suất baseline để hiển thị tham khảo, nếu đã có.
        label_names: Tên các intent ứng với cột xác suất.
        threshold: Ngưỡng UNKNOWN của baseline để hiển thị tham khảo.

    Returns:
        Hàng đợi có ID nguồn, độ ưu tiên và các tín hiệu hỗ trợ review.
    """
    required = {"text", "source_split", "source_row", "source_topic"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"Candidate thiếu cột: {sorted(required - set(candidates.columns))}")
    result = candidates.copy().reset_index(drop=True)
    result["source_case_id"] = result["source_split"].astype(str) + ":" + result["source_row"].astype(str)
    if result["source_case_id"].duplicated().any():
        raise ValueError("source_case_id của candidate phải duy nhất")
    result["duplicate_group"] = duplicate_groups(result["text"].astype(str).tolist())
    result["group_size"] = result["duplicate_group"].map(result["duplicate_group"].value_counts())
    focus = set(config["focus_topics"])
    secondary = set(config["secondary_topics"])
    result["priority"] = np.select(
        [result["source_topic"].isin(focus), result["source_topic"].isin(secondary)],
        ["ưu tiên", "mở rộng"],
        default="khác",
    )
    result["text_length"] = result["text"].astype(str).str.len()
    cues = [str(cue).casefold() for cue in config["request_cues"]]
    result["request_like"] = result["text"].astype(str).map(
        lambda value: "?" in value or any(cue in value.casefold() for cue in cues)
    )
    minimum = int(config["min_length"])
    maximum = int(config["max_length"])
    result["quality_hint"] = np.select(
        [result["text_length"] < minimum, result["text_length"] > maximum, ~result["request_like"]],
        ["quá ngắn", "quá dài", "cần kiểm tra dạng yêu cầu"],
        default="có dấu hiệu yêu cầu",
    )
    if probabilities is not None:
        if len(probabilities) != len(result) or label_names is None or threshold is None:
            raise ValueError("Xác suất baseline cần khớp candidate, label_names và threshold")
        top = probabilities.argmax(axis=1)
        result["baseline_top1"] = [label_names[index] for index in top]
        result["baseline_confidence"] = probabilities.max(axis=1)
        result["baseline_unknown"] = result["baseline_confidence"] < threshold
    else:
        result["baseline_top1"] = ""
        result["baseline_confidence"] = np.nan
        result["baseline_unknown"] = False
    order = {"ưu tiên": 0, "mở rộng": 1, "khác": 2}
    result["_rank"] = result["priority"].map(order)
    result = result.sort_values(
        ["_rank", "request_like", "text_length", "source_case_id"],
        ascending=[True, False, True, True],
        kind="stable",
    ).drop(columns="_rank")
    return result.reset_index(drop=True)


def duplicate_groups(
    texts: list[str], source_texts: list[str] | None = None, similarity: float = 0.92
) -> list[int]:
    """Nhóm câu trùng hoặc gần trùng theo bản review và câu nguồn.

    Args:
        texts: Các câu đã review.
        source_texts: Câu nguồn theo cùng thứ tự, nếu có.
        similarity: Ngưỡng cosine ký tự để gom câu dài gần trùng.

    Returns:
        ID nhóm theo vị trí; các câu cùng nhóm không được tách validation/test.
    """
    if source_texts is not None and len(source_texts) != len(texts):
        raise ValueError("Số câu nguồn và câu review không khớp")
    parent = list(range(len(texts)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        parent[root(right)] = root(left)

    for values in [texts, source_texts] if source_texts is not None else [texts]:
        keys = [review_key(value) for value in values]
        seen: dict[str, int] = {}
        for index, key in enumerate(keys):
            if key in seen:
                join(index, seen[key])
            else:
                seen[key] = index
        eligible = [index for index, key in enumerate(keys) if len(key) >= 20]
        if len(eligible) < 2:
            continue
        vectors = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform([keys[i] for i in eligible])
        scores = cosine_similarity(vectors)
        for row in range(len(eligible)):
            for column in np.flatnonzero(scores[row, row + 1 :] >= similarity) + row + 1:
                join(eligible[row], eligible[int(column)])
    return [root(index) for index in range(len(texts))]


def split_review_cases(
    approved: pd.DataFrame,
    source_texts: dict[str, str],
    per_split: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Tách đúng 50/50 theo nhóm câu gần trùng, không dùng confidence để chọn mẫu.

    Args:
        approved: Các dòng đã được người review duyệt, có text và source_case_id.
        source_texts: Mapping ID nguồn sang câu gốc.
        per_split: Số mẫu trong mỗi split.
        seed: Seed quyết định thứ tự nhóm.

    Returns:
        Hai danh sách source_case_id validation và test.

    Raises:
        ValueError: Khi không thể chia đủ số lượng mà vẫn giữ nguyên nhóm.
    """
    if len(approved) != 2 * per_split or approved["source_case_id"].duplicated().any():
        raise ValueError("Số mẫu approved hoặc source_case_id không hợp lệ")
    ids = approved["source_case_id"].astype(str).tolist()
    originals = [source_texts[case_id] for case_id in ids]
    groups = duplicate_groups(approved["text"].astype(str).tolist(), originals)
    members: dict[int, list[str]] = {}
    for case_id, group in zip(ids, groups, strict=True):
        members.setdefault(group, []).append(case_id)
    ordered = sorted(
        members.values(),
        key=lambda group: hashlib.sha256(f"{seed}:{min(group)}".encode()).hexdigest(),
    )
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, group in enumerate(ordered):
        for total, selected in list(reachable.items()):
            updated = total + len(group)
            if updated <= per_split and updated not in reachable:
                reachable[updated] = (*selected, index)
    if per_split not in reachable:
        raise ValueError("Không thể tách 50/50 mà không chia đôi nhóm câu gần trùng")
    chosen = set(reachable[per_split])
    validation = sorted(case_id for index, group in enumerate(ordered) if index in chosen for case_id in group)
    test = sorted(case_id for index, group in enumerate(ordered) if index not in chosen for case_id in group)
    return validation, test


def validate_approved_rows(
    review: pd.DataFrame,
    candidates: pd.DataFrame,
    in_domain_texts: list[str],
    expected_total: int,
    min_length: int,
    max_length: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Kiểm tra số lượng, nguồn gốc, nội dung và trùng lặp của câu OOD đã duyệt.

    Args:
        review: CSV quyết định của người review.
        candidates: Candidate nguồn từ UTS2017_Bank.
        in_domain_texts: Câu từ mọi split Banking77-VN.
        expected_total: Số câu OOD cần dùng.
        min_length: Độ dài tối thiểu của câu đã sửa.
        max_length: Độ dài tối đa của câu đã sửa.

    Returns:
        Các dòng approved và mapping ID nguồn sang câu gốc.

    Raises:
        ValueError: Khi phát hiện thiếu nguồn, trùng câu hoặc lẫn in-domain.
    """
    required = {"text", "topic", "source_case_id", "approved_ood"}
    if not required.issubset(review.columns):
        raise ValueError(f"File review thiếu cột: {sorted(required - set(review.columns))}")
    approved = review[review["approved_ood"].astype(str).str.lower().isin({"1", "true", "yes", "y"})].copy()
    if len(approved) != expected_total:
        raise ValueError(f"Cần đúng {expected_total} câu OOD đã approved, nhận {len(approved)}")
    if approved["source_case_id"].duplicated().any():
        raise ValueError("source_case_id được duyệt bị trùng")
    source = candidates.copy()
    source["source_case_id"] = source["source_split"].astype(str) + ":" + source["source_row"].astype(str)
    if source["source_case_id"].duplicated().any():
        raise ValueError("source_case_id trong candidate bị trùng")
    source = source.set_index("source_case_id")
    ids = approved["source_case_id"].astype(str)
    unknown = set(ids) - set(source.index)
    if unknown:
        raise ValueError(f"source_case_id không có trong candidate: {sorted(unknown)[:3]}")
    actual_topics = source.loc[ids, "source_topic"].astype(str).to_numpy()
    if not np.array_equal(actual_topics, approved["topic"].astype(str).to_numpy()):
        raise ValueError("topic của dòng được duyệt khác topic nguồn")
    lengths = approved["text"].astype(str).str.len()
    if ((lengths < min_length) | (lengths > max_length)).any():
        raise ValueError(f"Câu được duyệt phải dài từ {min_length} đến {max_length} ký tự")
    keys = approved["text"].map(review_key)
    if keys.eq("").any() or keys.duplicated().any():
        raise ValueError("Câu OOD được duyệt rỗng hoặc trùng nhau")
    if set(keys) & {review_key(text) for text in in_domain_texts}:
        raise ValueError("Câu OOD được duyệt trùng câu Banking77-VN")
    return approved, source["text"].astype(str).to_dict()


def save_review_decision(path: str | Path, case_id: str, text: str, status: str, note: str) -> None:
    """Lưu một quyết định vào CSV hiện có bằng cách thay file nguyên tử.

    Args:
        path: File review CSV hiện có.
        case_id: ID ứng viên duy nhất.
        text: Câu đã sửa hoặc câu nguồn.
        status: ``approved``, ``rejected`` hoặc ``needs_rewrite``.
        note: Ghi chú của người review.

    Raises:
        ValueError: Khi ID, trạng thái hoặc câu duyệt không hợp lệ.
    """
    import os
    import tempfile

    if status not in {"approved", "rejected", "needs_rewrite"}:
        raise ValueError("Trạng thái review không hợp lệ")
    if status == "approved" and not review_key(text):
        raise ValueError("Câu được duyệt không được rỗng")
    destination = Path(path)
    frame = pd.read_csv(destination, dtype=str, keep_default_na=False)
    mask = frame["source_case_id"].eq(case_id)
    if int(mask.sum()) != 1:
        raise ValueError(f"Không tìm thấy duy nhất source_case_id={case_id}")
    frame.loc[mask, "text"] = text.strip()
    frame.loc[mask, "reviewer_note"] = note.strip()
    frame.loc[mask, "review_status"] = status
    frame.loc[mask, "approved_ood"] = "1" if status == "approved" else "0" if status == "rejected" else ""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=destination.parent, delete=False, encoding="utf-8", newline=""
    ) as temporary:
        frame.to_csv(temporary, index=False)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination)
