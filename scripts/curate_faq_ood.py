"""Chọn biến thể FAQ đạt tiêu chí cục bộ và chia theo nhóm câu nguồn."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.review import duplicate_groups, review_key

INPUT = ROOT / "data/interim/faq_ood_100_candidates.csv"
RAW = ROOT / "data/interim/faq_generation_deepseek.jsonl"
API_REPAIRS = ROOT / "data/interim/faq_single_issue_rewrites.json"
CURATED = ROOT / "data/processed/faq_ood_curated"
AUDIT = ROOT / "reports/faq_ood_local_audit.csv"
IN_DOMAIN = ROOT / "data/processed/in_domain"
REPLACEMENTS = {
    "faq:0007:g3": (5, "Câu cũ thêm yêu cầu về thủ tục trả nợ, không có trong câu nguồn."),
    "faq:0010": (4, "Câu nguồn trộn mở tài khoản thường với gửi tiết kiệm; chọn câu chỉ hỏi tiền gửi."),
    "faq:0010:g2": (5, "Câu cũ dễ hiểu thành hỏi số dư tài khoản thay vì mức gửi tiết kiệm tối thiểu."),
    "faq:0022:g3": (6, "Câu cũ hỏi điều kiện có lãi suất tốt hơn nói chung, chưa rõ chương trình khuyến mãi."),
    "faq:0025:g3": (6, "Câu cũ đổi thời gian áp dụng thành thời gian nhận quà, lệch ý câu nguồn."),
}
VALIDATION_GROUPS = {"LOAN": 4, "SAVING": 3, "INTEREST_RATE": 2, "PROMOTION": 3}


def load_alternatives(path: Path) -> dict[str, list[str]]:
    """Đọc toàn bộ biến thể thô đã lưu trong lượt gọi DeepSeek trước.

    Args:
        path: File JSONL phản hồi sinh câu.

    Returns:
        Mapping `seed_id` sang danh sách câu theo thứ tự phản hồi.
    """
    alternatives: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        alternatives.setdefault(item["seed_id"], []).extend(item["questions"])
    return alternatives


def curate_candidates(
    candidates: pd.DataFrame,
    alternatives: dict[str, list[str]],
    api_repairs: dict[str, list[dict[str, str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thay câu lệch nguồn và câu nhiều ý bằng biến thể có truy vết.

    Args:
        candidates: 100 câu đã sinh, có `source_case_id` và `seed_id`.
        alternatives: Danh sách biến thể thô theo `seed_id`.
        api_repairs: Câu một ý DeepSeek viết sau lượt audit API, nếu có.

    Returns:
        Bộ 100 câu tạm duyệt và bảng audit gồm câu bị thay cùng câu thay thế.

    Raises:
        ValueError: Khi thiếu câu nguồn/biến thể hoặc chọn trùng ID, câu.
    """
    if len(candidates) != 100 or candidates["source_case_id"].duplicated().any():
        raise ValueError("Bộ đầu vào cần đúng 100 câu và ID duy nhất")
    curated = candidates.copy()
    curated["review_action"] = "kept"
    curated["replaced_source_case_id"] = ""
    audit_rows: list[dict[str, str]] = []
    for case_id, (alt_number, reason) in REPLACEMENTS.items():
        matching = curated.index[curated["source_case_id"] == case_id]
        if len(matching) != 1:
            raise ValueError(f"Thiếu ID cần thay: {case_id}")
        index = matching[0]
        old = curated.loc[index].to_dict()
        seed_id = str(old["seed_id"])
        options = alternatives.get(seed_id, [])
        if len(options) < alt_number:
            raise ValueError(f"Thiếu biến thể thứ {alt_number} cho {seed_id}")
        replacement = " ".join(options[alt_number - 1].split())
        audit_rows.append(
            {"source_case_id": case_id, "seed_id": seed_id, "text": str(old["text"]),
             "decision": "replaced", "reason": reason, "replacement_id": f"{seed_id}:alt{alt_number}"}
        )
        curated.at[index, "source_case_id"] = f"{seed_id}:alt{alt_number}"
        curated.at[index, "text"] = replacement
        curated.at[index, "origin"] = "deepseek_generated"
        curated.at[index, "review_action"] = "replacement"
        curated.at[index, "replaced_source_case_id"] = case_id
        curated.at[index, "model"] = "deepseek-flash"
    for seed_id, replacements in (api_repairs or {}).items():
        for number, replacement in enumerate(replacements, start=1):
            case_id = replacement["source_case_id"]
            matching = curated.index[curated["source_case_id"] == case_id]
            if len(matching) != 1 or str(curated.at[matching[0], "seed_id"]) != seed_id:
                raise ValueError(f"ID câu sửa nhiều ý không khớp nhóm: {case_id}")
            index = matching[0]
            old_text = str(curated.at[index, "text"])
            new_id = f"{seed_id}:api{number}"
            audit_rows.append(
                {"source_case_id": case_id, "seed_id": seed_id, "text": old_text,
                 "decision": "replaced", "reason": "DeepSeek audit: câu gộp nhiều nhu cầu; viết lại một ý.",
                 "replacement_id": new_id}
            )
            curated.at[index, "source_case_id"] = new_id
            curated.at[index, "text"] = " ".join(replacement["text"].split())
            curated.at[index, "origin"] = "deepseek_generated"
            curated.at[index, "review_action"] = "api_repair"
            curated.at[index, "replaced_source_case_id"] = case_id
            curated.at[index, "model"] = "deepseek-flash"
    curated["review_status"] = "provisional_ood"
    if curated["source_case_id"].duplicated().any() or curated["text"].map(review_key).duplicated().any():
        raise ValueError("Bộ sau thay thế có ID hoặc câu trùng")
    if not curated["text"].str.len().between(20, 300).all():
        raise ValueError("Bộ sau thay thế có câu ngoài giới hạn 20–300 ký tự")
    pii_pattern = r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{7,}\b|https?://"
    if curated["text"].str.contains(pii_pattern, flags=re.I, regex=True).any():
        raise ValueError("Bộ sau thay thế có email, chuỗi số dài hoặc URL")
    if not curated.groupby("seed_id").size().eq(4).all():
        raise ValueError("Mỗi nhóm nguồn phải còn đúng bốn câu")
    audit_rows.extend(
        {"source_case_id": str(row.source_case_id), "seed_id": str(row.seed_id), "text": str(row.text),
         "decision": "provisional_ood", "reason": "Rà nội dung cục bộ: đúng topic, ngoài 77 intent và đủ ý hỗ trợ.",
         "replacement_id": ""}
        for row in curated.itertuples()
    )
    return curated, pd.DataFrame(audit_rows)


def split_by_seed(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia 12/13 nhóm nguồn có cân bằng topic, tuyệt đối không tách biến thể.

    Args:
        frame: Bộ câu đã curated, mỗi `seed_id` có bốn câu.
        seed: Seed chọn nhóm validation.

    Returns:
        Hai split 48/52 câu theo `seed_id`.

    Raises:
        ValueError: Khi nhóm nguồn sai số lượng hoặc có topic không nhất quán.
    """
    group_topics = frame.groupby("seed_id")["topic"].agg(["first", "nunique", "size"])
    if not group_topics["nunique"].eq(1).all() or not group_topics["size"].eq(4).all():
        raise ValueError("Mỗi nhóm nguồn cần bốn câu cùng topic")
    rng = np.random.default_rng(seed)
    validation_ids: set[str] = set()
    for topic, count in VALIDATION_GROUPS.items():
        ids = sorted(group_topics.index[group_topics["first"] == topic])
        if len(ids) < count:
            raise ValueError(f"Không đủ nhóm {topic}")
        validation_ids.update(rng.permutation(ids)[:count].tolist())
    validation = frame.loc[frame["seed_id"].isin(validation_ids)].copy()
    test = frame.loc[~frame["seed_id"].isin(validation_ids)].copy()
    if len(validation) != 48 or len(test) != 52 or set(validation.seed_id) & set(test.seed_id):
        raise ValueError("Chia nhóm không đạt 48/52 hoặc có rò rỉ seed_id")
    return validation.reset_index(drop=True), test.reset_index(drop=True)


def main() -> int:
    """Xuất bảng audit cục bộ và hai split FAQ tạm duyệt.

    Returns:
        Mã 0 khi kiểm tra dữ liệu và ghi file thành công.
    """
    frame = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    api_repairs = json.loads(API_REPAIRS.read_text(encoding="utf-8")) if API_REPAIRS.exists() else {}
    curated, audit = curate_candidates(frame, load_alternatives(RAW), api_repairs)
    domain_keys: set[str] = set()
    for split in ("train", "model_selection", "calibration", "threshold", "test"):
        domain_keys.update(
            pd.read_parquet(IN_DOMAIN / f"{split}.parquet", columns=["text"])["text"].map(review_key)
        )
    if set(curated["text"].map(review_key)) & domain_keys:
        raise ValueError("Bộ FAQ có câu trùng chính xác với Banking77-VN")
    validation, test = split_by_seed(curated)
    groups = duplicate_groups(curated["text"].tolist())
    memberships: dict[int, set[str]] = {}
    split_names = ["validation" if row.seed_id in set(validation.seed_id) else "test" for row in curated.itertuples()]
    for group_id, split_name in zip(groups, split_names, strict=True):
        memberships.setdefault(group_id, set()).add(split_name)
    if any(len(names) > 1 for names in memberships.values()):
        raise ValueError("Có nhóm gần trùng chữ bị tách validation/test")
    CURATED.mkdir(parents=True, exist_ok=True)
    curated.to_csv(CURATED / "all.csv", index=False, encoding="utf-8-sig")
    validation.to_parquet(CURATED / "validation.parquet", index=False)
    test.to_parquet(CURATED / "test.parquet", index=False)
    audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")
    replacements = len(REPLACEMENTS) + sum(map(len, api_repairs.values()))
    print(f"Curated {len(curated)} câu; validation {len(validation)}, test {len(test)}; thay {replacements} câu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
