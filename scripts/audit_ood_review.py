"""Rút nhãn OOD sai rõ ràng và đánh dấu tập ngân hàng cần kiểm tra lại."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.config import load_config

# Các ID được đối chiếu trực tiếp với nội dung nguồn và câu đã duyệt ngày 2026-09-20.
# Chỉ liệt kê trường hợp sai rõ ràng; các câu còn lại vẫn cần kiểm tra độc lập.
PRAISE_OR_EXPERIENCE = set(
    """train:58 train:59 train:80 train:156 train:159 train:160 train:200 train:206
    train:312 train:313 train:330 train:540 train:585 train:772 train:775 train:901
    train:904 train:923 train:956 train:1081 train:1091 train:1092 train:1096
    train:1098 train:1140 train:1160 train:1247 train:1248 train:1249 train:1322
    train:1344 train:1364 train:1374 train:1396 train:1429 train:1445 train:1696
    test:12 test:13 test:158 test:261""".split()
)
OUTSIDE_FOUR_TOPICS = set(
    """train:108 train:763 train:847 train:849 train:985 train:996 train:1107
    train:1254 train:1320 train:1333 train:1970 test:7 test:71""".split()
)
KNOWN_INTENT_OR_MIXED = set(
    """train:139 train:930 train:1163 train:1311 train:1440 test:38""".split()
)
ADDITIONAL_REJECT = set(
    """train:31 train:109 train:145 train:773 train:933 train:1376 train:1571
    train:1956 test:64""".split()
)
NEEDS_REWRITE = set(
    """train:46 train:47 train:105 train:110 train:112 train:116 train:118
    train:133 train:148 train:221 train:815 train:820 train:823 train:841
    train:873 train:880 train:926 train:1210 train:1300 train:1369 test:16""".split()
)


def main() -> int:
    """Sao lưu review, rút các nhãn sai rõ ràng và khoá metric OOD ngân hàng."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config(ROOT / "configs/default.yaml")
    ood = config["data"]["ood"]
    review_path = ROOT / ood["bank_review_file"]
    quality_path = ROOT / ood["bank_quality_file"]
    backup = review_path.with_name("ood_bank_review_before_quality_audit.csv")
    if not backup.exists():
        shutil.copy2(review_path, backup)
    frame = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    groups = {
        "lời khen/chia sẻ trải nghiệm, không có yêu cầu hỗ trợ": PRAISE_OR_EXPERIENCE,
        "ngoài bốn chủ đề OOD mục tiêu hoặc yêu cầu quá chung": OUTSIDE_FOUR_TOPICS,
        "thuộc intent cũ hoặc trộn nhiều vấn đề": KNOWN_INTENT_OR_MIXED,
        "nhận xét mơ hồ hoặc không nêu nhu cầu hỗ trợ cụ thể": ADDITIONAL_REJECT,
    }
    all_ids = set().union(*groups.values())
    if len(all_ids) != sum(len(ids) for ids in groups.values()):
        raise ValueError("Một source_case_id nằm trong nhiều nhóm kiểm tra")
    if not all_ids.issubset(set(frame["source_case_id"])):
        raise ValueError("Danh sách audit có ID không tồn tại trong file review")
    if NEEDS_REWRITE & all_ids:
        raise ValueError("Một ID vừa bị loại vừa cần viết lại")
    changes = 0
    for reason, ids in groups.items():
        mask = frame["source_case_id"].isin(ids) & frame["approved_ood"].eq("1")
        changes += int(mask.sum())
        frame.loc[mask, "approved_ood"] = "0"
        frame.loc[mask, "review_status"] = "rejected"
        frame.loc[mask, "reviewer_note"] = frame.loc[mask, "reviewer_note"].map(
            lambda note, category=reason: f"{note} | quality_audit: {category}".strip(" |")[:300]
        )
    rewrite_mask = frame["source_case_id"].isin(NEEDS_REWRITE) & frame["approved_ood"].eq("1")
    rewrite_count = int(rewrite_mask.sum())
    frame.loc[rewrite_mask, "approved_ood"] = ""
    frame.loc[rewrite_mask, "review_status"] = "needs_rewrite"
    frame.loc[rewrite_mask, "reviewer_note"] = frame.loc[rewrite_mask, "reviewer_note"].map(
        lambda note: f"{note} | quality_audit: cần sửa chính tả/diễn đạt và giữ nguyên ý gốc".strip(" |")[:300]
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=review_path.parent, delete=False, encoding="utf-8", newline=""
    ) as temporary:
        frame.to_csv(temporary, index=False)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, review_path)
    approved_remaining = int(frame["approved_ood"].eq("1").sum())
    audit_notes = frame["reviewer_note"].str.contains("quality_audit:", regex=False)
    status = {
        "status": "pending_audit",
        "reason": "Phát hiện lời khen, trải nghiệm và vấn đề ngoài phạm vi trong nhãn DeepSeek approved",
        "clear_rejections": int((audit_notes & frame["approved_ood"].eq("0")).sum()),
        "needs_rewrite": int((audit_notes & frame["review_status"].eq("needs_rewrite")).sum()),
        "approved_remaining_unverified": approved_remaining,
        "previous_bank_metrics_valid": False,
    }
    quality_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Lượt này loại {changes}, chuyển {rewrite_count} câu sang cần viết lại; "
        f"còn {approved_remaining} câu tạm duyệt, chưa xác minh độc lập."
    )
    print(f"Trạng thái chất lượng: {quality_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
