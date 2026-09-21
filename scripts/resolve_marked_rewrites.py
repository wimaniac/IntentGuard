"""Biên tập các ứng viên OOD được đánh dấu rewrite=1 và ghi vết quyết định."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.review import review_key

REWRITES = {
    "train:47": (
        "Ngân hàng quảng cáo lãi suất 6–7%/năm, nhưng khi tôi đến hỏi thì được báo mức khác. "
        "Vì sao lãi suất thực tế khác quảng cáo?"
    ),
    "train:105": (
        "Tôi muốn vay vốn mở cửa hàng nhưng thấy thủ tục tại ngân hàng khá rườm rà. "
        "Ngân hàng có thể giải thích các bước cần thực hiện không?"
    ),
    "train:110": (
        "Tôi được biết có mức lãi suất vay 6,8%/năm, nhưng khi hỏi thì không được áp dụng. "
        "Điều kiện để hưởng mức lãi suất này là gì?"
    ),
    "train:133": (
        "Tôi là viên chức có lương ổn định nhưng được báo phải có hơn một năm làm việc theo hợp đồng mới được vay, "
        "dù tôi đã đi làm theo quyết định tuyển dụng từ ngày 1/1/2016. Điều kiện này được tính như thế nào?"
    ),
    "train:221": (
        "Tôi gửi tiền và đăng ký tại BIDV ngày 20/12, đồng thời làm thẻ Mastercard, "
        "nhưng không nhận được ưu đãi nào. Vì sao?"
    ),
    "train:820": (
        "Tôi hỏi về chương trình vay làm nhà, mua đất năm 2016 nhưng được tư vấn chỉ có vay tiêu dùng "
        "120 triệu đồng trong ba năm với lãi suất 12%/năm. Vì sao tôi chỉ được tư vấn vay tiêu dùng?"
    ),
    "train:873": "Để được vay vốn, có bắt buộc phải thế chấp sổ đỏ không?",
    "train:926": (
        "Tôi muốn thế chấp sổ đỏ để vay vốn kinh doanh nhưng gặp khó khăn khi đăng ký. "
        "Ngân hàng có thể hướng dẫn cách đăng ký khoản vay này không?"
    ),
    "train:1210": (
        "Thủ tục vay thế chấp sổ đỏ tại BIDV quá rườm rà và phức tạp. "
        "Ngân hàng có thể hướng dẫn rõ các yêu cầu không?"
    ),
    "train:1300": (
        "Tôi hỏi vay vốn nhưng được báo phải chờ đến đầu năm 2018 vì hiện ngân hàng chưa cho vay. "
        "Vì sao tôi phải chờ đến thời điểm đó?"
    ),
    "train:1369": (
        "Lãi suất được quảng cáo hấp dẫn, nhưng khi đến chi nhánh tôi được báo 10,5%/năm, chưa kể bảo hiểm. "
        "Vì sao mức lãi suất thực tế khác quảng cáo?"
    ),
}
REJECTIONS = {
    "train:327": "Trộn phí chuyển tiền (intent đã học) với ý định mua bảo hiểm; không thuộc bốn chủ đề OOD mục tiêu.",
}


def resolve_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Áp dụng quyết định cho các dòng đã gắn cờ; trả về CSV mới và bảng đối chiếu.

    Args:
        rows: Toàn bộ dòng review hiện tại, gồm cột ``rewrite``.

    Returns:
        Các dòng sau xử lý và 12 bản ghi đối chiếu nội dung nguồn với quyết định.

    Raises:
        ValueError: Khi cờ, trạng thái, ID hoặc câu sửa không khớp dữ liệu đã kiểm tra.
    """
    expected = set(REWRITES) | set(REJECTIONS)
    flagged = [row for row in rows if row.get("rewrite") == "1"]
    ids = [row["source_case_id"] for row in flagged]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"Danh sách rewrite=1 đã thay đổi: {sorted(set(ids) ^ expected)}")
    approved_keys = {review_key(row["text"]) for row in rows if row["approved_ood"] == "1"}
    records: list[dict[str, str]] = []
    for row in flagged:
        case_id = row["source_case_id"]
        if row["review_status"] != "needs_rewrite" or row["approved_ood"]:
            raise ValueError(f"Trạng thái hiện tại của {case_id} không còn là needs_rewrite")
        original = row["text"]
        if case_id in REWRITES:
            rewritten = REWRITES[case_id]
            key = review_key(rewritten)
            if not 20 <= len(rewritten) <= 300 or not key or key in approved_keys:
                raise ValueError(f"Câu sửa của {case_id} sai độ dài hoặc trùng câu đã duyệt")
            approved_keys.add(key)
            row["text"] = rewritten
            row["approved_ood"] = "1"
            row["review_status"] = "approved"
            reason = "Sửa lỗi chính tả/diễn đạt, giữ ý gốc và yêu cầu hỗ trợ thuộc bốn chủ đề OOD."
            decision = "approved_after_rewrite"
        else:
            rewritten = ""
            row["approved_ood"] = "0"
            row["review_status"] = "rejected"
            reason = REJECTIONS[case_id]
            decision = "rejected"
        row["rewrite"] = ""
        row["reviewer_note"] = f"{row['reviewer_note']} | rewrite_resolution: {reason}".strip(" |")[:300]
        records.append(
            {
                "source_case_id": case_id,
                "topic": row["topic"],
                "original_text": original,
                "rewritten_text": rewritten,
                "decision": decision,
                "reason": reason,
            }
        )
    return rows, records


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def main() -> int:
    """Xem trước hoặc áp dụng 12 quyết định rewrite, đồng thời giữ bản sao trước sửa."""
    parser = argparse.ArgumentParser(description="Xử lý 12 câu OOD đã được đánh dấu rewrite=1")
    parser.add_argument("--apply", action="store_true", help="Ghi thay đổi vào review CSV và bảng đối chiếu")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    review_path = ROOT / "data/raw/ood_bank_review.csv"
    report_path = ROOT / "reports/ood_rewrite_decisions.csv"
    status_path = ROOT / "data/ood_bank_quality_status.json"
    with review_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "rewrite" not in fieldnames:
        raise ValueError("File review chưa có cột rewrite")
    rows, records = resolve_rows(rows)
    approved = sum(row["approved_ood"] == "1" for row in rows)
    pending = sum(row["review_status"] == "needs_rewrite" for row in rows)
    print(f"Đã kiểm tra {len(records)} dòng: {len(REWRITES)} viết lại và tạm duyệt, {len(REJECTIONS)} loại.")
    print(f"Sau xử lý: {approved} tạm duyệt, {pending} cần viết lại; cổng chất lượng vẫn pending_audit.")
    if not args.apply:
        print("Chỉ xem trước; dùng --apply để ghi file.")
        return 0

    backup = review_path.with_name("ood_bank_review_before_marked_rewrites.csv")
    if backup.exists():
        raise FileExistsError(f"Bản sao trước xử lý đã tồn tại: {backup}")
    shutil.copy2(review_path, backup)
    _write_csv_atomic(review_path, fieldnames, rows)
    _write_csv_atomic(report_path, list(records[0]), records)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["reason"] = "Chưa đủ 100 câu OOD ngân hàng có chất lượng đã xác minh độc lập"
    status["clear_rejections"] = int(status["clear_rejections"]) + len(REJECTIONS)
    status["needs_rewrite"] = pending
    status["approved_remaining_unverified"] = approved
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã lưu bản sao: {backup}")
    print(f"Đã lưu bảng đối chiếu: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
