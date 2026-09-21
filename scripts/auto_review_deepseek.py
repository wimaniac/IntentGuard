"""Chạy DeepSeek duyệt tự động nhóm ứng viên OOD ngân hàng mục tiêu."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.auto_review import load_deepseek_key, rank_similar_candidates, run_auto_review
from intentguard.config import load_config, resolve_project_paths
from intentguard.data import load_processed_bundle, load_reviewed_bank_ood
from intentguard.llm_review import suggest_review
from intentguard.review import save_review_decision, split_review_cases, validate_approved_rows


def main() -> int:
    """Duyệt phần chưa xử lý đến đủ 100 câu, có thể chạy lại sau lỗi API."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-topic-keywords",
        action="store_true",
        help="Xét thêm ứng viên chủ đề khác có từ khoá vay/tiết kiệm/lãi suất/khuyến mãi",
    )
    parser.add_argument(
        "--secondary",
        action="store_true",
        help="Xét các ứng viên chưa duyệt từ nhãn nguồn OTHER và DISCOUNT",
    )
    parser.add_argument(
        "--similar-pool",
        type=int,
        default=0,
        metavar="N",
        help="Xét tối đa N ứng viên chưa xử lý gần các câu OOD đã duyệt",
    )
    args = parser.parse_args()
    config = load_config(ROOT / "configs/default.yaml")
    paths = resolve_project_paths(config, ROOT)
    ood = config["data"]["ood"]
    quality_path = ROOT / ood.get("bank_quality_file", "data/ood_bank_quality_status.json")
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if quality.get("status") != "verified":
            print("Tập OOD đang chờ audit nhãn; không chạy tiếp bộ duyệt tự động cũ", file=sys.stderr)
            return 2
    key = os.getenv("DEEPSEEK_API_KEY", "") or load_deepseek_key(ROOT / ".env")
    if not key:
        print("Thiếu DEEPSEEK_API_KEY trong môi trường hoặc .env", file=sys.stderr)
        return 2
    queue_path = paths["interim"] / "ood_bank_review_queue.csv"
    review_path = ROOT / ood["bank_review_file"]
    if not queue_path.exists() or not review_path.exists():
        print("Thiếu hàng đợi hoặc file review; chạy prepare_review_queue.py và mở trang review trước", file=sys.stderr)
        return 2
    queue = pd.read_csv(queue_path, dtype={"source_case_id": str}, keep_default_na=False)
    full_queue = queue.copy()
    focus_topics = set(ood["review"]["focus_topics"])
    candidate_ids = None
    if args.similar_pool > 0:
        review_for_ranking = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        ordered_ids = rank_similar_candidates(queue, review_for_ranking, args.similar_pool)
        candidate_ids = set(ordered_ids)
        queue = queue.set_index("source_case_id").loc[ordered_ids].reset_index()
        focus_topics = set(queue["source_topic"].astype(str))
        print(f"Đang xét {len(candidate_ids)} ứng viên gần các câu OOD đã duyệt.")
    elif args.cross_topic_keywords:
        pattern = (
            r"vay|trả góp|khoản vay|giải ngân|thế chấp|lãi suất|tiết kiệm|sổ tiết kiệm|"
            r"gửi tiền|tiền gửi|kỳ hạn|ưu đãi|khuyến mãi|khuyến mại|miễn phí|giảm giá|"
            r"quà tặng|voucher"
        )
        candidate_ids = set(
            queue.loc[
                ~queue["source_topic"].isin(focus_topics)
                & queue["text"].astype(str).str.contains(pattern, case=False, regex=True),
                "source_case_id",
            ].astype(str)
        )
        focus_topics = set(queue["source_topic"].astype(str))
        print(f"Đang xét {len(candidate_ids)} ứng viên có từ khoá mục tiêu nhưng chủ đề nguồn khác.")
    elif args.secondary:
        focus_topics = set(ood["review"]["secondary_topics"])
        print(f"Đang xét ứng viên chưa xử lý từ {sorted(focus_topics)}.")
    metadata = json.loads((paths["processed"] / "in_domain" / "metadata.json").read_text(encoding="utf-8"))
    bundle = load_processed_bundle(paths["processed"] / "in_domain")
    domain_texts = pd.concat(
        [bundle.train, bundle.model_selection, bundle.calibration, bundle.threshold, bundle.test],
        ignore_index=True,
    )["text"].astype(str).tolist()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    backup = review_path.with_name("ood_bank_review_before_deepseek.csv")
    if not backup.exists():
        shutil.copy2(review_path, backup)

    def judge(text: str, topic: str, labels: list[str]):
        """Gọi một câu qua DeepSeek bằng key chỉ giữ trong bộ nhớ tiến trình."""
        return suggest_review(text, topic, labels, key, model=model)

    reviewed = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    minimum = int(ood["review"]["min_length"])
    maximum = int(ood["review"]["max_length"])
    invalid = reviewed[
        reviewed["approved_ood"].eq("1")
        & ~reviewed["text"].str.len().between(minimum, maximum)
    ]
    for row in invalid.itertuples():
        suggestion = judge(str(row.text), str(row.topic), metadata["label_names"])
        replacement = (suggestion.rewritten_text or "").strip()
        if (
            suggestion.decision == "rewrite"
            and suggestion.in_scope
            and suggestion.support_request
            and suggestion.faithful_rewrite
            and minimum <= len(replacement) <= maximum
        ):
            save_review_decision(
                review_path,
                row.source_case_id,
                replacement,
                "approved",
                f"auto_repair:{model}; trước đó approved nhưng sai độ dài; {suggestion.reason[:120]}",
            )
        else:
            save_review_decision(
                review_path,
                row.source_case_id,
                str(row.text),
                "rejected",
                f"auto_repair:{model}; trước đó approved nhưng sai độ dài; {suggestion.reason[:120]}",
            )
        print(f"Đã kiểm tra lại câu approved sai độ dài: {row.source_case_id}")

    result = run_auto_review(
        queue=queue,
        review_path=review_path,
        label_names=metadata["label_names"],
        judge=judge,
        expected_total=int(ood["expected_total"]),
        min_length=minimum,
        max_length=maximum,
        focus_topics=focus_topics,
        in_domain_texts=domain_texts,
        model=model,
        candidate_ids=candidate_ids,
    )
    print(f"Hoàn tất lượt chạy: {result}")
    if result["remaining"] == 0:
        reviewed = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        approved, source_texts = validate_approved_rows(
            reviewed,
            full_queue,
            domain_texts,
            int(ood["expected_total"]),
            int(ood["review"]["min_length"]),
            int(ood["review"]["max_length"]),
        )
        validation, test = split_review_cases(
            approved,
            source_texts,
            int(ood["expected_per_split"]),
            int(config["project"]["seed"]),
        )
        (ROOT / ood["bank_validation_cases"]).write_text("\n".join(validation) + "\n", encoding="utf-8")
        (ROOT / ood["bank_test_cases"]).write_text("\n".join(test) + "\n", encoding="utf-8")
        load_reviewed_bank_ood(config, paths)
        print("Đã kiểm tra và xuất 50 validation / 50 test; nguồn nhãn gồm quyết định thủ công và DeepSeek.")
    return 0 if result["remaining"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
