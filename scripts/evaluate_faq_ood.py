"""Đánh giá chẩn đoán UNKNOWN trên 100 FAQ ngân hàng do người dùng thu thập."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.calibrate_and_evaluate import _backend_probabilities

from intentguard.calibration import apply_temperature, load_calibration
from intentguard.config import dump_json, load_config, resolve_project_paths
from intentguard.data import load_processed_bundle
from intentguard.faq_audit import validate_faq_label_audit
from intentguard.metrics import ood_metrics

FAQ_DIR = ROOT / "data/processed/faq_ood_real"
RAW = ROOT / "data/raw/vietnam_banks_faq_100.csv"
AUDIT = ROOT / "reports/faq_ood_label_audit.csv"


def cluster_recall_interval(
    confidence: np.ndarray, source_groups: list[str], threshold: float, repeats: int = 3000, seed: int = 42
) -> tuple[float, float]:
    """Ước lượng khoảng 95% cho recall bằng bootstrap theo URL nguồn.

    Args:
        confidence: Confidence top-1 của từng câu.
        source_groups: ID trang nguồn theo đúng thứ tự câu.
        threshold: Ngưỡng UNKNOWN đã chọn từ in-domain.
        repeats: Số lần lấy lại các nhóm FAQ.
        seed: Seed ngẫu nhiên để tái lập khoảng.

    Returns:
        Hai phân vị 2,5% và 97,5% của recall theo nhóm.
    """
    frame = pd.DataFrame({"source_group_id": source_groups, "unknown": np.asarray(confidence) < threshold})
    group_counts = frame.groupby("source_group_id")["unknown"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(group_counts), size=(repeats, len(group_counts)))
    selected = group_counts[indices]
    samples = selected[:, :, 0].sum(axis=1) / selected[:, :, 1].sum(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def evaluate_split(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    reference: np.ndarray,
    threshold: float,
    label_names: list[str],
) -> dict[str, Any]:
    """Tính metric, recall theo topic và một số lỗi chấp nhận nhầm.

    Args:
        frame: Câu FAQ trong split.
        probabilities: Xác suất đã hiệu chỉnh trên FAQ.
        reference: Xác suất in-domain cùng vai trò split.
        threshold: Ngưỡng UNKNOWN giữ nguyên từ in-domain.
        label_names: Tên 77 intent theo thứ tự cột xác suất.

    Returns:
        Metric chẩn đoán và ví dụ lỗi có nguồn gốc FAQ.
    """
    confidence = probabilities.max(axis=1)
    rejected = confidence < threshold
    low, high = cluster_recall_interval(confidence, frame["source_group_id"].tolist(), threshold)
    topics = {
        topic: {"count": int(mask.sum()), "unknown_recall": float(rejected[mask].mean())}
        for topic in sorted(frame["topic"].unique())
        for mask in [frame["topic"].eq(topic).to_numpy()]
    }
    accepted_indices = np.flatnonzero(~rejected)
    accepted_indices = accepted_indices[np.argsort(-confidence[accepted_indices])[:12]]
    errors = [
        {
            "source_case_id": str(frame.iloc[index]["source_case_id"]),
            "source_group_id": str(frame.iloc[index]["source_group_id"]),
            "topic": str(frame.iloc[index]["topic"]),
            "text": str(frame.iloc[index]["text"]),
            "text_original": str(frame.iloc[index]["text_original"]),
            "source_url": str(frame.iloc[index]["source_url"]),
            "predicted_intent": label_names[int(probabilities[index].argmax())],
            "confidence": float(confidence[index]),
        }
        for index in accepted_indices
    ]
    return {
        "samples": len(frame),
        "source_pages": int(frame["source_group_id"].nunique()),
        **ood_metrics(reference, probabilities, threshold),
        "ood_unknown_recall_cluster_ci95": [low, high],
        "topics": topics,
        "accepted_as_known_examples": errors,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Chẩn đoán OOD trên {report['source_rows']} FAQ ngân hàng có URL",
        "",
        f"Nguồn hiện hành: `{report['source_file']}` do người dùng thu thập, "
        f"{report['source_rows']} câu và {report['source_urls']} URL. Không có câu nào do LLM sinh trong tập này. "
        "Người dùng xác nhận URL; Codex rà lại từng nhãn OOD so với 77 intent. "
        "Đây là audit bằng AI, không phải bộ nhãn vàng do hai người gán nhãn độc lập.",
        "Metric FAQ cũng nằm trong `comparison.md` với nguồn gốc nhãn được ghi rõ; "
        "bảng dưới đây là phân tích chi tiết.",
        "",
        f"Input mô hình thay tên ngân hàng bằng `ngân hàng` ở {report['normalized_rows']} câu; "
        "câu gốc và URL được giữ trong artifact. "
        "Hai câu nhiều vế được thu gọn về một nhu cầu có sẵn trong câu gốc; "
        f"{report['audit']['semantic_duplicate_followups']} câu gần trùng ngữ nghĩa "
        "được giữ cùng split với câu tương ứng. "
        "Threshold và temperature đọc từ artifact đã chọn trên dữ liệu in-domain, không chọn lại bằng FAQ. "
        "Validation/test chia 50/50, mỗi topic 12–13 câu; các câu cùng URL nằm trọn trong một split. "
        "Khoảng 95% cho recall bootstrap theo URL vì nhiều câu cùng lấy từ một trang.",
        "",
        "| Backend | Split | Số câu | Số URL | Recall UNKNOWN | CI 95% theo URL | "
        "AUROC | AUPRC | Từ chối nhầm ID |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for backend, splits in report["backends"].items():
        for split, values in splits.items():
            low, high = values["ood_unknown_recall_cluster_ci95"]
            lines.append(
                f"| {backend} | {split} | {values['samples']} | {values['source_pages']} | "
                f"{values['ood_unknown_recall']:.4f} | [{low:.4f}, {high:.4f}] | "
                f"{values['auroc']:.4f} | {values['auprc']:.4f} | "
                f"{values['in_domain_false_rejection_rate']:.4f} |"
            )
    lines.extend(["", "## Recall UNKNOWN trên test theo topic", ""])
    for backend, splits in report["backends"].items():
        values = splits["test"]["topics"]
        lines.append(f"- **{backend}:** " + "; ".join(
            f"{topic} {item['unknown_recall']:.3f} ({item['count']} câu)" for topic, item in values.items()
        ))
    lines.extend([
        "", "## Giới hạn", "",
        "- Người dùng đã xác nhận URL là đúng. Codex chưa tự đối chiếu nguyên văn từng câu trên trang "
        "hoặc hiệu lực của từng chương trình ưu đãi.",
        "- Nhãn OOD và topic được Codex rà theo 77 intent; đây là lượt rà bằng AI, "
        "chưa có kiểm tra của hai người độc lập.",
        "- FAQ trên website có văn phong sạch hơn tin nhắn khách hàng thật. "
        "Không suy rộng recall ở đây thành hiệu năng vận hành.",
        "- AUPRC phụ thuộc tỷ lệ OOD/ID; không so trực tiếp AUPRC FAQ với MASSIVE.",
        "- Ví dụ được mô hình nhận thành intent đã học nằm trong file JSON cùng tên để phân tích lỗi.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Chạy cả hai backend trên FAQ nguồn mới mà không thay calibration hoặc báo cáo chính.

    Returns:
        Mã 0 khi tạo được báo cáo cho cả baseline và deep.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config(ROOT / "configs/default.yaml")
    paths = resolve_project_paths(config, ROOT)
    bundle = load_processed_bundle(paths["processed"] / "in_domain")
    frames = {split: pd.read_parquet(FAQ_DIR / f"{split}.parquet") for split in ("validation", "test")}
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    prepared = pd.concat([frame.assign(split=split) for split, frame in frames.items()], ignore_index=True)
    if len(raw) != 100 or len(prepared) != 100:
        raise ValueError("FAQ nguồn và hai split cần đúng 100 câu")
    raw_pairs = set(zip(raw["text"], raw["URL"], strict=True))
    prepared_pairs = set(zip(prepared["text_original"], prepared["source_url"], strict=True))
    if raw_pairs != prepared_pairs or len(raw_pairs) != 100:
        raise ValueError("FAQ đã xử lý không khớp câu gốc và URL hiện hành; chạy prepare_real_faq_ood.py")
    if set(frames["validation"].source_url) & set(frames["test"].source_url):
        raise ValueError("Một URL nguồn bị tách giữa validation và test")
    audit = pd.read_csv(AUDIT, dtype=str, keep_default_na=False)
    audit_stats = validate_faq_label_audit(prepared, audit)
    report: dict[str, Any] = {
        "status": "included_with_ai_review_provenance",
        "source_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "source_rows": len(raw),
        "source_urls": int(raw.URL.nunique()),
        "label_quality": "urls_confirmed_by_user; all_100_ood_labels_reviewed_by_codex_ai; not_two_human_gold",
        "audit_file": str(AUDIT.relative_to(ROOT)).replace("\\", "/"),
        "audit": audit_stats,
        "model_text": "bank_name_normalized",
        "normalized_rows": int(prepared.text.ne(prepared.text_original).sum()),
        "threshold_source": "in_domain_threshold_validation",
        "backends": {},
    }
    for backend in ("baseline", "deep"):
        calibration = load_calibration(paths["artifacts"] / f"calibration_{backend}.json")
        temperature, threshold = calibration["temperature"], calibration["threshold"]
        references = {
            "validation": apply_temperature(
                _backend_probabilities(backend, bundle.threshold, paths["artifacts"], config), temperature
            ),
            "test": apply_temperature(
                _backend_probabilities(backend, bundle.test, paths["artifacts"], config), temperature
            ),
        }
        report["backends"][backend] = {}
        for split, frame in frames.items():
            probabilities = apply_temperature(
                _backend_probabilities(backend, frame, paths["artifacts"], config), temperature
            )
            report["backends"][backend][split] = evaluate_split(
                frame, probabilities, references[split], threshold, bundle.label_names
            )
            print(f"{backend} {split}: recall {report['backends'][backend][split]['ood_unknown_recall']:.4f}")
    dump_json(report, paths["reports"] / "faq_ood_diagnostic.json")
    _write_markdown(report, paths["reports"] / "faq_ood_diagnostic.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
