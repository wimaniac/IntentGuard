"""CLI hiệu chỉnh confidence, chọn threshold và tạo báo cáo so sánh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.calibration import (
    apply_temperature,
    choose_threshold,
    fit_temperature,
    save_calibration,
)
from intentguard.config import dump_json, load_config, resolve_project_paths
from intentguard.data import load_processed_bundle
from intentguard.faq_audit import validate_faq_label_audit
from intentguard.metrics import classification_metrics, ece, nll, ood_metrics
from intentguard.models.deep import load_deep_model, predict_probabilities


def _backend_probabilities(backend: str, frame: pd.DataFrame, artifact_dir: Path, config: dict[str, Any]) -> np.ndarray:
    """Tính raw probabilities mà không chạm vào test ngoài lần evaluate."""
    if backend == "baseline":
        import joblib

        model = joblib.load(artifact_dir / "baseline.joblib")
        return model.predict_proba(frame["text"].astype(str).tolist())
    model, tokenizer, device, metadata = load_deep_model(artifact_dir)
    deep_config = metadata["config"]
    return predict_probabilities(
        model,
        tokenizer,
        frame.assign(label=frame.get("label", 0)),
        int(deep_config["max_length"]),
        int(deep_config["batch_size"]),
        device,
    )


def _error_summary(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, label_names: list[str]
) -> dict[str, Any]:
    """Lấy các ví dụ lỗi có cấu trúc, không nhúng dữ liệu cụ thể vào mã."""
    calibrated = probabilities
    predictions = calibrated.argmax(axis=1)
    confidence = calibrated.max(axis=1)
    result: dict[str, Any] = {
        "in_domain_rejected": [],
        "confusion_pairs": [],
    }
    for index in np.flatnonzero(confidence < threshold)[:20]:
        result["in_domain_rejected"].append(
            {
                "text": str(frame.iloc[index]["text"]),
                "true_intent": label_names[int(frame.iloc[index]["label"])],
                "top_intent": label_names[int(predictions[index])],
                "confidence": float(confidence[index]),
            }
        )
    if "label" in frame.columns:
        pairs: dict[tuple[str, str], int] = {}
        for true, predicted in zip(frame["label"].astype(int), predictions, strict=True):
            if true != predicted:
                key = (label_names[int(true)], label_names[int(predicted)])
                pairs[key] = pairs.get(key, 0) + 1
        result["confusion_pairs"] = [
            {"true_intent": pair[0], "predicted_intent": pair[1], "count": count}
            for pair, count in sorted(pairs.items(), key=lambda item: -item[1])[:20]
        ]
    return result


def _in_domain_reference(name: str, validation: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Ghép OOD validation với ID validation, OOD test với ID test."""
    return validation if name.endswith("_validation") else test


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    """Ghi báo cáo tóm tắt có thể đọc bằng tiếng Việt."""
    lines = [
        "# Báo cáo IntentGuard",
        "",
        "Báo cáo này được sinh từ artifact và cấu hình hiện hành; test gốc chỉ được đọc ở bước này.",
        "",
        "## In-domain",
        "",
        "| Backend | Accuracy | Macro-F1 | Top-3 | NLL sau calibration | ECE sau calibration | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, values in report["backends"].items():
        test = values["test_calibrated"]
        lines.append(
            f"| {backend} | {test['accuracy']:.4f} | {test['macro_f1']:.4f} | "
            f"{test['top_3_accuracy']:.4f} | {test['nll']:.4f} | {test['ece']:.4f} | "
            f"{values['threshold']:.6f} |"
        )
    if any(name.startswith("bank_faq_") for name in report["ood"]):
        ood_status_line = (
            "OOD ngân hàng dùng 100 FAQ có URL do người dùng xác nhận. Codex AI đã rà từng nhãn với 77 intent "
            "và thu gọn hai câu nhiều vế; nhãn này chưa phải bộ nhãn vàng do hai người gán nhãn độc lập. "
            "Các cặp gần trùng ngữ nghĩa được giữ cùng split."
        )
    elif any(name.startswith("bank_") for name in report["ood"]):
        ood_status_line = (
            "OOD ngân hàng mới gồm "
            f"{report['ood_label_provenance']['deepseek']} nhãn DeepSeek và "
            f"{report['ood_label_provenance']['manual']} nhãn duyệt thủ công. "
            "Metric trên tập này là kết quả với nhãn tự động, chưa phải đánh giá trên nhãn vàng độc lập."
        )
    elif report.get("ood_label_quality", {}).get("metric_decision") == "omit_current_report":
        ood_status_line = (
            "Bỏ qua metric OOD ngân hàng chính vì chưa có bộ nhãn vàng được xác minh độc lập; "
            "metric chính hiện chỉ gồm MASSIVE. Bộ FAQ ngân hàng có URL có báo cáo chẩn đoán riêng."
        )
    elif report.get("ood_label_quality", {}).get("status") == "pending_audit":
        ood_status_line = "OOD ngân hàng bị tạm rút vì kiểm tra chất lượng phát hiện nhãn sai; metric chỉ gồm MASSIVE."
    else:
        ood_status_line = "OOD ngân hàng mới đang chờ 100 câu hợp lệ; metric chỉ gồm MASSIVE."
    lines.extend(
        [
            "",
            "## OOD",
            "",
            ood_status_line,
            "",
            "```json",
            json.dumps(report["ood"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Chạy calibration và đánh giá backend đã được huấn luyện."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backends", nargs="+", default=["baseline", "deep"], choices=["baseline", "deep"])
    parser.add_argument("--artifact-dir", help="Thư mục artifact ứng viên, không ghi đè model hiện hành")
    parser.add_argument("--report-stem", default="comparison", help="Tên file báo cáo, không có đường dẫn")
    args = parser.parse_args()
    if Path(args.report_stem).name != args.report_stem or args.report_stem in {"", ".", ".."}:
        raise ValueError("report-stem chỉ được là tên file")
    config = load_config(ROOT / args.config)
    paths = resolve_project_paths(config, ROOT)
    bundle = load_processed_bundle(paths["processed"] / "in_domain")
    artifact_dir = ROOT / args.artifact_dir if args.artifact_dir else paths["artifacts"]
    reports: dict[str, Any] = {"backends": {}, "ood": {}, "data": bundle.metadata}
    quality_path = ROOT / config["data"]["ood"].get("bank_quality_file", "data/ood_bank_quality_status.json")
    bank_labels_valid = True
    faq_ai_reviewed = False
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        bank_labels_valid = quality.get("status") == "verified"
        faq_ai_reviewed = quality.get("status") == "faq_ai_reviewed"
        reports["ood_label_quality"] = quality
    ood_frames: dict[str, pd.DataFrame] = {}
    massive_path = paths["interim"] / "ood_massive.parquet"
    if massive_path.exists():
        massive = pd.read_parquet(massive_path)
        for split in massive["source_split"].dropna().unique():
            ood_frames[f"massive_{split}"] = massive[massive["source_split"] == split].reset_index(drop=True)
    for name in ["validation", "test"] if bank_labels_valid else []:
        path = paths["processed"] / f"ood_bank_{name}.parquet"
        if path.exists():
            ood_frames[f"bank_{name}"] = pd.read_parquet(path)
    if faq_ai_reviewed:
        faq_dir = ROOT / quality["faq_processed_dir"]
        faq_frames = {
            split: pd.read_parquet(faq_dir / f"{split}.parquet").assign(split=split)
            for split in ("validation", "test")
        }
        faq_all = pd.concat(faq_frames.values(), ignore_index=True)
        raw_faq = pd.read_csv(ROOT / quality["faq_source_file"], dtype=str, keep_default_na=False)
        raw_pairs = set(zip(raw_faq.text, raw_faq.URL, strict=True))
        prepared_pairs = set(zip(faq_all.text_original, faq_all.source_url, strict=True))
        if len(raw_faq) != 100 or len(raw_pairs) != 100 or raw_pairs != prepared_pairs:
            raise ValueError("FAQ đã xử lý không khớp 100 câu/URL nguồn hiện hành")
        audit = pd.read_csv(ROOT / quality["faq_label_audit_file"], dtype=str, keep_default_na=False)
        audit_stats = validate_faq_label_audit(faq_all, audit)
        ood_frames.update({f"bank_faq_{split}": frame for split, frame in faq_frames.items()})
        reports["ood_label_provenance"] = {
            "source_urls": "confirmed_by_user",
            "reviewer_type": "codex_ai",
            "not_two_human_gold": True,
            **audit_stats,
        }
    elif any(name.startswith("bank_") for name in ood_frames):
        review_path = ROOT / config["data"]["ood"]["bank_review_file"]
        reviewed = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        approved = reviewed[reviewed["approved_ood"].eq("1")]
        auto_count = int(approved["reviewer_note"].str.startswith("auto:").sum())
        reports["ood_label_provenance"] = {"deepseek": auto_count, "manual": len(approved) - auto_count}

    for backend in args.backends:
        if not (artifact_dir / f"{backend}.json").exists():
            print(f"Skip {backend}: chưa có artifact")
            continue
        calibration_probabilities = _backend_probabilities(backend, bundle.calibration, artifact_dir, config)
        threshold_probabilities = _backend_probabilities(backend, bundle.threshold, artifact_dir, config)
        temperature = fit_temperature(calibration_probabilities, bundle.calibration["label"].to_numpy())
        calibrated_threshold = apply_temperature(threshold_probabilities, temperature)
        threshold = choose_threshold(calibrated_threshold, float(config["calibration"]["max_rejection_rate"]))
        save_calibration(temperature, threshold, artifact_dir / f"calibration_{backend}.json")

        test_raw = _backend_probabilities(backend, bundle.test, artifact_dir, config)
        test_calibrated = apply_temperature(test_raw, temperature)
        backend_report: dict[str, Any] = {
            "temperature": temperature,
            "threshold": threshold,
            "calibration_before": {
                "nll": nll(bundle.calibration["label"].to_numpy(), calibration_probabilities),
                "ece": ece(
                    bundle.calibration["label"].to_numpy(),
                    calibration_probabilities,
                    int(config["calibration"]["ece_bins"]),
                ),
            },
            "calibration_after": {
                "nll": nll(
                    bundle.calibration["label"].to_numpy(),
                    apply_temperature(calibration_probabilities, temperature),
                ),
                "ece": ece(
                    bundle.calibration["label"].to_numpy(),
                    apply_temperature(calibration_probabilities, temperature),
                    int(config["calibration"]["ece_bins"]),
                ),
            },
            "test_calibrated": classification_metrics(
                bundle.test["label"].to_numpy(),
                test_calibrated,
                int(config["calibration"]["ece_bins"]),
            ),
            "threshold_validation": classification_metrics(
                bundle.threshold["label"].to_numpy(),
                calibrated_threshold,
                int(config["calibration"]["ece_bins"]),
            ),
            "errors": _error_summary(bundle.test, test_calibrated, threshold, bundle.label_names),
        }
        for name, ood_frame in ood_frames.items():
            ood_probabilities = apply_temperature(
                _backend_probabilities(backend, ood_frame, artifact_dir, config), temperature
            )
            reference = _in_domain_reference(name, calibrated_threshold, test_calibrated)
            backend_report.setdefault("ood", {})[name] = ood_metrics(reference, ood_probabilities, threshold)
            false_accept = np.flatnonzero(ood_probabilities.max(axis=1) >= threshold)[:20]
            backend_report.setdefault("ood_errors", {})[name] = [
                {
                    "text": str(ood_frame.iloc[index]["text"]),
                    "confidence": float(ood_probabilities[index].max()),
                }
                for index in false_accept
            ]
        reports["backends"][backend] = backend_report

    for name in ood_frames:
        reports["ood"][name] = {
            backend: values.get("ood", {}).get(name)
            for backend, values in reports["backends"].items()
            if name in values.get("ood", {})
        }
    dump_json(reports, paths["reports"] / f"{args.report_stem}.json")
    _write_markdown(reports, paths["reports"] / f"{args.report_stem}.md")
    print(f"Report written: {paths['reports'] / f'{args.report_stem}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
