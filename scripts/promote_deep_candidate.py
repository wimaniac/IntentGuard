"""Lưu artifact DL cũ và thay bằng candidate có macro-F1 model-selection cao hơn."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.config import load_config, resolve_project_paths
from intentguard.data import load_processed_bundle
from intentguard.models.deep import load_deep_model, predict_probabilities

MODEL_FILES = ("deep.json", "deep_classifier.pt", "train_embeddings.npy", "train_cases.parquet")
MODEL_DIRS = ("encoder", "tokenizer")


def _copy_model(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(source / name, destination / name)
    for name in MODEL_DIRS:
        shutil.copytree(source / name, destination / name, dirs_exist_ok=True)


def main() -> int:
    """Kiểm tra candidate, sao lưu DL cũ và cập nhật artifact đang dùng.

    Returns:
        Mã 0 khi candidate tốt hơn trên model-selection và đã nạp lại thành công.

    Raises:
        ValueError: Khi candidate không khớp nhãn hoặc không cải thiện model-selection.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--backup", default="data/experiments/deep_before_multilingual")
    args = parser.parse_args()
    candidate = (ROOT / args.candidate).resolve()
    backup = (ROOT / args.backup).resolve()
    if not candidate.is_relative_to(ROOT.resolve()) or not backup.is_relative_to(ROOT.resolve()):
        raise ValueError("Candidate và backup phải nằm trong workspace")
    if backup.exists():
        raise ValueError(f"Thư mục backup đã tồn tại: {backup}")
    config = load_config(ROOT / "configs/default.yaml")
    paths = resolve_project_paths(config, ROOT)
    active = paths["artifacts"]
    for name in (*MODEL_FILES, *MODEL_DIRS):
        if not (candidate / name).exists() or not (active / name).exists():
            raise FileNotFoundError(f"Thiếu artifact DL: {name}")
    candidate_meta = json.loads((candidate / "deep.json").read_text(encoding="utf-8"))
    active_meta = json.loads((active / "deep.json").read_text(encoding="utf-8"))
    if candidate_meta["label_names"] != active_meta["label_names"]:
        raise ValueError("Candidate không có cùng thứ tự 77 intent")
    old_score = float(active_meta["selection_macro_f1"])
    new_score = float(candidate_meta["selection_macro_f1"])
    if new_score <= old_score:
        raise ValueError(f"Candidate không cải thiện model-selection: {new_score:.4f} <= {old_score:.4f}")
    bundle = load_processed_bundle(paths["processed"] / "in_domain")
    model, tokenizer, device, metadata = load_deep_model(candidate)
    probabilities = predict_probabilities(
        model,
        tokenizer,
        bundle.model_selection,
        int(metadata["config"]["max_length"]),
        int(metadata["config"]["batch_size"]),
        device,
    )
    reloaded_score = float(f1_score(bundle.model_selection.label, probabilities.argmax(axis=1), average="macro"))
    if abs(reloaded_score - new_score) > 0.002:
        raise ValueError(f"Candidate nạp lại không khớp score: {reloaded_score:.4f} so với {new_score:.4f}")
    _copy_model(active, backup / "artifacts")
    if (active / "calibration_deep.json").exists():
        shutil.copy2(active / "calibration_deep.json", backup / "artifacts/calibration_deep.json")
    reports_backup = backup / "reports"
    reports_backup.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.json", "comparison.md", "faq_ood_diagnostic.json", "faq_ood_diagnostic.md"):
        source = paths["reports"] / name
        if source.exists():
            shutil.copy2(source, reports_backup / name)
    try:
        _copy_model(candidate, active)
        load_deep_model(active)
    except Exception:
        _copy_model(backup / "artifacts", active)
        raise
    print(f"Đã thay DL: model-selection macro-F1 {old_score:.4f} → {new_score:.4f}; backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
