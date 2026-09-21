"""CLI tạo hàng đợi review OOD ngân hàng từ candidate và tín hiệu baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.calibration import apply_temperature
from intentguard.config import load_config, resolve_project_paths
from intentguard.review import make_review_queue


def main() -> int:
    """Sinh CSV xếp ưu tiên; không tự thay đổi file quyết định của người review."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    paths = resolve_project_paths(config, ROOT)
    ood_config = config["data"]["ood"]
    candidates = pd.read_parquet(ROOT / ood_config["bank_candidate_file"])
    probabilities = None
    label_names = None
    threshold = None
    model_path = paths["artifacts"] / "baseline.joblib"
    calibration_path = paths["artifacts"] / "calibration_baseline.json"
    if model_path.exists() and calibration_path.exists():
        model = joblib.load(model_path)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        probabilities = apply_temperature(
            model.predict_proba(candidates["text"].astype(str).tolist()), float(calibration["temperature"])
        )
        label_names = model.label_names
        threshold = float(calibration["threshold"])
    queue = make_review_queue(candidates, ood_config["review"], probabilities, label_names, threshold)
    output = paths["interim"] / "ood_bank_review_queue.csv"
    queue.to_csv(output, index=False, encoding="utf-8")
    print(f"Đã tạo {len(queue)} ứng viên: {output}")
    print(queue["priority"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
