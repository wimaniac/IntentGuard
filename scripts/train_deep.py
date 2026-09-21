"""CLI fine-tune transformer và tạo embedding index chỉ từ split train."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.config import load_config, resolve_project_paths, set_global_seed
from intentguard.data import load_processed_bundle
from intentguard.models.deep import train_deep


def main() -> int:
    """Fine-tune backend DL trên dữ liệu đã chia sẵn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-name", help="Backbone khác cấu hình để thử nghiệm riêng")
    parser.add_argument("--artifact-dir", help="Thư mục artifact riêng; mặc định dùng cấu hình dự án")
    parser.add_argument("--max-length", type=int, help="Số token tối đa của thử nghiệm")
    parser.add_argument("--max-epochs", type=int, help="Số epoch tối đa của thử nghiệm")
    parser.add_argument("--learning-rate", type=float, help="Learning rate của thử nghiệm")
    parser.add_argument("--resume-from", help="Artifact DL đã lưu để tiếp tục fine-tune")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    paths = resolve_project_paths(config, ROOT)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    bundle = load_processed_bundle(paths["processed"] / "in_domain")
    model_config = config["models"]["deep"].copy()
    for key, value in (
        ("name", args.model_name),
        ("max_length", args.max_length),
        ("max_epochs", args.max_epochs),
        ("learning_rate", args.learning_rate),
    ):
        if value is not None:
            model_config[key] = value
    if int(model_config["max_length"]) < 8 or int(model_config["max_epochs"]) < 1:
        raise ValueError("max-length và max-epochs phải dương")
    if float(model_config["learning_rate"]) <= 0:
        raise ValueError("learning-rate phải dương")
    if args.resume_from:
        resume_path = (ROOT / args.resume_from).resolve()
        if not resume_path.is_relative_to(ROOT.resolve()):
            raise ValueError("resume-from phải nằm trong workspace")
        model_config["resume_from"] = str(resume_path)
    artifact_dir = ROOT / args.artifact_dir if args.artifact_dir else paths["artifacts"]
    if args.resume_from and artifact_dir.resolve() == resume_path:
        raise ValueError("artifact-dir phải khác resume-from để giữ checkpoint gốc")
    metadata = train_deep(
        bundle.train,
        bundle.model_selection,
        bundle.label_names,
        model_config,
        seed,
        artifact_dir,
    )
    print(f"Deep model ready: selection_macro_f1={metadata['selection_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
