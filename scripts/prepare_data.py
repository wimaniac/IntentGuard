"""CLI chuẩn bị dữ liệu in-domain và các nguồn OOD tách biệt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.config import load_config, resolve_project_paths, set_global_seed
from intentguard.data import (
    build_bank_ood_candidates,
    build_massive_ood,
    load_processed_bundle,
    load_reviewed_bank_ood,
    prepare_in_domain,
)


def main() -> int:
    """Chạy toàn bộ bước chuẩn bị dữ liệu có thể tái lập."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--review-only", action="store_true", help="Dùng artifact dữ liệu hiện có, không tải lại Hugging Face"
    )
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    paths = resolve_project_paths(config, ROOT)
    if args.review_only:
        bundle = load_processed_bundle(paths["processed"] / "in_domain")
        massive = pd.read_parquet(paths["interim"] / "ood_massive.parquet")
        candidate_path = paths["root"] / config["data"]["ood"]["bank_candidate_file"]
        candidates = pd.read_parquet(candidate_path)
    else:
        bundle = prepare_in_domain(config, paths, seed)
        massive = build_massive_ood(config, paths)
        candidates = build_bank_ood_candidates(config, paths)
    reviewed = None
    try:
        validation, test = load_reviewed_bank_ood(config, paths)
        reviewed = {"validation": len(validation), "test": len(test)}
    except FileNotFoundError as error:
        reviewed = {"status": "pending", "message": str(error)}
    print(
        json.dumps(
            {
                "in_domain": bundle.metadata["counts"],
                "massive_ood_rows": len(massive),
                "bank_candidate_rows": len(candidates),
                "reviewed_bank_ood": reviewed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
