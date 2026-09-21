"""CLI chọn và huấn luyện baseline TF-IDF + Logistic Regression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.config import load_config, resolve_project_paths, set_global_seed
from intentguard.data import load_processed_bundle
from intentguard.models.baseline import train_baseline


def main() -> int:
    """Huấn luyện baseline trên artifact dữ liệu đã chuẩn bị."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    paths = resolve_project_paths(config, ROOT)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    bundle = load_processed_bundle(paths["processed"] / "in_domain")
    model = train_baseline(
        bundle.train,
        bundle.model_selection,
        config["models"]["baseline"]["grid"],
        bundle.label_names,
        seed,
        paths["artifacts"],
    )
    bundle.train[["text", "label"]].assign(
        intent=lambda frame: frame["label"].map(dict(enumerate(bundle.label_names)))
    ).to_parquet(paths["artifacts"] / "train_cases.parquet", index=False)
    print(f"Baseline ready: selection_macro_f1={model.parameters['selection_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
