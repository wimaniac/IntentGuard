"""CLI chuẩn bị Banking77-VN và MASSIVE cho pipeline IntentGuard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.config import load_config, resolve_project_paths, set_global_seed
from intentguard.data import build_massive_ood, prepare_in_domain


def main() -> int:
    """Chuẩn bị các split in-domain và dữ liệu OOD MASSIVE.

    Returns:
        Mã 0 khi dữ liệu được tải, kiểm tra và ghi thành công.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    paths = resolve_project_paths(config, ROOT)
    bundle = prepare_in_domain(config, paths, seed)
    massive = build_massive_ood(config, paths)
    print(
        json.dumps(
            {
                "in_domain": bundle.metadata["counts"],
                "massive_ood_rows": len(massive),
                "faq_next_step": "uv run python scripts/prepare_real_faq_ood.py",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
