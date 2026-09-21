"""Đọc cấu hình YAML và chuẩn hoá các đường dẫn dùng trong pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc cấu hình YAML.

    Args:
        path: Đường dẫn tới file cấu hình.

    Returns:
        Dictionary cấu hình đã đọc.

    Raises:
        FileNotFoundError: Khi file cấu hình không tồn tại.
        ValueError: Khi nội dung YAML không phải mapping.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình phải là mapping YAML: {config_path}")
    return config


def resolve_project_paths(config: dict[str, Any], project_root: str | Path) -> dict[str, Path]:
    """Tạo các đường dẫn chuẩn từ cấu hình và thư mục project.

    Args:
        config: Cấu hình đã đọc.
        project_root: Thư mục gốc của project.

    Returns:
        Mapping tên đường dẫn tới đường dẫn tuyệt đối.
    """
    root = Path(project_root).resolve()
    project = config.setdefault("project", {})
    data_dir = root / project.get("data_dir", "data")
    paths = {
        "root": root,
        "data": data_dir,
        "raw": data_dir / "raw",
        "interim": data_dir / "interim",
        "processed": data_dir / "processed",
        "artifacts": root / project.get("artifact_dir", "data/artifacts"),
        "reports": root / project.get("report_dir", "reports"),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_global_seed(seed: int) -> None:
    """Đặt seed cho các thư viện tạo số ngẫu nhiên.

    Args:
        seed: Giá trị seed dùng chung.

    Returns:
        Không trả về giá trị.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dump_json(payload: Any, path: str | Path) -> None:
    """Ghi object JSON với encoding UTF-8.

    Args:
        payload: Dữ liệu có thể serialize thành JSON.
        path: File đích.

    Returns:
        Không trả về giá trị.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
