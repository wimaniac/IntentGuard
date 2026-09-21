"""Temperature scaling và chọn threshold UNKNOWN không phụ thuộc dữ liệu mẫu."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def _nll_for_temperature(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fit một temperature dương trên split calibration.

    Args:
        probabilities: Xác suất chưa hiệu chỉnh, shape ``(n_samples, n_classes)``.
        labels: Class index đúng, shape ``(n_samples,)``.

    Returns:
        Temperature tối ưu trong khoảng ổn định số học.
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    logits = np.log(probabilities)
    result = minimize_scalar(
        lambda value: _nll_for_temperature(logits, np.asarray(labels), float(value)),
        bounds=(0.05, 20.0),
        method="bounded",
    )
    return float(result.x)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Áp dụng temperature scaling và chuẩn hoá lại xác suất.

    Args:
        probabilities: Xác suất gốc.
        temperature: Temperature dương.

    Returns:
        Xác suất sau hiệu chỉnh, tổng mỗi hàng bằng một.
    """
    if temperature <= 0:
        raise ValueError("temperature phải dương")
    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def choose_threshold(probabilities: np.ndarray, max_rejection_rate: float) -> float:
    """Chọn threshold cao nhất không từ chối quá tỷ lệ in-domain cho phép.

    Args:
        probabilities: Xác suất đã hiệu chỉnh trên split threshold.
        max_rejection_rate: Tỷ lệ ``top_1 < threshold`` tối đa.

    Returns:
        Threshold được chọn; phép so sánh tại đúng biên là không UNKNOWN.
    """
    if not 0 <= max_rejection_rate <= 1:
        raise ValueError("max_rejection_rate phải nằm trong [0, 1]")
    top1 = np.asarray(probabilities, dtype=float).max(axis=1)
    candidates = np.unique(np.concatenate(([0.0], top1)))
    accepted: list[float] = []
    for candidate in candidates:
        rejected = float(np.mean(top1 < candidate))
        if rejected <= max_rejection_rate + 1e-12:
            accepted.append(float(candidate))
    if not accepted:
        return 0.0
    return max(accepted)


def save_calibration(temperature: float, threshold: float, path: str | Path) -> None:
    """Lưu tham số hiệu chỉnh và threshold thành JSON.

    Args:
        temperature: Temperature scaling.
        threshold: Ngưỡng UNKNOWN.
        path: File JSON đích.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"temperature": temperature, "threshold": threshold}, indent=2),
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> dict[str, float]:
    """Đọc tham số temperature và threshold.

    Args:
        path: File JSON đã lưu.

    Returns:
        Mapping có hai key ``temperature`` và ``threshold``.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
