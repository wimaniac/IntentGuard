"""Các metric phân loại, calibration và OOD của IntentGuard."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def top_k_accuracy(labels: np.ndarray, probabilities: np.ndarray, k: int = 3) -> float:
    """Tính top-k accuracy.

    Args:
        labels: Nhãn đúng dạng class index.
        probabilities: Xác suất theo class.
        k: Số ứng viên đầu.

    Returns:
        Tỷ lệ mẫu có nhãn đúng trong top-k.
    """
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == labels[:, None], axis=1)))


def nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Tính negative log-likelihood đa lớp."""
    values = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(values).mean())


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    """Tính expected calibration error theo confidence top-1."""
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return total


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> dict[str, float]:
    """Tổng hợp metric in-domain cho một backend.

    Args:
        labels: Nhãn đúng.
        probabilities: Xác suất theo class.
        bins: Số bin tính ECE.

    Returns:
        Dictionary accuracy, macro-F1, top-3, NLL và ECE.
    """
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "top_3_accuracy": top_k_accuracy(labels, probabilities, min(3, probabilities.shape[1])),
        "nll": nll(labels, probabilities),
        "ece": ece(labels, probabilities, bins),
    }


def ood_metrics(
    in_domain_probabilities: np.ndarray,
    ood_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Đo AUROC/AUPRC và hành vi UNKNOWN cho OOD.

    Args:
        in_domain_probabilities: Xác suất trên mẫu in-domain.
        ood_probabilities: Xác suất trên mẫu OOD.
        threshold: Ngưỡng từ chối dựa trên confidence top-1.

    Returns:
        Metric binary với lớp dương là OOD.
    """
    in_confidence = in_domain_probabilities.max(axis=1)
    ood_confidence = ood_probabilities.max(axis=1)
    scores = np.concatenate([1 - in_confidence, 1 - ood_confidence])
    labels = np.concatenate([np.zeros(len(in_confidence)), np.ones(len(ood_confidence))])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "ood_unknown_recall": float(np.mean(ood_confidence < threshold)),
        "in_domain_false_rejection_rate": float(np.mean(in_confidence < threshold)),
    }


def safe_metric_call(function: Any, *args: Any, **kwargs: Any) -> float | None:
    """Gọi metric binary an toàn khi split chỉ có một lớp.

    Args:
        function: Hàm metric sklearn.
        args: Đối số vị trí.
        kwargs: Đối số keyword.

    Returns:
        Giá trị metric hoặc None nếu không xác định.
    """
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return None
