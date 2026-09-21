"""Kiểm thử metric top-k và metric calibration."""

import numpy as np

from intentguard.metrics import classification_metrics, top_k_accuracy


def test_top_k_is_rank_based() -> None:
    probabilities = np.array([[0.1, 0.3, 0.6], [0.4, 0.35, 0.25]])
    labels = np.array([1, 2])
    assert top_k_accuracy(labels, probabilities, k=2) == 0.5


def test_classification_metrics_returns_expected_keys() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = classification_metrics(np.array([0, 1]), probabilities)
    assert {"accuracy", "macro_f1", "top_3_accuracy", "nll", "ece"} <= result.keys()
