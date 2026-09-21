"""Kiểm thử calibration và quy tắc UNKNOWN tại biên threshold."""

import numpy as np

from intentguard.calibration import apply_temperature, choose_threshold, fit_temperature


def test_temperature_outputs_normalized_probabilities() -> None:
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
    calibrated = apply_temperature(probabilities, 2.0)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert calibrated.shape == probabilities.shape


def test_threshold_respects_maximum_rejection_rate() -> None:
    probabilities = np.array([[0.95, 0.05], [0.80, 0.20], [0.60, 0.40], [0.55, 0.45]])
    threshold = choose_threshold(probabilities, max_rejection_rate=0.25)
    rejected = np.mean(probabilities.max(axis=1) < threshold)
    assert rejected <= 0.25
    assert np.isclose(threshold, 0.60)


def test_fit_temperature_is_positive() -> None:
    probabilities = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    assert fit_temperature(probabilities, np.array([0, 0, 1])) > 0
