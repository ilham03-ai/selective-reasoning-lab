from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes, dtype=float)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for idx in range(num_bins):
        left, right = bins[idx], bins[idx + 1]
        mask = (confidences >= left) & (confidences < right if idx < num_bins - 1 else confidences <= right)
        if not np.any(mask):
            continue
        bin_accuracy = correctness[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * mask.mean()
    return float(ece)


def calibration_curve_data(y_true: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> dict[str, np.ndarray]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = []
    bin_accuracy = []
    bin_confidence = []
    bin_counts = []
    for idx in range(num_bins):
        left, right = bins[idx], bins[idx + 1]
        mask = (confidences >= left) & (confidences < right if idx < num_bins - 1 else confidences <= right)
        if not np.any(mask):
            continue
        bin_centers.append((left + right) / 2.0)
        bin_accuracy.append(correctness[mask].mean())
        bin_confidence.append(confidences[mask].mean())
        bin_counts.append(mask.sum())
    return {
        "bin_centers": np.asarray(bin_centers, dtype=float),
        "bin_accuracy": np.asarray(bin_accuracy, dtype=float),
        "bin_confidence": np.asarray(bin_confidence, dtype=float),
        "bin_counts": np.asarray(bin_counts, dtype=float),
    }


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    predictions = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "brier_score": brier_score_multiclass(y_true, probs, probs.shape[1]),
        "ece": expected_calibration_error(y_true, probs),
    }


def selective_error_statistics(
    y_true: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    num_bins: int = 8,
) -> dict[str, np.ndarray]:
    predictions = probs.argmax(axis=1)
    errors = (predictions != y_true).astype(float)
    bins = np.quantile(uncertainty, np.linspace(0.0, 1.0, num_bins + 1))
    xs = []
    ys = []
    counts = []
    for idx in range(num_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == num_bins - 1:
            mask = (uncertainty >= left) & (uncertainty <= right)
        else:
            mask = (uncertainty >= left) & (uncertainty < right)
        if not np.any(mask):
            continue
        xs.append(float(uncertainty[mask].mean()))
        ys.append(float(errors[mask].mean()))
        counts.append(int(mask.sum()))
    return {
        "uncertainty_mean": np.asarray(xs, dtype=float),
        "error_rate": np.asarray(ys, dtype=float),
        "counts": np.asarray(counts, dtype=float),
    }


def mean_dict(records: Iterable[dict[str, float]]) -> dict[str, float]:
    records = list(records)
    if not records:
        return {}
    keys = records[0].keys()
    return {key: float(np.mean([record[key] for record in records])) for key in keys}
