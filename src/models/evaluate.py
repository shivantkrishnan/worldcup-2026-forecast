"""Evaluation helpers for probabilistic match outcome models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def multiclass_brier_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
    class_labels: list[str],
) -> float:
    """Compute the multiclass Brier score for ordered class probabilities."""
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    encoded = np.zeros_like(probabilities, dtype=float)

    for row_index, label in enumerate(y_true):
        encoded[row_index, label_to_index[label]] = 1.0

    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    class_labels: list[str],
) -> dict[str, float]:
    """Return primary probabilistic evaluation metrics."""
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=class_labels)),
        "brier_score": multiclass_brier_score(y_true, probabilities, class_labels),
    }
