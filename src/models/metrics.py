"""Probabilistic evaluation metrics for match outcome models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


def _one_hot_encode(y_true: pd.Series | list[str], class_labels: list[str]) -> np.ndarray:
    """One-hot encode labels in the provided class order."""
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    encoded = np.zeros((len(y_true), len(class_labels)), dtype=float)

    for row_index, label in enumerate(y_true):
        encoded[row_index, label_to_index[label]] = 1.0

    return encoded


def multiclass_brier_score(
    y_true: pd.Series | list[str],
    y_proba: np.ndarray,
    class_labels: list[str],
) -> float:
    """Return mean summed squared probability error across classes."""
    encoded = _one_hot_encode(y_true, class_labels)
    return float(np.mean(np.sum((y_proba - encoded) ** 2, axis=1)))


def evaluate_probabilistic_predictions(
    y_true: pd.Series | list[str],
    y_proba: np.ndarray,
    class_labels: list[str],
) -> dict[str, object]:
    """Evaluate probabilistic predictions with primary and secondary metrics."""
    predicted_labels = [class_labels[index] for index in np.argmax(y_proba, axis=1)]

    return {
        "log_loss": float(log_loss(y_true, y_proba, labels=class_labels)),
        "multiclass_brier_score": multiclass_brier_score(
            y_true,
            y_proba,
            class_labels,
        ),
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "class_labels": class_labels,
        "prediction_count": int(len(y_true)),
    }


def calibration_by_confidence_bin(
    y_true: pd.Series | list[str],
    y_proba: np.ndarray,
    class_labels: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize calibration by max predicted probability confidence bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    confidence = np.max(y_proba, axis=1)
    predicted_labels = np.array(class_labels, dtype=object)[np.argmax(y_proba, axis=1)]
    y_true_array = np.array(list(y_true), dtype=object)
    correct = predicted_labels == y_true_array

    rows: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)

        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "average_confidence": float(np.mean(confidence[mask]))
                if count
                else float("nan"),
                "empirical_accuracy": float(np.mean(correct[mask]))
                if count
                else float("nan"),
            }
        )

    return pd.DataFrame(rows)
