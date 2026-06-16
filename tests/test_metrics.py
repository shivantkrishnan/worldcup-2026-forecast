import numpy as np
import pandas as pd

from src.models.metrics import (
    calibration_by_confidence_bin,
    evaluate_probabilistic_predictions,
    multiclass_brier_score,
)


def test_multiclass_brier_score_known_examples() -> None:
    y_true = pd.Series(["a", "b"])
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    uncertain = np.array([[0.5, 0.5], [0.5, 0.5]])

    assert multiclass_brier_score(y_true, perfect, ["a", "b"]) == 0.0
    assert multiclass_brier_score(y_true, uncertain, ["a", "b"]) == 0.5


def test_evaluate_probabilistic_predictions_returns_expected_keys() -> None:
    metrics = evaluate_probabilistic_predictions(
        pd.Series(["a", "b"]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
        ["a", "b"],
    )

    assert set(metrics) == {
        "log_loss",
        "multiclass_brier_score",
        "accuracy",
        "class_labels",
        "prediction_count",
    }


def test_accuracy_is_computed_correctly() -> None:
    metrics = evaluate_probabilistic_predictions(
        pd.Series(["a", "b", "b"]),
        np.array([[0.8, 0.2], [0.9, 0.1], [0.3, 0.7]]),
        ["a", "b"],
    )

    assert metrics["accuracy"] == 2 / 3


def test_calibration_bin_counts_sum_to_prediction_count() -> None:
    calibration = calibration_by_confidence_bin(
        pd.Series(["a", "b", "a"]),
        np.array([[0.6, 0.4], [0.2, 0.8], [0.55, 0.45]]),
        ["a", "b"],
        n_bins=5,
    )

    assert calibration["count"].sum() == 3


def test_class_label_ordering_is_respected() -> None:
    y_true = pd.Series(["b"])
    proba = np.array([[0.0, 1.0]])

    assert multiclass_brier_score(y_true, proba, ["a", "b"]) == 0.0
    assert multiclass_brier_score(y_true, proba, ["b", "a"]) == 2.0
