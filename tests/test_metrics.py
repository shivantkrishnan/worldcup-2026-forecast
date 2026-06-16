import numpy as np
import pandas as pd

from src.models.metrics import (
    calibration_by_confidence_bin,
    classwise_calibration_table,
    evaluate_probabilistic_predictions,
    expected_calibration_error,
    multiclass_brier_score,
    summarize_calibration,
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


def test_ece_is_near_zero_for_perfectly_calibrated_toy_case() -> None:
    y_true = pd.Series(["a", "b"])
    y_proba = np.array([[0.5, 0.5], [0.5, 0.5]])

    assert expected_calibration_error(y_true, y_proba, ["a", "b"], n_bins=2) == 0.0


def test_ece_is_positive_for_overconfident_wrong_toy_case() -> None:
    y_true = pd.Series(["b", "b"])
    y_proba = np.array([[0.9, 0.1], [0.8, 0.2]])

    assert expected_calibration_error(y_true, y_proba, ["a", "b"], n_bins=2) > 0.0


def test_classwise_calibration_counts_sum_sensibly() -> None:
    y_true = pd.Series(["a", "b", "a"])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])

    table = classwise_calibration_table(y_true, y_proba, ["a", "b"], n_bins=2)

    assert table["count"].sum() == 6
    assert table.groupby("class_label")["count"].sum().to_dict() == {"a": 3, "b": 3}


def test_summarize_calibration_returns_expected_keys() -> None:
    summary = summarize_calibration(
        pd.Series(["a", "b"]),
        np.array([[0.6, 0.4], [0.3, 0.7]]),
        ["a", "b"],
        n_bins=2,
    )

    assert set(summary) == {
        "expected_calibration_error",
        "confidence_calibration_table",
        "classwise_calibration_table",
    }
