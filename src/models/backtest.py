"""Rolling-origin backtesting utilities for baseline match models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.splits import rolling_origin_splits, summarize_split
from src.features.feature_audit import get_feature_columns
from src.models.baseline import (
    DEFAULT_CLASS_LABELS,
    _predict_proba_in_class_order,
    build_calibrated_logistic_regression_pipeline,
    build_class_prior_baseline,
    build_logistic_regression_pipeline,
    predict_class_prior,
)
from src.models.metrics import (
    evaluate_probabilistic_predictions,
    summarize_calibration,
)

METRIC_KEYS = ["log_loss", "multiclass_brier_score", "accuracy"]
MODEL_PREFIXES = [
    "class_prior",
    "logistic_regression",
    "calibrated_logistic_regression",
]


def _target_distribution(df: pd.DataFrame, target_col: str = "result") -> dict[str, int]:
    """Return target distribution as plain Python ints."""
    return {
        str(label): int(count)
        for label, count in df[target_col].value_counts().to_dict().items()
    }


def _validate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_labels: list[str],
    include_calibrated: bool,
    calibration_cv: int,
) -> None:
    """Raise a clear error when a split cannot support baseline fitting."""
    if train_df.empty:
        raise ValueError("Train split has zero rows.")
    if test_df.empty:
        raise ValueError("Test split has zero rows.")
    if "result" not in train_df.columns or "result" not in test_df.columns:
        raise ValueError("Both train_df and test_df must include a result column.")

    train_classes = set(train_df["result"].dropna().unique())
    missing_classes = [label for label in class_labels if label not in train_classes]
    if missing_classes:
        raise ValueError(
            "Train split is missing target classes: "
            + ", ".join(sorted(missing_classes))
        )

    if include_calibrated:
        class_counts = train_df["result"].value_counts()
        underrepresented = [
            label
            for label in class_labels
            if int(class_counts.get(label, 0)) < calibration_cv
        ]
        if underrepresented:
            raise ValueError(
                f"Train split has fewer than {calibration_cv} rows for calibration "
                "CV classes: "
                + ", ".join(sorted(underrepresented))
            )


def _base_split_result(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_count: int,
) -> dict[str, object]:
    """Return date/count metadata shared by evaluated and skipped splits."""
    split_summary = summarize_split(train_df, test_df)
    return {
        "train_row_count": int(len(train_df)),
        "test_row_count": int(len(test_df)),
        "train_date_range": (
            split_summary["train_start_date"],
            split_summary["train_end_date"],
        ),
        "test_date_range": (
            split_summary["test_start_date"],
            split_summary["test_end_date"],
        ),
        "target_distribution_train": _target_distribution(train_df)
        if "result" in train_df.columns
        else {},
        "target_distribution_test": _target_distribution(test_df)
        if "result" in test_df.columns
        else {},
        "feature_count": int(feature_count),
    }


def train_and_evaluate_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    include_calibrated: bool = True,
    class_labels: list[str] | None = None,
) -> dict[str, object]:
    """Train and evaluate baseline models on one time-ordered split."""
    train = train_df.copy(deep=True)
    test = test_df.copy(deep=True)
    labels = (
        class_labels.copy() if class_labels is not None else DEFAULT_CLASS_LABELS.copy()
    )
    selected_features = (
        feature_columns.copy()
        if feature_columns is not None
        else get_feature_columns(train)
    )

    if not selected_features:
        raise ValueError("No numeric feature columns available for modeling.")

    _validate_split(
        train,
        test,
        labels,
        include_calibrated=include_calibrated,
        calibration_cv=3,
    )

    x_train = train[selected_features]
    y_train = train["result"]
    x_test = test[selected_features]
    y_test = test["result"]

    class_prior = build_class_prior_baseline(y_train, labels)
    class_prior_proba = predict_class_prior(len(test), class_prior)
    class_prior_metrics = evaluate_probabilistic_predictions(
        y_test,
        class_prior_proba,
        labels,
    )
    class_prior_calibration_summary = summarize_calibration(
        y_test,
        class_prior_proba,
        labels,
    )

    logistic_model = build_logistic_regression_pipeline()
    logistic_model.fit(x_train, y_train)
    logistic_proba = _predict_proba_in_class_order(logistic_model, x_test, labels)
    logistic_metrics = evaluate_probabilistic_predictions(
        y_test,
        logistic_proba,
        labels,
    )
    logistic_calibration_summary = summarize_calibration(
        y_test,
        logistic_proba,
        labels,
    )

    calibrated_metrics = None
    calibrated_calibration_summary = None
    if include_calibrated:
        calibrated_model = build_calibrated_logistic_regression_pipeline()
        calibrated_model.fit(x_train, y_train)
        calibrated_proba = _predict_proba_in_class_order(
            calibrated_model,
            x_test,
            labels,
        )
        calibrated_metrics = evaluate_probabilistic_predictions(
            y_test,
            calibrated_proba,
            labels,
        )
        calibrated_calibration_summary = summarize_calibration(
            y_test,
            calibrated_proba,
            labels,
        )

    result = _base_split_result(train, test, len(selected_features))
    result.update(
        {
            "skipped": False,
            "messages": ["Split evaluated successfully."],
            "feature_columns": selected_features,
            "class_labels": labels,
            "class_prior_metrics": class_prior_metrics,
            "class_prior_calibration_summary": class_prior_calibration_summary,
            "logistic_regression_metrics": logistic_metrics,
            "logistic_calibration_summary": logistic_calibration_summary,
            "calibrated_logistic_regression_metrics": calibrated_metrics,
            "calibrated_logistic_calibration_summary": (
                calibrated_calibration_summary
            ),
        }
    )
    return result


def _skipped_split_result(
    split_id: int,
    split: dict[str, object],
    message: str,
    feature_count: int,
) -> dict[str, object]:
    """Return a structured skipped result for invalid rolling windows."""
    train_df = split["train_df"]
    test_df = split["test_df"]
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    result = _base_split_result(train_df, test_df, feature_count)
    result.update(
        {
            "split_id": split_id,
            "skipped": True,
            "messages": [message],
            "class_prior_metrics": None,
            "class_prior_calibration_summary": None,
            "logistic_regression_metrics": None,
            "logistic_calibration_summary": None,
            "calibrated_logistic_regression_metrics": None,
            "calibrated_logistic_calibration_summary": None,
        }
    )
    return result


def run_rolling_origin_backtest(
    features_df: pd.DataFrame,
    initial_train_end_date: str = "2014-12-31",
    test_window_months: int = 24,
    step_months: int = 24,
    final_test_end_date: str = "2026-06-10",
    include_calibrated: bool = True,
) -> list[dict[str, object]]:
    """Run expanding-window rolling-origin backtests without writing files."""
    features = features_df.copy(deep=True)
    feature_columns = get_feature_columns(features)
    splits = rolling_origin_splits(
        features,
        initial_train_end_date=initial_train_end_date,
        test_window_months=test_window_months,
        step_months=step_months,
        final_test_end_date=final_test_end_date,
        date_col="match_date",
    )

    results: list[dict[str, object]] = []
    for split_id, split in enumerate(splits, start=1):
        train_df = split["train_df"]
        test_df = split["test_df"]
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        try:
            result = train_and_evaluate_on_split(
                train_df,
                test_df,
                feature_columns=feature_columns,
                include_calibrated=include_calibrated,
                class_labels=DEFAULT_CLASS_LABELS.copy(),
            )
            result["split_id"] = split_id
        except ValueError as error:
            result = _skipped_split_result(
                split_id,
                split,
                f"Split skipped: {error}",
                len(feature_columns),
            )
        results.append(result)

    return results


def _metric_value(metrics: dict[str, object] | None, metric_key: str) -> float:
    """Safely extract a metric from an optional metrics dictionary."""
    if not metrics:
        return float("nan")
    return float(metrics[metric_key])


def _ece_value(summary: dict[str, object] | None) -> float:
    """Safely extract expected calibration error from an optional summary."""
    if not summary:
        return float("nan")
    return float(summary["expected_calibration_error"])


def summarize_backtest_results(
    backtest_results: list[dict[str, object]],
) -> pd.DataFrame:
    """Return one summary row per rolling-origin split."""
    rows: list[dict[str, object]] = []
    for result in backtest_results:
        row: dict[str, object] = {
            "split_id": int(result["split_id"]),
            "skipped": bool(result.get("skipped", False)),
            "messages": " | ".join(str(message) for message in result["messages"]),
            "train_start_date": result["train_date_range"][0],
            "train_end_date": result["train_date_range"][1],
            "test_start_date": result["test_date_range"][0],
            "test_end_date": result["test_date_range"][1],
            "train_row_count": int(result["train_row_count"]),
            "test_row_count": int(result["test_row_count"]),
            "feature_count": int(result["feature_count"]),
        }

        for prefix, metrics_key, calibration_key in [
            (
                "class_prior",
                "class_prior_metrics",
                "class_prior_calibration_summary",
            ),
            (
                "logistic_regression",
                "logistic_regression_metrics",
                "logistic_calibration_summary",
            ),
            (
                "calibrated_logistic_regression",
                "calibrated_logistic_regression_metrics",
                "calibrated_logistic_calibration_summary",
            ),
        ]:
            metrics = result.get(metrics_key)
            calibration_summary = result.get(calibration_key)
            for metric_key in METRIC_KEYS:
                row[f"{prefix}_{metric_key}"] = _metric_value(metrics, metric_key)
            row[f"{prefix}_ece"] = _ece_value(calibration_summary)

        rows.append(row)

    return pd.DataFrame(rows)


def _mean_std(series: pd.Series) -> tuple[float | None, float | None]:
    """Return mean and sample std for a metric series, preserving empty metrics."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return (None, None)
    return (float(numeric.mean()), float(numeric.std(ddof=1)))


def _comparison_count_and_share(mask: pd.Series) -> tuple[int, float | None]:
    """Return count and share for a boolean comparison with missing values."""
    valid = mask.dropna()
    if valid.empty:
        return (0, None)
    count = int(valid.sum())
    return (count, float(count / len(valid)))


def _valid_comparison(
    summary_df: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> pd.Series:
    """Return nullable boolean comparison where both metric values exist."""
    valid = summary_df[[left_col, right_col]].notna().all(axis=1)
    comparison = pd.Series(pd.NA, index=summary_df.index, dtype="boolean")
    comparison.loc[valid] = summary_df.loc[valid, left_col] < summary_df.loc[
        valid, right_col
    ]
    return comparison


def aggregate_backtest_results(
    backtest_summary_df: pd.DataFrame,
) -> dict[str, object]:
    """Aggregate rolling-origin split metrics and model win counts."""
    messages: list[str] = []
    if backtest_summary_df.empty:
        return {
            "split_count": 0,
            "evaluated_split_count": 0,
            "model_metrics": {},
            "logistic_beats_class_prior_log_loss_count": 0,
            "logistic_beats_class_prior_log_loss_share": None,
            "calibrated_beats_logistic_log_loss_count": 0,
            "calibrated_beats_logistic_log_loss_share": None,
            "calibrated_beats_logistic_ece_count": 0,
            "calibrated_beats_logistic_ece_share": None,
            "best_overall_model_by_mean_log_loss": None,
            "messages": ["No backtest splits were available."],
        }

    evaluated = backtest_summary_df.loc[~backtest_summary_df["skipped"]].copy()
    split_count = int(len(backtest_summary_df))
    evaluated_split_count = int(len(evaluated))
    if evaluated_split_count < split_count:
        messages.append(
            f"{split_count - evaluated_split_count} split(s) were skipped."
        )

    model_metrics: dict[str, dict[str, float | None]] = {}
    for prefix in MODEL_PREFIXES:
        metrics: dict[str, float | None] = {}
        for metric_key in [*METRIC_KEYS, "ece"]:
            mean_value, std_value = _mean_std(evaluated[f"{prefix}_{metric_key}"])
            metrics[f"{metric_key}_mean"] = mean_value
            metrics[f"{metric_key}_std"] = std_value
        model_metrics[prefix] = metrics

    logistic_vs_prior = _valid_comparison(
        evaluated,
        "logistic_regression_log_loss",
        "class_prior_log_loss",
    )
    calibrated_vs_logistic_loss = _valid_comparison(
        evaluated,
        "calibrated_logistic_regression_log_loss",
        "logistic_regression_log_loss",
    )
    calibrated_vs_logistic_ece = _valid_comparison(
        evaluated,
        "calibrated_logistic_regression_ece",
        "logistic_regression_ece",
    )

    logistic_count, logistic_share = _comparison_count_and_share(logistic_vs_prior)
    calibrated_loss_count, calibrated_loss_share = _comparison_count_and_share(
        calibrated_vs_logistic_loss
    )
    calibrated_ece_count, calibrated_ece_share = _comparison_count_and_share(
        calibrated_vs_logistic_ece
    )

    mean_log_losses = {
        prefix: metrics["log_loss_mean"]
        for prefix, metrics in model_metrics.items()
        if metrics["log_loss_mean"] is not None
    }
    best_model = (
        min(mean_log_losses, key=mean_log_losses.get) if mean_log_losses else None
    )
    if best_model:
        messages.append(f"Best mean log loss model: {best_model}.")
    else:
        messages.append("No evaluated metrics were available.")

    return {
        "split_count": split_count,
        "evaluated_split_count": evaluated_split_count,
        "model_metrics": model_metrics,
        "logistic_beats_class_prior_log_loss_count": logistic_count,
        "logistic_beats_class_prior_log_loss_share": logistic_share,
        "calibrated_beats_logistic_log_loss_count": calibrated_loss_count,
        "calibrated_beats_logistic_log_loss_share": calibrated_loss_share,
        "calibrated_beats_logistic_ece_count": calibrated_ece_count,
        "calibrated_beats_logistic_ece_share": calibrated_ece_share,
        "best_overall_model_by_mean_log_loss": best_model,
        "messages": messages,
    }


def format_metric_mean_std(value: Any, std: Any) -> str:
    """Format an aggregate metric pair for scripts."""
    if value is None or pd.isna(value):
        return "n/a"
    if std is None or pd.isna(std):
        return f"{value:.6f} +/- n/a"
    return f"{value:.6f} +/- {std:.6f}"
