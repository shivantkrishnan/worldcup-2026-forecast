"""First probabilistic baseline modeling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.clean_results import (
    OUTCOME_DRAW,
    OUTCOME_TEAM_A_WIN,
    OUTCOME_TEAM_B_WIN,
)
from src.data.splits import chronological_train_test_split, summarize_split
from src.features.feature_audit import get_feature_columns
from src.models.metrics import (
    calibration_by_confidence_bin,
    evaluate_probabilistic_predictions,
)

DEFAULT_CLASS_LABELS = [
    OUTCOME_TEAM_A_WIN,
    OUTCOME_DRAW,
    OUTCOME_TEAM_B_WIN,
]


def _target_distribution(df: pd.DataFrame, target_col: str = "result") -> dict[str, int]:
    """Return target distribution as plain Python ints."""
    return {
        str(label): int(count)
        for label, count in df[target_col].value_counts().to_dict().items()
    }


def _predict_proba_in_class_order(
    model: Pipeline,
    features: pd.DataFrame,
    class_labels: list[str],
) -> np.ndarray:
    """Return model probabilities reordered into the requested class order."""
    raw_proba = model.predict_proba(features)
    model_classes = list(model.classes_)
    class_index = {label: index for index, label in enumerate(model_classes)}
    return np.column_stack([raw_proba[:, class_index[label]] for label in class_labels])


def build_class_prior_baseline(
    y_train: pd.Series,
    class_labels: list[str],
) -> np.ndarray:
    """Return class-prior probabilities in class_labels order."""
    counts = y_train.value_counts(normalize=True)
    return np.array([float(counts.get(label, 0.0)) for label in class_labels])


def predict_class_prior(n_rows: int, class_prior: np.ndarray) -> np.ndarray:
    """Return repeated class-prior probabilities for n_rows predictions."""
    return np.tile(class_prior, (n_rows, 1))


def build_logistic_regression_pipeline() -> Pipeline:
    """Build the first baseline logistic regression preprocessing/model pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def train_baseline_model(
    features_df: pd.DataFrame,
    test_start_date: str = "2022-01-01",
) -> dict[str, object]:
    """Train and evaluate class-prior and logistic-regression baselines."""
    feature_columns = get_feature_columns(features_df)
    train_df, test_df = chronological_train_test_split(
        features_df,
        test_start_date=test_start_date,
        date_col="match_date",
    )
    split_summary = summarize_split(train_df, test_df)

    x_train = train_df[feature_columns]
    y_train = train_df["result"]
    x_test = test_df[feature_columns]
    y_test = test_df["result"]

    class_labels = DEFAULT_CLASS_LABELS.copy()
    class_prior = build_class_prior_baseline(y_train, class_labels)
    class_prior_proba = predict_class_prior(len(test_df), class_prior)
    class_prior_metrics = evaluate_probabilistic_predictions(
        y_test,
        class_prior_proba,
        class_labels,
    )

    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(x_train, y_train)
    logistic_proba = _predict_proba_in_class_order(pipeline, x_test, class_labels)
    logistic_regression_metrics = evaluate_probabilistic_predictions(
        y_test,
        logistic_proba,
        class_labels,
    )
    calibration_table_logistic = calibration_by_confidence_bin(
        y_test,
        logistic_proba,
        class_labels,
    )

    return {
        "feature_columns": feature_columns,
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
        "target_distribution_train": _target_distribution(train_df),
        "target_distribution_test": _target_distribution(test_df),
        "class_prior": class_prior,
        "class_prior_metrics": class_prior_metrics,
        "logistic_regression_metrics": logistic_regression_metrics,
        "calibration_table_logistic": calibration_table_logistic,
        "fitted_pipeline": pipeline,
        "class_labels": class_labels,
    }
