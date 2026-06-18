"""In-memory forecast helpers for scheduled fixtures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build_features import build_modeling_features
from src.features.feature_audit import get_feature_columns
from src.models.baseline import (
    DEFAULT_CLASS_LABELS,
    build_calibrated_logistic_regression_pipeline,
)

SELECTED_ELO_K_FACTOR = 10.0
SELECTED_ELO_HOME_ADVANTAGE = 50.0
PROBABILITY_COLUMNS = ["p_team_a_win", "p_draw", "p_team_b_win"]


@dataclass(frozen=True)
class ForecastModelBundle:
    """Fitted selected-baseline model plus feature metadata."""

    model: object
    feature_columns: list[str]
    class_labels: list[str]
    include_elo: bool
    elo_k_factor: float
    elo_home_advantage: float


def _predict_proba_in_class_order(
    model: object,
    features: pd.DataFrame,
    class_labels: list[str],
) -> np.ndarray:
    """Return probabilities reordered into the configured class-label order."""
    probabilities = model.predict_proba(features)
    model_classes = list(model.classes_)
    class_index = {label: index for index, label in enumerate(model_classes)}
    return np.column_stack([probabilities[:, class_index[label]] for label in class_labels])


def train_selected_baseline(training_matches: pd.DataFrame) -> ForecastModelBundle:
    """Train the selected calibrated logistic baseline in memory only."""
    matches = training_matches.copy(deep=True)
    features = build_modeling_features(
        matches,
        include_elo=True,
        elo_k_factor=SELECTED_ELO_K_FACTOR,
        elo_home_advantage=SELECTED_ELO_HOME_ADVANTAGE,
    )
    feature_columns = get_feature_columns(features)
    model = build_calibrated_logistic_regression_pipeline()
    model.fit(features[feature_columns], features["result"])

    return ForecastModelBundle(
        model=model,
        feature_columns=feature_columns,
        class_labels=DEFAULT_CLASS_LABELS.copy(),
        include_elo=True,
        elo_k_factor=SELECTED_ELO_K_FACTOR,
        elo_home_advantage=SELECTED_ELO_HOME_ADVANTAGE,
    )


def predict_fixture_probabilities(
    model_bundle: ForecastModelBundle,
    fixture_feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return 3-class probabilities for fixture feature rows."""
    features = fixture_feature_rows.copy(deep=True)
    missing = [
        column for column in model_bundle.feature_columns if column not in features.columns
    ]
    if missing:
        raise ValueError(
            "Fixture feature rows are missing model feature columns: "
            + ", ".join(missing)
        )

    probabilities = _predict_proba_in_class_order(
        model_bundle.model,
        features[model_bundle.feature_columns],
        model_bundle.class_labels,
    )
    return pd.DataFrame(
        probabilities,
        columns=PROBABILITY_COLUMNS,
        index=features.index,
    )


def _confidence_label(probabilities: pd.Series) -> str:
    """Return a display-only confidence label, not a statistical interval."""
    sorted_probabilities = probabilities.sort_values(ascending=False)
    max_probability = float(sorted_probabilities.iloc[0])
    margin = float(sorted_probabilities.iloc[0] - sorted_probabilities.iloc[1])

    if max_probability >= 0.60 and margin >= 0.15:
        return "High"
    if max_probability >= 0.45 and margin >= 0.08:
        return "Medium"
    return "Low"


def _favorite_display(predicted_class: str, row: pd.Series) -> str:
    """Map a model class label to a user-facing favorite."""
    if predicted_class == "team_a_win":
        return str(row["team_a"])
    if predicted_class == "team_b_win":
        return str(row["team_b"])
    return "Draw"


def format_prediction_output(
    predictions: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Return user-facing fixture prediction output."""
    probabilities = predictions.copy(deep=True)
    fixture_rows = fixtures.copy(deep=True).reset_index(drop=True)
    probabilities = probabilities.reset_index(drop=True)

    if len(probabilities) != len(fixture_rows):
        raise ValueError("predictions and fixtures must have the same number of rows.")

    missing_probability_columns = [
        column for column in PROBABILITY_COLUMNS if column not in probabilities.columns
    ]
    if missing_probability_columns:
        raise ValueError(
            "Predictions are missing probability columns: "
            + ", ".join(missing_probability_columns)
        )

    output = fixture_rows[["match_id", "match_date", "team_a", "team_b"]].copy()
    output = pd.concat([output, probabilities[PROBABILITY_COLUMNS]], axis=1)
    probability_to_class = {
        "p_team_a_win": "team_a_win",
        "p_draw": "draw",
        "p_team_b_win": "team_b_win",
    }
    output["predicted_class"] = probabilities[PROBABILITY_COLUMNS].idxmax(axis=1).map(
        probability_to_class
    )
    output["favorite_display"] = output.apply(
        lambda row: _favorite_display(str(row["predicted_class"]), row),
        axis=1,
    )
    output["confidence_label"] = probabilities[PROBABILITY_COLUMNS].apply(
        _confidence_label,
        axis=1,
    )

    return output[
        [
            "match_id",
            "match_date",
            "team_a",
            "team_b",
            "p_team_a_win",
            "p_draw",
            "p_team_b_win",
            "predicted_class",
            "favorite_display",
            "confidence_label",
        ]
    ]
