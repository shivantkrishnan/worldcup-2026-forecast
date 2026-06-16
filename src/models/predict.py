"""Prediction helpers for trained match outcome models."""

from __future__ import annotations

import pandas as pd


def predict_match_probabilities(model: object, features: pd.DataFrame) -> pd.DataFrame:
    """Return class probabilities from a fitted scikit-learn style model."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba.")

    probabilities = model.predict_proba(features)
    class_labels = list(model.classes_)
    return pd.DataFrame(probabilities, columns=class_labels, index=features.index)
