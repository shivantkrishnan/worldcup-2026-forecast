"""Baseline model training entrypoint."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_model(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """Train a simple multinomial logistic regression baseline."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    multi_class="multinomial",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features, target)
    return model
