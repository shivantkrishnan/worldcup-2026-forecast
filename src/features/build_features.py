"""Feature-building utilities.

Rolling features must only use matches before the prediction date. This module
starts as a placeholder so that future feature work has a clear home.
"""

import pandas as pd


def build_match_features(results: pd.DataFrame) -> pd.DataFrame:
    """Return a feature dataframe for model training.

    The MVP implementation will add leakage-safe rolling team statistics here.
    """
    return results.copy()
