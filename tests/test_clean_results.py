import pandas as pd
import pytest

from src.data.clean_results import (
    OUTCOME_DRAW,
    OUTCOME_TEAM_A_WIN,
    OUTCOME_TEAM_B_WIN,
    clean_results,
    label_outcome,
)


def test_label_outcome_team_a_win() -> None:
    assert label_outcome(2, 1) == OUTCOME_TEAM_A_WIN


def test_label_outcome_draw() -> None:
    assert label_outcome(1, 1) == OUTCOME_DRAW


def test_label_outcome_team_b_win() -> None:
    assert label_outcome(0, 3) == OUTCOME_TEAM_B_WIN


def test_clean_results_adds_outcome_labels() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "team_a": ["A", "B", "C"],
            "team_b": ["D", "E", "F"],
            "team_a_score": [2, 1, 0],
            "team_b_score": [1, 1, 3],
        }
    )

    cleaned = clean_results(raw)

    assert cleaned["outcome"].tolist() == [
        OUTCOME_TEAM_A_WIN,
        OUTCOME_DRAW,
        OUTCOME_TEAM_B_WIN,
    ]


def test_clean_results_requires_expected_columns() -> None:
    raw = pd.DataFrame({"date": ["2024-01-01"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        clean_results(raw)
