import pandas as pd
import numpy as np

from scripts.compare_prediction_modes import compare_prediction_modes


def test_compare_prediction_modes_reports_probability_shifts() -> None:
    backfilled = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team_a": "Alpha",
                "team_b": "Beta",
                "p_team_a_win": 0.40,
                "p_draw": 0.30,
                "p_team_b_win": 0.30,
                "predicted_class": "team_a_win",
                "favorite_display": "Alpha",
                "confidence_label": "Low",
            }
        ]
    )
    live = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team_a": "Alpha",
                "team_b": "Beta",
                "p_team_a_win": 0.30,
                "p_draw": 0.25,
                "p_team_b_win": 0.45,
                "predicted_class": "team_b_win",
                "favorite_display": "Beta",
                "confidence_label": "Medium",
            }
        ]
    )

    comparison, team_summary = compare_prediction_modes(backfilled, live)

    assert len(comparison) == 1
    assert np.isclose(comparison.loc[0, "p_team_b_win_shift"], 0.15)
    assert bool(comparison.loc[0, "favorite_changed"]) is True
    assert bool(comparison.loc[0, "confidence_changed"]) is True
    assert set(team_summary["team"]) == {"Alpha", "Beta"}
