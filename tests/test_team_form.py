import pandas as pd

from src.data.clean_results import clean_results
from src.features.team_form import (
    build_match_level_features,
    build_team_match_panel,
    add_team_rolling_features,
)

CORE_TEAM_PANEL_COLUMNS = [
    "match_id",
    "match_date",
    "team",
    "opponent",
    "goals_for",
    "goals_against",
    "goal_diff",
    "points",
    "win",
    "draw",
    "loss",
    "is_team_a",
    "is_home",
    "is_neutral",
    "tournament",
    "result_from_team_perspective",
]


def make_canonical_matches() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-05",
                "2020-01-10",
                "2020-01-15",
                "2020-01-20",
                "2020-01-25",
            ],
            "home_team": ["Team A", "Team C", "Team A", "Team D", "Team A", "Team E"],
            "away_team": ["Team B", "Team A", "Team C", "Team A", "Team D", "Team A"],
            "home_score": [2, 1, 1, 0, 3, 2],
            "away_score": [0, 0, 1, 2, 1, 0],
            "tournament": ["Friendly"] * 6,
            "city": [f"City {index}" for index in range(6)],
            "country": ["Country"] * 6,
            "neutral": [False, True, False, False, False, False],
        }
    )
    return clean_results(raw)


def test_build_team_match_panel_creates_two_rows_per_match() -> None:
    canonical = make_canonical_matches()

    panel = build_team_match_panel(canonical)

    assert len(panel) == len(canonical) * 2


def test_team_perspective_goals_points_and_results_are_correct() -> None:
    canonical = make_canonical_matches()

    panel = build_team_match_panel(canonical)
    first_match_rows = panel.loc[panel["match_id"] == canonical.loc[0, "match_id"]]
    team_a_row = first_match_rows.loc[first_match_rows["team"] == "Team A"].iloc[0]
    team_b_row = first_match_rows.loc[first_match_rows["team"] == "Team B"].iloc[0]

    assert team_a_row["goals_for"] == 2
    assert team_a_row["goals_against"] == 0
    assert team_a_row["points"] == 3
    assert team_a_row["win"] == 1
    assert team_a_row["draw"] == 0
    assert team_a_row["loss"] == 0
    assert team_a_row["result_from_team_perspective"] == "win"

    assert team_b_row["goals_for"] == 0
    assert team_b_row["goals_against"] == 2
    assert team_b_row["points"] == 0
    assert team_b_row["win"] == 0
    assert team_b_row["draw"] == 0
    assert team_b_row["loss"] == 1
    assert team_b_row["result_from_team_perspective"] == "loss"


def test_is_home_is_correct_for_neutral_and_non_neutral_matches() -> None:
    canonical = make_canonical_matches()

    panel = build_team_match_panel(canonical)
    non_neutral_rows = panel.loc[panel["match_id"] == canonical.loc[0, "match_id"]]
    neutral_rows = panel.loc[panel["match_id"] == canonical.loc[1, "match_id"]]

    assert non_neutral_rows.loc[non_neutral_rows["team"] == "Team A", "is_home"].item()
    assert not non_neutral_rows.loc[
        non_neutral_rows["team"] == "Team B", "is_home"
    ].item()
    assert neutral_rows["is_home"].tolist() == [False, False]


def test_rolling_features_do_not_include_current_match() -> None:
    canonical = make_canonical_matches()
    panel = add_team_rolling_features(build_team_match_panel(canonical), windows=(5,))

    team_a_rows = panel.loc[panel["team"] == "Team A"].reset_index(drop=True)
    sixth_match = team_a_rows.iloc[5]

    assert sixth_match["rolling_points_per_match_5"] == 2.0
    assert sixth_match["rolling_goals_for_avg_5"] == 1.6
    assert sixth_match["rolling_goals_against_avg_5"] == 0.6
    assert sixth_match["rolling_win_rate_5"] == 0.6


def test_add_team_rolling_features_preserves_core_panel_columns() -> None:
    canonical = make_canonical_matches()
    panel = add_team_rolling_features(build_team_match_panel(canonical), windows=(5,))

    for column in CORE_TEAM_PANEL_COLUMNS:
        assert column in panel.columns


def test_first_match_has_missing_prior_history_features() -> None:
    canonical = make_canonical_matches()
    panel = add_team_rolling_features(build_team_match_panel(canonical), windows=(5,))

    first_team_a_match = panel.loc[panel["team"] == "Team A"].iloc[0]

    assert first_team_a_match["matches_played_before"] == 0
    assert pd.isna(first_team_a_match["days_since_last_match"])
    assert pd.isna(first_team_a_match["rolling_points_per_match_5"])
    assert pd.isna(first_team_a_match["expanding_points_per_match"])


def test_days_since_last_match_uses_previous_match_only() -> None:
    canonical = make_canonical_matches()
    panel = add_team_rolling_features(build_team_match_panel(canonical), windows=(5,))

    team_a_rows = panel.loc[panel["team"] == "Team A"].reset_index(drop=True)

    assert team_a_rows.loc[1, "days_since_last_match"] == 4
    assert team_a_rows.loc[2, "days_since_last_match"] == 5


def test_build_match_level_features_returns_one_row_per_match() -> None:
    canonical = make_canonical_matches()

    features = build_match_level_features(canonical, windows=(5,))

    assert len(features) == len(canonical)
    assert features["match_id"].tolist() == canonical["match_id"].tolist()


def test_match_level_differential_features_are_team_a_minus_team_b() -> None:
    canonical = make_canonical_matches()

    features = build_match_level_features(canonical, windows=(5,))
    row = features.loc[features["match_id"] == canonical.loc[5, "match_id"]].iloc[0]

    expected = (
        row["team_a_matches_played_before"] - row["team_b_matches_played_before"]
    )
    assert row["matches_played_before_diff"] == expected


def test_input_dataframes_are_not_mutated() -> None:
    canonical = make_canonical_matches()
    canonical_original = canonical.copy(deep=True)
    panel = build_team_match_panel(canonical)
    panel_original = panel.copy(deep=True)

    build_team_match_panel(canonical)
    add_team_rolling_features(panel, windows=(5,))
    build_match_level_features(canonical, windows=(5,))

    pd.testing.assert_frame_equal(canonical, canonical_original)
    pd.testing.assert_frame_equal(panel, panel_original)
