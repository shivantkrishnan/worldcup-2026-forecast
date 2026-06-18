import pandas as pd

from src.features.build_features import build_modeling_features
from src.features.elo import (
    ELO_FEATURE_COLUMNS,
    add_elo_features_to_matches,
    build_elo_features,
    expected_score,
    result_to_score,
    update_elo_pair,
)


def make_canonical_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "match_date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
            "team_a": ["Team A", "Team A", "Team A", "Team C"],
            "team_b": ["Team B", "Team C", "Team B", "Team B"],
            "team_a_goals": [2, 3, 1, 1],
            "team_b_goals": [0, 1, 1, 2],
            "result": ["team_a_win", "team_a_win", "draw", "team_b_win"],
            "tournament": ["Friendly"] * 4,
            "city": ["City"] * 4,
            "country": ["Country"] * 4,
            "neutral": [False] * 4,
            "is_neutral": [False] * 4,
            "goal_diff_team_a": [2, 2, 0, -1],
            "total_goals": [2, 4, 2, 3],
            "training_cutoff_date": ["2026-06-10"] * 4,
            "is_baseline_train_eligible": [True] * 4,
        }
    )


def test_expected_score_returns_half_for_equal_ratings() -> None:
    assert expected_score(1500.0, 1500.0) == 0.5


def test_result_to_score_maps_all_result_labels() -> None:
    assert result_to_score("team_a_win") == 1.0
    assert result_to_score("draw") == 0.5
    assert result_to_score("team_b_win") == 0.0


def test_update_elo_pair_increases_winner_and_decreases_loser() -> None:
    updated_a, updated_b = update_elo_pair(1500.0, 1500.0, score_a=1.0)

    assert updated_a > 1500.0
    assert updated_b < 1500.0


def test_k_factor_changes_update_magnitude() -> None:
    low_k_a, _ = update_elo_pair(1500.0, 1500.0, score_a=1.0, k_factor=10.0)
    high_k_a, _ = update_elo_pair(1500.0, 1500.0, score_a=1.0, k_factor=30.0)

    assert high_k_a - 1500.0 > low_k_a - 1500.0


def test_draw_updates_ratings_toward_each_other() -> None:
    updated_a, updated_b = update_elo_pair(1600.0, 1400.0, score_a=0.5)

    assert updated_a < 1600.0
    assert updated_b > 1400.0


def test_first_match_for_unseen_teams_uses_initial_rating() -> None:
    features = build_elo_features(make_canonical_matches())
    first_match = features.loc[features["match_id"] == "m1"].iloc[0]

    assert first_match["elo_team_a_pre"] == 1500.0
    assert first_match["elo_team_b_pre"] == 1500.0
    assert first_match["elo_expected_score_team_a"] == 0.5
    assert first_match["elo_effective_diff_team_a_minus_team_b"] == 0.0
    assert first_match["elo_home_advantage_applied"] == 0.0
    assert first_match["elo_matches_before_team_a"] == 0
    assert first_match["elo_matches_before_team_b"] == 0


def test_current_match_result_does_not_affect_own_pre_match_features() -> None:
    features = build_elo_features(make_canonical_matches(), k_factor=20.0)
    first_match = features.loc[features["match_id"] == "m1"].iloc[0]

    assert first_match["elo_team_a_pre"] == 1500.0
    assert first_match["elo_team_b_pre"] == 1500.0
    assert first_match["elo_diff_team_a_minus_team_b"] == 0.0


def test_later_matches_reflect_prior_date_block_updates() -> None:
    features = build_elo_features(make_canonical_matches(), k_factor=20.0)
    third_match = features.loc[features["match_id"] == "m3"].iloc[0]

    assert third_match["elo_team_a_pre"] == 1520.0
    assert third_match["elo_team_b_pre"] == 1490.0
    assert third_match["elo_matches_before_team_a"] == 2
    assert third_match["elo_matches_before_team_b"] == 1


def test_same_date_matches_are_emitted_before_same_date_updates() -> None:
    features = build_elo_features(make_canonical_matches(), k_factor=20.0)
    second_same_date_match = features.loc[features["match_id"] == "m2"].iloc[0]

    assert second_same_date_match["elo_team_a_pre"] == 1500.0
    assert second_same_date_match["elo_team_b_pre"] == 1500.0
    assert second_same_date_match["elo_matches_before_team_a"] == 0
    assert second_same_date_match["elo_matches_before_team_b"] == 0


def test_home_advantage_changes_expected_score_on_non_neutral_match() -> None:
    features = build_elo_features(make_canonical_matches(), home_advantage=100.0)
    first_match = features.loc[features["match_id"] == "m1"].iloc[0]

    assert first_match["elo_home_advantage_applied"] == 100.0
    assert first_match["elo_effective_diff_team_a_minus_team_b"] == 100.0
    assert first_match["elo_expected_score_team_a"] > 0.5


def test_home_advantage_does_not_apply_on_neutral_match() -> None:
    canonical = make_canonical_matches()
    canonical.loc[0, "is_neutral"] = True
    canonical.loc[0, "neutral"] = True

    features = build_elo_features(canonical, home_advantage=100.0)
    first_match = features.loc[features["match_id"] == "m1"].iloc[0]

    assert first_match["elo_home_advantage_applied"] == 0.0
    assert first_match["elo_effective_diff_team_a_minus_team_b"] == 0.0
    assert first_match["elo_expected_score_team_a"] == 0.5


def test_home_advantage_does_not_mutate_underlying_pre_match_rating() -> None:
    features = build_elo_features(make_canonical_matches(), home_advantage=100.0)
    first_match = features.loc[features["match_id"] == "m1"].iloc[0]

    assert first_match["elo_team_a_pre"] == 1500.0
    assert first_match["elo_team_b_pre"] == 1500.0


def test_elo_output_has_one_row_per_input_match() -> None:
    canonical = make_canonical_matches()

    features = build_elo_features(canonical)

    assert len(features) == len(canonical)
    assert features["match_id"].tolist() == ["m1", "m2", "m3", "m4"]


def test_build_elo_features_does_not_mutate_input_dataframe() -> None:
    canonical = make_canonical_matches()
    original = canonical.copy(deep=True)

    build_elo_features(canonical)

    pd.testing.assert_frame_equal(canonical, original)


def test_add_elo_features_to_matches_preserves_rows_and_adds_elo_columns() -> None:
    canonical = make_canonical_matches()

    with_elo = add_elo_features_to_matches(canonical)

    assert len(with_elo) == len(canonical)
    assert with_elo["match_id"].tolist() == canonical["match_id"].tolist()
    for column in ELO_FEATURE_COLUMNS:
        assert column in with_elo.columns


def test_build_modeling_features_can_optionally_include_elo_features() -> None:
    features = build_modeling_features(
        make_canonical_matches(),
        windows=(2,),
        include_elo=True,
    )

    for column in ELO_FEATURE_COLUMNS:
        assert column in features.columns


def test_build_modeling_features_can_pass_elo_parameters() -> None:
    features = build_modeling_features(
        make_canonical_matches(),
        windows=(2,),
        include_elo=True,
        elo_k_factor=30.0,
        elo_home_advantage=75.0,
    )
    first_match = features.loc[features["match_id"] == "m1"].iloc[0]

    assert first_match["elo_home_advantage_applied"] == 75.0
