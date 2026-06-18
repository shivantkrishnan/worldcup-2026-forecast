import pandas as pd

from src.features.fixture_features import build_fixture_feature_rows


def make_completed_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-01",
                "team_a": "Alpha",
                "team_b": "Beta",
                "team_a_goals": 2,
                "team_b_goals": 0,
                "result": "team_a_win",
                "tournament": "Friendly",
                "is_neutral": False,
            },
            {
                "match_id": "m2",
                "match_date": "2026-06-10",
                "team_a": "Alpha",
                "team_b": "Gamma",
                "team_a_goals": 0,
                "team_b_goals": 1,
                "result": "team_b_win",
                "tournament": "Friendly",
                "is_neutral": False,
            },
            {
                "match_id": "m3_same_day",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Delta",
                "team_a_goals": 3,
                "team_b_goals": 0,
                "result": "team_a_win",
                "tournament": "Friendly",
                "is_neutral": False,
            },
        ]
    )


def test_fixture_feature_builder_does_not_require_scores_or_result() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "f1",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Beta",
                "tournament": "FIFA World Cup",
            }
        ]
    )

    features = build_fixture_feature_rows(make_completed_matches(), fixtures)

    assert len(features) == 1
    assert features.loc[0, "match_id"] == "f1"


def test_fixture_feature_builder_returns_one_row_per_fixture() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "f1",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Beta",
            },
            {
                "match_id": "f2",
                "match_date": "2026-06-21",
                "team_a": "Gamma",
                "team_b": "Delta",
            },
        ]
    )

    features = build_fixture_feature_rows(make_completed_matches(), fixtures)

    assert len(features) == len(fixtures)


def test_fixture_features_use_only_prior_completed_matches() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "f1",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Beta",
            }
        ]
    )

    features = build_fixture_feature_rows(make_completed_matches(), fixtures)

    assert features.loc[0, "team_a_matches_played_before"] == 2


def test_same_date_completed_matches_are_excluded() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "f1",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Beta",
            }
        ]
    )

    features = build_fixture_feature_rows(make_completed_matches(), fixtures)

    assert features.loc[0, "team_a_matches_played_before"] == 2
    assert features.loc[0, "team_a_expanding_points_per_match"] == 1.5


def test_feature_cutoff_date_is_respected() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "f1",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Beta",
            }
        ]
    )

    features = build_fixture_feature_rows(
        make_completed_matches(),
        fixtures,
        feature_cutoff_date="2026-06-05",
    )

    assert features.loc[0, "team_a_matches_played_before"] == 1
    assert features.loc[0, "team_a_expanding_points_per_match"] == 3.0


def test_future_2026_world_cup_fixtures_default_to_neutral() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "f1",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Beta",
                "tournament": "FIFA World Cup",
            }
        ]
    )

    features = build_fixture_feature_rows(
        make_completed_matches(),
        fixtures,
        include_elo=True,
        elo_home_advantage=50.0,
    )

    assert bool(features.loc[0, "is_neutral"])
    assert features.loc[0, "elo_home_advantage_applied"] == 0.0


def test_fixture_feature_builder_does_not_mutate_inputs() -> None:
    completed = make_completed_matches()
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "f1",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Beta",
                "tournament": "FIFA World Cup",
            }
        ]
    )
    original_completed = completed.copy(deep=True)
    original_fixtures = fixtures.copy(deep=True)

    build_fixture_feature_rows(completed, fixtures)

    pd.testing.assert_frame_equal(completed, original_completed)
    pd.testing.assert_frame_equal(fixtures, original_fixtures)
