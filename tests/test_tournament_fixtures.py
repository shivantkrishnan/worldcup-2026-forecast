import pandas as pd
import pytest

from src.data.tournament_fixtures import (
    load_tournament_fixtures,
    normalize_tournament_fixtures,
    validate_tournament_fixtures,
)


def make_fixture_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "wc2026_a1",
                "match_date": "2026-06-12",
                "team_a": "Team A",
                "team_b": "Team B",
                "group": "A",
                "stage": "group",
            },
            {
                "match_id": "wc2026_a2",
                "match_date": "2026-06-13",
                "team_a": "Team C",
                "team_b": "Team D",
                "group": "A",
                "stage": "group",
                "neutral": "false",
            },
        ]
    )


def test_fixture_validation_catches_duplicate_match_id() -> None:
    fixtures = make_fixture_rows()
    fixtures.loc[1, "match_id"] = fixtures.loc[0, "match_id"]

    with pytest.raises(ValueError, match="unique"):
        validate_tournament_fixtures(fixtures)


@pytest.mark.parametrize("column", ["team_a", "team_b", "group", "stage"])
def test_fixture_validation_catches_missing_required_values(column: str) -> None:
    fixtures = make_fixture_rows()
    fixtures.loc[0, column] = None

    with pytest.raises(ValueError, match=column if column != "group" else "group"):
        validate_tournament_fixtures(fixtures)


def test_fixture_validation_catches_team_a_equal_team_b() -> None:
    fixtures = make_fixture_rows()
    fixtures.loc[0, "team_b"] = "Team A"

    with pytest.raises(ValueError, match="different"):
        validate_tournament_fixtures(fixtures)


def test_fixture_date_parsing_works() -> None:
    normalized = normalize_tournament_fixtures(make_fixture_rows())

    assert pd.api.types.is_datetime64_any_dtype(normalized["match_date"])


def test_missing_neutral_defaults_to_true_for_2026_world_cup_fixtures() -> None:
    normalized = normalize_tournament_fixtures(make_fixture_rows())

    assert bool(normalized.loc[0, "is_neutral"]) is True
    assert bool(normalized.loc[1, "is_neutral"]) is False


def test_load_tournament_fixtures_reads_normalizes_and_validates(tmp_path) -> None:
    path = tmp_path / "fixtures_2026.csv"
    make_fixture_rows().to_csv(path, index=False)

    loaded = load_tournament_fixtures(path)

    assert len(loaded) == 2
    assert "is_neutral" in loaded.columns
