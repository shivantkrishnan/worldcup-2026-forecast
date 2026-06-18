import pandas as pd

from scripts import ingest_official_results_2026 as ingest


def make_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-11",
                "team_a": "Mexico",
                "team_b": "South Africa",
                "group": "A",
            },
            {
                "match_id": "m2",
                "match_date": "2026-06-11",
                "team_a": "South Korea",
                "team_b": "Czech Republic",
                "group": "A",
            },
            {
                "match_id": "m3",
                "match_date": "2026-06-17",
                "team_a": "Portugal",
                "team_b": "DR Congo",
                "group": "K",
            },
        ]
    )


def fifa_match(
    match_id: str,
    local_date: str,
    group: str,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
) -> dict:
    return {
        "IdMatch": match_id,
        "StageName": [{"Locale": "en-GB", "Description": "First Stage"}],
        "GroupName": [{"Locale": "en-GB", "Description": f"Group {group}"}],
        "LocalDate": f"{local_date}T13:00:00Z",
        "Home": {"ShortClubName": home},
        "Away": {"ShortClubName": away},
        "HomeTeamScore": home_goals,
        "AwayTeamScore": away_goals,
    }


def test_build_results_from_fifa_payload_maps_aliases_and_scores() -> None:
    payload = {
        "Results": [
            fifa_match("4001", "2026-06-11", "A", "Mexico", "South Africa", 2, 0),
            fifa_match("4002", "2026-06-11", "A", "Korea Republic", "Czechia", 2, 1),
            fifa_match("4003", "2026-06-17", "K", "Portugal", "Congo DR", 1, 1),
        ]
    }

    results, omitted, orientation_notes = ingest.build_results_from_fifa_payload(
        make_fixtures(),
        payload,
        source_url="https://api.fifa.com/example",
        last_updated="2026-06-18",
    )

    assert omitted == []
    assert orientation_notes == []
    assert list(results["match_id"]) == ["m1", "m2", "m3"]
    assert list(results["team_a"]) == ["Mexico", "South Korea", "Portugal"]
    assert list(results["team_b"]) == ["South Africa", "Czech Republic", "DR Congo"]
    assert list(results["result"]) == ["team_a_win", "team_a_win", "draw"]


def test_build_results_from_fifa_payload_reverses_scores_to_local_orientation() -> None:
    payload = {
        "Results": [
            fifa_match("4001", "2026-06-11", "A", "South Africa", "Mexico", 1, 3),
        ]
    }

    results, omitted, orientation_notes = ingest.build_results_from_fifa_payload(
        make_fixtures(),
        payload,
        source_url="https://api.fifa.com/example",
        last_updated="2026-06-18",
    )

    assert omitted == []
    assert orientation_notes == [
        "score orientation reversed for Mexico vs South Africa"
    ]
    assert results.loc[0, "team_a_goals"] == 3
    assert results.loc[0, "team_b_goals"] == 1
    assert results.loc[0, "result"] == "team_a_win"


def test_build_results_from_fifa_payload_omits_unmapped_rows() -> None:
    payload = {
        "Results": [
            fifa_match("4004", "2026-06-11", "A", "Alpha", "Beta", 1, 0),
        ]
    }

    results, omitted, _ = ingest.build_results_from_fifa_payload(
        make_fixtures(),
        payload,
        source_url="https://api.fifa.com/example",
        last_updated="2026-06-18",
    )

    assert results.empty
    assert omitted == [
        "2026-06-11 Group A: Alpha vs Beta could not be mapped to local fixtures."
    ]
