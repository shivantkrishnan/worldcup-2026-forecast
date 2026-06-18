"""Ingest completed 2026 World Cup results from FIFA's official public API."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tournament_fixtures import load_tournament_fixtures  # noqa: E402
from src.data.tournament_results import (  # noqa: E402
    load_tournament_results,
    validate_tournament_results,
)
from src.utils.config import FIXTURES_2026_PATH, RESULTS_2026_PATH  # noqa: E402

FIFA_CALENDAR_API_URL = "https://api.fifa.com/api/v3/calendar/matches"
FIFA_COMPETITION_ID = "17"
FIFA_2026_SEASON_ID = "285023"
RESULT_COLUMNS = [
    "match_id",
    "match_date",
    "team_a",
    "team_b",
    "team_a_goals",
    "team_b_goals",
    "result",
    "status",
    "went_to_extra_time",
    "went_to_penalties",
    "team_a_penalties",
    "team_b_penalties",
    "source",
    "last_updated",
]
TEAM_ALIASES = {
    "congo dr": "DR Congo",
    "czechia": "Czech Republic",
    "korea republic": "South Korea",
    "usa": "United States",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "cabo verde": "Cape Verde",
    "ir iran": "Iran",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch official completed 2026 World Cup results and write CSV."
    )
    parser.add_argument(
        "--fixtures",
        default=FIXTURES_2026_PATH,
        help="Path to local fixtures_2026.csv.",
    )
    parser.add_argument(
        "--output",
        default=RESULTS_2026_PATH,
        help="Output path for results_2026.csv.",
    )
    parser.add_argument(
        "--from-date",
        default="2026-06-11",
        help="First fixture date to request from FIFA, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--to-date",
        default=date.today().isoformat(),
        help="Last fixture date to request from FIFA, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print results without writing the output CSV.",
    )
    return parser.parse_args(argv)


def build_fifa_results_url(from_date: str, to_date: str) -> str:
    """Return the official FIFA calendar API URL for the result window."""
    query = urlencode(
        {
            "language": "en",
            "count": "100",
            "idCompetition": FIFA_COMPETITION_ID,
            "idSeason": FIFA_2026_SEASON_ID,
            "from": from_date,
            "to": to_date,
        }
    )
    return f"{FIFA_CALENDAR_API_URL}?{query}"


def fetch_fifa_payload(url: str) -> dict[str, Any]:
    """Fetch JSON from FIFA's public API."""
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _description(value: list[dict[str, Any]] | None) -> str:
    """Return the English description from a localized FIFA value."""
    if not value:
        return ""
    for item in value:
        if str(item.get("Locale", "")).casefold().startswith("en"):
            return str(item.get("Description", "")).strip()
    return str(value[0].get("Description", "")).strip()


def normalize_team_name(team_name: str) -> str:
    """Normalize FIFA display names to the local fixture naming convention."""
    stripped = team_name.strip()
    return TEAM_ALIASES.get(stripped.casefold(), stripped)


def _team_key(team_name: str) -> str:
    """Return a stable lookup key for team names."""
    return normalize_team_name(team_name).casefold()


def result_label(team_a_goals: int, team_b_goals: int) -> str:
    """Return the canonical result label from Team A's perspective."""
    if team_a_goals > team_b_goals:
        return "team_a_win"
    if team_a_goals < team_b_goals:
        return "team_b_win"
    return "draw"


def _group_code(fifa_match: dict[str, Any]) -> str:
    """Return the single-letter group code from a FIFA match object."""
    description = _description(fifa_match.get("GroupName"))
    return description.replace("Group", "").strip()


def _is_completed_group_match(fifa_match: dict[str, Any]) -> bool:
    """Return whether a FIFA match object represents a completed group match."""
    stage_name = _description(fifa_match.get("StageName")).casefold()
    has_score = (
        fifa_match.get("HomeTeamScore") is not None
        and fifa_match.get("AwayTeamScore") is not None
    )
    return "first stage" in stage_name and has_score


def _fixture_lookup_key(match_date: str, group: str, team_a: str, team_b: str) -> tuple:
    """Return the local fixture/result matching key."""
    teams = tuple(sorted([_team_key(team_a), _team_key(team_b)]))
    return match_date, group, teams


def _build_fixture_lookup(fixtures: pd.DataFrame) -> dict[tuple, pd.Series]:
    """Return fixture rows keyed by local date, group, and unordered teams."""
    lookup: dict[tuple, pd.Series] = {}
    for _, fixture in fixtures.iterrows():
        match_date = pd.Timestamp(fixture["match_date"]).date().isoformat()
        key = _fixture_lookup_key(
            match_date,
            str(fixture["group"]),
            str(fixture["team_a"]),
            str(fixture["team_b"]),
        )
        lookup[key] = fixture
    return lookup


def _score_for_local_orientation(
    fixture: pd.Series,
    fifa_home: str,
    fifa_away: str,
    home_goals: int,
    away_goals: int,
) -> tuple[int, int, str | None]:
    """Return goals in local fixture orientation plus any correction note."""
    local_team_a = str(fixture["team_a"])
    local_team_b = str(fixture["team_b"])
    fifa_home_local = normalize_team_name(fifa_home)
    fifa_away_local = normalize_team_name(fifa_away)

    if _team_key(local_team_a) == _team_key(fifa_home_local):
        return home_goals, away_goals, None
    if _team_key(local_team_a) == _team_key(fifa_away_local):
        return away_goals, home_goals, (
            f"score orientation reversed for {local_team_a} vs {local_team_b}"
        )
    return home_goals, away_goals, (
        f"could not verify score orientation for {local_team_a} vs {local_team_b}"
    )


def build_results_from_fifa_payload(
    fixtures: pd.DataFrame,
    payload: dict[str, Any],
    source_url: str,
    last_updated: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build local results rows from a FIFA API payload.

    Returns results, omitted-match notes, and orientation-correction notes.
    """
    fixture_lookup = _build_fixture_lookup(fixtures)
    rows: list[dict[str, Any]] = []
    omitted: list[str] = []
    orientation_notes: list[str] = []

    for fifa_match in payload.get("Results", []):
        if not _is_completed_group_match(fifa_match):
            continue

        local_date = pd.Timestamp(fifa_match["LocalDate"]).date().isoformat()
        group = _group_code(fifa_match)
        home_team = normalize_team_name(
            str(fifa_match.get("Home", {}).get("ShortClubName", ""))
        )
        away_team = normalize_team_name(
            str(fifa_match.get("Away", {}).get("ShortClubName", ""))
        )
        home_goals = int(fifa_match["HomeTeamScore"])
        away_goals = int(fifa_match["AwayTeamScore"])
        lookup_key = _fixture_lookup_key(local_date, group, home_team, away_team)
        fixture = fixture_lookup.get(lookup_key)

        if fixture is None:
            omitted.append(
                f"{local_date} Group {group}: {home_team} vs {away_team} "
                "could not be mapped to local fixtures."
            )
            continue

        team_a_goals, team_b_goals, orientation_note = _score_for_local_orientation(
            fixture,
            home_team,
            away_team,
            home_goals,
            away_goals,
        )
        if orientation_note is not None:
            orientation_notes.append(orientation_note)
            if orientation_note.startswith("could not verify"):
                omitted.append(orientation_note)
                continue

        rows.append(
            {
                "match_id": fixture["match_id"],
                "match_date": pd.Timestamp(fixture["match_date"]).date().isoformat(),
                "team_a": fixture["team_a"],
                "team_b": fixture["team_b"],
                "team_a_goals": team_a_goals,
                "team_b_goals": team_b_goals,
                "result": result_label(team_a_goals, team_b_goals),
                "status": "completed",
                "went_to_extra_time": "false",
                "went_to_penalties": "false",
                "team_a_penalties": "",
                "team_b_penalties": "",
                "source": (
                    f"FIFA official API IdMatch {fifa_match.get('IdMatch')} "
                    f"({source_url})"
                ),
                "last_updated": last_updated,
            }
        )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        ["match_date", "match_id"],
        kind="mergesort",
    )
    return results, omitted, orientation_notes


def write_results_csv(results: pd.DataFrame, output_path: str | Path) -> None:
    """Write official-source results with the project schema header."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False, columns=RESULT_COLUMNS, quoting=csv.QUOTE_MINIMAL)


def main(argv: list[str] | None = None) -> int:
    """Fetch official completed results, validate them, and optionally write CSV."""
    args = parse_args(argv)
    source_url = build_fifa_results_url(args.from_date, args.to_date)

    try:
        fixtures = load_tournament_fixtures(args.fixtures)
        payload = fetch_fifa_payload(source_url)
        results, omitted, orientation_notes = build_results_from_fifa_payload(
            fixtures,
            payload,
            source_url=source_url,
            last_updated=date.today().isoformat(),
        )
        validate_tournament_results(results, fixtures_or_predictions=fixtures)
    except Exception as error:  # noqa: BLE001 - CLI should report validation/fetch errors.
        print(f"Unable to ingest official results: {error}")
        return 1

    print("Official Results Ingestion")
    print("==========================")
    print(f"source: {source_url}")
    print(f"completed result rows: {len(results)}")
    if len(results):
        print(
            results[
                [
                    "match_id",
                    "match_date",
                    "team_a",
                    "team_b",
                    "team_a_goals",
                    "team_b_goals",
                    "result",
                ]
            ].to_string(index=False)
        )

    print("\nOrientation notes")
    print("=================")
    if orientation_notes:
        for note in orientation_notes:
            print(f"- {note}")
    else:
        print("No orientation corrections were needed.")

    print("\nOmitted completed FIFA rows")
    print("===========================")
    if omitted:
        for note in omitted:
            print(f"- {note}")
    else:
        print("None.")

    if args.dry_run:
        print("\nDry run only; no results file was written.")
        return 0

    write_results_csv(results, args.output)
    loaded_results = load_tournament_results(args.output, fixtures_or_predictions=fixtures)
    print(f"\nWrote and validated {len(loaded_results)} rows at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
