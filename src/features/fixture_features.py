"""Leakage-safe feature rows for scheduled fixtures."""

from __future__ import annotations

from typing import Any

from collections import defaultdict

import pandas as pd

from src.features.elo import (
    ELO_FEATURE_COLUMNS,
    expected_score,
    result_to_score,
    update_elo_pair,
)
from src.features.team_form import (
    ROLLING_BASE_COLUMNS,
    ROLLING_FEATURE_NAME_MAP,
    build_team_match_panel,
    get_team_form_feature_columns,
)

REQUIRED_FIXTURE_COLUMNS = {"match_id", "match_date", "team_a", "team_b"}
LIVE_RESULT_COLUMNS = {
    "match_id",
    "match_date",
    "team_a",
    "team_b",
    "team_a_goals",
    "team_b_goals",
    "result",
}


def _coerce_bool(value: Any) -> bool | None:
    """Coerce common boolean-like values, preserving missing as unknown."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return bool(value)


def _is_2026_world_cup_fixture(row: pd.Series) -> bool:
    """Return whether a fixture appears to be a 2026 World Cup match."""
    match_year = pd.Timestamp(row["match_date"]).year
    tournament = str(row.get("tournament", "")).strip().casefold()
    has_tournament_context = "group" in row.index or "stage" in row.index
    return match_year == 2026 and ("world cup" in tournament or has_tournament_context)


def _fixture_neutral_flag(row: pd.Series) -> bool:
    """Return the neutral-site flag used for fixture feature generation."""
    is_neutral = _coerce_bool(row.get("is_neutral"))
    if is_neutral is not None:
        return is_neutral

    neutral = _coerce_bool(row.get("neutral"))
    if neutral is not None:
        return neutral

    if _is_2026_world_cup_fixture(row):
        # Generic historical home advantage is not a host-country model. USA,
        # Canada, and Mexico effects should be explicit tournament-state features later.
        return True

    return False


def _validate_fixture_columns(fixtures: pd.DataFrame) -> None:
    """Raise a clear error when required fixture columns are missing."""
    missing = REQUIRED_FIXTURE_COLUMNS.difference(fixtures.columns)
    if missing:
        raise ValueError(
            "Missing required fixture columns: " + ", ".join(sorted(missing))
        )


def _fixture_tournament(fixture: pd.Series) -> str:
    """Return the fixture tournament/display label."""
    tournament = str(fixture.get("tournament", "Scheduled Fixture")).strip()
    if not tournament or tournament == "nan":
        tournament = "Scheduled Fixture"
    return tournament


def _base_fixture_row(fixture: pd.Series) -> dict[str, object]:
    """Return metadata columns for one fixture feature row."""
    return {
        "match_id": fixture["match_id"],
        "match_date": fixture["match_date"],
        "team_a": str(fixture["team_a"]).strip(),
        "team_b": str(fixture["team_b"]).strip(),
        "tournament": _fixture_tournament(fixture),
        "is_neutral": _fixture_neutral_flag(fixture),
    }


def _prepare_completed_matches(
    completed_matches: pd.DataFrame,
    feature_cutoff_date: str | None,
) -> pd.DataFrame:
    """Return completed matches allowed by the optional feature cutoff."""
    history = completed_matches.copy(deep=True)
    history["match_date"] = pd.to_datetime(history["match_date"], errors="raise")

    if feature_cutoff_date is not None:
        cutoff = pd.Timestamp(feature_cutoff_date)
        history = history.loc[history["match_date"] <= cutoff].copy()
    return history


def build_live_feature_history(
    historical_completed_matches: pd.DataFrame,
    tournament_results: pd.DataFrame,
    feature_cutoff_date: str,
    tournament_label: str = "FIFA World Cup 2026",
) -> pd.DataFrame:
    """Append completed 2026 results for feature construction only.

    The returned history is intended for fixture feature generation, not model
    fitting. Results are filtered through the explicit feature cutoff, and only
    completed score-bearing rows are appended.
    """
    history = historical_completed_matches.copy(deep=True)
    history["match_date"] = pd.to_datetime(history["match_date"], errors="raise")

    results = tournament_results.copy(deep=True)
    missing = LIVE_RESULT_COLUMNS.difference(results.columns)
    if missing:
        raise ValueError(
            "Missing required tournament result columns for live features: "
            + ", ".join(sorted(missing))
        )

    results["match_date"] = pd.to_datetime(results["match_date"], errors="raise")
    if "status" in results.columns:
        status = results["status"].astype("string").str.strip().str.casefold()
        results = results.loc[status.eq("completed")].copy(deep=True)

    cutoff = pd.Timestamp(feature_cutoff_date)
    results = results.loc[results["match_date"] <= cutoff].copy(deep=True)

    if results.empty:
        return history

    live_rows = pd.DataFrame(
        {
            "match_id": results["match_id"].astype(str).str.strip(),
            "match_date": results["match_date"],
            "team_a": results["team_a"].astype(str).str.strip(),
            "team_b": results["team_b"].astype(str).str.strip(),
            "team_a_goals": pd.to_numeric(
                results["team_a_goals"],
                errors="raise",
            ),
            "team_b_goals": pd.to_numeric(
                results["team_b_goals"],
                errors="raise",
            ),
            "result": results["result"].astype(str).str.strip(),
            "tournament": tournament_label,
            "is_neutral": True,
        }
    )

    combined = pd.concat([history, live_rows], ignore_index=True, sort=False)
    if combined["match_id"].duplicated().any():
        duplicate_ids = sorted(
            combined.loc[combined["match_id"].duplicated(keep=False), "match_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "Live feature history contains duplicate match_id values: "
            + ", ".join(duplicate_ids)
        )
    return combined


def _team_feature_values(
    team_history: pd.DataFrame,
    fixture_date: pd.Timestamp,
    windows: tuple[int, ...],
) -> dict[str, float | int]:
    """Return leakage-safe team-form values before one fixture date."""
    prior = team_history.loc[team_history["match_date"] < fixture_date].copy()
    values: dict[str, float | int] = {
        "matches_played_before": int(len(prior)),
        "days_since_last_match": float("nan"),
        "expanding_points_per_match": float("nan"),
        "expanding_goal_diff_avg": float("nan"),
        "expanding_win_rate": float("nan"),
    }
    for window in windows:
        for feature_prefix in ROLLING_FEATURE_NAME_MAP.values():
            values[f"{feature_prefix}_{window}"] = float("nan")

    if prior.empty:
        return values

    prior = prior.sort_values(["match_date", "match_id"], kind="mergesort")
    values["days_since_last_match"] = int(
        (fixture_date - prior["match_date"].max()).days
    )
    values["expanding_points_per_match"] = float(prior["points"].mean())
    values["expanding_goal_diff_avg"] = float(prior["goal_diff"].mean())
    values["expanding_win_rate"] = float(prior["win"].mean())

    for window in windows:
        if len(prior) < window:
            continue
        rolling_source = prior.tail(window)
        for base_column, feature_prefix in ROLLING_FEATURE_NAME_MAP.items():
            values[f"{feature_prefix}_{window}"] = float(
                rolling_source[base_column].mean()
            )
    return values


def _build_team_form_fixture_features(
    completed_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """Return fixture rows with team-form features and differentials."""
    if completed_matches.empty:
        panel = pd.DataFrame(columns=["team", "match_date", "match_id", *ROLLING_BASE_COLUMNS])
    else:
        panel = build_team_match_panel(completed_matches)
    panel["match_date"] = pd.to_datetime(panel["match_date"], errors="raise")
    team_histories = {
        str(team): group.sort_values(["match_date", "match_id"], kind="mergesort")
        for team, group in panel.groupby("team", sort=False)
    }
    feature_columns = get_team_form_feature_columns(windows)

    rows: list[dict[str, object]] = []
    for _, fixture in fixtures.iterrows():
        fixture_date = pd.Timestamp(fixture["match_date"])
        row = _base_fixture_row(fixture)
        team_a_history = team_histories.get(str(fixture["team_a"]).strip(), panel.iloc[0:0])
        team_b_history = team_histories.get(str(fixture["team_b"]).strip(), panel.iloc[0:0])
        team_a_values = _team_feature_values(team_a_history, fixture_date, windows)
        team_b_values = _team_feature_values(team_b_history, fixture_date, windows)

        for feature_column in feature_columns:
            row[f"team_a_{feature_column}"] = team_a_values[feature_column]
            row[f"team_b_{feature_column}"] = team_b_values[feature_column]
            row[f"{feature_column}_diff"] = (
                team_a_values[feature_column] - team_b_values[feature_column]
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _row_is_neutral(row: object) -> bool:
    """Return neutral-site status for completed-match Elo updates."""
    if hasattr(row, "is_neutral"):
        value = _coerce_bool(getattr(row, "is_neutral"))
        return bool(value)
    if hasattr(row, "neutral"):
        value = _coerce_bool(getattr(row, "neutral"))
        return bool(value)
    return False


def _completed_home_advantage(row: object, home_advantage: float) -> float:
    """Return home advantage for a completed match update."""
    return 0.0 if _row_is_neutral(row) else float(home_advantage)


def _build_fixture_elo_features(
    completed_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    initial_rating: float,
    k_factor: float,
    home_advantage: float,
) -> pd.DataFrame:
    """Return pre-fixture Elo features without fixture placeholder updates."""
    completed = completed_matches.copy(deep=True)
    completed["match_date"] = pd.to_datetime(completed["match_date"], errors="raise")
    completed = completed.sort_values(["match_date", "match_id"], kind="mergesort")
    fixture_rows = fixtures.copy(deep=True).sort_values(
        ["match_date", "match_id"],
        kind="mergesort",
    )

    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}
    feature_rows: list[dict[str, object]] = []
    completed_by_date = {
        date: group
        for date, group in completed.groupby("match_date", sort=True)
    }
    fixtures_by_date = {
        date: group
        for date, group in fixture_rows.groupby("match_date", sort=True)
    }

    all_dates = sorted(set(completed_by_date).union(fixtures_by_date))
    for match_date in all_dates:
        start_ratings = ratings.copy()
        start_counts = match_counts.copy()

        if match_date in fixtures_by_date:
            for fixture in fixtures_by_date[match_date].itertuples(index=False):
                team_a = str(getattr(fixture, "team_a")).strip()
                team_b = str(getattr(fixture, "team_b")).strip()
                rating_a = start_ratings.get(team_a, initial_rating)
                rating_b = start_ratings.get(team_b, initial_rating)
                applied_home_advantage = (
                    0.0
                    if bool(getattr(fixture, "is_neutral"))
                    else float(home_advantage)
                )
                expected_a = expected_score(rating_a + applied_home_advantage, rating_b)
                feature_rows.append(
                    {
                        "match_id": getattr(fixture, "match_id"),
                        "elo_team_a_pre": float(rating_a),
                        "elo_team_b_pre": float(rating_b),
                        "elo_diff_team_a_minus_team_b": float(rating_a - rating_b),
                        "elo_effective_diff_team_a_minus_team_b": float(
                            rating_a + applied_home_advantage - rating_b
                        ),
                        "elo_expected_score_team_a": expected_a,
                        "elo_home_advantage_applied": applied_home_advantage,
                        "elo_matches_before_team_a": int(start_counts.get(team_a, 0)),
                        "elo_matches_before_team_b": int(start_counts.get(team_b, 0)),
                    }
                )

        if match_date not in completed_by_date:
            continue

        rating_deltas: defaultdict[str, float] = defaultdict(float)
        count_deltas: defaultdict[str, int] = defaultdict(int)
        for row in completed_by_date[match_date].itertuples(index=False):
            team_a = str(getattr(row, "team_a"))
            team_b = str(getattr(row, "team_b"))
            rating_a = start_ratings.get(team_a, initial_rating)
            rating_b = start_ratings.get(team_b, initial_rating)
            applied_home_advantage = _completed_home_advantage(row, home_advantage)
            expected_a = expected_score(rating_a + applied_home_advantage, rating_b)
            updated_a, updated_b = update_elo_pair(
                rating_a,
                rating_b,
                result_to_score(str(getattr(row, "result"))),
                k_factor=k_factor,
                expected_score_a=expected_a,
            )
            rating_deltas[team_a] += updated_a - rating_a
            rating_deltas[team_b] += updated_b - rating_b
            count_deltas[team_a] += 1
            count_deltas[team_b] += 1

        for team, delta in rating_deltas.items():
            ratings[team] = start_ratings.get(team, initial_rating) + delta
        for team, count_delta in count_deltas.items():
            match_counts[team] = start_counts.get(team, 0) + count_delta

    return pd.DataFrame(feature_rows, columns=["match_id", *ELO_FEATURE_COLUMNS])


def build_fixture_feature_rows(
    completed_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    include_elo: bool = True,
    elo_k_factor: float = 10.0,
    elo_home_advantage: float = 50.0,
    feature_cutoff_date: str | None = None,
) -> pd.DataFrame:
    """Build selected-baseline feature rows for scheduled fixtures.

    Fixtures do not need scores or result labels. For each fixture, completed
    match history is filtered to rows strictly before the fixture date. With
    date-only timestamps, same-date completed matches are excluded because their
    kickoff order is unknown.
    """
    _validate_fixture_columns(fixtures)

    completed = _prepare_completed_matches(completed_matches, feature_cutoff_date)
    fixture_rows = fixtures.copy(deep=True)
    fixture_rows["match_date"] = pd.to_datetime(
        fixture_rows["match_date"],
        errors="raise",
    )
    fixture_rows = fixture_rows.sort_values(
        ["match_date", "match_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    fixture_rows["is_neutral"] = fixture_rows.apply(_fixture_neutral_flag, axis=1)
    fixture_rows["tournament"] = fixture_rows.apply(_fixture_tournament, axis=1)

    if fixture_rows.empty:
        return pd.DataFrame()

    features = _build_team_form_fixture_features(
        completed,
        fixture_rows,
        windows=(5, 10),
    )
    if include_elo:
        elo_features = _build_fixture_elo_features(
            completed,
            fixture_rows,
            initial_rating=1500.0,
            k_factor=elo_k_factor,
            home_advantage=elo_home_advantage,
        )
        features = features.merge(
            elo_features,
            on="match_id",
            how="left",
            validate="one_to_one",
        )

    return features
