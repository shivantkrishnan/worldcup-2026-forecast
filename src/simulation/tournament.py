"""Monte Carlo group-stage tournament simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.simulation.scorelines import sample_scoreline_from_probabilities

OUTCOME_TEAM_A_WIN = "team_a_win"
OUTCOME_DRAW = "draw"
OUTCOME_TEAM_B_WIN = "team_b_win"
VALID_OUTCOMES = [OUTCOME_TEAM_A_WIN, OUTCOME_DRAW, OUTCOME_TEAM_B_WIN]
PROBABILITY_COLUMNS = ["p_team_a_win", "p_draw", "p_team_b_win"]
REQUIRED_COLUMNS = {"match_id", "group", "team_a", "team_b", *PROBABILITY_COLUMNS}
COMPLETED_COLUMN = "is_completed"
ACTUAL_RESULT_COLUMN = "actual_result"
SUMMARY_COLUMNS = [
    "team",
    "group",
    "simulations",
    "group_winner_prob",
    "top_2_prob",
    "third_place_prob",
    "best_third_place_advance_prob",
    "advance_prob",
    "avg_points",
    "avg_goals_for",
    "avg_goals_against",
    "avg_goal_difference",
    "avg_group_rank",
]


def validate_fixture_probability_table(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a copy of fixture-level probability rows."""
    missing = REQUIRED_COLUMNS.difference(fixtures.columns)
    if missing:
        raise ValueError(
            "Missing required fixture probability columns: "
            + ", ".join(sorted(missing))
        )

    validated = fixtures.copy(deep=True)
    if validated["match_id"].duplicated().any():
        raise ValueError("match_id values must be unique for simulation.")

    if validated["group"].isna().any():
        raise ValueError("group must be present for every fixture.")
    if validated["team_a"].isna().any() or validated["team_b"].isna().any():
        raise ValueError("team_a and team_b must be non-null for every fixture.")

    for column in PROBABILITY_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="raise")

    if (validated[PROBABILITY_COLUMNS] < 0).any().any():
        raise ValueError("Fixture probabilities must be nonnegative.")

    probability_sums = validated[PROBABILITY_COLUMNS].sum(axis=1)
    invalid_sum = ~np.isclose(probability_sums, 1.0, atol=1e-6)
    if invalid_sum.any():
        raise ValueError("Fixture probabilities must sum to 1 for every match.")

    if COMPLETED_COLUMN in validated.columns:
        validated[COMPLETED_COLUMN] = validated[COMPLETED_COLUMN].map(_coerce_bool)
        completed = validated[COMPLETED_COLUMN]
        if completed.any():
            if ACTUAL_RESULT_COLUMN not in validated.columns:
                raise ValueError(
                    "Completed fixture rows require an actual_result column."
                )
            missing_actual = completed & validated[ACTUAL_RESULT_COLUMN].isna()
            if missing_actual.any():
                raise ValueError(
                    "Completed fixture rows require non-null actual_result values."
                )
            invalid_actual = completed & ~validated[ACTUAL_RESULT_COLUMN].isin(
                VALID_OUTCOMES
            )
            if invalid_actual.any():
                raise ValueError(
                    "actual_result must be one of: " + ", ".join(VALID_OUTCOMES)
                )

            for column in ["team_a_goals", "team_b_goals"]:
                if column not in validated.columns:
                    raise ValueError(
                        f"Completed fixture rows require a {column} column."
                    )
                validated[column] = pd.to_numeric(
                    validated[column],
                    errors="coerce",
                )
                missing_scores = completed & validated[column].isna()
                if missing_scores.any():
                    raise ValueError(
                        f"Completed fixture rows require non-null {column}."
                    )

            for _, row in validated.loc[completed].iterrows():
                goals_a = int(row["team_a_goals"])
                goals_b = int(row["team_b_goals"])
                if goals_a > goals_b:
                    expected_result = OUTCOME_TEAM_A_WIN
                elif goals_a < goals_b:
                    expected_result = OUTCOME_TEAM_B_WIN
                else:
                    expected_result = OUTCOME_DRAW
                if row[ACTUAL_RESULT_COLUMN] != expected_result:
                    raise ValueError(
                        "Completed fixture scores must be consistent with "
                        "actual_result."
                    )

    return validated


def _coerce_bool(value: object) -> bool:
    """Coerce common boolean values from CSV-backed prediction tables."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _row_is_completed(row: pd.Series) -> bool:
    """Return whether a fixture row has a fixed completed result."""
    if COMPLETED_COLUMN not in row.index:
        return False
    return _coerce_bool(row[COMPLETED_COLUMN])


def sample_match_outcome(row: pd.Series, rng: np.random.Generator) -> str:
    """Sample one W/D/L outcome from fixture probabilities."""
    if _row_is_completed(row):
        return str(row[ACTUAL_RESULT_COLUMN])
    probabilities = row[PROBABILITY_COLUMNS].astype(float).to_numpy()
    return str(rng.choice(VALID_OUTCOMES, p=probabilities))


def _team_metadata(row: object, prefix: str) -> dict[str, float]:
    """Return optional team-level tie-break metadata from a fixture row."""
    conduct_column = f"{prefix}_team_conduct_score"
    ranking_column = f"{prefix}_fifa_ranking"
    conduct_score = (
        getattr(row, conduct_column)
        if hasattr(row, conduct_column)
        else np.nan
    )
    fifa_ranking = (
        getattr(row, ranking_column)
        if hasattr(row, ranking_column)
        else np.nan
    )
    return {
        "team_conduct_score": float(conduct_score)
        if not pd.isna(conduct_score)
        else np.nan,
        "fifa_ranking": float(fifa_ranking)
        if not pd.isna(fifa_ranking)
        else np.nan,
    }


def _initial_group_table(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return zeroed standings rows for every team/group in the fixtures."""
    team_rows: list[dict[str, object]] = []
    for row in fixtures.itertuples(index=False):
        group = getattr(row, "group")
        team_rows.append(
            {
                "group": group,
                "team": getattr(row, "team_a"),
                **_team_metadata(row, "team_a"),
            }
        )
        team_rows.append(
            {
                "group": group,
                "team": getattr(row, "team_b"),
                **_team_metadata(row, "team_b"),
            }
        )

    teams = pd.DataFrame(team_rows).drop_duplicates(["group", "team"])
    for column in [
        "played",
        "points",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_difference",
    ]:
        teams[column] = 0
    return teams.reset_index(drop=True)


def _add_team_result(
    table: pd.DataFrame,
    group: Any,
    team: Any,
    points: int,
    win: int,
    draw: int,
    loss: int,
    goals_for: int = 0,
    goals_against: int = 0,
) -> None:
    """Mutate one simulation table row with a sampled result."""
    mask = table["group"].eq(group) & table["team"].eq(team)
    table.loc[mask, "played"] += 1
    table.loc[mask, "points"] += points
    table.loc[mask, "wins"] += win
    table.loc[mask, "draws"] += draw
    table.loc[mask, "losses"] += loss
    table.loc[mask, "goals_for"] += goals_for
    table.loc[mask, "goals_against"] += goals_against
    table.loc[mask, "goal_difference"] += goals_for - goals_against


def _add_team_result_to_record(
    records: dict[tuple[Any, Any], dict[str, Any]],
    group: Any,
    team: Any,
    points: int,
    win: int,
    draw: int,
    loss: int,
    goals_for: int = 0,
    goals_against: int = 0,
) -> None:
    """Update one team standing record without repeated pandas indexing."""
    record = records[(group, team)]
    record["played"] += 1
    record["points"] += points
    record["wins"] += win
    record["draws"] += draw
    record["losses"] += loss
    record["goals_for"] += goals_for
    record["goals_against"] += goals_against
    record["goal_difference"] += goals_for - goals_against


def _records_to_group_table(
    records: dict[tuple[Any, Any], dict[str, Any]],
) -> pd.DataFrame:
    """Convert keyed standing records back to a group table dataframe."""
    return pd.DataFrame(
        [
            {"group": group, "team": team, **values}
            for (group, team), values in records.items()
        ]
    )


def _completed_score(row: pd.Series) -> tuple[int, int]:
    """Return completed goals when present, otherwise placeholder zeros."""
    if not _row_is_completed(row):
        return 0, 0
    if "team_a_goals" not in row.index or "team_b_goals" not in row.index:
        return 0, 0
    return int(row["team_a_goals"]), int(row["team_b_goals"])


def simulate_fixture_results_once(
    fixtures: pd.DataFrame,
    rng: np.random.Generator,
    scoreline_distributions: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return one simulated result row per fixture.

    Completed rows keep their actual result and score. Remaining rows sample a
    W/D/L class from model probabilities, then sample a scoreline conditional on
    that class.
    """
    validated = validate_fixture_probability_table(fixtures)
    rows: list[dict[str, object]] = []
    for _, row in validated.iterrows():
        is_fixed_result = _row_is_completed(row)
        if is_fixed_result:
            sampled_result = str(row[ACTUAL_RESULT_COLUMN])
            team_a_goals = int(row["team_a_goals"])
            team_b_goals = int(row["team_b_goals"])
        else:
            sampled_result, team_a_goals, team_b_goals = sample_scoreline_from_probabilities(
                row,
                rng,
                scoreline_distributions=scoreline_distributions,
            )

        rows.append(
            {
                "match_id": row["match_id"],
                "group": row["group"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "sampled_result": sampled_result,
                "team_a_goals": team_a_goals,
                "team_b_goals": team_b_goals,
                "is_fixed_result": is_fixed_result,
            }
        )
    return pd.DataFrame(rows)


def _add_head_to_head_metrics(
    group_table: pd.DataFrame,
    group_matches: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add head-to-head tie-break metrics for teams tied on points."""
    output = group_table.copy(deep=True)
    for column in [
        "head_to_head_points",
        "head_to_head_goal_difference",
        "head_to_head_goals_for",
    ]:
        output[column] = 0

    if group_matches is None or group_matches.empty:
        return output

    for _, tied_teams in output.groupby("points", sort=False):
        if len(tied_teams) <= 1:
            continue
        team_set = set(tied_teams["team"])
        head_to_head_matches = group_matches.loc[
            group_matches["team_a"].isin(team_set)
            & group_matches["team_b"].isin(team_set)
        ]
        if head_to_head_matches.empty:
            continue

        metrics = {
            team: {
                "points": 0,
                "goal_difference": 0,
                "goals_for": 0,
            }
            for team in team_set
        }
        for match in head_to_head_matches.itertuples(index=False):
            team_a = getattr(match, "team_a")
            team_b = getattr(match, "team_b")
            goals_a = int(getattr(match, "team_a_goals"))
            goals_b = int(getattr(match, "team_b_goals"))
            metrics[team_a]["goals_for"] += goals_a
            metrics[team_b]["goals_for"] += goals_b
            metrics[team_a]["goal_difference"] += goals_a - goals_b
            metrics[team_b]["goal_difference"] += goals_b - goals_a
            if goals_a > goals_b:
                metrics[team_a]["points"] += 3
            elif goals_a < goals_b:
                metrics[team_b]["points"] += 3
            else:
                metrics[team_a]["points"] += 1
                metrics[team_b]["points"] += 1

        for team, team_metrics in metrics.items():
            mask = output["team"].eq(team)
            output.loc[mask, "head_to_head_points"] = team_metrics["points"]
            output.loc[mask, "head_to_head_goal_difference"] = team_metrics[
                "goal_difference"
            ]
            output.loc[mask, "head_to_head_goals_for"] = team_metrics["goals_for"]

    return output


def rank_group_table(
    group_results: pd.DataFrame,
    tie_breaker_order: list[str] | None = None,
    rng: np.random.Generator | None = None,
    group_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank one group table with official-style group-stage tie-breakers.

    Head-to-head metrics are applied for teams tied on points. Multi-team ties
    are handled by aggregating matches among the tied teams, which is a clean
    approximation but does not recursively reapply head-to-head criteria after a
    subset remains tied.
    """
    tie_breakers = tie_breaker_order or [
        "points",
        "head_to_head_points",
        "head_to_head_goal_difference",
        "head_to_head_goals_for",
        "goal_difference",
        "goals_for",
        "team_conduct_score",
        "fifa_ranking",
    ]
    ranked = group_results.copy(deep=True)
    random_generator = rng if rng is not None else np.random.default_rng(0)
    ranked = _add_head_to_head_metrics(ranked, group_matches)
    for optional_column in ["team_conduct_score", "fifa_ranking"]:
        if optional_column not in ranked.columns:
            ranked[optional_column] = np.nan
    ranked["_random_tiebreak"] = random_generator.random(len(ranked))
    ranked["_team_conduct_sort"] = ranked["team_conduct_score"].fillna(-np.inf)
    ranked["_fifa_ranking_sort"] = ranked["fifa_ranking"].fillna(np.inf)

    sort_columns = []
    ascending = []
    for column in tie_breakers:
        if column == "team_conduct_score":
            sort_columns.append("_team_conduct_sort")
            ascending.append(False)
        elif column == "fifa_ranking":
            sort_columns.append("_fifa_ranking_sort")
            ascending.append(True)
        elif column in ranked.columns:
            sort_columns.append(column)
            ascending.append(False)
    sort_columns.extend(["_random_tiebreak", "team"])
    ascending.extend([False, True])

    ranked = ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ranked["group_rank"] = np.arange(1, len(ranked) + 1)
    return ranked.drop(
        columns=["_random_tiebreak", "_team_conduct_sort", "_fifa_ranking_sort"]
    )


def _apply_best_third_place_advancement(
    ranked_table: pd.DataFrame,
    n_best_third_place: int,
    rng: np.random.Generator,
    tie_breaker_order: list[str] | None,
) -> pd.DataFrame:
    """Mark best third-place teams as advanced using placeholder tie-breakers."""
    if n_best_third_place <= 0:
        return ranked_table

    output = ranked_table.copy(deep=True)
    candidates = output.loc[output["group_rank"].eq(3)].copy()
    if candidates.empty:
        return output

    ranked_candidates = rank_third_place_teams(candidates, rng=rng)
    selected = ranked_candidates.head(n_best_third_place)[["group", "team"]]
    selected_keys = set(zip(selected["group"], selected["team"]))
    selected_mask = output.apply(
        lambda row: (row["group"], row["team"]) in selected_keys,
        axis=1,
    )
    output.loc[selected_mask, "advanced"] = True
    output.loc[selected_mask, "best_third_place_advanced"] = True
    return output


def rank_third_place_teams(
    third_place_teams: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Rank third-place teams across groups using 2026-style criteria."""
    ranked = third_place_teams.copy(deep=True)
    random_generator = rng if rng is not None else np.random.default_rng(0)
    ranked["_team_conduct_sort"] = ranked.get(
        "team_conduct_score",
        pd.Series([np.nan] * len(ranked), index=ranked.index),
    ).fillna(-np.inf)
    ranked["_fifa_ranking_sort"] = ranked.get(
        "fifa_ranking",
        pd.Series([np.nan] * len(ranked), index=ranked.index),
    ).fillna(np.inf)
    ranked["_random_tiebreak"] = random_generator.random(len(ranked))
    ranked = ranked.sort_values(
        [
            "points",
            "goal_difference",
            "goals_for",
            "_team_conduct_sort",
            "_fifa_ranking_sort",
            "_random_tiebreak",
            "team",
        ],
        ascending=[False, False, False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked.drop(
        columns=["_team_conduct_sort", "_fifa_ranking_sort", "_random_tiebreak"]
    )


def simulate_group_stage_once(
    fixtures: pd.DataFrame,
    rng: np.random.Generator,
    points_for_win: int = 3,
    points_for_draw: int = 1,
    top_n_per_group: int = 2,
    include_best_third_place: bool = True,
    n_best_third_place: int = 8,
    tie_breaker_order: list[str] | None = None,
    scoreline_distributions: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Simulate one group stage and return one standings row per team."""
    validated = validate_fixture_probability_table(fixtures)
    table = _initial_group_table(validated)
    records = {
        (row["group"], row["team"]): {
            key: value
            for key, value in row.items()
            if key not in {"group", "team"}
        }
        for row in table.to_dict("records")
    }
    match_results = simulate_fixture_results_once(
        validated,
        rng=rng,
        scoreline_distributions=scoreline_distributions,
    )

    for _, row in match_results.iterrows():
        outcome = row["sampled_result"]
        group = row["group"]
        team_a = row["team_a"]
        team_b = row["team_b"]
        team_a_goals = int(row["team_a_goals"])
        team_b_goals = int(row["team_b_goals"])

        if outcome == OUTCOME_TEAM_A_WIN:
            _add_team_result_to_record(
                records,
                group,
                team_a,
                points_for_win,
                1,
                0,
                0,
                goals_for=team_a_goals,
                goals_against=team_b_goals,
            )
            _add_team_result_to_record(
                records,
                group,
                team_b,
                0,
                0,
                0,
                1,
                goals_for=team_b_goals,
                goals_against=team_a_goals,
            )
        elif outcome == OUTCOME_TEAM_B_WIN:
            _add_team_result_to_record(
                records,
                group,
                team_a,
                0,
                0,
                0,
                1,
                goals_for=team_a_goals,
                goals_against=team_b_goals,
            )
            _add_team_result_to_record(
                records,
                group,
                team_b,
                points_for_win,
                1,
                0,
                0,
                goals_for=team_b_goals,
                goals_against=team_a_goals,
            )
        else:
            _add_team_result_to_record(
                records,
                group,
                team_a,
                points_for_draw,
                0,
                1,
                0,
                goals_for=team_a_goals,
                goals_against=team_b_goals,
            )
            _add_team_result_to_record(
                records,
                group,
                team_b,
                points_for_draw,
                0,
                1,
                0,
                goals_for=team_b_goals,
                goals_against=team_a_goals,
            )

    table = _records_to_group_table(records)
    ranked_groups = [
        rank_group_table(
            group_table,
            tie_breaker_order=tie_breaker_order,
            rng=rng,
            group_matches=match_results.loc[match_results["group"].eq(group)],
        )
        for group, group_table in table.groupby("group", sort=True)
    ]
    ranked = pd.concat(ranked_groups, ignore_index=True)
    ranked["third_place"] = ranked["group_rank"].eq(3)
    ranked["best_third_place_advanced"] = False
    ranked["advanced"] = ranked["group_rank"] <= top_n_per_group

    if include_best_third_place:
        ranked = _apply_best_third_place_advancement(
            ranked,
            n_best_third_place=n_best_third_place,
            rng=rng,
            tie_breaker_order=tie_breaker_order,
        )

    return ranked[
        [
            "team",
            "group",
            "played",
            "points",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_difference",
            "third_place",
            "best_third_place_advanced",
            "group_rank",
            "advanced",
        ]
    ]


def simulate_group_stage(
    fixtures: pd.DataFrame,
    n_simulations: int = 1000,
    random_seed: int = 42,
    points_for_win: int = 3,
    points_for_draw: int = 1,
    top_n_per_group: int = 2,
    include_best_third_place: bool = True,
    n_best_third_place: int = 8,
    tie_breaker_order: list[str] | None = None,
    scoreline_distributions: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Run repeated group-stage simulations and return team-level results."""
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    if top_n_per_group <= 0:
        raise ValueError("top_n_per_group must be a positive integer.")
    if n_best_third_place < 0:
        raise ValueError("n_best_third_place cannot be negative.")

    validated = validate_fixture_probability_table(fixtures)
    rng = np.random.default_rng(random_seed)
    simulations: list[pd.DataFrame] = []

    for simulation_id in range(1, n_simulations + 1):
        result = simulate_group_stage_once(
            validated,
            rng=rng,
            points_for_win=points_for_win,
            points_for_draw=points_for_draw,
            top_n_per_group=top_n_per_group,
            include_best_third_place=include_best_third_place,
            n_best_third_place=n_best_third_place,
            tie_breaker_order=tie_breaker_order,
            scoreline_distributions=scoreline_distributions,
        )
        result["simulation_id"] = simulation_id
        simulations.append(result)

    return pd.concat(simulations, ignore_index=True)


def summarize_advancement_probabilities(
    simulation_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize simulated group-stage advancement probabilities."""
    required = {
        "team",
        "group",
        "simulation_id",
        "points",
        "group_rank",
        "advanced",
        "goals_for",
        "goals_against",
        "goal_difference",
        "third_place",
        "best_third_place_advanced",
    }
    missing = required.difference(simulation_results.columns)
    if missing:
        raise ValueError(
            "Missing required simulation result columns: "
            + ", ".join(sorted(missing))
        )

    grouped = simulation_results.groupby(["team", "group"], sort=True)
    summary = grouped.agg(
        simulations=("simulation_id", "nunique"),
        group_winner_prob=("group_rank", lambda values: float((values == 1).mean())),
        top_2_prob=("group_rank", lambda values: float((values <= 2).mean())),
        third_place_prob=("third_place", lambda values: float(values.astype(bool).mean())),
        best_third_place_advance_prob=(
            "best_third_place_advanced",
            lambda values: float(values.astype(bool).mean()),
        ),
        advance_prob=("advanced", lambda values: float(values.astype(bool).mean())),
        avg_points=("points", "mean"),
        avg_goals_for=("goals_for", "mean"),
        avg_goals_against=("goals_against", "mean"),
        avg_goal_difference=("goal_difference", "mean"),
        avg_group_rank=("group_rank", "mean"),
    ).reset_index()

    return summary[SUMMARY_COLUMNS].sort_values(
        ["group", "advance_prob", "group_winner_prob", "team"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
