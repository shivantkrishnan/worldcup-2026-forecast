"""Monte Carlo group-stage tournament simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

OUTCOME_TEAM_A_WIN = "team_a_win"
OUTCOME_DRAW = "draw"
OUTCOME_TEAM_B_WIN = "team_b_win"
VALID_OUTCOMES = [OUTCOME_TEAM_A_WIN, OUTCOME_DRAW, OUTCOME_TEAM_B_WIN]
PROBABILITY_COLUMNS = ["p_team_a_win", "p_draw", "p_team_b_win"]
REQUIRED_COLUMNS = {"match_id", "group", "team_a", "team_b", *PROBABILITY_COLUMNS}
SUMMARY_COLUMNS = [
    "team",
    "group",
    "simulations",
    "group_winner_prob",
    "top_2_prob",
    "advance_prob",
    "avg_points",
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

    return validated


def sample_match_outcome(row: pd.Series, rng: np.random.Generator) -> str:
    """Sample one W/D/L outcome from fixture probabilities."""
    probabilities = row[PROBABILITY_COLUMNS].astype(float).to_numpy()
    return str(rng.choice(VALID_OUTCOMES, p=probabilities))


def _initial_group_table(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return zeroed standings rows for every team/group in the fixtures."""
    team_rows: list[dict[str, object]] = []
    for row in fixtures.itertuples(index=False):
        group = getattr(row, "group")
        team_rows.append({"group": group, "team": getattr(row, "team_a")})
        team_rows.append({"group": group, "team": getattr(row, "team_b")})

    teams = pd.DataFrame(team_rows).drop_duplicates(["group", "team"])
    for column in [
        "points",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
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
) -> None:
    """Mutate one simulation table row with a sampled result."""
    mask = table["group"].eq(group) & table["team"].eq(team)
    table.loc[mask, "points"] += points
    table.loc[mask, "wins"] += win
    table.loc[mask, "draws"] += draw
    table.loc[mask, "losses"] += loss


def rank_group_table(
    group_results: pd.DataFrame,
    tie_breaker_order: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Rank one group table using a temporary seeded random final tie-break."""
    tie_breakers = tie_breaker_order or ["points", "wins"]
    ranked = group_results.copy(deep=True)
    random_generator = rng if rng is not None else np.random.default_rng(0)
    ranked["_random_tiebreak"] = random_generator.random(len(ranked))
    ranked = ranked.sort_values(
        [*tie_breakers, "_random_tiebreak", "team"],
        ascending=[False for _ in tie_breakers] + [False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["group_rank"] = np.arange(1, len(ranked) + 1)
    return ranked.drop(columns=["_random_tiebreak"])


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

    ranked_candidates = rank_group_table(
        candidates,
        tie_breaker_order=tie_breaker_order,
        rng=rng,
    )
    selected = ranked_candidates.head(n_best_third_place)[["group", "team"]]
    selected_keys = set(zip(selected["group"], selected["team"]))
    selected_mask = output.apply(
        lambda row: (row["group"], row["team"]) in selected_keys,
        axis=1,
    )
    output.loc[selected_mask, "advanced"] = True
    return output


def simulate_group_stage_once(
    fixtures: pd.DataFrame,
    rng: np.random.Generator,
    points_for_win: int = 3,
    points_for_draw: int = 1,
    top_n_per_group: int = 2,
    include_best_third_place: bool = False,
    n_best_third_place: int = 0,
    tie_breaker_order: list[str] | None = None,
) -> pd.DataFrame:
    """Simulate one group stage and return one standings row per team."""
    validated = validate_fixture_probability_table(fixtures)
    table = _initial_group_table(validated)

    for _, row in validated.iterrows():
        outcome = sample_match_outcome(row, rng)
        group = row["group"]
        team_a = row["team_a"]
        team_b = row["team_b"]

        if outcome == OUTCOME_TEAM_A_WIN:
            _add_team_result(table, group, team_a, points_for_win, 1, 0, 0)
            _add_team_result(table, group, team_b, 0, 0, 0, 1)
        elif outcome == OUTCOME_TEAM_B_WIN:
            _add_team_result(table, group, team_a, 0, 0, 0, 1)
            _add_team_result(table, group, team_b, points_for_win, 1, 0, 0)
        else:
            _add_team_result(table, group, team_a, points_for_draw, 0, 1, 0)
            _add_team_result(table, group, team_b, points_for_draw, 0, 1, 0)

    ranked_groups = [
        rank_group_table(
            group_table,
            tie_breaker_order=tie_breaker_order,
            rng=rng,
        )
        for _, group_table in table.groupby("group", sort=True)
    ]
    ranked = pd.concat(ranked_groups, ignore_index=True)
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
            "points",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
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
    include_best_third_place: bool = False,
    n_best_third_place: int = 0,
    tie_breaker_order: list[str] | None = None,
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
        )
        result["simulation_id"] = simulation_id
        simulations.append(result)

    return pd.concat(simulations, ignore_index=True)


def summarize_advancement_probabilities(
    simulation_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize simulated group-stage advancement probabilities."""
    required = {"team", "group", "simulation_id", "points", "group_rank", "advanced"}
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
        advance_prob=("advanced", lambda values: float(values.astype(bool).mean())),
        avg_points=("points", "mean"),
        avg_group_rank=("group_rank", "mean"),
    ).reset_index()

    return summary[SUMMARY_COLUMNS].sort_values(
        ["group", "advance_prob", "group_winner_prob", "team"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
