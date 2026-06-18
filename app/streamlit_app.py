"""Consumer-facing Streamlit dashboard for World Cup 2026 forecasts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simulate_group_stage import prepare_simulation_fixture_table  # noqa: E402
from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.data.tournament_fixtures import load_tournament_fixtures  # noqa: E402
from src.data.tournament_results import load_tournament_results  # noqa: E402
from src.presentation.dashboard import (  # noqa: E402
    build_current_group_table,
    format_number,
    format_percent,
    get_teams_from_fixtures,
    prepare_match_table,
    prepare_probability_summary,
)
from src.presentation.match_display import build_match_display_table  # noqa: E402
from src.simulation.scorelines import build_empirical_scoreline_distributions  # noqa: E402
from src.simulation.tournament import (  # noqa: E402
    simulate_group_stage,
    summarize_advancement_probabilities,
    validate_fixture_probability_table,
)
from src.utils.config import (  # noqa: E402
    DEFAULT_TRAINING_CUTOFF_DATE,
    FIXTURE_PREDICTIONS_2026_PATH,
    FIXTURES_2026_PATH,
    RESULTS_2026_PATH,
)

LIVE_PREDICTIONS_PATH = PROJECT_ROOT / "data/tournament/fixture_predictions_2026_live.csv"
BACKFILLED_PREDICTIONS_PATH = PROJECT_ROOT / FIXTURE_PREDICTIONS_2026_PATH
FIXTURES_PATH = PROJECT_ROOT / FIXTURES_2026_PATH
RESULTS_PATH = PROJECT_ROOT / RESULTS_2026_PATH
NAV_ITEMS = [
    "Overview",
    "Groups",
    "Matches",
    "Teams",
    "Simulation",
    "Methodology",
]


def _inject_styles() -> None:
    """Apply restrained dashboard styling."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }
        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            color: inherit;
            color: var(--text-color, inherit);
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
        }
        .status-banner {
            border: 1px solid rgba(70, 140, 92, 0.45);
            border-left: 4px solid #4b9462;
            background: rgba(70, 140, 92, 0.10);
            color: inherit;
            color: var(--text-color, inherit);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            margin: 0.6rem 0 1.1rem 0;
        }
        .method-card {
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: rgba(128, 128, 128, 0.08);
            background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            color: inherit;
            color: var(--text-color, inherit);
            margin-bottom: 0.8rem;
        }
        .method-card h3 {
            margin-top: 0;
            margin-bottom: 0.35rem;
        }
        .method-card p {
            margin-bottom: 0;
        }
        .method-cta {
            border-color: rgba(80, 130, 190, 0.42);
            background: rgba(80, 130, 190, 0.10);
            color: inherit;
            color: var(--text-color, inherit);
        }
        .methodology-article {
            max-width: 920px;
            line-height: 1.68;
            color: inherit;
            color: var(--text-color, inherit);
        }
        .methodology-article h2 {
            margin-top: 2rem;
            margin-bottom: 0.55rem;
            font-size: 1.35rem;
        }
        .methodology-article p {
            margin-bottom: 0.85rem;
            font-size: 1rem;
        }
        .methodology-callout {
            border-left: 4px solid #355f8d;
            background: rgba(80, 130, 190, 0.10);
            color: inherit;
            color: var(--text-color, inherit);
            padding: 0.85rem 1rem;
            margin: 1rem 0 1.25rem 0;
            border-radius: 0 8px 8px 0;
        }
        .methodology-callout p {
            margin-bottom: 0;
        }
        .small-muted {
            color: inherit;
            color: var(--text-color, inherit);
            font-size: 0.92rem;
            opacity: 0.72;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 8px;
            padding: 0.25rem 0.35rem;
            margin-bottom: 0.25rem;
            background: rgba(128, 128, 128, 0.08);
            background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            color: inherit;
            color: var(--text-color, inherit);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            border-color: rgba(80, 130, 190, 0.55);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _data_path(path: Path) -> str:
    """Return a short display path relative to the project root."""
    if str(path) in {"", "."}:
        return "-"
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _coerce_bool(value: object) -> bool:
    """Coerce common CSV-backed boolean values."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _requested_page_from_query() -> str:
    """Return the requested dashboard page from the URL query string."""
    requested_page = st.query_params.get("page", NAV_ITEMS[0])
    if isinstance(requested_page, list):
        requested_page = requested_page[0] if requested_page else NAV_ITEMS[0]
    return requested_page if requested_page in NAV_ITEMS else NAV_ITEMS[0]


@st.cache_data(show_spinner=False)
def cached_load_fixtures(path: str) -> pd.DataFrame:
    """Load tournament fixtures with project validation."""
    return load_tournament_fixtures(path)


@st.cache_data(show_spinner=False)
def cached_load_results(path: str, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Load tournament results with orientation validation."""
    return load_tournament_results(path, fixtures_or_predictions=fixtures)


@st.cache_data(show_spinner=False)
def cached_load_predictions(
    live_path: str,
    fallback_path: str,
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """Load live predictions, falling back to backfilled predictions if needed."""
    live = Path(live_path)
    fallback = Path(fallback_path)
    metadata = {
        "source_path": "",
        "source_kind": "missing",
        "warning": "",
    }
    if live.exists():
        predictions = validate_fixture_probability_table(pd.read_csv(live))
        metadata["source_path"] = str(live)
        metadata["source_kind"] = "live"
        return predictions, metadata
    if fallback.exists():
        predictions = validate_fixture_probability_table(pd.read_csv(fallback))
        metadata["source_path"] = str(fallback)
        metadata["source_kind"] = "backfilled_ex_ante"
        metadata["warning"] = (
            "Live prediction file is missing. Showing backfilled ex-ante "
            "probabilities until live predictions are generated."
        )
        return predictions, metadata
    metadata["warning"] = "No fixture prediction file was found."
    return None, metadata


@st.cache_data(show_spinner=False)
def cached_scoreline_distributions() -> tuple[dict[str, pd.DataFrame], str]:
    """Load empirical scoreline distributions when historical data is available."""
    try:
        completed_matches = load_baseline_training_matches()
    except FileNotFoundError:
        return {}, "Fallback scoreline distribution"

    distributions = build_empirical_scoreline_distributions(completed_matches)
    if not distributions:
        return {}, "Fallback scoreline distribution"
    return distributions, "Empirical historical scoreline distribution"


@st.cache_data(show_spinner="Running group-stage simulation...")
def cached_simulation(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    results: pd.DataFrame | None,
    n_simulations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run group-stage simulation from the current prediction/result state."""
    if results is not None and not results.empty:
        simulation_fixtures = prepare_simulation_fixture_table(
            fixtures,
            predictions,
            results,
        )
    else:
        if len(predictions) < len(fixtures):
            raise ValueError(
                "Remaining-only live prediction files require results_2026.csv "
                "so completed fixtures can be fixed."
            )
        simulation_fixtures = validate_fixture_probability_table(predictions)

    scoreline_distributions, scoreline_source = cached_scoreline_distributions()
    simulation_results = simulate_group_stage(
        simulation_fixtures,
        n_simulations=n_simulations,
        random_seed=42,
        top_n_per_group=2,
        include_best_third_place=True,
        n_best_third_place=8,
        scoreline_distributions=scoreline_distributions,
    )
    summary = summarize_advancement_probabilities(simulation_results)
    completed = (
        simulation_fixtures["is_completed"].map(_coerce_bool)
        if "is_completed" in simulation_fixtures.columns
        else pd.Series([False] * len(simulation_fixtures), index=simulation_fixtures.index)
    )
    metadata = {
        "n_simulations": n_simulations,
        "fixed_completed_matches": int(completed.sum()),
        "sampled_remaining_matches": int((~completed).sum()),
        "scoreline_source": scoreline_source,
        "scoreline_simulation_used": "yes",
    }
    return summary, metadata


def _show_setup_block(title: str, body: str, command: str | None = None) -> None:
    """Render a clear setup message."""
    st.warning(title)
    st.write(body)
    if command:
        st.code(command, language="bash")


def _metadata_value(predictions: pd.DataFrame | None, column: str) -> str:
    """Return a compact metadata value from prediction rows."""
    if predictions is None or predictions.empty or column not in predictions.columns:
        return "-"
    values = sorted(predictions[column].dropna().astype(str).unique())
    return ", ".join(values) if values else "-"


def _status_counts(display_table: pd.DataFrame) -> dict[str, int]:
    """Return display status counts with zero defaults."""
    counts = display_table["display_status"].value_counts().to_dict()
    return {
        "completed": int(counts.get("completed", 0)),
        "scheduled": int(counts.get("scheduled", 0)),
        "prediction_missing": int(counts.get("prediction_missing", 0)),
    }


def _bar_chart(
    summary: pd.DataFrame,
    value_column: str,
    label: str,
    n: int = 12,
) -> None:
    """Render a sorted Streamlit bar chart."""
    if summary.empty or value_column not in summary.columns:
        st.info("Simulation output is not available for this chart.")
        return
    chart_data = (
        summary.sort_values(value_column, ascending=False, kind="mergesort")
        .head(n)
        .set_index("team")[[value_column]]
        .rename(columns={value_column: label})
    )
    st.bar_chart(chart_data)


def _render_probability_table(summary: pd.DataFrame, sort_column: str, n: int = 12) -> None:
    """Render a formatted probability table."""
    if summary.empty:
        st.info("Simulation output is not available.")
        return
    table = (
        summary.sort_values(sort_column, ascending=False, kind="mergesort")
        .head(n)
        .pipe(prepare_probability_summary)
    )
    st.dataframe(table, hide_index=True, width="stretch")


def _render_overview(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame | None,
    prediction_metadata: dict[str, str],
    display_table: pd.DataFrame,
    simulation_summary: pd.DataFrame | None,
    simulation_metadata: dict[str, Any] | None,
) -> None:
    """Render the dashboard overview."""
    st.title("World Cup 2026 Forecasting Dashboard")
    st.caption(
        "Live group-stage forecasts using completed results plus model probabilities "
        "for remaining fixtures."
    )

    counts = _status_counts(display_table)
    teams = get_teams_from_fixtures(fixtures)
    groups = sorted(fixtures["group"].dropna().astype(str).unique())
    metric_columns = st.columns(6)
    metric_columns[0].metric("Completed matches", counts["completed"])
    metric_columns[1].metric("Scheduled matches", counts["scheduled"])
    metric_columns[2].metric("Teams tracked", len(teams))
    metric_columns[3].metric("Groups tracked", len(groups))
    metric_columns[4].metric(
        "Prediction mode",
        _metadata_value(predictions, "forecast_mode"),
    )
    metric_columns[5].metric(
        "Feature cutoff",
        _metadata_value(predictions, "feature_cutoff_date"),
    )

    st.markdown(
        """
        <div class="status-banner">
        Completed matches are fixed from official results. Scheduled matches use
        live model probabilities. The scoreline layer is approximate and exists
        to make group-table mechanics use goal difference and goals scored.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="method-card method-cta">
        <h3>How the model works</h3>
        <p>
        The dashboard separates completed results from scheduled-match
        predictions, keeps 2026 results out of baseline training, and reports
        probabilistic forecasts rather than single-score promises. Open the
        Methodology page for the full applied-science explanation.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Methodology", type="secondary"):
        st.query_params["page"] = "Methodology"
        st.rerun()

    if prediction_metadata.get("warning"):
        st.warning(prediction_metadata["warning"])
        st.code(
            "python scripts/generate_fixture_predictions.py --forecast-mode live "
            "--results data/tournament/results_2026.csv "
            "--output data/tournament/fixture_predictions_2026_live.csv",
            language="bash",
        )

    if simulation_metadata:
        meta_columns = st.columns(4)
        meta_columns[0].metric("Simulations", f"{simulation_metadata['n_simulations']:,}")
        meta_columns[1].metric("Fixed results", simulation_metadata["fixed_completed_matches"])
        meta_columns[2].metric("Sampled matches", simulation_metadata["sampled_remaining_matches"])
        meta_columns[3].metric("Scoreline layer", "On")

    if simulation_summary is None or simulation_summary.empty:
        st.info("Simulation outputs are unavailable until prediction data can be loaded.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Most likely to advance")
        _bar_chart(simulation_summary, "advance_prob", "Advance probability")
        _render_probability_table(simulation_summary, "advance_prob", n=10)
    with right:
        st.subheader("Most likely group winners")
        _bar_chart(simulation_summary, "group_winner_prob", "Group winner probability")
        _render_probability_table(simulation_summary, "group_winner_prob", n=10)

    scheduled = display_table.loc[display_table["display_status"].eq("scheduled")].copy()
    if not scheduled.empty:
        scheduled["max_probability"] = scheduled[
            ["p_team_a_win", "p_draw", "p_team_b_win"]
        ].max(axis=1)
        closest = scheduled.sort_values(
            ["max_probability", "match_date"],
            ascending=[True, True],
            kind="mergesort",
        ).head(8)
        st.subheader("Upcoming matches with the most uncertainty")
        st.dataframe(
            prepare_match_table(closest),
            hide_index=True,
            width="stretch",
        )

    with st.expander("What this means"):
        st.write(
            "Advancement probability is the share of simulated group-stage worlds "
            "where a team reaches the next round. It is not the same as being the "
            "favorite in a single match. Draws, goal difference, and third-place "
            "comparisons all matter."
        )


def _render_groups(
    fixtures: pd.DataFrame,
    current_table: pd.DataFrame,
    display_table: pd.DataFrame,
    simulation_summary: pd.DataFrame | None,
) -> None:
    """Render the group-focused view."""
    st.title("Groups")
    groups = sorted(fixtures["group"].dropna().astype(str).unique())
    controls = st.columns([1.1, 1.3, 1])
    selected_group = controls[0].selectbox("Group", groups)
    status_filter = controls[1].radio(
        "Matches",
        ["All", "Completed", "Scheduled", "Prediction missing"],
        horizontal=True,
    )
    show_audit = controls[2].checkbox("Show audit probabilities", value=False)

    st.subheader(f"Group {selected_group} table")
    group_table = current_table.loc[current_table["group"].eq(selected_group)]
    st.dataframe(group_table, hide_index=True, width="stretch")

    if simulation_summary is not None and not simulation_summary.empty:
        st.subheader("Forecast probabilities")
        group_summary = simulation_summary.loc[
            simulation_summary["group"].astype(str).eq(selected_group)
        ].sort_values("advance_prob", ascending=False, kind="mergesort")
        st.dataframe(
            prepare_probability_summary(group_summary),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Matches")
    matches = display_table.loc[display_table["group"].astype(str).eq(selected_group)].copy()
    status_map = {
        "Completed": "completed",
        "Scheduled": "scheduled",
        "Prediction missing": "prediction_missing",
    }
    if status_filter != "All":
        matches = matches.loc[matches["display_status"].eq(status_map[status_filter])]
    if matches.empty:
        st.info("No matches match the selected filters.")
    else:
        st.dataframe(
            prepare_match_table(matches, show_audit_probabilities=show_audit),
            hide_index=True,
            width="stretch",
        )


def _render_matches(display_table: pd.DataFrame) -> None:
    """Render searchable/filterable match table."""
    st.title("Matches")
    groups = sorted(display_table["group"].dropna().astype(str).unique())
    teams = sorted(
        pd.concat([display_table["team_a"], display_table["team_b"]])
        .dropna()
        .astype(str)
        .unique()
    )

    controls = st.columns([1.3, 1.4, 1.4, 1.2])
    selected_groups = controls[0].multiselect("Groups", groups, default=groups)
    selected_statuses = controls[1].multiselect(
        "Status",
        ["completed", "scheduled", "prediction_missing"],
        default=["completed", "scheduled", "prediction_missing"],
    )
    selected_team = controls[2].selectbox("Team", ["All teams", *teams])
    show_audit = controls[3].checkbox("Audit probabilities", value=False)

    mode_options = sorted(display_table["forecast_mode"].dropna().astype(str).unique())
    selected_modes = st.multiselect("Forecast mode", mode_options, default=mode_options)

    filtered = display_table.copy(deep=True)
    if selected_groups:
        filtered = filtered.loc[filtered["group"].astype(str).isin(selected_groups)]
    if selected_statuses:
        filtered = filtered.loc[filtered["display_status"].isin(selected_statuses)]
    if selected_team != "All teams":
        filtered = filtered.loc[
            filtered["team_a"].astype(str).eq(selected_team)
            | filtered["team_b"].astype(str).eq(selected_team)
        ]
    if selected_modes:
        filtered = filtered.loc[
            filtered["forecast_mode"].isna()
            | filtered["forecast_mode"].astype(str).isin(selected_modes)
        ]

    if filtered.empty:
        st.info("No matches match the selected filters.")
        return

    st.dataframe(
        prepare_match_table(filtered, show_audit_probabilities=show_audit),
        hide_index=True,
        width="stretch",
    )


def _render_teams(
    fixtures: pd.DataFrame,
    current_table: pd.DataFrame,
    display_table: pd.DataFrame,
    simulation_summary: pd.DataFrame | None,
) -> None:
    """Render team-focused view."""
    st.title("Teams")
    teams = get_teams_from_fixtures(fixtures)
    selected_team = st.selectbox("Team", teams)

    team_group_rows = current_table.loc[current_table["team"].eq(selected_team)]
    group = str(team_group_rows.iloc[0]["group"]) if not team_group_rows.empty else "-"
    cols = st.columns(5)
    cols[0].metric("Group", group)
    if not team_group_rows.empty:
        row = team_group_rows.iloc[0]
        cols[1].metric("Current rank", int(row["rank"]))
        cols[2].metric("Points", int(row["points"]))
        cols[3].metric("Goal difference", int(row["goal_difference"]))
        cols[4].metric("Played", int(row["played"]))

    if simulation_summary is not None and not simulation_summary.empty:
        team_summary = simulation_summary.loc[simulation_summary["team"].eq(selected_team)]
        if not team_summary.empty:
            forecast = team_summary.iloc[0]
            st.subheader("Forecast outlook")
            forecast_cols = st.columns(5)
            forecast_cols[0].metric("Advance", format_percent(forecast["advance_prob"]))
            forecast_cols[1].metric(
                "Group winner",
                format_percent(forecast["group_winner_prob"]),
            )
            forecast_cols[2].metric("Top two", format_percent(forecast["top_2_prob"]))
            forecast_cols[3].metric(
                "Best third route",
                format_percent(forecast["best_third_place_advance_prob"]),
            )
            forecast_cols[4].metric(
                "Avg GD",
                format_number(forecast["avg_goal_difference"], decimals=2),
            )

    team_matches = display_table.loc[
        display_table["team_a"].astype(str).eq(selected_team)
        | display_table["team_b"].astype(str).eq(selected_team)
    ].copy()

    completed = team_matches.loc[team_matches["display_status"].eq("completed")]
    scheduled = team_matches.loc[team_matches["display_status"].eq("scheduled")]

    left, right = st.columns(2)
    with left:
        st.subheader("Completed results")
        if completed.empty:
            st.info("No completed results for this team yet.")
        else:
            st.dataframe(
                prepare_match_table(completed),
                hide_index=True,
                width="stretch",
            )
    with right:
        st.subheader("Remaining fixtures")
        if scheduled.empty:
            st.info("No scheduled predictions for this team.")
        else:
            st.dataframe(
                prepare_match_table(scheduled),
                hide_index=True,
                width="stretch",
            )


def _render_simulation(
    simulation_summary: pd.DataFrame | None,
    simulation_metadata: dict[str, Any] | None,
) -> None:
    """Render simulation outputs."""
    st.title("Simulation")
    if simulation_summary is None or simulation_summary.empty or simulation_metadata is None:
        st.warning("Simulation output is unavailable.")
        return

    cols = st.columns(5)
    cols[0].metric("Simulations", f"{simulation_metadata['n_simulations']:,}")
    cols[1].metric("Fixed completed matches", simulation_metadata["fixed_completed_matches"])
    cols[2].metric("Sampled remaining matches", simulation_metadata["sampled_remaining_matches"])
    cols[3].metric("Scoreline simulation", "On")
    cols[4].metric("Scoreline source", simulation_metadata["scoreline_source"])

    st.subheader("Advancement probabilities")
    _bar_chart(simulation_summary, "advance_prob", "Advance probability", n=16)
    _render_probability_table(simulation_summary, "advance_prob", n=16)

    left, right = st.columns(2)
    with left:
        st.subheader("Group winners")
        _bar_chart(simulation_summary, "group_winner_prob", "Group winner probability")
    with right:
        st.subheader("Best third-place route")
        third = simulation_summary.loc[
            simulation_summary["best_third_place_advance_prob"] > 0
        ]
        _bar_chart(third, "best_third_place_advance_prob", "Best third advance")

    st.subheader("Average table outcomes")
    table = simulation_summary[
        [
            "team",
            "group",
            "avg_points",
            "avg_goal_difference",
            "top_2_prob",
            "advance_prob",
        ]
    ].copy()
    table["avg_points"] = table["avg_points"].map(lambda value: format_number(value, 2))
    table["avg_goal_difference"] = table["avg_goal_difference"].map(
        lambda value: format_number(value, 2)
    )
    table["top_2_prob"] = table["top_2_prob"].map(format_percent)
    table["advance_prob"] = table["advance_prob"].map(format_percent)
    st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Simulation caveats"):
        st.write(
            "The match model predicts win, draw, and loss probabilities. Exact "
            "scorelines are sampled from an empirical conditional layer so group "
            "tables can use goal difference and goals scored. Knockout paths are "
            "not simulated yet, and probabilities should not be treated as betting "
            "advice."
        )


def _render_methodology(prediction_metadata: dict[str, str]) -> None:
    """Render methodology as a consumer-facing technical article."""
    st.title("Methodology")
    st.caption(
        "How the dashboard turns historical results, live tournament state, "
        "and calibrated match probabilities into group-stage forecasts."
    )

    st.markdown(
        f"""
        <div class="methodology-article">
        <h2>What This Dashboard Is Estimating</h2>
        <p>
        The dashboard estimates uncertainty around 2026 World Cup group-stage
        matches and group outcomes. At the match level, the model produces a
        probability for exactly three outcomes: <strong>Team A win</strong>,
        <strong>draw</strong>, and <strong>Team B win</strong>. It is not a
        pick engine. A team can be the favorite and still fail to win because a
        forecast is a distribution over possible outcomes, not a promise about
        one future.
        </p>
        <p>
        That distinction matters in soccer. Even strong teams draw. A side that
        is better over a season can be held scoreless for 90 minutes. A set
        piece, a red card, finishing variance, or a conservative tactical setup
        can move a match away from the most likely outcome. The job of the
        model is to put sensible probabilities on those possibilities.
        </p>

        <div class="methodology-callout">
        <p>
        <strong>Core target:</strong> estimate the probability of Team A win,
        draw, and Team B win before the match is played, using only information
        that would have been available by the feature cutoff.
        </p>
        </div>

        <h2>Forecasting Probabilities, Not Picking Winners</h2>
        <p>
        A World Cup group is a linked set of uncertain events. A single match
        forecast is useful, but advancement depends on the joint path of many
        fixtures. Draws change incentives. Goal difference and goals scored can
        decide ranking. Third-place qualification means a result in one group
        can affect the value of a result in another group.
        </p>
        <p>
        In economic terms, a forecast is an information asset. Its value comes
        from the distribution over possible states, not just the single most
        likely state. A 42 percent win probability, 31 percent draw probability,
        and 27 percent loss probability is more useful than simply saying
        "favorite." It tells us how often the group table may branch into
        different worlds.
        </p>

        <h2>The Information Set</h2>
        <p>
        Forecasts have to respect time. The selected baseline model is trained
        only on matches available through
        <strong>{DEFAULT_TRAINING_CUTOFF_DATE}</strong>. Completed 2026 World
        Cup matches can update the live tournament state and the live feature
        state, but they do not retrain the baseline model. This is a deliberate
        separation between learning a historical relationship and conditioning
        on facts that have occurred during the tournament.
        </p>
        <p>
        The same timing discipline appears in economic forecasting and causal
        inference: the model should not use information that was unavailable at
        prediction time. In machine learning language, that is leakage
        prevention. A feature that knows the result, directly or indirectly, is
        not a feature. It is the answer leaking into the input.
        </p>

        <h2>Historical Training Data Versus Tournament State</h2>
        <p>
        The project keeps several data layers separate because they answer
        different questions. Historical international results train and validate
        the model. The 2026 fixture file defines the scheduled matches. The
        official results file defines the live tournament state for completed
        matches. Generated prediction files hold model probabilities. The
        display table decides what the user should see for each fixture.
        </p>
        <p>
        This separation prevents two common mistakes. First, completed World Cup
        results should not be silently blended into baseline training. Second,
        a completed match should not be displayed as though it is still a live
        future prediction. Once a result exists, the score is a fact; the model
        probability attached to that match belongs in audit context.
        </p>

        <h2>Rolling Form as a State Variable</h2>
        <p>
        National teams evolve. Player pools change, managers change, injuries
        matter, and some teams enter a tournament in better rhythm than their
        long-run reputation suggests. Rolling form features treat recent
        performance as a state variable: recent points per match, goals for,
        goals against, goal difference, win rate, draw rate, and loss rate
        summarize observable team condition before a fixture.
        </p>
        <p>
        These features are built from a long team-match panel with two rows per
        match, one from each team's perspective. Every rolling and expanding
        statistic is shifted so the current match outcome is excluded. If a team
        plays on June 18, the feature values for that match can use prior
        matches only. This is the practical difference between a useful form
        measure and a leaked label.
        </p>

        <h2>Elo and Opponent-Adjusted Strength</h2>
        <p>
        Rolling form captures recent outcomes, but it does not fully account for
        opponent quality. Elo-style ratings provide an opponent-adjusted strength
        signal. Beating a strong team says more than beating a weak team.
        Losing narrowly to an elite opponent should not be treated the same as
        losing to a side with much lower prior strength.
        </p>
        <p>
        Elo behaves like a compact belief-updating rule: results move ratings
        more when they are surprising relative to prior expectations. That is
        close to Bayesian and economic intuition without pretending this is a
        full structural model of team quality. The current selected Elo variant
        uses a conservative update rate and a 50-point non-neutral historical
        home adjustment because that combination performed best on rolling-origin
        log loss among the tested variants.
        </p>
        <p>
        The home adjustment is used to learn from historical non-neutral
        matches. It is not a blanket assumption that every 2026 host receives
        the same generic advantage. The current World Cup fixtures are treated
        as neutral by default unless a future venue or host-country feature is
        explicitly introduced.
        </p>

        <h2>Why Calibrated Logistic Regression</h2>
        <p>
        The first selected model is sigmoid-calibrated logistic regression over
        leakage-safe rolling-form and Elo features. This is a deliberately
        restrained baseline. It outputs full class probabilities, is stable
        enough for repeated validation, and remains interpretable enough to
        audit when probabilities look surprising.
        </p>
        <p>
        Starting here avoids premature black-box complexity. In a setting with
        changing teams, sparse international fixtures, and a small number of
        World Cup matches, a stable model with explicit assumptions can be more
        valuable than a flexible model that is difficult to diagnose. Stronger
        models should be added later, but they should have to beat this baseline
        on probability quality, not just produce more elaborate outputs.
        </p>

        <h2>Validation and Backtesting</h2>
        <p>
        Random train/test splits are not the primary validation design because
        football team strength changes over time. Random splits can train on
        future-era information and test on older matches, which gives an
        unrealistic picture of forecasting performance. The project uses
        time-aware holdouts and rolling-origin backtesting so each validation
        window resembles the actual forecasting problem: learn from the past,
        predict the future.
        </p>
        <p>
        Rolling-origin validation is especially useful because it asks whether
        a modeling choice is stable across several historical cutoffs rather
        than lucky in one holdout. Tournament-specific backtesting over prior
        World Cups and major tournaments remains an important next validation
        priority before the simulation layer is treated as mature.
        </p>

        <h2>Calibration and Probability Quality</h2>
        <p>
        Calibration asks whether predicted probabilities mean what they say. If
        the model gives many events a 70 percent probability, those events
        should happen about 70 percent of the time over a large enough set of
        comparable predictions. This matters because the dashboard is consumed
        as probabilities, not just rankings.
        </p>
        <p>
        The selected baseline was chosen primarily because it improved
        rolling-origin mean log loss after adding Elo features and calibration.
        Calibration caveats remain visible because probability quality is not a
        single number. Log loss, Brier score, expected calibration error, and
        calibration-bin behavior can disagree, and that disagreement is itself
        useful diagnostic information.
        </p>

        <h2>Pre-Tournament, Backfilled, and Live Forecasts</h2>
        <p>
        The dashboard distinguishes forecast modes. A
        <strong>pre_tournament</strong> forecast uses only the original
        pre-tournament information set and simulates from the starting state. A
        <strong>backfilled_ex_ante</strong> forecast reconstructs what the model
        would have said using an earlier cutoff; it is useful for audit, but it
        is not a true logged live prediction. A <strong>live</strong> forecast
        updates tournament state and feature state with completed matches while
        leaving the baseline model itself unchanged.
        </p>
        <p>
        This distinction is important for trust. Completed matches should affect
        standings and remaining paths, but they should not be used to pretend
        the model knew something before kickoff that it only learned afterward.
        </p>

        <h2>Tournament Simulation</h2>
        <p>
        Match probabilities become advancement probabilities through Monte Carlo
        simulation. The app fixes completed results from the official results
        file, samples remaining group-stage matches from model probabilities,
        applies group ranking rules, and repeats that process many times.
        Advancement probability is the share of simulated worlds in which a team
        reaches the next round.
        </p>
        <p>
        This is scenario analysis. It translates match-level uncertainty into
        tournament-state uncertainty. A team may have a modest chance to win any
        one remaining match but still advance often because draws are enough, or
        because other teams split points in favorable ways.
        </p>

        <h2>Scoreline Simulation</h2>
        <p>
        The core model predicts win, draw, and loss probabilities, not exact
        scores. The scoreline layer samples plausible scores conditional on the
        sampled match result so that group tables can use goal difference and
        goals scored. This makes the standings mechanics more realistic than
        treating all wins or draws as identical.
        </p>
        <p>
        The scoreline layer should be read carefully. It supports table
        mechanics; it is not a calibrated exact-score model. Exact goals, player
        props, shot volume, and in-match dynamics are outside the current
        model's claim.
        </p>

        <h2>Display Semantics</h2>
        <p>
        The display layer separates facts from forecasts. <strong>Completed</strong>
        means the row is showing an actual official result. <strong>Scheduled</strong>
        means the row is showing a current prediction for an unplayed fixture.
        <strong>Audit probability</strong> means a model probability attached to
        a completed match for evaluation. <strong>Feature cutoff</strong> means
        the latest match date allowed into the live feature state.
        </p>
        <p>
        This prevents a subtle product problem: completed matches may still have
        model probabilities, but those probabilities are no longer current
        forecasts. They are evidence for evaluating the model, not a substitute
        for the score.
        </p>

        <h2>Current Limitations and Future Extensions</h2>
        <p>
        The dashboard does not yet simulate the knockout bracket. It does not
        include player availability, injuries, expected lineups, market odds,
        paid data feeds, or live event data such as shots, red cards, possession,
        or expected goals. The scoreline layer is approximate. Calibration
        caveats remain. The app is designed for forecasting interpretation and
        portfolio demonstration, not betting advice.
        </p>
        <p>
        The natural next extensions are tournament-specific backtesting, richer
        Elo variants, player-level squad features, market benchmarks, and a
        true knockout simulation once bracket logic and fixture placeholders are
        handled explicitly.
        </p>

        <h2>How to Read the Dashboard</h2>
        <p>
        Start with the Overview for tournament state: completed matches,
        scheduled matches, simulation count, and the current feature cutoff.
        Use Groups to see standings and group-specific advancement paths. Use
        Matches to inspect individual scheduled predictions and completed
        results. Use Teams to see one team's current table position and forecast
        outlook. Use Simulation to focus on advancement, group-winner, and
        third-place qualification probabilities.
        </p>
        <p>
        The most important habit is to read every number as a probability over
        possible states. A high probability is not certainty. A low probability
        is not impossibility. The dashboard is meant to make uncertainty visible
        enough to reason about it.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Current data state"):
        st.write(f"Prediction file: {_data_path(Path(prediction_metadata.get('source_path', '')))}")
        st.write(f"Prediction source type: {prediction_metadata.get('source_kind', '-')}")


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="World Cup 2026 Forecasting Dashboard",
        layout="wide",
    )
    _inject_styles()

    st.sidebar.title("World Cup 2026")
    st.sidebar.markdown("### Navigation")
    requested_page = _requested_page_from_query()
    selected_page = st.sidebar.radio(
        "Dashboard page",
        NAV_ITEMS,
        index=NAV_ITEMS.index(requested_page),
        label_visibility="collapsed",
    )
    if selected_page != requested_page:
        st.query_params["page"] = selected_page

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Simulation")
    simulation_count = st.sidebar.slider(
        "Simulation count",
        min_value=100,
        max_value=1000,
        value=300,
        step=100,
    )
    st.sidebar.caption(
        "Higher counts reduce simulation noise but take longer to compute."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Local entrypoint: app/streamlit_app.py")

    try:
        fixtures = cached_load_fixtures(str(FIXTURES_PATH))
    except FileNotFoundError:
        _show_setup_block(
            "Missing fixture file",
            "The dashboard needs the manually maintained 2026 group-stage fixtures.",
            "python scripts/generate_fixture_predictions.py --forecast-mode live "
            "--results data/tournament/results_2026.csv "
            "--output data/tournament/fixture_predictions_2026_live.csv",
        )
        return
    except ValueError as error:
        st.error(f"Fixture validation failed: {error}")
        return

    results: pd.DataFrame | None = None
    if RESULTS_PATH.exists():
        try:
            results = cached_load_results(str(RESULTS_PATH), fixtures)
        except (FileNotFoundError, ValueError) as error:
            st.warning(f"Results could not be loaded: {error}")
    else:
        st.warning(
            "results_2026.csv is missing. Completed matches cannot be fixed until "
            "the official result file is available."
        )

    predictions, prediction_metadata = cached_load_predictions(
        str(LIVE_PREDICTIONS_PATH),
        str(BACKFILLED_PREDICTIONS_PATH),
    )
    if predictions is None:
        st.warning("No prediction file is available.")
        st.code(
            "python scripts/generate_fixture_predictions.py --forecast-mode live "
            "--results data/tournament/results_2026.csv "
            "--output data/tournament/fixture_predictions_2026_live.csv",
            language="bash",
        )

    display_table = build_match_display_table(
        fixtures,
        predictions=predictions,
        results=results,
    )
    current_table = build_current_group_table(fixtures, results)

    simulation_summary: pd.DataFrame | None = None
    simulation_metadata: dict[str, Any] | None = None
    if predictions is not None:
        try:
            simulation_summary, simulation_metadata = cached_simulation(
                fixtures,
                predictions,
                results,
                simulation_count,
            )
        except (FileNotFoundError, ValueError) as error:
            st.warning(f"Simulation could not be computed: {error}")

    if selected_page == "Overview":
        _render_overview(
            fixtures,
            predictions,
            prediction_metadata,
            display_table,
            simulation_summary,
            simulation_metadata,
        )
    elif selected_page == "Groups":
        _render_groups(fixtures, current_table, display_table, simulation_summary)
    elif selected_page == "Matches":
        _render_matches(display_table)
    elif selected_page == "Teams":
        _render_teams(fixtures, current_table, display_table, simulation_summary)
    elif selected_page == "Simulation":
        _render_simulation(simulation_summary, simulation_metadata)
    else:
        _render_methodology(prediction_metadata)


if __name__ == "__main__":
    main()
