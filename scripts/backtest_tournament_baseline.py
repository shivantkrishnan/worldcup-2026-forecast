"""Run tournament-specific baseline validation without writing artifacts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.features.build_features import build_modeling_features  # noqa: E402
from src.models.backtest import format_metric_mean_std, train_and_evaluate_on_split  # noqa: E402

DEFAULT_WORLD_CUP_YEARS = (2002, 2006, 2010, 2014, 2018, 2022)
DEFAULT_TOURNAMENT_LABEL = "FIFA World Cup"
DEFAULT_MIN_TEST_MATCHES = 10


@dataclass(frozen=True)
class TournamentModelSetup:
    """Feature/model setup evaluated in tournament holdouts."""

    name: str
    include_elo: bool
    elo_k_factor: float = 20.0
    elo_home_advantage: float = 0.0


MODEL_SETUPS = (
    TournamentModelSetup(name="no_elo_calibrated_logistic", include_elo=False),
    TournamentModelSetup(
        name="simple_elo_calibrated_logistic",
        include_elo=True,
        elo_k_factor=20.0,
        elo_home_advantage=0.0,
    ),
    TournamentModelSetup(
        name="selected_elo_calibrated_logistic",
        include_elo=True,
        elo_k_factor=10.0,
        elo_home_advantage=50.0,
    ),
)


def _normalize_label(value: object) -> str:
    """Normalize a tournament label for robust equality checks."""
    return str(value).strip().casefold()


def add_tournament_year(
    matches: pd.DataFrame,
    tournament_label: str = DEFAULT_TOURNAMENT_LABEL,
) -> pd.DataFrame:
    """Return a copy with tournament_year filled for the requested tournament."""
    output = matches.copy(deep=True)
    output["match_date"] = pd.to_datetime(output["match_date"], errors="raise")
    is_target_tournament = output["tournament"].map(_normalize_label).eq(
        _normalize_label(tournament_label)
    )
    output["tournament_year"] = pd.NA
    output.loc[is_target_tournament, "tournament_year"] = output.loc[
        is_target_tournament, "match_date"
    ].dt.year
    return output


def extract_tournament_years(
    matches: pd.DataFrame,
    tournament_label: str = DEFAULT_TOURNAMENT_LABEL,
    excluded_years: tuple[int, ...] = (2026,),
) -> list[int]:
    """Return sorted tournament years present in the data, excluding 2026 by default."""
    with_year = add_tournament_year(matches, tournament_label=tournament_label)
    years = pd.to_numeric(with_year["tournament_year"], errors="coerce").dropna()
    excluded = set(excluded_years)
    return sorted({int(year) for year in years if int(year) not in excluded})


def _tournament_mask(
    df: pd.DataFrame,
    tournament_year: int,
    tournament_label: str,
) -> pd.Series:
    """Return rows for one tournament year and label."""
    dates = pd.to_datetime(df["match_date"], errors="raise")
    is_label = df["tournament"].map(_normalize_label).eq(
        _normalize_label(tournament_label)
    )
    return is_label & dates.dt.year.eq(int(tournament_year))


def make_tournament_holdout_split(
    features_df: pd.DataFrame,
    tournament_year: int,
    tournament_label: str = DEFAULT_TOURNAMENT_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Return train/test rows for a tournament-specific holdout."""
    features = features_df.copy(deep=True)
    features["match_date"] = pd.to_datetime(features["match_date"], errors="raise")
    test_mask = _tournament_mask(features, tournament_year, tournament_label)
    test_df = features.loc[test_mask].sort_values(
        ["match_date", "match_id"],
        kind="mergesort",
    )
    if test_df.empty:
        raise ValueError(f"No {tournament_label} rows found for {tournament_year}.")

    tournament_start = pd.Timestamp(test_df["match_date"].min())
    train_df = features.loc[features["match_date"] < tournament_start].sort_values(
        ["match_date", "match_id"],
        kind="mergesort",
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        tournament_start,
    )


def _build_feature_tables(
    matches: pd.DataFrame,
    model_setups: tuple[TournamentModelSetup, ...],
    windows: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    """Build one leakage-safe feature table for each model setup."""
    return {
        setup.name: build_modeling_features(
            matches,
            windows=windows,
            include_elo=setup.include_elo,
            elo_k_factor=setup.elo_k_factor,
            elo_home_advantage=setup.elo_home_advantage,
        )
        for setup in model_setups
    }


def _evaluated_row(
    tournament_year: int,
    setup: TournamentModelSetup,
    tournament_start: pd.Timestamp,
    test_row_count: int,
    result: dict[str, object],
) -> dict[str, object]:
    """Flatten one evaluated tournament/model result."""
    metrics = result["calibrated_logistic_regression_metrics"]
    calibration = result["calibrated_logistic_calibration_summary"]
    assert isinstance(metrics, dict)
    assert isinstance(calibration, dict)

    return {
        "tournament_year": int(tournament_year),
        "model_setup": setup.name,
        "elo_k_factor": setup.elo_k_factor if setup.include_elo else pd.NA,
        "elo_home_advantage": (
            setup.elo_home_advantage if setup.include_elo else pd.NA
        ),
        "tournament_start_date": tournament_start.date().isoformat(),
        "test_match_count": int(test_row_count),
        "feature_count": int(result["feature_count"]),
        "skipped": False,
        "message": "evaluated",
        "log_loss": float(metrics["log_loss"]),
        "multiclass_brier_score": float(metrics["multiclass_brier_score"]),
        "accuracy": float(metrics["accuracy"]),
        "ece": float(calibration["expected_calibration_error"]),
    }


def _skipped_row(
    tournament_year: int,
    setup: TournamentModelSetup,
    message: str,
    test_match_count: int = 0,
) -> dict[str, object]:
    """Return a skipped tournament/model result row."""
    return {
        "tournament_year": int(tournament_year),
        "model_setup": setup.name,
        "elo_k_factor": setup.elo_k_factor if setup.include_elo else pd.NA,
        "elo_home_advantage": (
            setup.elo_home_advantage if setup.include_elo else pd.NA
        ),
        "tournament_start_date": pd.NA,
        "test_match_count": int(test_match_count),
        "feature_count": pd.NA,
        "skipped": True,
        "message": message,
        "log_loss": float("nan"),
        "multiclass_brier_score": float("nan"),
        "accuracy": float("nan"),
        "ece": float("nan"),
    }


def aggregate_tournament_backtest(summary_df: pd.DataFrame) -> dict[str, object]:
    """Aggregate tournament-holdout metrics by model setup."""
    if summary_df.empty:
        return {
            "evaluated_tournament_count": 0,
            "tournament_years_used": [],
            "model_metrics": {},
            "selected_beats_simple_elo_log_loss_count": 0,
            "selected_beats_no_elo_log_loss_count": 0,
            "messages": ["No tournament backtest rows were available."],
        }

    evaluated = summary_df.loc[~summary_df["skipped"]].copy()
    model_metrics: dict[str, dict[str, float | None]] = {}
    for model_setup, group in evaluated.groupby("model_setup", sort=True):
        metrics: dict[str, float | None] = {}
        for metric in ["log_loss", "multiclass_brier_score", "accuracy", "ece"]:
            numeric = pd.to_numeric(group[metric], errors="coerce").dropna()
            metrics[f"{metric}_mean"] = (
                float(numeric.mean()) if not numeric.empty else None
            )
            metrics[f"{metric}_std"] = (
                float(numeric.std(ddof=1)) if len(numeric) > 1 else None
            )
        model_metrics[str(model_setup)] = metrics

    pivot = evaluated.pivot_table(
        index="tournament_year",
        columns="model_setup",
        values="log_loss",
        aggfunc="first",
    )
    selected_vs_simple = (
        pivot["selected_elo_calibrated_logistic"]
        < pivot["simple_elo_calibrated_logistic"]
        if {
            "selected_elo_calibrated_logistic",
            "simple_elo_calibrated_logistic",
        }.issubset(pivot.columns)
        else pd.Series(dtype=bool)
    )
    selected_vs_no_elo = (
        pivot["selected_elo_calibrated_logistic"]
        < pivot["no_elo_calibrated_logistic"]
        if {
            "selected_elo_calibrated_logistic",
            "no_elo_calibrated_logistic",
        }.issubset(pivot.columns)
        else pd.Series(dtype=bool)
    )

    skipped_count = int(summary_df["skipped"].sum())
    messages = [
        f"{skipped_count} tournament/model row(s) skipped."
        if skipped_count
        else "All available tournament/model rows evaluated."
    ]

    return {
        "evaluated_tournament_count": int(evaluated["tournament_year"].nunique()),
        "tournament_years_used": [
            int(year) for year in sorted(evaluated["tournament_year"].unique())
        ],
        "model_metrics": model_metrics,
        "selected_beats_simple_elo_log_loss_count": int(selected_vs_simple.sum()),
        "selected_beats_no_elo_log_loss_count": int(selected_vs_no_elo.sum()),
        "messages": messages,
    }


def run_tournament_backtest(
    baseline_matches: pd.DataFrame,
    tournament_years: tuple[int, ...] = DEFAULT_WORLD_CUP_YEARS,
    tournament_label: str = DEFAULT_TOURNAMENT_LABEL,
    min_test_matches: int = DEFAULT_MIN_TEST_MATCHES,
    model_setups: tuple[TournamentModelSetup, ...] = MODEL_SETUPS,
    windows: tuple[int, ...] = (5, 10),
) -> dict[str, object]:
    """Evaluate selected baselines on held-out tournament-year test sets."""
    if 2026 in tournament_years:
        tournament_years = tuple(year for year in tournament_years if year != 2026)

    matches = baseline_matches.copy(deep=True)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="raise")
    feature_tables = _build_feature_tables(matches, model_setups, windows)

    rows: list[dict[str, object]] = []
    for tournament_year in tournament_years:
        test_match_count = int(
            _tournament_mask(matches, tournament_year, tournament_label).sum()
        )
        if test_match_count < min_test_matches:
            message = (
                f"Skipped {tournament_year}: only {test_match_count} "
                f"{tournament_label} matches found."
            )
            rows.extend(
                _skipped_row(tournament_year, setup, message, test_match_count)
                for setup in model_setups
            )
            continue

        for setup in model_setups:
            features = feature_tables[setup.name]
            try:
                train_df, test_df, tournament_start = make_tournament_holdout_split(
                    features,
                    tournament_year=tournament_year,
                    tournament_label=tournament_label,
                )
                result = train_and_evaluate_on_split(train_df, test_df)
                rows.append(
                    _evaluated_row(
                        tournament_year,
                        setup,
                        tournament_start,
                        len(test_df),
                        result,
                    )
                )
            except ValueError as error:
                rows.append(
                    _skipped_row(
                        tournament_year,
                        setup,
                        f"Skipped {tournament_year}/{setup.name}: {error}",
                        test_match_count,
                    )
                )

    summary = pd.DataFrame(rows)
    return {
        "summary": summary,
        "aggregate": aggregate_tournament_backtest(summary),
    }


def _print_aggregate_metrics(aggregate: dict[str, object]) -> None:
    """Print aggregate model metrics."""
    print("\nAggregate Metrics By Model Setup")
    print("================================")
    model_metrics = aggregate["model_metrics"]
    assert isinstance(model_metrics, dict)
    if not model_metrics:
        print("No evaluated model metrics.")
        return

    rows: list[dict[str, object]] = []
    for model_setup, metrics in model_metrics.items():
        assert isinstance(metrics, dict)
        row = {"model_setup": model_setup}
        for metric in ["log_loss", "multiclass_brier_score", "accuracy", "ece"]:
            row[metric] = format_metric_mean_std(
                metrics[f"{metric}_mean"],
                metrics[f"{metric}_std"],
            )
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def print_tournament_backtest_report(report: dict[str, object]) -> None:
    """Print a compact tournament-specific validation report."""
    summary = report["summary"]
    aggregate = report["aggregate"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(aggregate, dict)

    print("Tournament-Specific Baseline Backtest")
    print("=====================================")
    print(f"tournament label: {DEFAULT_TOURNAMENT_LABEL}")
    print(f"default years: {', '.join(str(year) for year in DEFAULT_WORLD_CUP_YEARS)}")
    print(f"evaluated tournament years: {aggregate['tournament_years_used']}")

    display_columns = [
        "tournament_year",
        "model_setup",
        "test_match_count",
        "log_loss",
        "multiclass_brier_score",
        "accuracy",
        "ece",
        "skipped",
        "message",
    ]
    print("\nPer-Tournament Metrics")
    print("======================")
    if summary.empty:
        print("No tournament rows were available.")
    else:
        print(summary[display_columns].to_string(index=False))

    _print_aggregate_metrics(aggregate)

    print("\nModel Comparisons")
    print("=================")
    print(
        "selected K=10/home=50 beats simple K=20/home=0 on log loss: "
        f"{aggregate['selected_beats_simple_elo_log_loss_count']}/"
        f"{aggregate['evaluated_tournament_count']}"
    )
    print(
        "selected K=10/home=50 beats no-Elo on log loss: "
        f"{aggregate['selected_beats_no_elo_log_loss_count']}/"
        f"{aggregate['evaluated_tournament_count']}"
    )

    print("\nMessages")
    print("========")
    for message in aggregate["messages"]:
        print(f"- {message}")

    print("\nNo model artifacts or processed feature files were written.")


def main() -> int:
    """Run tournament-specific baseline validation."""
    try:
        baseline_matches = load_baseline_training_matches()
    except FileNotFoundError as error:
        print(f"Missing historical results data: {error}")
        return 1

    report = run_tournament_backtest(baseline_matches)
    print_tournament_backtest_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
