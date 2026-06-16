# Methodology

## Project Objective

The project builds a portfolio-grade World Cup 2026 forecasting dashboard for international football. The first baseline is a team-level probabilistic model that predicts three match outcomes from `team_a`'s perspective:

- `team_a_win`
- `draw`
- `team_b_win`

The system should support pre-tournament forecasts, live forecasts, and prediction audits while preserving leakage-safe data boundaries.

## Data Sources and Separation of Training/Current Tournament State

The v1 historical training source is a manually downloaded international football results CSV placed at:

```text
data/raw/results.csv
```

Raw data is not committed. Current 2026 World Cup fixtures and results are maintained separately under:

```text
data/tournament/
```

Historical match data is used for baseline training. Current tournament data is used for standings, tournament state, prediction audit, and live simulations, but not for first-baseline training.

## Canonical Match Schema

Raw historical rows use `home_team` and `away_team`. Cleaned canonical rows use `team_a` and `team_b`.

For the v1 raw dataset:

- `team_a` maps to `home_team`.
- `team_b` maps to `away_team`.
- Neutral-site matches still use the raw file's listed first team as `team_a`.
- Result labels are always from `team_a`'s perspective.

Canonical cleaned matches represent completed matches with scores. Scoreless fixture rows are excluded from canonical completed-match output and handled through tournament-state files.

## Data Quality and Duplicate-Resolution Policy

Canonical data quality checks run in memory before feature engineering. They check for:

- Duplicate `match_id` values.
- Negative scores.
- Null required values.
- Invalid result labels.
- Non-boolean neutral flags.
- Baseline cutoff inconsistencies.

Duplicate canonical `match_id` groups are handled explicitly:

- If duplicate rows agree on outcome fields, keep one row and quarantine extras as metadata duplicates.
- If duplicate rows disagree on score or result fields, quarantine the full duplicate group by default.
- Conflicting duplicates are not silently dropped.

Quarantine is in-memory for now.

## Training Cutoff Logic

The first baseline trains only on matches with:

```text
match_date <= 2026-06-10
```

Matches after that date may be useful for live tournament state and prediction audit, but they are excluded from first-baseline model training.

## Validation Philosophy

Validation should mimic real forecasting. The default approach is time-aware validation, not random splitting. Historical matches before a cutoff are used to predict future matches after that cutoff.

This reduces the risk of accidentally evaluating on a setting where information from a later football era influences training for an earlier one.

## Feature-Engineering Principles

Features must be leakage-safe. Rolling features must use only matches before the prediction date. Feature tables should preserve explicit date or timestamp cutoffs.

The first team-form feature layer transforms canonical matches into a long team-match panel, shifts team outcomes by one match, and then computes rolling or expanding features from prior matches only.

Initial features should be simple and interpretable before adding complexity. Each new feature group should be justified by football intuition and tested by whether it improves probabilistic forecast quality.

## Model Evaluation Metrics

Primary metrics:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics.

Accuracy is secondary because the dashboard forecasts probabilities, not just labels.

## Calibration and Uncertainty

Forecasts should be evaluated as probabilities. Calibration matters because users need to know whether a predicted 60 percent chance behaves like a real 60 percent chance over comparable matches.

The project should surface uncertainty and avoid overclaiming precision, especially for live forecasts, player availability, and tournament simulations.

## Player-Level Extension Methodology

Player-level data is a later modular extension after the team-level baseline is evaluated. Player form features should be position-normalized, recency-weighted, and shrunk for small samples.

Recent friendlies are stronger evidence of selection and availability than of true player quality. Player identity matching is a major risk and should be handled through a registry.

Player features should only remain if they improve log loss, Brier score, and calibration without introducing leakage.

## Market/Odds Extension Methodology

Market or odds data may be considered later as a benchmark or feature source. It should be documented carefully because odds combine public information, bookmaker margin, market liquidity, and implied probabilities.

Any market extension should separate:

- Pure model forecasts.
- Market-implied baselines.
- Combined or adjusted forecasts.

The project should be clear about whether odds are used for benchmarking, calibration comparison, or as model inputs.

## Live Forecast vs Pre-Tournament Forecast

`pre_tournament_forecast` uses only historical data before the tournament cutoff and forecasts from the original tournament state.

`live_forecast` uses the pre-tournament trained model while updating standings and tournament state from completed World Cup matches.

`prediction_audit` compares predicted probabilities against completed match outcomes and distinguishes true pre-match logged predictions from backfilled ex-ante predictions.

## Limitations and Future Improvements

Current limitations:

- Only the first leakage-safe team-form feature layer has been implemented.
- No model has been trained yet.
- Raw data is manually downloaded and locally maintained.
- Duplicate quarantine is in-memory.
- Current tournament files are manually maintained.

Future improvements:

- Elo-style and opponent-adjusted team strength features.
- Backtesting over prior World Cups and major tournaments.
- Calibrated model comparisons.
- Player-level live form extension.
- Market-implied probability benchmarks.
- Reproducible processed data artifacts once the policy is defined.
