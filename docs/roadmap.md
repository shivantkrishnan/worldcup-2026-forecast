# Project Roadmap

Concise roadmap and extension backlog for the World Cup 2026 Forecasting Dashboard.

## Current Completed Milestones

- Repository foundation.
- Local historical data pipeline.
- Canonical match schema.
- Duplicate resolution and data audit.
- Leakage-safe rolling team-form features.
- Feature readiness audit.
- Baseline logistic model.
- Calibration diagnostics.
- Rolling-origin backtesting.
- Baseline model card and model selection report.
- Leakage-safe Elo features.
- Elo feature-set evaluation and updated selected baseline.
- Elo K-factor and home/neutral-site variant evaluation.

## Current Selected Baseline

- Model: sigmoid-calibrated logistic regression.
- Features: rolling team-form plus pre-match Elo with K=10 and a 50-point non-neutral home adjustment.
- Primary selection metric: rolling-origin mean log loss.
- Caveat: ECE/calibration-bin alignment did not improve on average.

## Near-Term Core Modeling Roadmap

Continue refining Elo variants:

- Tournament-specific validation of the selected K/home setup.
- Margin-of-victory adjustment.
- Tournament-weighted Elo.

After each variant:

- Rerun feature audit.
- Rerun single-holdout evaluation.
- Rerun rolling-origin backtest.
- Update the model selection report.
- Compare whether the selected baseline changes.

## Tournament Forecasting Roadmap

- Implement manually maintained 2026 fixtures/results ingestion.
- Generate match-level predictions for known fixtures.
- Build Monte Carlo group-stage and knockout simulation.
- Output group qualification, knockout advancement, finalist, and champion probabilities.
- Distinguish pre-tournament, live, and backfilled forecasts.

## Product/UI Roadmap

- Streamlit match predictor.
- Display favorite plus full 3-class probabilities.
- Show model caveats and selected baseline summary.
- Show tournament simulation outputs.
- Show prediction audit after results are known.

## Supplemental / Later Extensions

- Player-level squad and form features.
- Market/odds benchmark.
- Live in-match prediction model using score, minute, red cards, shots, xG, possession, pass completion, corners, and other live stats.
- Bootstrapped uncertainty intervals.
- Tournament-specific backtesting over prior World Cups or major tournaments.
- Richer event-data features if reliable data becomes available.

## Not Yet in Scope

- Betting advice.
- Paid data integration.
- Live API ingestion.
- Final production deployment.
- Model artifacts written by default.
