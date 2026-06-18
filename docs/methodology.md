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

The first tournament-state ingestion layer expects a manually maintained `data/tournament/fixtures_2026.csv`. It is validated separately from the historical training data and can be used to generate fixture predictions without retraining on 2026 World Cup results.

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

After the first 2022-2026 holdout, baseline models are also evaluated with rolling-origin backtests. Each split trains on past data and tests on a future window, then the origin moves forward. This checks whether model improvements are stable across time rather than tied to one validation period.

## Feature-Engineering Principles

Features must be leakage-safe. Rolling features must use only matches before the prediction date. Feature tables should preserve explicit date or timestamp cutoffs.

The first team-form feature layer transforms canonical matches into a long team-match panel, shifts team outcomes by one match, and then computes rolling or expanding features from prior matches only.

The second team-level feature family is Elo-style team strength. Elo ratings are opponent-adjusted pre-match state variables. Ratings are emitted before updating with the current result, and when only date-level timestamps are available, updates are applied after the full date block so same-date results cannot leak into same-date features.

Elo variants may adjust expected score with a temporary home-advantage term for non-neutral matches. The home term affects only the match-level expected score and update calculation; it does not permanently inflate the team's underlying rating. Neutral-site matches receive no home bonus unless a separate host or venue model is introduced later.

Home advantage is considered because historical non-neutral international matches can include crowd, travel, familiarity, venue, and local-context effects. If that context is ignored, Elo can over-credit a home team's underlying strength for wins that partly reflect location. The current adjustment is therefore a historical-learning device, not a claim that generic home advantage should drive 2026 World Cup forecasts. Most 2026 fixtures should be treated as neutral by default, and USA, Canada, or Mexico host effects should later be modeled as explicit tournament-state or venue features.

Initial features should be simple and interpretable before adding complexity. Each new feature group should be justified by football intuition and tested by whether it improves probabilistic forecast quality.

Feature readiness is audited before baseline training. The audit uses the same time-aware split philosophy as model validation and reports target balance, feature missingness, high-missingness features, fully missing features, and excluded non-numeric feature candidates.

Missing rolling-history values are expected for early team histories and full-window rolling features. The later model pipeline must handle them explicitly through documented imputation, filtering, or model choices.

## Model Evaluation Metrics

Primary metrics:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics.

Accuracy is secondary because the dashboard forecasts probabilities, not just labels.

The first baseline compares a class-prior probability model against a multinomial logistic regression model. Missing rolling-history values are handled inside the scikit-learn pipeline with train-fitted median imputation plus missingness indicators, followed by scaling and logistic regression.

The baseline evaluation also compares an internally calibrated logistic regression variant. Calibration is fitted only on training data through internal cross-validation, never on the test set.

Rolling-origin backtests refit preprocessing, imputation, scaling, logistic regression, and calibration independently inside each training window.

Tournament-specific backtesting is now a separate validation layer. It holds out prior FIFA World Cups, trains only on matches before each tournament starts, and tests only on that tournament's matches. This checks whether the selected broad all-match baseline remains credible in the match environment the dashboard ultimately forecasts.

The first tournament-specific validation over the 2002 through 2022 FIFA World Cups supports the selected K=10/home=50 Elo setup by mean log loss, but it also reinforces the calibration caveat because ECE remains worse than the no-Elo setup.

Based on the completed single-holdout and rolling-origin results, sigmoid-calibrated logistic regression is the current selected model family. The selected feature set now includes leakage-safe rolling team-form plus pre-match Elo features.

The first forecast output layer trains this selected baseline in memory and produces scheduled-fixture probabilities without requiring fixture scores or outcomes. Fixture features use only completed matches strictly before the fixture date; with date-only timestamps, same-date completed matches are excluded unless a future timestamped system proves they occurred earlier.

Fixture prediction generation can print predictions or explicitly write `data/tournament/fixture_predictions_2026.csv` when requested. Written prediction rows include model/training metadata so later prediction audit can distinguish model version, training cutoff, and feature cutoff.

Forecast outputs distinguish `pre_tournament`, `backfilled_ex_ante`, and `live` modes. `feature_cutoff_date` is mandatory metadata because it defines the information set used to build fixture features. Backfilled ex-ante predictions can reconstruct pre-tournament probabilities after the fact, but they are not the same as true timestamped live predictions.

The first Monte Carlo simulation layer consumes fixture-level probabilities and simulates group-stage outcomes only. When completed 2026 results are present in `results_2026.csv`, those matches are fixed in every run and only remaining matches are sampled. Completed 2026 results can update tournament state and live simulation state, but they cannot train the first baseline model.

Completed 2026 results should come from official or clearly source-attributed sources. Unverified results are omitted rather than inferred, because a wrong fixed result is more damaging to live simulation state than a missing one.

The simulator samples `team_a_win`, `draw`, or `team_b_win`, awards points, and estimates group-winner/top-two/advancement probabilities. Because scorelines are not modeled yet, group ranking currently uses points, wins, and a seeded random tie-break placeholder rather than official goal-difference tie-break rules.

Selection is based primarily on rolling-origin log loss stability. Adding simple Elo improved the selected model's rolling-origin mean log loss from `1.201547` to `1.197724`. The first K/home variant grid selected K=10 with a 50-point non-neutral home adjustment, improving rolling-origin mean log loss further to `1.186855`.

K=10 is interpreted as an empirically selected smoothing parameter for sparse, noisy international football results, with rolling form features carrying more of the short-run momentum signal.

This selection is provisional. Expected calibration error does not consistently improve, and the selected K/home variant worsens mean ECE versus simple Elo, so calibration caveats remain part of any dashboard or report language.

## Calibration and Uncertainty

Forecasts should be evaluated as probabilities. Calibration matters because users need to know whether a predicted 60 percent chance behaves like a real 60 percent chance over comparable matches.

Calibration diagnostics include expected calibration error, confidence-bin calibration, and classwise calibration tables. These diagnostics are reported alongside log loss and Brier score because accuracy can improve even when probability quality worsens.

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

Scheduled fixture predictions output the full 3-class probability vector first. Favorite labels and confidence labels are display aids layered on top of those probabilities, not replacements for calibrated probability reporting.

Group-stage simulation uses those full probabilities directly. Draw probability is especially important because draws affect both teams' points and cannot be recovered from a favorite-only forecast.

## Limitations and Future Improvements

Current limitations:

- Only the first leakage-safe team-form feature layer has been implemented.
- A simple leakage-safe Elo feature family has been implemented and selected into the baseline feature set by rolling-origin mean log loss.
- Only the first simple baseline model has been implemented.
- Missing feature values are handled by median imputation plus missingness indicators in the baseline pipeline.
- Calibration diagnostics and the first sigmoid-calibrated logistic comparison are available.
- Rolling-origin backtesting is available for baseline model stability checks.
- The current selected baseline is sigmoid-calibrated logistic regression over rolling team-form plus pre-match Elo features with K=10 and a 50-point non-neutral home adjustment.
- Raw data is manually downloaded and locally maintained.
- Duplicate quarantine is in-memory.
- Current tournament files are manually maintained.

Future improvements:

- Improve Elo-style and opponent-adjusted team strength features.
- Backtesting over prior World Cups and major tournaments.
- Calibrated model comparisons.
- Hyperparameter tuning and stronger model families.
- Player-level live form extension.
- Market-implied probability benchmarks.
- Reproducible processed data artifacts once the policy is defined.
