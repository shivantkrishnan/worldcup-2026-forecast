# Live Forecasting Design

The project supports both pre-tournament and live forecasting for the 2026 World Cup. The first baseline model must remain a clean ex-ante model: it trains only on data available before the tournament began.

Default training cutoff date: `2026-06-10`

Completed 2026 World Cup matches may update tournament state, standings, evaluation, and simulations. They must not be used to train the first baseline model.

## Forecast Modes

### A. `pre_tournament_forecast`

Purpose: create the original ex-ante tournament forecast from the starting state.

Data rules:

- Uses historical international football data before `2026-06-10`.
- Does not use completed 2026 World Cup matches.
- Uses fixtures and tournament structure as known before kickoff.

Outputs:

- Match-level probabilities for scheduled matches.
- Full tournament simulations from the original starting state.
- Baseline artifacts tagged with `training_cutoff_date = 2026-06-10`.

### B. `live_forecast`

Purpose: update forecasts as the tournament progresses without retraining the first baseline model on tournament results.

Data rules:

- Uses the pre-tournament trained model.
- Uses completed 2026 World Cup results to update standings and tournament state.
- Predicts remaining matches from the current state.
- Does not add completed 2026 World Cup matches to the first baseline model training set.

Outputs:

- Current group standings.
- Current bracket or stage state when available.
- Remaining-match probabilities.
- Live simulations from the current tournament state.

### C. `prediction_audit`

Purpose: evaluate forecast quality against completed match outcomes.

Data rules:

- Compares predicted probabilities to actual completed match outcomes.
- Distinguishes true pre-match logged predictions from backfilled ex-ante predictions.
- Treats backfilled rows as useful for analysis, but not equivalent to timestamped pre-match predictions.

Metrics:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics by probability or confidence bins.

Prediction logging rules:

- Predictions made going forward must be timestamped before kickoff.
- Matches already completed before prediction logging begins should be marked `is_backfilled = true`.
- Prediction rows should include `feature_cutoff_timestamp` and `training_cutoff_date` so audits can verify what information was available.

## Leakage Boundary

For the first baseline model, the training data boundary is `2026-06-10`.

Allowed after that boundary:

- Updating standings from completed World Cup matches.
- Updating tournament state.
- Evaluating logged predictions.
- Running live simulations from the current state.

Not allowed for the first baseline:

- Retraining on completed 2026 World Cup matches.
- Building training features that include completed World Cup results.
- Replacing a pre-match probability with a post-match probability without marking it as backfilled or audited.
