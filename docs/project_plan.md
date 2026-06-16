# Project Plan

## MVP Milestones

### 1. Repository Foundation

- Create project structure.
- Add durable project instructions.
- Add dependency list.
- Add starter tests.

### 2. Data Ingestion

- Define expected raw results schema.
- Load local CSV or Parquet files from `data/raw/`.
- Validate required columns.
- Keep raw data out of Git.
- Define tournament fixture, result, and prediction-log schemas.

### 3. Cleaning and Labels

- Standardize dates, team names, and score columns.
- Convert raw `home_team` and `away_team` fields into the canonical `team_a` and `team_b` schema.
- Drop or quarantine incomplete records.
- Create 3-class labels:
  - `team_a_win`
  - `draw`
  - `team_b_win`
- Add deterministic `match_id`, goal summaries, and baseline cutoff eligibility.

### 4. Leakage-Safe Features

- Use time-aware train/test splits by default before modeling.
- Sort matches by date.
- Build rolling team features from prior matches only.
- Include simple features first:
  - rolling goals for
  - rolling goals against
  - rolling win rate
  - rolling draw rate
  - match count before date

### 5. Baseline Model

- Use a time-aware train/test split.
- Default holdout: train before `2022-01-01`, test on `2022-01-01` through the `2026-06-10` baseline cutoff.
- Train a multinomial logistic regression baseline.
- Use `2026-06-10` as the default training cutoff date.
- Do not train the first baseline model on completed 2026 World Cup matches.
- Calibrate probabilities when appropriate.
- Save model artifacts only after the artifact policy is defined.

### 6. Evaluation

- Report log loss.
- Report multiclass Brier score.
- Plot or tabulate calibration by confidence bins.
- Compare to simple baselines.
- Add prediction audit output for completed matches.
- Distinguish timestamped pre-match predictions from backfilled ex-ante predictions.

### 7. Live Forecasting State

- Use completed 2026 World Cup matches for current standings and tournament state.
- Use completed 2026 World Cup matches for evaluation and prediction audit.
- Use the pre-tournament trained model for remaining-match live forecasts.
- Keep live tournament state separate from baseline training data.

### 8. Streamlit Predictor

- Load trained model and feature metadata.
- Let users choose Team A and Team B.
- Display probabilities for Team A win, draw, and Team B win.
- Show model caveats and evaluation summary.
- Show whether predictions are pre-tournament, live, or audit records.

## Extensions

### Modeling

- Elo or Glicko-style team strength features.
- Hierarchical features by tournament and confederation.
- Gradient boosting comparison.
- Probability calibration comparisons.
- Player-level live form extension after the team-level baseline is evaluated.
- Player identity registry and roster-aware feature aggregation.
- Position-specific player feature normalization with small-sample shrinkage.

### Product

- Group-stage simulator.
- Knockout bracket simulator.
- Team profile pages.
- Interactive calibration and feature inspection.
- Live forecast view based on current standings.
- Prediction audit dashboard with backfilled-row filters.
- Player availability and squad form view once player data is validated.

### Engineering

- Add data validation checks.
- Add reproducible pipeline commands.
- Add CI for tests and linting.
- Add model cards and experiment tracking.
- Add player schema validators before any player-data ingestion.
- Maintain methodology and decision docs continuously so they can become the dashboard methodology page.
