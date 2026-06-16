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
- Drop or quarantine incomplete records.
- Create 3-class labels:
  - `team_a_win`
  - `draw`
  - `team_b_win`

### 4. Leakage-Safe Features

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

### Product

- Group-stage simulator.
- Knockout bracket simulator.
- Team profile pages.
- Interactive calibration and feature inspection.
- Live forecast view based on current standings.
- Prediction audit dashboard with backfilled-row filters.

### Engineering

- Add data validation checks.
- Add reproducible pipeline commands.
- Add CI for tests and linting.
- Add model cards and experiment tracking.
