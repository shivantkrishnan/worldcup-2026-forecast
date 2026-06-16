# World Cup 2026 Forecasting Dashboard

Portfolio-grade forecasting dashboard for international football matches, targeting the 2026 FIFA World Cup.

## Objective

Build an end-to-end machine learning pipeline:

1. Load historical international football results.
2. Clean and validate match records.
3. Build leakage-safe rolling features using only matches before each prediction date.
4. Train a calibrated 3-class match outcome model using a default training cutoff date of `2026-06-10`:
   - Team A win
   - Draw
   - Team B win
5. Support both pre-tournament and live forecasting.
6. Evaluate predictions with probabilistic metrics and audit logs.
7. Serve predictions in a Streamlit match predictor.

Primary evaluation metrics:

- Log loss
- Brier score
- Calibration curves

## Project Structure

```text
.
├── app/
├── data/
│   ├── player/
│   ├── predictions/
│   ├── tournament/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   │   └── player_features/
│   ├── models/
│   ├── player/
│   └── utils/
└── tests/
```

## Setup

Create and activate a virtual environment:

```bash
conda deactivate
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run the Streamlit app:

```bash
python -m streamlit run app/streamlit_app.py
```

Run the historical data audit:

```bash
python scripts/audit_historical_results.py
```

The audit prints descriptive diagnostics only and does not write processed data.

Run the baseline model training script:

```bash
python scripts/train_baseline_model.py
```

The training script prints class-prior, logistic regression, and calibrated logistic regression metrics plus calibration diagnostics. It does not write model artifacts by default.

Methodology and project decisions are tracked continuously in:

```text
docs/methodology.md
docs/decision_log.md
docs/methodology_checklist.md
```

These notes are intended to become the methodology section of the dashboard or website.

## Environment Troubleshooting

If your prompt shows both `(.venv)` and `(base)`, deactivate Conda before rebuilding the virtual environment:

```bash
conda deactivate
```

Use module commands from inside the project venv so Python, pip, and Streamlit all come from the same environment:

```bash
which python
python -m pip --version
python -m streamlit run app/streamlit_app.py
```

## Data Policy

Raw datasets belong in `data/raw/` and must not be committed. Processed datasets belong in `data/processed/` and should be reproducible from scripts whenever possible.

Historical results are cleaned into a canonical `team_a`/`team_b` match schema before feature engineering. The canonical schema is documented in `docs/canonical_match_schema.md`.

Tournament fixtures, current tournament state, and prediction logs have their own proposed schemas in `docs/data_schemas.md`.

Player-level data is a planned modular extension after the team-level baseline. Its proposed strategy and schemas live in `docs/player_data_strategy.md`, `docs/player_form_index.md`, and `docs/player_data_schemas.md`.

External APIs are intentionally out of scope for the first version.

## Data Setup

Historical training data is expected at:

```text
data/raw/results.csv
```

Download the international football results CSV manually and place it at that path. The expected columns are:

```text
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
```

Current 2026 World Cup tournament state is maintained separately:

```text
data/tournament/fixtures_2026.csv
data/tournament/results_2026.csv
```

Use manually maintained, FIFA-sourced fixture and result files for current tournament state. Completed 2026 World Cup matches may be used for standings, evaluation, prediction audit, and live simulation state, but they must not be used to train the first baseline model.

## Forecasting Modes

The project distinguishes between three forecast modes:

- `pre_tournament_forecast`: trains only on historical data available before `2026-06-10` and produces ex-ante predictions from the original tournament state.
- `live_forecast`: uses the pre-tournament trained model while allowing completed 2026 World Cup matches to update standings, bracket state, and remaining-match simulations.
- `prediction_audit`: compares predicted probabilities against actual completed outcomes using log loss, Brier score, and calibration diagnostics.

Completed 2026 World Cup matches are allowed for standings, tournament state, evaluation, and live simulation state. They must not be used to train the first baseline model.

Predictions for matches already completed before logging begins should be marked as backfilled. Predictions made going forward should be timestamped before kickoff.

## Validation Strategy

Model evaluation should use time-aware splits by default. The first holdout design trains on matches before `2022-01-01` and tests on baseline-eligible matches from `2022-01-01` through `2026-06-10`.

Random splits are not the primary evaluation design because they can leak context across football eras.

## MVP Roadmap

1. Define canonical schema for historical international match results.
2. Implement cleaning and outcome labeling into the canonical `team_a`/`team_b` schema.
3. Add canonical data quality checks, duplicate resolution, and time-aware split utilities.
4. Build leakage-safe rolling team features.
5. Train a baseline multinomial logistic regression model with `training_cutoff_date = 2026-06-10`.
6. Define fixtures, results, and prediction-log schemas for live forecasting.
7. Evaluate log loss, Brier score, and calibration.
8. Add prediction audit output that distinguishes logged pre-match predictions from backfilled predictions.
9. Save model artifacts and prediction metadata.
10. Build a simple Streamlit match predictor.

## Extension Ideas

- Add Elo-style ratings.
- Add tournament-stage and confederation features.
- Compare multiple calibrated models.
- Add model cards and dataset documentation.
- Add simulation tools for group and knockout stages.
- Add player-level live form features after the team-level baseline is evaluated.
