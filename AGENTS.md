# AGENTS.md

Durable instructions for future Codex work on the World Cup 2026 Forecasting Dashboard.

## Project Mission

Build a portfolio-grade, end-to-end forecasting dashboard for international football matches:

historical international football results -> cleaned data -> leakage-safe features -> calibrated match outcome model -> Streamlit match predictor.

The first model predicts a 3-class result:

- `team_a_win`
- `draw`
- `team_b_win`

## Modeling Rules

- Optimize and report probabilistic quality first: log loss, Brier score, and calibration.
- Treat accuracy as secondary.
- Prevent target leakage. Rolling or historical features must only use matches played before the prediction date.
- Keep train/test splits time-aware.
- Do not use external APIs until the project explicitly adds that milestone.
- Do not commit raw datasets.
- Keep model outputs interpretable enough for a portfolio reviewer to understand.

## Data Rules

- Store manually downloaded source files in `data/raw/`.
- Store cleaned or derived files in `data/processed/`.
- Do not commit raw data files, large binary artifacts, local notebooks with embedded outputs, or secrets.
- Prefer reproducible scripts over manual notebook-only transformations.

## Code Style

- Keep Python typed, readable, and boring in the best way.
- Prefer small pure functions for cleaning, features, evaluation, and prediction helpers.
- Add concise comments only where they explain non-obvious choices.
- Use `pathlib.Path` for filesystem paths.
- Avoid global mutable state.
- Make leakage assumptions explicit in function names, docstrings, or comments.

## Testing Expectations

- Add or update tests for any behavior that affects labels, features, splits, metrics, or predictions.
- Use small synthetic dataframes in tests when possible.
- Before handing off substantive changes, run:

```bash
pytest
```

## App Expectations

- The Streamlit app should be usable as a match predictor, not a static report.
- Show predicted class probabilities and calibration-aware caveats.
- Keep the first MVP simple: select Team A, Team B, neutral-site flag if available, and show probabilities.

## Collaboration Notes

- Preserve user changes. Never revert unrelated work.
- Keep changes scoped to the requested milestone.
- When adding dependencies, update `requirements.txt` and explain why.
