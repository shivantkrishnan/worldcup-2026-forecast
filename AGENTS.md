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
- Compute team-form features from a long team-match panel, using shift/lag logic before rolling or expanding calculations so current-match outcomes are excluded.
- Compute rating-style features as pre-match state variables. Ratings update only after the match or date block being predicted; if only date-level timestamps are available, same-date results must not affect features for other matches on that date.
- Use time-aware train/test splits by default. Random splits are not primary because football team strength changes over time and random validation can leak context across eras.
- Do not use external APIs until the project explicitly adds that milestone.
- Do not commit raw datasets.
- Keep model outputs interpretable enough for a portfolio reviewer to understand.

## Methodology Rules

- Maintain detailed methodology documentation throughout development.
- Update `docs/methodology.md`, `docs/decision_log.md`, and related docs whenever making methodological, statistical, ML, econometric, data-source, validation, or feature-design decisions.
- Write methodology notes rigorously enough to become the dashboard or website methodology section.
- Emphasize statistical validity, leakage prevention, calibration, model uncertainty, economic intuition, and the rationale for each modeling choice.

## Data Rules

- The v1 historical training source is a manually downloaded international football results CSV at `data/raw/results.csv`.
- The canonical match schema uses `team_a`/`team_b`; for raw historical results, `team_a` is `home_team` and `team_b` is `away_team`.
- Canonical result labels are always from `team_a`'s perspective: `team_a_win`, `draw`, `team_b_win`.
- Canonical cleaned matches represent completed matches with scores; scoreless fixture rows are excluded and handled through tournament-state files.
- Current 2026 World Cup fixtures and results are maintained separately under `data/tournament/`.
- The first baseline trains only on historical matches with `date <= 2026-06-10`.
- Completed 2026 World Cup matches may update standings, tournament state, evaluation, and live simulations, but not first-baseline training.
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
