# Data Sources

This project uses manually maintained local data files for v1. It does not scrape websites, call APIs, or download Kaggle data automatically.

## Historical Match Source

The v1 historical training source is a manually downloaded international football results CSV placed at:

```text
data/raw/results.csv
```

Expected columns:

```text
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
```

This dataset is used to train the first team-level baseline model, after cleaning and leakage-safe feature engineering are implemented.

## Current World Cup Source

Current 2026 World Cup fixtures and results are maintained separately under:

```text
data/tournament/fixtures_2026.csv
data/tournament/results_2026.csv
```

For now, these files should be updated manually from FIFA official schedule and results pages.

## Why Raw Data Is Not Committed

Raw datasets are not committed because they may be large, change over time, have licensing restrictions, or be easier to replace from their original source. The repository should track code, docs, schemas, and lightweight placeholder files, not raw datasets.

## Training Data vs Tournament State

Historical training data and current tournament state serve different purposes.

Historical training data:

- Comes from `data/raw/results.csv`.
- Feeds the baseline training pipeline.
- Must be filtered to matches with `date <= 2026-06-10` for the first baseline.

Current tournament state:

- Comes from `data/tournament/fixtures_2026.csv` and `data/tournament/results_2026.csv`.
- Updates current group standings, bracket state, prediction audit, and live simulation state.
- Does not enter first-baseline model training.

## Leakage Boundary

Completed 2026 World Cup matches are allowed for:

- Current group standings.
- Tournament state.
- Evaluation and prediction audit.
- Live simulation state.

Completed 2026 World Cup matches are not allowed for training the first baseline model. The first baseline must train only on historical matches with `date <= 2026-06-10`.
