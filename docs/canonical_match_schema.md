# Canonical Match Schema

The project converts raw historical results into a canonical match schema before feature engineering or modeling.

## Why `team_a` and `team_b`

The model predicts outcomes from a consistent orientation. Instead of mixing home/away naming throughout the pipeline, cleaned match records use:

- `team_a`
- `team_b`

For the v1 raw historical dataset, `team_a` corresponds to `home_team` and `team_b` corresponds to `away_team`. For neutral-site records, `team_a` is still the listed first team from the raw file.

## Result Labels

Result labels are always from `team_a`'s perspective:

- `team_a_win`
- `draw`
- `team_b_win`

This keeps target labels stable for both non-neutral and neutral matches.

## Baseline Training Cutoff

The first baseline model trains only on historical matches with:

```text
match_date <= 2026-06-10
```

Current World Cup matches after `2026-06-10` are excluded from first-baseline training. They may be used for standings, tournament state, prediction audit, and live simulation state.

## Leakage Rule

Future feature engineering must only use information available before the match being predicted. Rolling features, player features, tournament-state features, and audit records should all preserve clear date or timestamp cutoffs.

## Required Columns

The canonical cleaned output columns are:

```text
match_id
match_date
team_a
team_b
team_a_goals
team_b_goals
result
tournament
city
country
neutral
is_neutral
goal_diff_team_a
total_goals
training_cutoff_date
is_baseline_train_eligible
```
