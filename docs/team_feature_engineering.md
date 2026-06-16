# Team Feature Engineering

The first feature layer is a leakage-safe team-form feature set built from canonical completed matches. It does not train a model and does not write processed feature data to disk.

## Long Team-Match Panel

Canonical matches have one row per match with `team_a` and `team_b`. Team-form features are easier and safer to compute after converting this table into a long team-match panel with two rows per match:

- One row from `team_a`'s perspective.
- One row from `team_b`'s perspective.

Each team row contains goals for, goals against, goal difference, points, win/draw/loss indicators, opponent, venue flags, tournament, and result from that team's perspective.

This representation lets rolling features be computed consistently for each team regardless of whether the team appeared as `team_a` or `team_b`.

## Leakage-Safe Rolling Form

Rolling form must be shifted before rolling or expanding calculations. For a match on date `t`, features may use only matches before `t`.

The implementation uses lagged team outcomes before computing rolling and expanding means. This prevents the current match's goals, points, or result from appearing in the features used to predict that same match.

## Team-Level vs Match-Level Differential Features

Team-level features describe one team's prior form, such as its rolling points per match.

Match-level differential features compare the two teams in a fixture:

```text
feature_diff = team_a_feature - team_b_feature
```

Differentials are useful because the model predicts the match outcome from `team_a`'s perspective. A positive differential means `team_a` has a stronger recent-history value than `team_b` for that feature.

## Recent Form Is Useful but Noisy

Recent form windows can capture meaningful changes in team performance, fitness, or tactical stability. They are also noisy because international teams play uneven schedules, opponents vary in strength, and small samples can overreact to a few matches.

The first feature set starts with interpretable rolling and expanding summaries. More complex features should be added only after they improve log loss, Brier score, and calibration.

## Feature Readiness Audit

Before baseline model training, the match-level feature table is audited in memory. The audit reports:

- Numeric candidate feature columns.
- Train/test row counts using the default time-aware split.
- Target distributions in train and test windows.
- Missingness by feature overall, in train, and in test.
- Fully missing and high-missingness features.
- Non-numeric candidate columns excluded from model features.

Missing rolling-history values are expected for early-team-history rows and rolling windows. The model pipeline must handle these values explicitly later rather than relying on silent defaults.

## Economic and Statistical Intuition

Forecasts should mimic information available before the match. A bettor, analyst, coach, or dashboard user cannot know the current match's result before kickoff. The feature pipeline therefore treats each match as a forecast made with prior information only.

## First Feature Set

| Feature | Level | Description |
| --- | --- | --- |
| `matches_played_before` | Team | Number of prior matches available for the team. |
| `days_since_last_match` | Team | Days since the team's previous match. |
| `rolling_points_per_match_5` | Team | Average points from the team's prior 5 matches. |
| `rolling_points_per_match_10` | Team | Average points from the team's prior 10 matches. |
| `rolling_goals_for_avg_5` | Team | Average goals scored in the team's prior 5 matches. |
| `rolling_goals_against_avg_5` | Team | Average goals conceded in the team's prior 5 matches. |
| `rolling_goal_diff_avg_5` | Team | Average goal difference in the team's prior 5 matches. |
| `rolling_win_rate_5` | Team | Win rate across the team's prior 5 matches. |
| `rolling_draw_rate_5` | Team | Draw rate across the team's prior 5 matches. |
| `rolling_loss_rate_5` | Team | Loss rate across the team's prior 5 matches. |
| `expanding_points_per_match` | Team | Average points across all prior team matches. |
| `expanding_goal_diff_avg` | Team | Average goal difference across all prior team matches. |
| `expanding_win_rate` | Team | Win rate across all prior team matches. |

The implementation can calculate the rolling averages and rates for each requested rolling window. The default windows are 5 and 10 matches.

## Elo Team Strength Features

Elo-style ratings are the second team-level feature family. They summarize opponent-adjusted team strength as a pre-match state variable.

Elo features are emitted before rating updates. Because the current dataset has match dates but not reliable kickoff timestamps, ratings are updated after each date block. This prevents same-date results from affecting features for other matches on that same date.

The first Elo implementation is intentionally simple: initial rating `1500.0`, K-factor `20.0`, standard expected-score formula, and draw score `0.5`. It is ready for later model evaluation but is not yet part of the default baseline feature table.

## Later Feature Sets

Later feature sets may add:

- Squad strength and player live form.
- Market or odds-implied expectations.
- Tournament-specific context.
- Confederation, travel, rest, and venue context.

Each extension should preserve the same leakage rule: features must only use information available before the match being predicted.
