# Elo Feature Engineering

Elo-style ratings are a compact way to summarize team strength from prior results. A team's rating rises after better-than-expected outcomes and falls after worse-than-expected outcomes.

## What Elo Measures

Elo measures relative team strength as a state variable. The expected score for `team_a` against `team_b` is:

```text
1 / (1 + 10 ** ((rating_b - rating_a) / 400))
```

Higher-rated teams are expected to earn more match result points than lower-rated teams.

## Why Elo Adds Opponent-Adjusted Strength

Rolling form features summarize recent results, goals, and points. They do not directly account for opponent quality. A win against a strong team and a win against a weak team can look identical in simple rolling points.

Elo adds opponent adjustment because the rating update depends on both teams' pre-match ratings. Beating a strong opponent is more informative than beating a weak opponent, and losing to a weak opponent is more damaging than losing to a strong opponent.

## Pre-Match Feature Rule

Elo ratings must be emitted as pre-match features. The current match result must never affect its own feature row.

For each match, the feature table stores:

- `elo_team_a_pre`
- `elo_team_b_pre`
- `elo_diff_team_a_minus_team_b`
- `elo_expected_score_team_a`
- `elo_matches_before_team_a`
- `elo_matches_before_team_b`

These values represent what was known before the match or date block.

## Date-Block Updates

The historical dataset currently has match dates but not reliable kickoff timestamps. When only date-level timestamps are available, same-date results should not affect features for other matches on that date.

The implementation therefore processes matches in date blocks:

1. Sort by `match_date` and `match_id`.
2. Emit pre-match Elo features for every match on the date using ratings as of the start of that date.
3. After all rows for that date are emitted, apply rating updates from that date's results.

This prevents same-date leakage while still updating the rating state for future dates.

## Defaults

The first implementation uses:

- Initial rating: `1500.0`
- K-factor: `20.0`

Unseen teams start at the initial rating. The K-factor controls how quickly ratings move after each result.

## Draw Handling

The result label is from `team_a`'s perspective:

- `team_a_win` -> score `1.0`
- `draw` -> score `0.5`
- `team_b_win` -> score `0.0`

For a draw, the higher-rated team loses rating points and the lower-rated team gains rating points because the lower-rated team outperformed expectation.

## Limitations

This is intentionally a simple Elo implementation. Current limitations:

- No home advantage term.
- No margin-of-victory adjustment.
- No tournament weighting.
- No recency decay beyond ordinary rating movement.
- No confederation or travel adjustment.
- No uncertainty estimate for teams with little history.
- Same-date updates are date-blocked because kickoff times are unavailable.

## Future Improvements

Future Elo-style extensions may include:

- Home advantage tuning.
- Margin-of-victory adjustment.
- Tournament weighting by match importance.
- Confederation adjustments.
- Separate ratings for neutral and non-neutral contexts.
- Glicko-style uncertainty or rating deviation.

Each extension should be evaluated with the same feature audit, single-holdout validation, rolling-origin backtesting, and model selection process.
