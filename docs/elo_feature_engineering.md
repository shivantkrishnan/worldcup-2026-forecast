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
- `elo_effective_diff_team_a_minus_team_b`
- `elo_expected_score_team_a`
- `elo_home_advantage_applied`
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
- Home advantage: `0.0`

Unseen teams start at the initial rating. The K-factor controls how quickly ratings move after each result.

## Home And Neutral-Site Handling

Home advantage is modeled as a temporary expected-score adjustment, not as a permanent rating increase.

For non-neutral matches, `home_advantage` is added to `team_a`'s effective rating because `team_a` maps to the home team in the raw historical dataset. For neutral-site matches, no home advantage is applied.

The historical rationale is that non-neutral international matches can include crowd, travel, familiarity, venue, and local-context effects. A fixed home adjustment helps the rating update avoid treating every home result as pure evidence of underlying team strength.

The emitted raw rating difference remains:

```text
elo_team_a_pre - elo_team_b_pre
```

The emitted effective rating difference includes only the match-context adjustment used for expected score:

```text
elo_team_a_pre + elo_home_advantage_applied - elo_team_b_pre
```

This keeps underlying team strength separate from match-location context.

For 2026 World Cup forecasting, this does not mean applying generic home advantage to every match. Most World Cup fixtures should be treated as neutral by default. Host-country effects for USA, Canada, and Mexico should be added later as separate tournament-state or venue features if the project chooses to model them.

## Draw Handling

The result label is from `team_a`'s perspective:

- `team_a_win` -> score `1.0`
- `draw` -> score `0.5`
- `team_b_win` -> score `0.0`

For a draw, the higher-rated team loses rating points and the lower-rated team gains rating points because the lower-rated team outperformed expectation.

## Limitations

This is intentionally a simple Elo implementation. Current limitations:

- Home advantage is currently a simple fixed rating-point adjustment.
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

## Evaluation Summary

The first simple Elo feature set was evaluated against the rolling-form-only baseline using the same sigmoid-calibrated logistic regression model family.

Elo increased the feature count from 57 to 65 numeric features after adding effective-difference and home-adjustment columns. The simple K=20/home=0 setup improved the selected model's rolling-origin mean log loss from `1.201547` to `1.197724`.

The first K/home variant grid selected K=10 with a 50-point non-neutral home adjustment. This improved rolling-origin mean log loss to `1.186855` and beat simple Elo on log loss in all 6 rolling windows.

The calibration caveat remains: the selected K/home variant worsened mean ECE versus simple Elo. The selected baseline includes Elo features because log loss is the primary selection metric, but further calibration and rating refinement are still needed.
