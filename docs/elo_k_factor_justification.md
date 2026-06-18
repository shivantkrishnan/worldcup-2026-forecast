# Elo K-Factor Justification

## Why K-Factor Needs Justification

K controls how quickly Elo ratings react to new results. A larger K moves ratings more after each match, while a smaller K keeps ratings smoother and slower-moving.

Different domains require different K choices. Chess, club football, international football, and national-team tournaments have different schedules, noise levels, and competitive structures.

This project should not import chess or FIDE conventions mechanically. K is treated as a forecasting parameter and selected using out-of-sample validation on international football matches.

## Why K=10 Is Plausible For International Football

International fixtures are sparse and irregular. National teams may play only a limited number of matches per year, and those matches can vary sharply in opponent quality, competition, location, and importance.

Friendlies, qualifiers, and tournaments also have different incentives. A friendly may involve experimentation or squad rotation, while a qualifier or knockout match may reflect a stronger competitive signal.

Single-match results can be noisy. A lower K prevents one upset, unusual friendly, travel-heavy fixture, or rotated squad from over-moving national-team strength.

This project also already includes leakage-safe rolling team-form features. Those rolling features capture short-run momentum, while Elo can serve as a slower-moving opponent-adjusted team-strength signal.

## Empirical Selection Rule

K was selected from a compact grid using rolling-origin mean log loss:

- K-factor: `10`, `20`, `30`.
- Home advantage: `0`, `50`, `75`, `100`.

Selection was not based on in-sample fit. Each rolling-origin split refits preprocessing, model training, and calibration only on historical training rows, then evaluates on a future window.

Log loss is primary because the dashboard forecasts probabilities. A model should be rewarded for assigning higher probability to outcomes that actually occur.

Brier score, accuracy, and ECE remain supporting diagnostics. They help identify tradeoffs, especially cases where probability assignment improves but calibration-bin alignment worsens.

## Why This Is Still Provisional

The grid is compact, not exhaustive. It gives a disciplined first comparison, not a final rating-system proof.

Rolling-origin windows are broad and not yet tournament-specific. Future World Cup and major-tournament backtests may change the selected K.

ECE worsens relative to simple Elo. The selected K=10/home=50 variant improves rolling-origin mean log loss but does not solve calibration-bin alignment.

Margin-of-victory, tournament weighting, and host-country effects have not yet been tested. These may change both the preferred K-factor and the right interpretation of home or venue effects.

## World Cup Interpretation

The historical home adjustment helps model non-neutral historical matches. It is applied only to the match-level expected-score calculation and does not permanently inflate a team's underlying rating.

For actual 2026 World Cup forecasting, most matches should be treated as neutral unless a separate host/venue model is explicitly enabled.

USA, Canada, and Mexico host effects should be handled later as a distinct tournament-state feature, not as generic home advantage. That keeps historical home advantage separate from World Cup-specific venue and host-country context.
