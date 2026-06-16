# Player Data Strategy

Player-level data is a planned modular extension to the team-level World Cup forecasting system. The first baseline remains team-level so the project can establish a clean, leakage-safe benchmark before adding noisier and harder-to-maintain player inputs.

## Why Player-Level Data Is Useful

Team strength changes when player availability, minutes, role, and form change. A team-level model can learn broad historical strength, but it may miss major short-term context such as:

- Key players returning from injury or missing the squad.
- A goalkeeper losing starting status.
- A striker getting regular club minutes before the tournament.
- A national team settling on a preferred midfield or defensive unit.
- Recent friendlies revealing likely selection, rotations, or tactical plans.

Player-level features can later help the model represent current squad quality, depth, and role-specific form more directly than team-level historical results alone.

## Why Player-Level Data Is Risky

Player data is harder to use safely than team-level match results.

- Identity matching is error-prone across sources because names, birth dates, clubs, and identifiers can disagree.
- Short-run goals and assists are noisy and can overstate true ability.
- Club performance does not transfer perfectly to international football.
- Friendlies are useful selection signals, but weak quality signals.
- Injuries, lineups, and late roster changes can be unavailable or inconsistent.
- Event-level metrics are source-dependent and may not cover all players or leagues evenly.
- Post-match information can easily leak into features if as-of dates are not enforced.

Every player feature must have a clear `as_of_date` or timestamp boundary. No post-match information should be used to predict that match.

## Source Tiers

### Tier 1: Practical Player Data

Use first because it is relatively understandable and easier to explain in a portfolio project.

Examples:

- Transfermarkt-style appearance data.
- Minutes.
- Starts.
- Goals.
- Assists.
- Cards.
- Club.
- Competition.
- Market value.
- National-team appearances.
- Recent international friendly involvement.

Tier 1 can support simple availability, minutes, squad depth, and national-team integration features without requiring advanced event data.

### Tier 2: Public/Basic Player Stats

Use after Tier 1 once identity matching and as-of-date rules are reliable.

Examples:

- Standard player season stats.
- Shooting.
- Passing.
- Defensive stats.
- Goalkeeper stats.
- Possibly FBref-style data if accessible and stable.

Tier 2 can improve role-specific features, but coverage and stability should be checked before relying on it.

### Tier 3: Open Event Data for Methodology and Backtesting

Use primarily to demonstrate methodology and test richer feature ideas.

Examples:

- StatsBomb open data.
- Wyscout public data.
- Event-level concepts such as xG, xA, xT, VAEP, progressive actions, and defensive actions.

Tier 3 may not fully cover current World Cup squads, but it can help design, validate, and explain advanced player feature engineering.

### Tier 4: Paid/Current APIs Later

Use only after the baseline and public-data extension are stable.

Examples:

- Current event-level player data.
- Injuries.
- Lineups.
- Advanced club and international statistics.

Tier 4 may improve live forecasting, but it adds dependency, cost, licensing, and reproducibility concerns.

## Why the Team-Level Baseline Comes First

The baseline must be a clean reference point. It should train only on team-level historical results and use leakage-safe rolling features. This makes it possible to answer the core question later:

Did player-level data improve probabilistic forecasts?

The player extension should be evaluated by out-of-sample log loss, multiclass Brier score, and calibration diagnostics. If player features do not improve those metrics, they should not be kept merely because they are interesting.

## How Player Data Can Improve Forecasts Later

Player-level data may add signal through:

- Expected starting XI strength.
- Top-14 rotation strength.
- Squad depth.
- Unit-level attacking, midfield, defensive, and goalkeeper form.
- Availability penalties for injuries, suspensions, or missing squad members.
- National-team cohesion from recent shared minutes and appearances.

These features should aggregate to team-level inputs so the first model architecture can evolve without turning the dashboard into a player-by-player prediction system too early.

## Club Form vs International and Friendly Form

Club form should be treated as a broad signal of player quality, role, and match fitness. It usually has more minutes and richer competition context, but it may not map directly to a national-team role.

International competitive matches should be treated as stronger evidence of national-team role and fit because they reflect the player in the actual team context.

Recent friendlies should be treated differently:

- Stronger signal for availability, selection likelihood, and tactical integration.
- Weaker signal for true player quality.
- Useful for detecting likely starters or recently favored substitutes.
- Not enough on their own to upgrade a player's underlying strength.

Recent goals and assists should be shrunk toward a longer-run baseline because short-run finishing and assist totals are noisy.

## Player Identity Registry

Player identity matching is a major project risk. The project should maintain `data/player/player_registry.csv` as the central lookup table for player identifiers across sources.

The registry should track:

- Canonical player name.
- Normalized name.
- Date of birth.
- Nationality and national team.
- Current club.
- Primary and secondary positions.
- Source-specific identifiers.
- Last update timestamp.

Feature engineering should join through `player_id` rather than raw names whenever possible.
