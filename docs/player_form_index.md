# Player Live Form Index

The Player Live Form Index is a proposed future feature layer. It is not part of the first team-level baseline and should not be implemented until the baseline, schemas, and leakage checks are stable.

The index should summarize player availability, recent involvement, and role-specific form as of a specific date. Every score must be computed with data available before the prediction timestamp.

## Core Components

### `availability_score`

Measures whether a player is expected to be available for selection.

Potential inputs:

- Included in tournament squad.
- Recent club and national-team appearances.
- Known injuries, suspensions, or absences when reliable data is available.
- Recent matchday squad involvement.

### `recent_minutes_score`

Measures match fitness and current involvement.

Potential inputs:

- Minutes in recent club matches.
- Minutes in recent international matches.
- Starts versus substitute appearances.
- Recency-weighted minutes.

### `attacking_form_score`

Measures attacking contribution for forwards and attacking midfielders.

Potential inputs:

- Goals.
- Shots or xG when available.
- Touches or actions in attacking areas when available.
- Penalty and non-penalty goals separated when possible.

Short-run goals should be shrunk because finishing form is noisy.

### `creative_form_score`

Measures chance creation and progression.

Potential inputs:

- Assists.
- Key passes or xA when available.
- Progressive passes.
- Progressive carries.
- Crosses or final-third entries when available.

### `defensive_form_score`

Measures defensive contribution for defenders and midfielders.

Potential inputs:

- Tackles, interceptions, blocks, and clearances.
- Defensive actions adjusted by team possession where available.
- Aerial duels.
- Pressures or pressure regains where event data exists.

### `goalkeeper_form_score`

Measures goalkeeper-specific contribution.

Potential inputs:

- Starts and minutes.
- Goals conceded, adjusted cautiously for team context.
- Save percentage.
- Post-shot xG performance when available.
- Claims, sweeping, or distribution metrics when available.

Goalkeeper scores should not reuse outfield attacking or defensive formulas.

### `discipline_penalty`

Penalizes cards and availability risks.

Potential inputs:

- Recent yellow and red cards.
- Suspensions.
- Accumulated tournament cards when live rules are modeled.

### `international_integration_score`

Measures how recently and consistently a player has appeared for the national team.

Potential inputs:

- Recent international minutes.
- Starts in competitive internationals.
- Friendly involvement as a selection signal.
- Shared minutes with likely starters when available.

Friendlies should be stronger evidence of selection and availability than of true player quality.

### `club_strength_adjustment`

Adjusts club performance for competition context.

Potential inputs:

- Club strength or league strength.
- Competition tier.
- Domestic versus continental matches.
- Opponent quality when available.

This adjustment should prevent overvaluing raw production from weaker contexts or undervaluing steady minutes in stronger competitions.

## Weighting and Normalization

### Recency Decay

Recent matches should usually matter more than older matches. Exponential weighting is a natural default:

```text
weight = exp(-days_since_match / half_life)
```

The half-life should be tuned or documented by feature family. Availability may need a shorter half-life than underlying quality.

### Position-Specific Normalization

Player scores must be normalized by position or role. Goals matter differently for forwards, center backs, and goalkeepers. Defensive actions are not comparable across tactical roles without adjustment.

Minimum position groups:

- Goalkeeper.
- Center back.
- Fullback or wingback.
- Defensive or central midfielder.
- Attacking midfielder or winger.
- Forward.

### Shrinkage for Small Samples

Small-sample player form should be pulled toward a prior. This is especially important for:

- Goals.
- Assists.
- Save percentage.
- Penalty-related outcomes.
- Any metric based on very few minutes.

Shrinkage should make the index less reactive to a single hot or cold match.

## Aggregating Player Scores to Team Features

The model should consume team-level aggregate features rather than hundreds of raw player columns.

### `expected_xi_strength`

Estimated strength of the likely starting XI as of the forecast timestamp.

### `top_14_strength`

Estimated strength of the likely starters plus primary substitutes. Useful because World Cup matches often depend on rotation and late substitutions.

### `squad_depth_score`

Captures quality and availability beyond the first-choice lineup.

### `attacking_unit_form`

Aggregates relevant attacking scores for forwards, wingers, and attacking midfielders.

### `midfield_progression_form`

Aggregates creative and progression-oriented midfield contributions.

### `defensive_unit_form`

Aggregates defender and defensive midfielder availability and defensive form.

### `goalkeeper_form`

Uses the likely starting goalkeeper or a weighted goalkeeper pool.

### `injury_or_availability_penalty`

Team-level penalty for missing, doubtful, suspended, or low-minute players.

### `national_team_cohesion`

Measures recent shared international involvement, squad stability, and friendly integration. This should be treated as a tactical familiarity signal, not a pure quality signal.

## Evaluation Standard

Player features should only be kept if they improve the forecasting system on:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics.

They should also pass leakage checks proving that no post-match data is included for the prediction being evaluated.
