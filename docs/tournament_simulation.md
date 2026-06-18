# Tournament Simulation

The first tournament simulation layer consumes fixture-level match probabilities and estimates group-stage advancement probabilities with Monte Carlo sampling.

When available, fixture probabilities are loaded from:

```text
data/tournament/fixture_predictions_2026.csv
```

That file is generated explicitly from the manually maintained fixture file and selected baseline model. If it is missing, the simulation script uses a synthetic example only for smoke testing.

When available, completed results are loaded from:

```text
data/tournament/results_2026.csv
```

Completed result rows are fixed in every simulation run. They are not sampled
from prediction probabilities.

The simulator now prints forecast-mode metadata from the prediction file:

- forecast mode values present,
- number of `is_backfilled` rows,
- whether past/completed fixture rows are being sampled as predictions.

If backfilled rows are present, the run is not a true live simulation. It is a
simulation over reconstructed ex-ante probabilities unless completed 2026
results are fixed from `data/tournament/results_2026.csv`.

## Backfilled vs Live Simulation

A backfilled ex-ante simulation samples every fixture from reconstructed
probabilities, including matches whose calendar dates have already passed.
That is useful for methodology and demos, but it is not a true live state.

A true live group-stage simulation combines:

- fixed completed results from `results_2026.csv`,
- generated probabilities from `fixture_predictions_2026.csv` for remaining matches.

The current live simulation still uses simplified group ranking because
scoreline/GD simulation and official tie-break rules are future work.

## Why Fixture Probabilities Feed Simulation

Tournament outcomes are paths through many uncertain matches. A deterministic favorite list cannot represent that uncertainty. Fixture probabilities let the simulator sample many plausible group-stage worlds and count how often each team wins its group, finishes in the top two, or advances.

## Why Full 3-Class Probabilities Matter

The simulator needs the full probability distribution:

- `p_team_a_win`
- `p_draw`
- `p_team_b_win`

Draw probability is essential in group play because draws affect both teams' points and can materially change qualification paths. A favorite-only output would lose that information.

## How Group Outcomes Are Sampled

For each simulated tournament run:

1. Each group-stage fixture samples one of `team_a_win`, `draw`, or `team_b_win`.
2. Teams receive 3 points for a win, 1 point for a draw, and 0 points for a loss.
3. Group tables track points, wins, draws, losses, and placeholder goal columns.
4. Teams are ranked within each group.
5. Advancement is assigned by configurable rules.

The current defaults advance the top 2 teams per group.

## Temporary Tie-Break Simplification

The current model predicts only win/draw/loss probabilities, not scorelines. Because goal difference and goals scored are not simulated yet, the first tie-break order is deliberately simple:

1. Points.
2. Wins.
3. Seeded random tie-break.

This is a temporary placeholder. It should be replaced by scoreline or goal-difference simulation plus official tournament tie-break rules before making polished World Cup claims.

## Advancement Probabilities

The summary output reports:

- `group_winner_prob`: share of simulations where the team ranks first in its group.
- `top_2_prob`: share of simulations where the team ranks first or second.
- `advance_prob`: share of simulations where the team advances under the configured rules.
- `avg_points`: average simulated group-stage points.
- `avg_group_rank`: average simulated group rank.

If best-third-place qualification is enabled, `advance_prob` can exceed `top_2_prob`.

## Current Scope

This is not yet a full World Cup simulator.

Implemented:

- Group-stage W/D/L sampling from fixture probabilities.
- Configurable points for wins and draws.
- Configurable top-N advancement per group.
- Optional best-third-place advancement.
- In-memory output only.

Not implemented yet:

- Scoreline or goal-difference simulation.
- Official FIFA tie-break rules.
- Knockout bracket simulation.
- Penalty shootout logic.
- Prediction logging and audit integration.

## Next Requirements

Next steps before a complete tournament simulator:

- Add or maintain real 2026 fixtures at `data/tournament/fixtures_2026.csv`.
- Generate real fixture predictions at `data/tournament/fixture_predictions_2026.csv`.
- Maintain completed 2026 results in `data/tournament/results_2026.csv`.
- Add a scoreline or goal-difference model.
- Implement official group tie-break rules.
- Build knockout bracket simulation.
- Integrate prediction logging and post-result audit.
