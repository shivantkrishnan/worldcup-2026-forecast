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

The presentation layer should follow the same boundary: completed fixtures are
displayed as actual results, while their probability rows are shown only in a
prediction-audit context. Scheduled fixtures without results are the only rows
displayed as current predictions.

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

The current live simulation samples approximate scorelines for unplayed matches
so goal difference and goals scored can affect group and third-place ranking.
This scoreline layer is for table mechanics only; it is not a separately
validated goals model.

`results_2026.csv` should be maintained from official or clearly
source-attributed completed results. The current ingestion helper uses FIFA's
official public calendar API and omits any row that cannot be mapped safely to
the local fixture orientation.

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

1. Completed fixtures from `results_2026.csv` keep their actual result and score.
2. Unplayed fixtures sample one of `team_a_win`, `draw`, or `team_b_win` from the selected model probabilities.
3. A scoreline is sampled conditional on the sampled result class.
4. Teams receive 3 points for a win, 1 point for a draw, and 0 points for a loss.
5. Group tables track played, points, wins, draws, losses, goals for, goals against, and goal difference.
6. Teams are ranked within each group.
7. Advancement is assigned by configurable rules.

The current 2026-style defaults advance the top 2 teams per group plus the 8 best third-place teams.

## Conditional Scorelines

The selected match model still predicts only W/D/L probabilities. The simulator adds a transparent conditional scoreline layer:

- Build empirical scoreline distributions from historical completed matches when local historical data is available.
- Condition the sampled scoreline on the sampled class: Team A win, draw, or Team B win.
- Use documented fallback scorelines if empirical distributions are unavailable.
- Keep completed 2026 results fixed and never resample them.

The empirical scoreline layer is not calibrated as a goals model yet. It exists so group tables can use plausible goal difference and goals scored instead of random tie-breaks alone.

## Ranking Criteria

Within each group, teams are ranked using:

1. Points.
2. Head-to-head points among tied teams, where feasible.
3. Head-to-head goal difference among tied teams, where feasible.
4. Head-to-head goals scored among tied teams, where feasible.
5. Overall goal difference.
6. Overall goals scored.
7. Optional `team_conduct_score`, if supplied.
8. Optional `fifa_ranking`, if supplied.
9. Seeded random fallback.

The head-to-head implementation aggregates matches among tied teams. It does
not yet recursively reapply head-to-head criteria after a multi-team tie is
partially broken.

Third-place teams are ranked across groups by:

1. Points.
2. Goal difference.
3. Goals for.
4. Optional `team_conduct_score`.
5. Optional `fifa_ranking`.
6. Seeded random fallback.

## Advancement Probabilities

The summary output reports:

- `group_winner_prob`: share of simulations where the team ranks first in its group.
- `top_2_prob`: share of simulations where the team ranks first or second.
- `third_place_prob`: share of simulations where the team ranks third.
- `best_third_place_advance_prob`: share of simulations where the team advances as a best third-place team.
- `advance_prob`: share of simulations where the team advances under the configured rules.
- `avg_points`: average simulated group-stage points.
- `avg_goals_for`: average simulated goals for.
- `avg_goals_against`: average simulated goals against.
- `avg_goal_difference`: average simulated goal difference.
- `avg_group_rank`: average simulated group rank.

If best-third-place qualification is enabled, `advance_prob` can exceed `top_2_prob`.

## Current Scope

This is not yet a full World Cup simulator.

Implemented:

- Group-stage W/D/L sampling from fixture probabilities.
- Conditional scoreline sampling for unplayed fixtures.
- Configurable points for wins and draws.
- Configurable top-N advancement per group.
- Best-third-place advancement.
- Official-style group and third-place ranking with documented limitations.
- In-memory output only.

Not implemented yet:

- Knockout bracket simulation.
- Penalty shootout logic.
- Fully calibrated goals model.
- Recursive multi-team tie-break handling.
- Prediction logging and audit integration.

## Next Requirements

Next steps before a complete tournament simulator:

- Add or maintain real 2026 fixtures at `data/tournament/fixtures_2026.csv`.
- Generate real fixture predictions at `data/tournament/fixture_predictions_2026.csv`.
- Maintain completed 2026 results in `data/tournament/results_2026.csv`.
- Validate or improve the scoreline layer with historical tournament scorelines.
- Build knockout bracket simulation.
- Integrate prediction logging and post-result audit.
