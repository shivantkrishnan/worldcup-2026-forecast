# Tournament Simulation

The tournament simulation layer consumes fixture-level match probabilities and
estimates group-stage and knockout-path probabilities with Monte Carlo sampling.

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
- generated live probabilities for remaining matches.

The simulation script can now accept a prediction file containing only
remaining fixtures, for example:

```text
data/tournament/fixture_predictions_2026_live.csv
```

When a fixture is missing from that prediction file, it must be covered by a
completed result. If any unplayed fixture lacks both a result and prediction
probabilities, the simulation fails instead of filling in a silent fallback.

The current live simulation samples approximate scorelines for unplayed matches
so goal difference and goals scored can affect group and third-place ranking.
This scoreline layer is for table mechanics only; it is not a separately
validated goals model.

The full-tournament simulator extends this path after the group stage. It
assigns Round-of-32 participants, samples knockout winners through the final,
and reports probabilities for reaching each stage and winning the tournament.

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

## Knockout Simulation

The full-tournament simulator currently reports:

- `reach_round_of_32_prob`
- `reach_round_of_16_prob`
- `reach_quarterfinal_prob`
- `reach_semifinal_prob`
- `reach_final_prob`
- `champion_prob`

It can also retain optional path traces. Trace rows keep one team per
simulation with final group position, stage reach flags, knockout opponents by
round, model-implied advancement probabilities in those knockout matches, and
whether the team advanced. These traces support champion-probability
decomposition without changing the model or the sampled tournament logic.

For each Monte Carlo run:

1. Completed group fixtures are fixed from `results_2026.csv`.
2. Remaining group fixtures are sampled from live prediction probabilities.
3. Group tables are ranked with the same group-stage rules.
4. The top two teams in each group and the eight best third-place teams enter
   the Round of 32.
5. Knockout matches are sampled through the final.

The Round-of-32 slot pools and third-place assignments are encoded in
`src/simulation/knockout_bracket.py` from the FIFA World Cup 26 Regulations,
Annex C:

```text
https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf
```

Annex C lists the 495 possible combinations of the eight best third-placed
teams and their Round-of-32 assignments. The project stores that table in
`src/simulation/third_place_assignment_2026.csv` and validates at load time
that all 495 valid eight-group combinations are covered, that each assignment
uses every qualifying third-place group exactly once, and that each group is
assigned only to an eligible slot. Non-eight-group inputs or invalid group
labels fail explicitly; there is no deterministic constrained fallback for
valid combinations.

## Knockout Match Probabilities

Knockout matchups are hypothetical until the bracket is known. Locally, when
`data/raw/results.csv` is available, `scripts/simulate_tournament.py` can build
neutral-site arbitrary matchup probabilities with the selected calibrated
logistic model and the live feature state through the feature cutoff.

The deployed Streamlit app intentionally does not commit raw historical data.
When raw data is unavailable, the app can still render a first-pass full
tournament view using a clearly labeled snapshot-strength fallback. That
fallback derives neutral matchup strength from committed group-stage prediction
probabilities and completed results. It is not the same as the selected model
and should be read as a product/demo approximation.

## Knockout Path Decomposition

Champion probability is a product of several forces:

- Group-stage state and the chance to win or escape the group.
- Official bracket placement, including the Annex C third-place mapping.
- The distribution of likely opponents generated by the simulation.
- Model-implied head-to-head advancement probabilities.
- Knockout randomness, including regular-time draw mass split evenly for
  advancement.

The path explorer estimates likely opponents by counting how often each
opponent appears in a team's simulated path, conditional on that team reaching
the round. These are not fixed scheduled opponents until the real bracket is
known. The head-to-head table reports the model-implied probability that the
selected team advances against common simulated opponents; when the public app
uses the deploy-safe fallback, those matchup probabilities are labeled as
approximate.

Path difficulty is summarized with transparent proxies: average model-implied
advancement probability in knockout matches played, average opponent champion
or finalist probability, and the expected number of elite opponents faced. These
diagnostics help separate real path effects from possible model artifacts such
as overconfident matchup probabilities or the snapshot-strength fallback.

## Knockout Draw Handling

The match model predicts regular-time `team_a_win`, `draw`, and `team_b_win`.
Knockout advancement needs one team. The first-pass rule is:

```text
p_team_a_advance = p_team_a_win + 0.5 * p_draw
p_team_b_advance = p_team_b_win + 0.5 * p_draw
```

This splits regular-time draw mass evenly between both teams, approximating
extra time and penalties symmetrically. A future extension should model
extra-time and penalty-shootout outcomes explicitly.

## Current Scope

This is now a first-pass full World Cup simulator, with documented limitations.

Implemented:

- Group-stage W/D/L sampling from fixture probabilities.
- Conditional scoreline sampling for unplayed fixtures.
- Configurable points for wins and draws.
- Configurable top-N advancement per group.
- Best-third-place advancement.
- Official-style group and third-place ranking with documented limitations.
- Round-of-32 participant assignment.
- Official FIFA Annex C third-place assignment table.
- First-pass full knockout simulation through champion.
- Optional knockout path traces and champion-probability decomposition.
- Neutral arbitrary-knockout probability generation locally when raw data is
  available.
- Deploy-safe snapshot-strength fallback for public UI rendering.
- In-memory output only.

Not implemented yet:

- Penalty shootout logic.
- Fully calibrated goals model.
- Recursive multi-team tie-break handling.
- Prediction logging and audit integration.

## Next Requirements

Next steps before a complete tournament simulator:

- Add or maintain real 2026 fixtures at `data/tournament/fixtures_2026.csv`.
- Generate real fixture predictions at `data/tournament/fixture_predictions_2026.csv`.
- Maintain completed 2026 results in `data/tournament/results_2026.csv`.
- Generate live remaining-fixture predictions separately, such as
  `data/tournament/fixture_predictions_2026_live.csv`.
- Validate or improve the scoreline layer with historical tournament scorelines.
- Validate knockout-path and knockout-probability estimates with prior
  tournament backtests.
- Integrate prediction logging and post-result audit.
