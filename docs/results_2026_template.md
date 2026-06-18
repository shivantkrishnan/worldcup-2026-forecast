# `results_2026.csv` Template

Use this header for manually maintained completed 2026 World Cup results:

```csv
match_id,match_date,team_a,team_b,team_a_goals,team_b_goals,result,status,went_to_extra_time,went_to_penalties,team_a_penalties,team_b_penalties,source,last_updated
```

Small non-official example shape:

```csv
match_id,match_date,team_a,team_b,team_a_goals,team_b_goals,result,status,went_to_extra_time,went_to_penalties,team_a_penalties,team_b_penalties,source,last_updated
example_group_a_01,2026-06-11,Team A,Team B,2,1,team_a_win,completed,false,false,,,manual_example,2026-06-18
example_group_a_02,2026-06-12,Team C,Team D,0,0,draw,completed,false,false,,,manual_example,2026-06-18
```

The example rows are placeholders only. Do not treat them as official results.

Rules:

- Keep `match_id` aligned with `data/tournament/fixtures_2026.csv` and generated fixture predictions.
- Keep `team_a` and `team_b` in the same orientation as the fixture row.
- Use result labels from Team A's perspective: `team_a_win`, `draw`, or `team_b_win`.
- Include only completed matches for now.
- Do not use completed 2026 results to retrain the first baseline model.
