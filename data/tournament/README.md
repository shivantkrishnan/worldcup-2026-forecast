# Tournament Data

Maintain current 2026 World Cup tournament files here:

```text
data/tournament/fixtures_2026.csv
data/tournament/results_2026.csv
data/tournament/fixture_predictions_2026.csv
```

For now, update these files manually using FIFA official schedule and results pages as the source of truth.

Completed 2026 World Cup results may be used for standings, tournament state, prediction audit, and live simulation state. They must not be used to train the first baseline model.

Use `docs/fixtures_2026_template.md` for the fixture CSV header. Generate fixture predictions with:

```bash
python scripts/generate_fixture_predictions.py --output data/tournament/fixture_predictions_2026.csv
```

Prediction files are not written by default; the `--output` flag is required.
