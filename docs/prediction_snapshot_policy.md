# Prediction Snapshot Policy

## Purpose

Generated prediction CSVs are normally reproducible outputs. In the usual local
workflow, they should be regenerated from source data, feature code, and model
code rather than treated as hand-authored inputs.

For the public Streamlit demo, the project makes one deliberate exception:
`data/tournament/fixture_predictions_2026_live.csv` is committed as a small live
prediction snapshot so the deployed app can run from a fresh clone without
access to local raw training data or a scheduled prediction job.

## What Is Committed

The public demo may commit:

- `data/tournament/fixtures_2026.csv`
- `data/tournament/results_2026.csv`
- `data/tournament/fixture_predictions_2026_live.csv`

The live prediction snapshot includes forecast metadata such as forecast mode,
training cutoff date, feature cutoff date, model name, selected baseline label,
and whether rows are backfilled. It should be refreshed intentionally when
official results are updated, forecast features change, or the selected model
changes.

## What Remains Uncommitted

Raw historical training data remains uncommitted. The historical source file
belongs at `data/raw/results.csv` for local development, but it should not be
included in the public repository.

Backfilled prediction files, including
`data/tournament/fixture_predictions_2026.csv`, remain uncommitted unless the
project explicitly decides to publish a separate prediction snapshot for audit
or reconstruction. Model artifacts, local caches, large raw files, secrets, and
notebooks with embedded outputs should also remain uncommitted.

## Interpretation

The committed live prediction file is a public demo snapshot, not raw training
data and not a permanent model artifact. It lets Streamlit Community Cloud serve
the dashboard without retraining the model at startup. Completed World Cup
results in `results_2026.csv` may condition live standings, audits, and
simulation state, but they do not retrain the first baseline model.

Future production versions could replace this snapshot with a scheduled
prediction job, a backend service, or a versioned prediction store. Until then,
snapshot updates should be intentional, documented by commit history, and made
only after the relevant source fixtures/results or model code have been
validated.
