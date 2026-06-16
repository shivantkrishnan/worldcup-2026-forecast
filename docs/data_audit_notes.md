# Data Audit Notes

The historical-results audit is descriptive only. It prints local dataset diagnostics to stdout and does not write processed files.

The audit also runs in-memory canonical data quality checks for duplicate match IDs, negative scores, missing required values, invalid result labels, boolean neutral flags, and baseline cutoff consistency.

Before quality checks, canonical duplicate matches are resolved in memory. If duplicate rows agree on outcome fields, the first row is kept and extra rows are quarantined as `metadata_duplicate`. If duplicate rows disagree on score or result fields, the full duplicate group is quarantined as `conflicting_duplicate`.

Raw data remains uncommitted. The expected local historical dataset is:

```text
data/raw/results.csv
```

No feature tables, model artifacts, tournament simulations, or app outputs are created by the audit.

The `2026-06-10` cutoff is used only to separate first-baseline training data from later or current matches. Matches after the cutoff may be useful for evaluation and live tournament state, but they are excluded from the first baseline training set.

Canonical cleaned matches require completed scores so result labels can be assigned. Scoreless fixture rows in the raw CSV are excluded from canonical completed-match output and should be maintained through tournament-state files instead.

Quarantined duplicate rows are not written to disk yet. Conflicting duplicates are excluded from modeling data unless they are manually reviewed and resolved in a later data-curation step.
