# World Cup 2026 Forecasting Dashboard

Portfolio-grade forecasting dashboard for international football matches, targeting the 2026 FIFA World Cup.

## Objective

Build an end-to-end machine learning pipeline:

1. Load historical international football results.
2. Clean and validate match records.
3. Build leakage-safe rolling features using only matches before each prediction date.
4. Train a calibrated 3-class match outcome model:
   - Team A win
   - Draw
   - Team B win
5. Evaluate with probabilistic metrics.
6. Serve predictions in a Streamlit match predictor.

Primary evaluation metrics:

- Log loss
- Brier score
- Calibration curves

## Project Structure

```text
.
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
└── tests/
```

## Setup

Create and activate a virtual environment:

```bash
conda deactivate
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run the Streamlit app:

```bash
python -m streamlit run app/streamlit_app.py
```

## Environment Troubleshooting

If your prompt shows both `(.venv)` and `(base)`, deactivate Conda before rebuilding the virtual environment:

```bash
conda deactivate
```

Use module commands from inside the project venv so Python, pip, and Streamlit all come from the same environment:

```bash
which python
python -m pip --version
python -m streamlit run app/streamlit_app.py
```

## Data Policy

Raw datasets belong in `data/raw/` and must not be committed. Processed datasets belong in `data/processed/` and should be reproducible from scripts whenever possible.

External APIs are intentionally out of scope for the first version.

## MVP Roadmap

1. Define canonical schema for historical international match results.
2. Implement cleaning and outcome labeling.
3. Build leakage-safe rolling team features.
4. Train a baseline multinomial logistic regression model.
5. Evaluate log loss, Brier score, and calibration.
6. Save model artifacts and prediction metadata.
7. Build a simple Streamlit match predictor.

## Extension Ideas

- Add Elo-style ratings.
- Add tournament-stage and confederation features.
- Compare multiple calibrated models.
- Add model cards and dataset documentation.
- Add simulation tools for group and knockout stages.
