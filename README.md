# Ethiopian Banks Review Analysis

Reliable, modular analysis of Google Play reviews for Ethiopian banking apps. Built for finance stakeholders who value transparency, risk reduction, and reproducibility.

## What this repo contains
- Data preprocessing (`src/preprocessor.py`)
- Sentiment analysis with safe fallbacks (`src/sentiment.py`)
- Thematic analysis and topic modeling (`src/thematic.py`)
- Pipelines to produce figures and reports (`src/pipeline_task2.py`, `src/pipeline_task4.py`)
- Visualizations (`src/visualization.py`)
- Streamlit stakeholder dashboard (`app.py`)
- Minimal tests and CI for reliability (`tests/`, `.github/workflows/ci.yml`)

## Quick start
1) Create a Python 3.11+ environment and install deps.
2) Place raw CSVs in `data/raw/`.
3) Run preprocessing to generate `data/processed/bank_reviews_processed.csv`.
4) Run the analysis pipeline to generate `data/task2` outputs.
5) Run the reporting pipeline to produce figures and `reports/final_report.md`.

## Run the end-to-end pipeline

Using the CLI (writes outputs to `data/processed` and `data/task2`):

```bash
python -m src.cli ensemble
```

Options: `ensemble` (default), `transformer`, `vader`, `textblob`.

## Launch the dashboard

```bash
streamlit run app.py
```

The app loads `data/task2/final_results.csv` if present, otherwise falls back to `data/processed/bank_reviews_processed.csv`.

## Notes
- Transformer models are optional and imported lazily; robust fallbacks ensure the pipeline runs without them.
- NLTK and spaCy resources are fetched on-demand with graceful fallbacks in constrained environments.

## Quality

- Linting (ruff), formatting (black), typing (mypy) configured via `pyproject.toml`.
- GitHub Actions CI runs lint, type check, and tests on pushes and PRs.
