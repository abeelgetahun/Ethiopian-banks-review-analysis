# Ethiopian Banks Review Analysis

A robust, modular pipeline to analyze Google Play reviews of Ethiopian banks, designed to highlight reliability and risk reduction for finance stakeholders.

## What this repo contains
- Data preprocessing (`src/preprocessor.py`)
- Sentiment analysis with safe fallbacks (`src/sentiment.py`)
- Thematic analysis and topic modeling (`src/thematic.py`)
- Pipelines to produce figures and reports (`src/pipeline_task2.py`, `src/pipeline_task4.py`)
- Visualizations (`src/visualization.py`)
- Minimal tests and CI for reliability (`tests/`, `.github/workflows/ci.yml`)

## Quick start
1) Create a Python 3.11+ environment and install deps.
2) Place raw CSVs in `data/raw/`.
3) Run preprocessing to generate `data/processed/bank_reviews_processed.csv`.
4) Run the analysis pipeline to generate `data/task2` outputs.
5) Run the reporting pipeline to produce figures and `reports/final_report.md`.

## Commands (optional)
- Preprocess: python -m src.preprocessor
- Task 2: python -m src.pipeline_task2
- Task 4: python -m src.pipeline_task4

## Notes
- Transformer models are optional and imported lazily to avoid heavy installs in constrained environments; VADER/TextBlob fallback ensures the pipeline still runs.
- NLTK resources are auto-downloaded in CI.
