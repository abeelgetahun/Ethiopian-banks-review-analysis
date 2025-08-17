# Week 12 Second Interim Report (Aug 17, 2025)

## Plan to improve the project
- Refactor for reliability and maintainability.
- Add tests and CI to prove correctness.
- Align pipelines and file paths for reproducible runs.
- Prepare clear outputs (figures + markdown report) for non-technical stakeholders.

## Tasks to be done
- [x] Fix path mismatches across pipelines
- [x] Make heavy NLP deps optional with safe fallbacks
- [x] Add unit tests for core modules
- [x] Add GitHub Actions CI to run tests
- [ ] Simple Streamlit dashboard (next)
- [ ] Add SHAP/Explainability (next)

## Progress against plan
- Done today:
  - Updated `src/sentiment.py` to lazy-import transformers with VADER fallback.
  - Fixed `src/pipeline_task2.py` input path and `src/pipeline_task4.py` loader filenames.
  - Added tests: `tests/test_preprocessor.py`, `tests/test_sentiment.py`.
  - Added CI workflow: `.github/workflows/ci.yml`.
  - Updated `requirements.txt` with missing deps.
  - Wrote `README.md` with quickstart.
- Deferred:
  - Streamlit dashboard and SHAP due to time; planned next.

## Github proof of work
- See commits on branch `task-4` and CI status under Actions.

## Reschedule plan
- Build a minimal Streamlit dashboard to explore sentiments and themes (1 day).
- Add model explainability for classifier variant or rule transparency for themes (0.5 day).
- Expand tests for thematic module and visualization save paths (0.5 day).
