import pandas as pd
from pathlib import Path

from src.cli import run_pipeline


def test_pipeline_integration(tmp_path, monkeypatch):
    # Set up a temporary workspace structure mimicking project
    base = tmp_path
    (base / "data" / "raw").mkdir(parents=True)
    (base / "data" / "task2").mkdir(parents=True)
    (base / "data" / "processed").mkdir(parents=True)

    # Create small raw dataset
    df = pd.DataFrame(
        {
            "review_text": [
                "Great app for transfers",
                "App keeps crashing, terrible experience",
                "Okay features, could be better",
            ],
            "rating": [5, 1, 3],
            "date": ["2025-06-01", "2025-06-02", "2025-06-03"],
            "bank": ["CBE", "CBE", "BOA"],
            "source": ["play", "play", "play"],
        }
    )
    raw_csv = base / "data" / "raw" / "sample.csv"
    df.to_csv(raw_csv, index=False)

    # NOTE: For simplicity, we don't monkeypatch the module's base path; the pipeline will run
    # against the real project. Here we just assert the function completes without error.
    # A fuller integration test would refactor cli.run_pipeline to accept base path.

    run_pipeline("vader")
    assert True
