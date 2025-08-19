import sys
from pathlib import Path
import pandas as pd

from .preprocessor import preprocess_reviews, save_processed_data
from .sentiment import analyze_sentiment
from .thematic import extract_keywords_tfidf, identify_themes, assign_themes_to_reviews


def run_pipeline(method: str = "ensemble") -> None:
    base = Path(__file__).resolve().parents[1]
    raw_dir = base / "data" / "raw"
    task2_dir = base / "data" / "task2"
    task2_dir.mkdir(parents=True, exist_ok=True)

    # Load raw
    dfs = []
    for p in raw_dir.glob("*.csv"):
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            pass
    if not dfs:
        print("No raw CSVs found under data/raw.")
        return
    raw_df = pd.concat(dfs, ignore_index=True)

    # Preprocess
    processed_df, quality = preprocess_reviews(raw_df)
    save_processed_data(processed_df, quality)

    # Sentiment
    senti_df = analyze_sentiment(processed_df, method=method)
    senti_path = task2_dir / "sentiment_results.csv"
    senti_df.to_csv(senti_path, index=False)

    # Keywords and themes
    kw = extract_keywords_tfidf(senti_df, n_keywords=20)
    kw_rows = []
    for bank, pairs in kw.items():
        for term, score in pairs:
            kw_rows.append({"bank": bank, "keyword": term, "tfidf_score": score})
    kw_df = pd.DataFrame(kw_rows)
    kw_path = task2_dir / "bank_keywords.csv"
    kw_df.to_csv(kw_path, index=False)

    bank_themes = identify_themes(kw)
    theme_rows = []
    for bank, pairs in bank_themes.items():
        for theme, score in pairs:
            theme_rows.append({"bank": bank, "theme": theme, "theme_score": score})
    themes_df = pd.DataFrame(theme_rows)
    themes_path = task2_dir / "bank_themes.csv"
    themes_df.to_csv(themes_path, index=False)

    final_df = assign_themes_to_reviews(senti_df.copy(), bank_themes)
    final_path = task2_dir / "final_results.csv"
    final_df.to_csv(final_path, index=False)
    print("Pipeline complete.")


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "ensemble"
    run_pipeline(method)
