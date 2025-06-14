import pandas as pd
import re
from datetime import datetime
import os
import json

def preprocess_reviews(df):
    """
    Preprocess the scraped review data
    """
    # Fallback renaming in case scraper failed to rename
    rename_map = {
        'content': 'review_text',
        'score': 'rating',
        'at': 'date'
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required_columns = ['review_text', 'rating', 'date', 'bank', 'source']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print("Available columns:", list(df.columns))
        raise KeyError(f"Missing required columns: {missing}")

    processed_df = df.copy()

    processed_df['review_text'] = processed_df['review_text'].fillna('')
    processed_df = processed_df.drop_duplicates(subset=['review_text', 'bank'])

    processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
    processed_df = processed_df.dropna(subset=['date'])  # remove bad dates
    processed_df['date'] = processed_df['date'].dt.strftime('%Y-%m-%d')

    processed_df['review_text'] = processed_df['review_text'].apply(
        lambda x: re.sub(r'[^\w\s]', ' ', str(x))
    )
    processed_df['review_text'] = processed_df['review_text'].apply(
        lambda x: re.sub(r'\s+', ' ', x).strip()
    )

    processed_df['review_id'] = range(1, len(processed_df) + 1)
    processed_df = processed_df[['review_id', 'review_text', 'rating', 'date', 'bank', 'source']]

    quality_report = {
        "total_reviews": len(processed_df),
        "reviews_per_bank": processed_df['bank'].value_counts().to_dict(),
        "missing_values": processed_df.isnull().sum().to_dict(),
        "duplicate_count": len(df) - len(processed_df),
        "rating_distribution": processed_df['rating'].value_counts().to_dict()
    }

    return processed_df, quality_report

def save_processed_data(df, report):
    os.makedirs('../data/processed', exist_ok=True)

    processed_data_path = "../data/processed/bank_reviews_processed.csv"
    df.to_csv(processed_data_path, index=False)

    report_path = "../data/processed/quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    print(f"Processed data saved to {processed_data_path}")
    print(f"Quality report saved to {report_path}")

    return processed_data_path, report_path

def main():
    raw_data_dir = "../data/raw"
    if not os.path.exists(raw_data_dir):
        print(f"Raw data directory {raw_data_dir} does not exist.")
        return

    raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not raw_files:
        print(f"No CSV files found in {raw_data_dir}")
        return

    all_dfs = []
    for file in raw_files:
        file_path = os.path.join(raw_data_dir, file)
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if not all_dfs:
        print("No valid data loaded.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    processed_df, quality_report = preprocess_reviews(combined_df)
    save_processed_data(processed_df, quality_report)

    return processed_df

if __name__ == "__main__":
    main()
