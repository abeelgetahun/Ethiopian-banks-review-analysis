import pandas as pd
import re
from datetime import datetime
import os

def preprocess_reviews(df):
    """
    Preprocess the scraped review data
    """
    # Check for required columns
    required_columns = ['review_text', 'rating', 'date', 'bank', 'source']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print("Available columns:", list(df.columns))
        raise KeyError(f"Missing required columns: {missing}")

    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    processed_df['review_text'] = processed_df['review_text'].fillna('')
    
    # Remove duplicates
    processed_df = processed_df.drop_duplicates(subset=['review_text', 'bank'])
    
    # Normalize dates to YYYY-MM-DD format
    processed_df['date'] = pd.to_datetime(processed_df['date']).dt.strftime('%Y-%m-%d')
    
    # Remove special characters and extra whitespace from review text
    processed_df['review_text'] = processed_df['review_text'].apply(
        lambda x: re.sub(r'[^\w\s]', ' ', str(x))
    )
    processed_df['review_text'] = processed_df['review_text'].apply(
        lambda x: re.sub(r'\s+', ' ', str(x)).strip()
    )
    
    # Add a unique ID for each review
    processed_df['review_id'] = range(1, len(processed_df) + 1)
    
    # Reorder columns
    processed_df = processed_df[['review_id', 'review_text', 'rating', 'date', 'bank', 'source']]
    
    # Create data quality report
    quality_report = {
        "total_reviews": len(processed_df),
        "reviews_per_bank": processed_df['bank'].value_counts().to_dict(),
        "missing_values": processed_df.isnull().sum().to_dict(),
        "duplicate_count": len(df) - len(processed_df),
        "rating_distribution": processed_df['rating'].value_counts().to_dict()
    }
    
    return processed_df, quality_report

def save_processed_data(df, report):
    """
    Save the processed data and quality report
    """
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_data_path = f"data/processed/bank_reviews_processed_{timestamp}.csv"
    df.to_csv(processed_data_path, index=False)
    
    # Save quality report
    report_path = f"data/processed/quality_report_{timestamp}.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Processed data saved to {processed_data_path}")
    print(f"Quality report saved to {report_path}")
    
    return processed_data_path, report_path

def main():
    # Look for the latest raw data file
    raw_data_dir = "data/raw"
    if not os.path.exists(raw_data_dir):
        print(f"Raw data directory {raw_data_dir} does not exist.")
        return
    
    raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not raw_files:
        print(f"No CSV files found in {raw_data_dir}")
        return
    
    # Sort files by creation time and get the latest
    latest_file = sorted(raw_files, key=lambda x: os.path.getctime(os.path.join(raw_data_dir, x)))[-1]
    raw_data_path = os.path.join(raw_data_dir, latest_file)
    
    print(f"Loading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Preprocess the data
    processed_df, quality_report = preprocess_reviews(df)
    
    # Save the processed data
    save_processed_data(processed_df, quality_report)
    
    return processed_df

if __name__ == "__main__":
    main()