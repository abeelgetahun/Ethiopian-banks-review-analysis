from google_play_scraper import app, reviews, Sort
import pandas as pd
import time
from datetime import datetime
import os

def scrape_app_reviews(app_id, country="us", lang="en", count=500):
    """
    Scrape reviews for a specific app from Google Play Store
    """
    try:
        # Get app details
        app_details = app(app_id)
        app_name = app_details['title']

        print(f"Scraping reviews for {app_name}...")

        # Get reviews
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            count=count,
            sort=Sort.MOST_RELEVANT
        )

        time.sleep(2)  # prevent rate limit issues

        df = pd.DataFrame(result)

        if df.empty:
            print(f"No reviews found for {app_name}")
            return pd.DataFrame()

        # Add extra info
        df['bank'] = app_name
        df['source'] = 'Google Play'

        # Rename for consistency
        df = df.rename(columns={
            'content': 'review_text',
            'score': 'rating',
            'at': 'date'
        })

        # Keep only needed columns
        df = df[['review_text', 'rating', 'date', 'bank', 'source']]

        print(f"Successfully scraped {len(df)} reviews for {app_name}")
        return df

    except Exception as e:
        print(f"Error scraping {app_id}: {str(e)}")
        return pd.DataFrame()

def main():
    app_ids = {
        "com.combanketh.mobilebanking": "Commercial Bank of Ethiopia",
        "com.boa.boaMobileBanking": "Bank of Abyssinia",
        "com.dashen.dashensuperapp": "Dashen Bank"
    }

    os.makedirs("data/raw", exist_ok=True)

    for app_id, bank_name in app_ids.items():
        df_new = scrape_app_reviews(app_id)
        if df_new.empty:
            continue

        bank_filename = bank_name.replace(' ', '_').lower()
        file_path = f"data/raw/{bank_filename}_reviews.csv"

        if os.path.exists(file_path):
            try:
                df_existing = pd.read_csv(file_path)
                combined_df = pd.concat([df_existing, df_new], ignore_index=True)
                combined_df.drop_duplicates(subset=['review_text', 'bank'], inplace=True)
                print(f"Updated existing file for {bank_name}")
            except Exception as e:
                print(f"Error reading existing file for {bank_name}, will overwrite: {e}")
                combined_df = df_new
        else:
            combined_df = df_new
            print(f"Creating new file for {bank_name}")

        combined_df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")

    print("All reviews scraped and updated.")


if __name__ == "__main__":
    main()
