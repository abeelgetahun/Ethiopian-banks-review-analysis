from google_play_scraper import app, reviews, Sort
import pandas as pd
import time
from datetime import datetime
import json
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
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            count=count,
            sort=Sort.MOST_RELEVANT  # Use the enum, not a string
        )
        
        # Delay to prevent hitting rate limits
        time.sleep(2)
        
        # Convert to DataFrame
        df = pd.DataFrame(result)
        
        # Add bank/app name
        df['bank'] = app_name
        df['source'] = 'Google Play'
        
        print(f"Successfully scraped {len(df)} reviews for {app_name}")
        return df
    
    except Exception as e:
        print(f"Error scraping {app_id}: {str(e)}")
        return pd.DataFrame()

def main():
    # App IDs for the three Ethiopian banks
    # You may need to find the correct app IDs for these banks
    app_ids = {
        "com.combanketh.mobilebanking": "Commercial Bank of Ethiopia",
        "com.boa.boaMobileBanking": "Bank of Abyssinia",
        "com.dashen.dashensuperapp": "Dashen Bank"
    }
    
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each bank separately
    for app_id, bank_name in app_ids.items():
        df = scrape_app_reviews(app_id)
        if not df.empty:
            # Rename columns to match project requirements
            df = df.rename(columns={
                'content': 'review_text',
                'score': 'rating',
                'at': 'date'
            })
            
            # Select only necessary columns
            df = df[['review_text', 'rating', 'date', 'bank', 'source']]
            
            # Generate a bank-specific filename
            # Replace spaces with underscores and make it lowercase for a cleaner filename
            bank_filename = bank_name.replace(' ', '_').lower()
            file_path = f"data/raw/{bank_filename}_reviews_{timestamp}.csv"
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            print(f"Reviews for {bank_name} saved to {file_path}")
    
    print("All bank reviews have been saved to separate CSV files.")

if __name__ == "__main__":
    main()