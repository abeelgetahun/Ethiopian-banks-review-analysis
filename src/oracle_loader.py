# src/oracle_loader.py

import pandas as pd
from database import OracleConnection
from typing import Dict
import os
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

class DataLoader:
    def __init__(self):
        self.db = OracleConnection()
        
    def load_data(self, processed_df: pd.DataFrame, app_ids: Dict[str, str]):
        try:
            self.db.connect()
            
            # Recreate tables (optional - comment out if you want to preserve existing data)
            self.db.drop_tables()
            self.db.create_tables()
            
            # Insert banks data
            self._insert_banks(app_ids)
            
            # Get bank_id mapping
            bank_mapping = self._get_bank_mapping()
            
            # Insert reviews data
            self._insert_reviews(processed_df, bank_mapping)
            
            print("Data loading completed successfully")
            
        except Exception as e:
            print(f"Error in data loading process: {e}")
            raise
        finally:
            self.db.close()

    def _insert_banks(self, app_ids: Dict[str, str]):
        try:
            for app_id, bank_name in app_ids.items():
                self.db.cursor.execute(
                    "INSERT INTO banks (bank_name, app_id) VALUES (:1, :2)",
                    (bank_name, app_id)
                )
            self.db.connection.commit()
            print("Banks data inserted successfully")
        except Exception as e:
            print(f"Error inserting banks data: {e}")
            raise

    def _get_bank_mapping(self) -> Dict[str, int]:
        self.db.cursor.execute("SELECT bank_id, bank_name FROM banks")
        return dict(self.db.cursor.fetchall())

    def _insert_reviews(self, df: pd.DataFrame, bank_mapping: Dict[str, int]):
        try:
            for _, row in df.iterrows():
                bank_id = bank_mapping[row['bank']]
                self.db.cursor.execute("""
                    INSERT INTO reviews 
                    (review_id, bank_id, review_text, rating, review_date, source)
                    VALUES (:1, :2, :3, :4, TO_DATE(:5, 'YYYY-MM-DD'), :6)
                """, (
                    int(row['review_id']),
                    bank_id,
                    row['review_text'],
                    int(row['rating']),
                    row['date'],
                    row['source']
                ))
            
            self.db.connection.commit()
            print(f"Inserted {len(df)} reviews successfully")
        except Exception as e:
            print(f"Error inserting reviews data: {e}")
            raise

def main():
    # Define data path relative to project root
    data_dir = Path(project_root) / "data" / "processed"
    processed_data_path = data_dir / "bank_reviews_processed.csv"

    if not processed_data_path.exists():
        print(f"Processed data file not found: {processed_data_path}")
        return

    df = pd.read_csv(processed_data_path)
    
    app_ids = {
        "com.combanketh.mobilebanking": "Commercial Bank of Ethiopia",
        "com.boa.boaMobileBanking": "Bank of Abyssinia",
        "com.dashen.dashensuperapp": "Dashen Bank"
    }
    
    loader = DataLoader()
    loader.load_data(df, app_ids)

if __name__ == "__main__":
    main()