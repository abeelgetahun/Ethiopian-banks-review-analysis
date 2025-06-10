# src/database.py

import cx_Oracle
import os
from typing import Dict, Optional
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config.db_config import ORACLE_CONFIG, TABLE_SCHEMAS

class OracleConnection:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.config = ORACLE_CONFIG

    def connect(self):
        try:
            self.connection = cx_Oracle.connect(
                user=self.config['user'],
                password=self.config['password'],
                dsn=self.config['dsn'],
                encoding=self.config['encoding']
            )
            self.cursor = self.connection.cursor()
            print("Connected to Oracle Database")
        except Exception as e:
            print(f"Error connecting to Oracle Database: {e}")
            raise

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Database connection closed")

    def create_tables(self):
        try:
            # Create BANKS table
            self.cursor.execute(TABLE_SCHEMAS['banks'])
            
            # Create REVIEWS table
            self.cursor.execute(TABLE_SCHEMAS['reviews'])

            self.connection.commit()
            print("Tables created successfully")
        except cx_Oracle.DatabaseError as e:
            print(f"Error creating tables: {e}")
            raise

    def drop_tables(self):
        try:
            # Drop tables in correct order due to foreign key constraints
            self.cursor.execute("DROP TABLE reviews")
            self.cursor.execute("DROP TABLE banks")
            self.connection.commit()
            print("Tables dropped successfully")
        except cx_Oracle.DatabaseError as e:
            print(f"Error dropping tables: {e}")
            raise