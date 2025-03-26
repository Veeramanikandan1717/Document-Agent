import sqlite3
import pandas as pd
import logging
from typing import Dict, Any

class SQLAgent:
    def __init__(self, db_path: str):
        """
        Initialize the connection to the SQLite database.
        Uses check_same_thread=False to allow multi-threaded access.
        """
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Returns rows as dictionaries
            self.cursor = self.conn.cursor()  # ✅ Fix: Initialize the cursor
            self.logger.info(f"✅ Successfully connected to database: {db_path}")
        except Exception as e:
            self.logger.error(f"❌ Database Connection Error: {e}")

    def get_table_names(self):
        """Fetch all table names from the SQLite database."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        return [table[0] for table in tables]  # Extract table names from tuples

    
    def execute_query(self, query: str):
        """
        Execute the given SQL query and return the results.
        If it's a SELECT query, returns a list of dictionaries; otherwise, commits changes.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            
            if query.strip().upper().startswith("SELECT"):
                results = cursor.fetchall()
                self.logger.info(f"✅ Query executed successfully. Rows fetched: {len(results)}")
                return [dict(row) for row in results]
            else:
                self.conn.commit()  # For INSERT, UPDATE, DELETE operations
                self.logger.info("✅ Query executed successfully. Changes committed.")
                return "Query executed successfully."
        except Exception as e:
            self.logger.error(f"SQL Execution Error: {e}")
            return f"SQL Execution Error: {e}"
    
    def store_csv_data(self, csv_file, table_name: str) -> str:
        """
        Reads a CSV file (as a file-like object) and stores its contents into a table.
        If the table already exists, it will be replaced.
        """
        try:
            # Ensure the file pointer is at the beginning
            csv_file.seek(0)
            # Read CSV file into a DataFrame
            df = pd.read_csv(csv_file)
            if df.empty:
                self.logger.warning("⚠️ The uploaded CSV file is empty.")
                return "The uploaded CSV file is empty."
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            self.conn.commit()
            self.logger.info(f"✅ Data from CSV stored successfully in table '{table_name}'.")
            return f"Data from CSV stored in table '{table_name}' successfully."
        except Exception as e:
            self.logger.error(f"Error storing CSV data: {e}")
            return f"Error storing CSV data: {e}"
