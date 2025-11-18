# src/data/fetch_db.py

import psycopg2
import pandas as pd

from src.config.settings import Settings

# Load global settings instance
settings = Settings()


def fetch_exchange_rates():
    """
    Fetch all USD→NGN exchange rate records from the database
    and return them as a pandas DataFrame.
    """
    try:
        conn = psycopg2.connect(settings.SUPABASE_DB_URL, sslmode="require")

        query = """
            SELECT date, rate
            FROM ngn_us_exchange_rates
            ORDER BY date ASC;
        """

        df = pd.read_sql_query(query, conn)
        return df

    except Exception as e:
        print("❌ Error fetching data:", e)
        return None

    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    df = fetch_exchange_rates()
    if df is not None:
        print(f"Total records fetched: {len(df)}")
    else:
        print("No data fetched.")
