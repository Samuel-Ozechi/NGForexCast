# src/data/ingest.py

import pandas as pd
from sqlalchemy import create_engine, text
from src.config.settings import Settings

settings = Settings()

def fetch_exchange_rates(start_date=None, end_date=None):
    """
    Fetch USD→NGN exchange rate records from the database.
    Filters by start_date and end_date if provided.
    """
    try:
        # Create SQLAlchemy engine
        engine = create_engine(settings.SUPABASE_DB_URL)

        # Base query
        query = "SELECT date, rate FROM ngn_us_exchange_rates WHERE 1=1"
        params = {}

        # Dynamically append filters
        if start_date:
            query += " AND date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND date <= :end_date"
            params['end_date'] = end_date

        query += " ORDER BY date ASC;"

        # Use text() to handle parameterized SQL safely
        df = pd.read_sql(text(query), engine, params=params)
        return df

    except Exception as e:
        print("❌ Error fetching data:", e)
        return None
    
## Alternative simpler version without date filtering
# def fetch_exchange_rates():
#     """
#     Fetch all USD→NGN exchange rate records from the database
#     and return them as a pandas DataFrame.
#     """
#     try:
#         # Create SQLAlchemy engine using your Postgres URL
#         engine = create_engine(settings.SUPABASE_DB_URL)

#         query = """
#             SELECT date, rate
#             FROM ngn_us_exchange_rates
#             ORDER BY date ASC;
#         """

#         df = pd.read_sql(query, engine)
#         return df

#     except Exception as e:
#         print("❌ Error fetching data:", e)
#         return None


if __name__ == "__main__":
    df = fetch_exchange_rates()
    if df is not None:
        print(f"Total records fetched: {len(df)}")
        print(df.head(4))
    else:
        print("No data fetched.")
