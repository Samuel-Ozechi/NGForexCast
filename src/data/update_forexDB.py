# src/data/load_forexDB.py
import os
import requests
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  


SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")  # Supabase PostgreSQL connection URL
API_KEY = os.getenv("API_KEY")   # Your API key for the exchange rate service
API_URL = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/USD"  # API endpoint for USD to NGN rates


def fetch_exchange_rate():
    """Fetch current USD→NGN rate from the API."""
    response = requests.get(API_URL)
    data = response.json()
    rate = float(data['conversion_rates']["NGN"])
    date_string = data.get("time_last_update_utc")
    format_code = "%a, %d %b %Y %H:%M:%S %z"
    datetime_object = datetime.strptime(date_string, format_code)
    date = datetime_object.date()

    return date, rate

def store_exchange_rate(date, rate):
    """Insert rate into Supabase PostgreSQL."""
    conn = psycopg2.connect(SUPABASE_DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO exchange_rates (date, rate)
        VALUES (%s, %s)
        """,
        (date, rate)
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Stored {date}: {rate}")

def main():
    date, rate = fetch_exchange_rate()
    store_exchange_rate(date, rate)

if __name__ == "__main__":
    main()
