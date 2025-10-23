# src/data/load_forexDB.py
import os
import requests
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  


SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")  # Supabase PostgreSQL connection URL
API_URL = "https://api.exchangerate.host/latest?base=USD&symbols=NGN" # API endpoint for USD to NGN rates

def fetch_exchange_rate():
    """Fetch current USD→NGN rate from the API."""
    response = requests.get(API_URL)
    data = response.json()
    rate = data["rates"]["NGN"]
    timestamp = datetime.fromisoformat(data["date"])
    return "USD", "NGN", rate, timestamp

def store_exchange_rate(base, target, rate, timestamp):
    """Insert rate into Supabase PostgreSQL."""
    conn = psycopg2.connect(SUPABASE_DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO exchange_rates (base_currency, target_currency, rate, ts)
        VALUES (%s, %s, %s, %s)
        """,
        (base, target, rate, timestamp)
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Stored {base}->{target}: {rate} at {timestamp}")

def main():
    base, target, rate, ts = fetch_exchange_rate()
    store_exchange_rate(base, target, rate, ts)

if __name__ == "__main__":
    main()
