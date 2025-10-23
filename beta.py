import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

import pandas as pd
from pathlib import Path

RAW = Path("data/01_raw/usd_ngn_rates.csv")

# read everything as text to avoid parsing surprises
df = pd.read_csv(RAW, dtype=str)

# normalize column names
df.columns = df.columns.str.strip()

# parse Date (day-first format like "23/10/2025"), keep invalid as NaT
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# clean Rate: remove thousands separators/quotes/whitespace then convert to float
df["Rate"] = (
    df["Rate"]
    .astype(str)
    .str.strip()
    .str.replace(r'[^0-9\.\-]', "", regex=True)  # leaves digits, dot, minus
)
df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")


# save cleaned file
df.to_csv(RAW, index=False)

print(f"Read {len(df)} rows, wrote {len(df)} cleaned rows to {RAW}")