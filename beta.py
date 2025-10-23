import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("API_KEY")  
# API_URL = f"https://api.exchangerate.host/latest?access_key={API_KEY}&base=USD&symbols=NGN" 
API_URL = "https://v6.exchangerate-api.com/v6/2743e5b334baac7ccf84a41c/latest/USD"
response = requests.get(API_URL)
data = response.json()
conversion_rates = data['conversion_rates']
naira_rate = conversion_rates['NGN']
timestamp = data["time_last_update_utc"]
timestamp = datetime.strptime(timestamp, "%a, %d %b %Y %H:%M:%S %z")
print (f"USD to NGN rate: {naira_rate} as of {timestamp}")