# src/api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from src.model.predict import ForexInference

app = FastAPI(title="NGN Forex Forecast API")

service = ForexInference()

class ForecastRequest(BaseModel):
    weeks: int = 12


@app.get("/")
def health():
    return {"status": "online"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    preds = service.recursive_forecast(req.weeks)
    return {"horizon": req.weeks, "forecast": preds} 

# if __name__ == "__main__":
#     # For local testing
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)