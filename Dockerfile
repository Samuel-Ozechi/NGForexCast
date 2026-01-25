# 1. Base Image
FROM python:3.9-slim

# 2. System Dependencies (Essential for LightGBM)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup App Directory
WORKDIR /app

# 4. Install Python Dependencies
# Make sure your requirements.txt includes uvicorn, fastapi, lightgbm, joblib, etc.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Project Code
COPY . .

# 6. Default command for the API (Prefect will override this automatically)
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]