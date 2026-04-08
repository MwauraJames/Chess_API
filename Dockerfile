FROM python:3.10-slim

WORKDIR /app

# Install system dependencies first (faster)
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .

# Install Python packages with timeout
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy code after (doesn't invalidate pip cache)
COPY main.py .
COPY model.py .
COPY best.pt .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]