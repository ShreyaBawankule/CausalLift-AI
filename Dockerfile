# Production Multi-Stage Container for Causal Uplift Intelligence Platform
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and pre-trained artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY reports/ ./reports/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default launch command: starts the interactive executive dashboard
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
