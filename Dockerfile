# ============================================================
# Stock Sentiment Predictor — Production Dockerfile
# ============================================================
# Build:  docker build -t stock-predictor .
# Run:    docker run -p 8000:8000 --env-file .env stock-predictor
# ============================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create app directory
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download TextBlob corpora (needed at runtime)
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run with gunicorn for production (uvicorn workers)
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-"]
