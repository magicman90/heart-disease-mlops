# ---------------------------------------------------------------------------
# Heart Disease Risk Prediction API - Production Image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by scientific python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Create a non-root user for security best-practice, and make sure it
# actually owns /app (needed since app.py writes api_requests.log there;
# without this chown, the container crashes with PermissionError on startup)
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

WORKDIR /app/src
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
