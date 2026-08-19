FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Ensure data and logs directories exist
RUN mkdir -p /app/data /app/logs /app/artifacts

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Run FastAPI via Uvicorn
CMD ["uvicorn", "src.trust_api:app", "--host", "0.0.0.0", "--port", "8000"]
