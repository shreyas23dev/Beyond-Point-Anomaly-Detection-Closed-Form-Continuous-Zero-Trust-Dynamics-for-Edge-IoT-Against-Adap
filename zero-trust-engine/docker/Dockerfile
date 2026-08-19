FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/

RUN mkdir -p logs artifacts data

EXPOSE 8001

CMD ["uvicorn", "src.zta_api:app", "--host", "0.0.0.0", "--port", "8001"]
