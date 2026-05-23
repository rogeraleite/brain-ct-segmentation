FROM python:3.11-slim

WORKDIR /app

# libgomp1: required by scipy/numpy for OpenMP parallelism
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/  ./src/
COPY api/  ./api/
COPY models/best_model.pth ./models/best_model.pth

# Make src/ importable without installing it as a package
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
