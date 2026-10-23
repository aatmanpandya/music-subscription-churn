# Build from the repo root, after exporting the champion bundle:
#   python -m pipeline.export_champion
#   docker build -f service/Dockerfile -t kkbox-churn-api .
#   docker run --rm -p 8000:8000 kkbox-churn-api
# Pinned by version tag and digest (python 3.12.15, multi-arch index) for reproducible builds.
FROM python:3.12.15-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_BUNDLE_DIR=/app/model_bundle CHURN_NTHREAD=2
WORKDIR /app

COPY service/requirements-service.txt /tmp/requirements-service.txt
RUN pip install --no-cache-dir -r /tmp/requirements-service.txt

# Only the inference-time code: no MLflow, no training, no scikit-learn.
COPY pipeline/__init__.py pipeline/schema.py pipeline/features.py pipeline/calibration.py pipeline/scorer.py pipeline/
COPY service/__init__.py service/app.py service/
COPY service/model_bundle /app/model_bundle

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
