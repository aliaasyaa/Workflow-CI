FROM python:3.9-slim

WORKDIR /app

# Install dependencies langsung via pip (tanpa conda)
RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn \
    pandas \
    numpy \
    cloudpickle

# Salin model artifacts yang sudah di-download saat CI
COPY model_export /app/model

# Expose port untuk MLflow model serving
EXPOSE 5000

# Serve model saat container dijalankan
CMD ["mlflow", "models", "serve", "--model-uri", "/app/model", "--host", "0.0.0.0", "--port", "5000", "--env-manager", "local"]
