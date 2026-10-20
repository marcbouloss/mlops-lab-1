FROM python:3.11-slim

# pinned to the version that created mlflow.db (Labs 2-3), so the schema matches
RUN pip install --no-cache-dir mlflow==3.16.0

EXPOSE 5000

# --artifacts-destination (not --default-artifact-root): artifacts are served by this
# server over HTTP (mlflow-artifacts:/...), so the inference container can download them
# without access to this container's /mlflow-data volume.
# --allowed-hosts: mlflow 3 rejects Host headers it doesn't know; other services call it as mlflow:5000.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow:5000,localhost:*,127.0.0.1:*"]
