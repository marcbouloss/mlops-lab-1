# mlops-lab-1 — Food-11 image classification

Reproducible MLOps pipeline for the [Food-11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
image classification task (11 food categories).

- **Code**: GitHub — `marcbouloss/mlops-lab-1`
- **Data**: DagsHub — `urpx-prog/mlops-lab-1` (DVC remote)
- **Experiments + model registry**: MLflow, as a local tracking server (Labs 2-3) and as the `mlflow` service of the Docker Compose stack (Lab 4)

## Labs

All labs build on the same project, so the code lives in one place; each lab has its own answers file.

| Lab | Topic | Answers | Main code | Snapshot |
|---|---|---|---|---|
| 1 | git + dvc, data preparation | [lab/lab1.md](lab/lab1.md) | [`src/food11/data.py`](src/food11/data.py), [`data.dvc`](data.dvc) | tag [`lab1`](https://github.com/marcbouloss/mlops-lab-1/tree/lab1) |
| 2 | training + MLflow tracking | [lab/lab2.md](lab/lab2.md), [reports/lab2/](reports/lab2) | [`src/food11/train.py`](src/food11/train.py) | tag [`lab2`](https://github.com/marcbouloss/mlops-lab-1/tree/lab2) |
| 3 | model registry + Docker serving | [lab/lab3.md](lab/lab3.md), [reports/lab3/](reports/lab3) | [`src/food11/serve.py`](src/food11/serve.py), [`Dockerfile`](Dockerfile), [`.dockerignore`](.dockerignore) | tag [`lab3`](https://github.com/marcbouloss/mlops-lab-1/tree/lab3) |
| 4 | Docker Compose: mlflow + inference + frontend | [lab/lab4.md](lab/lab4.md), [reports/lab4/](reports/lab4) | [`docker-compose.yml`](docker-compose.yml), [`mlflow/Dockerfile`](mlflow/Dockerfile), [`frontend/`](frontend) | tag [`lab4`](https://github.com/marcbouloss/mlops-lab-1/tree/lab4) |

## Stack

| Concern | Tool |
|---|---|
| Environment / deps | `uv` |
| Data versioning | `DVC` (remote: DagsHub) |
| Pipeline orchestration | `dvc.yaml` (preprocess → train → evaluate) |
| Experiment tracking + registry | MLflow 3.16: local server on `127.0.0.1:5000` (Labs 2-3), `mlflow` Compose service with a named volume (Lab 4) |
| Model | torchvision transfer learning (MobileNetV3-Small / ResNet18) |
| Serving | FastAPI + uvicorn, loads `models:/food11@champion` (Lab 3) |
| Containers | Docker multi-stage image (Lab 3), Docker Compose: mlflow + inference + Streamlit frontend (Lab 4) |

## Layout

```
data/food11_raw/              # raw images                      \
data/food11_processed/        # 128x128, one folder per class    } DVC-tracked (data.dvc)
data/food11_processed_mini/   # same, <=100 images per class     /
src/food11/data.py     # Lab 1: raw -> food11_processed(_mini)
src/food11/train.py    # Lab 2: resnet18 + MLflow tracking
src/food11/serve.py    # Lab 3: FastAPI serving models:/food11@champion
Dockerfile             # Lab 3: multi-stage image (uv builder -> python slim)
docker-compose.yml     # Lab 4: mlflow + inference + frontend (docker compose up --build)
mlflow/Dockerfile      # Lab 4: tracking server + registry image (state in the mlflow-data volume)
frontend/              # Lab 4: Streamlit upload page (app.py) + its Dockerfile
processed/             # split manifests (CSV), pipeline output
src/preprocess.py      # stage 1: raw folders -> train/val/test.csv
src/train.py           # stage 2: transfer learning -> models/model.pt
src/evaluate.py        # stage 3: metrics + confusion matrix
params.yaml            # all hyperparameters
dvc.yaml               # pipeline definition
metrics/ plots/        # tracked metrics and plots
```

## Setup

```bash
uv sync
uv run dvc pull        # fetch raw data from DagsHub
```

## Run the pipeline

```bash
uv run dvc repro       # runs only stages whose deps/params changed
uv run dvc metrics show
uv run dvc plots show
```

## Experiment tracking (MLflow)

**Local tracking server** (Labs 2-3), from the repo root. `src/food11/train.py` and
`src/food11/serve.py` use it by default:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"
```

**Full stack with Docker Compose** (Lab 4): mlflow UI at http://127.0.0.1:5000, upload
page at http://127.0.0.1:8501. Its state lives in the `mlflow-data` volume. See
[lab/lab4.md](lab/lab4.md) for how the volume was seeded with the Lab 2-3 runs.

```bash
docker compose up --build -d
docker compose down          # keeps the mlflow-data volume (add -v to delete it)
```

**Optional, DVC pipeline only:** `src/train.py` (`dvc repro`) logs to whatever
`MLFLOW_TRACKING_URI` points to, for example the DagsHub-hosted MLflow:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/urpx-prog/mlops-lab-1.mlflow
export MLFLOW_TRACKING_USERNAME=<dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```

## Tuning

Edit `params.yaml` (e.g. `train.arch`, `train.epochs`, `train.subset_fraction`
for a fast smoke run), then `uv run dvc repro`. Compare runs with
`uv run dvc exp show`.
