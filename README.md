# mlops-lab-1 — Food-11 image classification

Reproducible MLOps pipeline for the [Food-11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
image classification task (11 food categories).

- **Code**: GitHub — `marcbouloss/mlops-lab-1`
- **Data + experiments**: DagsHub — `urpx-prog/mlops-lab-1` (DVC remote + MLflow tracking)

## Stack

| Concern | Tool |
|---|---|
| Environment / deps | `uv` |
| Data versioning | `DVC` (remote: DagsHub) |
| Pipeline orchestration | `dvc.yaml` (preprocess → train → evaluate) |
| Experiment tracking | MLflow (hosted on DagsHub) |
| Model | torchvision transfer learning (MobileNetV3-Small / ResNet18) |

## Layout

```
data/food11_raw/      # raw images, DVC-tracked (data.dvc)
processed/        # split manifests (CSV), pipeline output
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

### Experiment tracking (optional)

Set DagsHub credentials so training logs to MLflow:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/urpx-prog/mlops-lab-1.mlflow
export MLFLOW_TRACKING_USERNAME=<dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```

## Tuning

Edit `params.yaml` (e.g. `train.arch`, `train.epochs`, `train.subset_fraction`
for a fast smoke run), then `uv run dvc repro`. Compare runs with
`uv run dvc exp show`.
