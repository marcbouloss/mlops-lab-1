"""Shared helpers for the Food-11 DVC pipeline."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

# Food-11 class id -> human readable label
CLASSES = [
    "Bread",              # 0
    "Dairy product",      # 1
    "Dessert",            # 2
    "Egg",                # 3
    "Fried food",         # 4
    "Meat",               # 5
    "Noodles-Pasta",      # 6
    "Rice",               # 7
    "Seafood",            # 8
    "Soup",               # 9
    "Vegetable-Fruit",    # 10
]

SPLIT_DIRS = {"train": "training", "val": "validation", "test": "evaluation"}


def load_params(section: str | None = None) -> dict:
    with open("params.yaml", "r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    return params[section] if section else params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def label_from_filename(name: str) -> int:
    """Food-11 files are named '<classid>_<index>.jpg'."""
    return int(Path(name).stem.split("_")[0])


def maybe_init_mlflow(run_name: str):
    """Enable DagsHub/MLflow tracking when credentials are present.

    Returns the mlflow module if tracking is active, else None. Set
    MLFLOW_TRACKING_URI + MLFLOW_TRACKING_USERNAME/PASSWORD (a DagsHub token),
    or DAGSHUB_TOKEN, to turn this on.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri and os.getenv("DAGSHUB_TOKEN"):
        try:
            import dagshub

            dagshub.init(repo_owner="urpx-prog", repo_name="mlops-lab-1", mlflow=True)
            uri = os.getenv("MLFLOW_TRACKING_URI")
        except Exception as exc:  # pragma: no cover
            print(f"[mlflow] dagshub init failed: {exc}")
            return None
    if not uri:
        print("[mlflow] no tracking URI configured - skipping experiment logging")
        return None
    try:
        import mlflow

        mlflow.set_experiment("food11-classification")
        mlflow.start_run(run_name=run_name)
        return mlflow
    except Exception as exc:  # pragma: no cover
        print(f"[mlflow] disabled: {exc}")
        return None
