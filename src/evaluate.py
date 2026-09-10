"""Stage 3: evaluate the trained model on the held-out evaluation split."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.common import CLASSES, load_params, maybe_init_mlflow
from src.dataset import Food11Dataset
from src.train import build_model

METRICS_DIR = Path("metrics")
PLOTS_DIR = Path("plots")


@torch.no_grad()
def main() -> None:
    p = load_params("evaluate")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    METRICS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    ckpt = torch.load("models/model.pt", map_location=device)
    model = build_model(ckpt["arch"], len(CLASSES), freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = Food11Dataset("processed/test.csv", load_params("train")["img_size"], train=False)
    dl = DataLoader(ds, batch_size=p["batch_size"], num_workers=4)

    y_true, y_pred = [], []
    loss_sum, n = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y in dl:
        x = x.to(device)
        out = model(x).cpu()
        loss_sum += criterion(out, y).item() * x.size(0)
        n += x.size(0)
        y_pred.extend(out.argmax(1).tolist())
        y_true.extend(y.tolist())

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    metrics = {
        "test_accuracy": acc,
        "test_loss": loss_sum / n,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    with open(METRICS_DIR / "eval_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(METRICS_DIR / "classification_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=120)

    print(json.dumps(metrics, indent=2))

    mlflow = maybe_init_mlflow(run_name="evaluate")
    if mlflow:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(PLOTS_DIR / "confusion_matrix.png"))
        mlflow.end_run()


if __name__ == "__main__":
    main()
