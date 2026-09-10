"""Stage 2: transfer-learning trainer for Food-11."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models

from src.common import CLASSES, load_params, maybe_init_mlflow, set_seed
from src.dataset import Food11Dataset

MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")


def build_model(arch: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        if freeze_backbone:
            for p in m.features.parameters():
                p.requires_grad = False
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for p in m.parameters():
                p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return m


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


def main() -> None:
    p = load_params("train")
    set_seed(p["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)

    train_ds = Food11Dataset("processed/train.csv", p["img_size"], train=True)
    val_ds = Food11Dataset("processed/val.csv", p["img_size"], train=False)

    if p["subset_fraction"] < 1.0:
        g = torch.Generator().manual_seed(p["seed"])
        k = int(len(train_ds) * p["subset_fraction"])
        idx = torch.randperm(len(train_ds), generator=g)[:k].tolist()
        train_ds = Subset(train_ds, idx)
        print(f"using subset: {k} training images")

    dl_kw = dict(batch_size=p["batch_size"], num_workers=p["num_workers"], pin_memory=(device == "cuda"))
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    model = build_model(p["arch"], len(CLASSES), p["freeze_backbone"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad],
        lr=p["lr"],
        weight_decay=p["weight_decay"],
    )

    mlflow = maybe_init_mlflow(run_name=f"{p['arch']}-e{p['epochs']}")
    if mlflow:
        mlflow.log_params(p)

    history = []
    best_acc = 0.0
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = run_loss / seen
        val_loss, val_acc = evaluate(model, val_dl, device, criterion)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(row)
        if mlflow:
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "arch": p["arch"]}, MODELS_DIR / "model.pt")

    pd.DataFrame(history).to_csv(METRICS_DIR / "training_history.csv", index=False)
    with open(METRICS_DIR / "train_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"best_val_acc": best_acc, "final_val_acc": history[-1]["val_acc"]}, fh, indent=2)

    if mlflow:
        mlflow.log_artifact(str(MODELS_DIR / "model.pt"))
        mlflow.end_run()
    print(f"best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
