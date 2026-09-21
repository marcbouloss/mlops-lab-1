"""Lab 2: fine-tune resnet18 on Food-11 and track the run with MLflow.

    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASETS = {"processed": "processed/food11_processed", "mini": "processed/food11_processed_mini"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over loader; trains if an optimizer is given. Returns (loss, accuracy)."""
    model.train(optimizer is not None)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="mini")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = DATASETS[args.dataset]
    train_ds = datasets.ImageFolder(f"{root}/train", TRAIN_TF)
    val_ds = datasets.ImageFolder(f"{root}/val", EVAL_TF)
    test_ds = datasets.ImageFolder(f"{root}/test", EVAL_TF)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, **kw)
    test_dl = DataLoader(test_ds, **kw)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))  # 1000 -> 11 classes
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")
    with mlflow.start_run():
        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "resnet18",
            "optimizer": "adam",
            "seed": args.seed,
        })
        for epoch in range(1, args.epochs + 1):
            train_loss, _ = run_epoch(model, train_dl, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_dl, criterion, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        _, test_acc = run_epoch(model, test_dl, criterion, device)
        mlflow.log_metric("test_accuracy", test_acc)
        # mlflow 3 saves as pt2 (traced graph) by default, which needs an example input
        example = test_ds[0][0].unsqueeze(0).numpy()
        mlflow.pytorch.log_model(model.cpu().eval(), "model", input_example=example)
        print(f"test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
