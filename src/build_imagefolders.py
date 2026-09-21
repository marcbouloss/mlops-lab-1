"""Stage: materialise the CSV splits as ImageFolder trees (<split>/<class>/<img>.jpg).

Outputs:
    processed/food11_processed/{train,val,test}/<class>/...       full splits
    processed/food11_processed_mini/{train,val,test}/<class>/...  small per-class subset
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from src.common import load_params

OUT = Path("processed")


def place(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # hardlink: no extra disk in the workspace
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    p = load_params("imagefolders")
    for name, per_class in (("food11_processed", None), ("food11_processed_mini", p["mini_per_class"])):
        root = OUT / name
        shutil.rmtree(root, ignore_errors=True)
        for split in ("train", "val", "test"):
            df = pd.read_csv(OUT / f"{split}.csv")  # already shuffled by preprocess
            if per_class:
                df = df.groupby("label").head(per_class[split])
            for fp, cls in zip(df["filepath"], df["class_name"]):
                place(fp, root / split / cls / Path(fp).name)
            print(f"{name}/{split}: {len(df)} images")


if __name__ == "__main__":
    main()
