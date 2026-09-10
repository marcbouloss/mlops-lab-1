"""Stage 1: turn the raw flat Food-11 folders into split manifests.

Raw layout (DVC-tracked):
    data/food11_raw/{training,validation,evaluation}/<classid>_<idx>.jpg

Output:
    data/processed/{train,val,test}.csv   columns: filepath,label,class_name
    data/processed/summary.json           per-split class counts
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common import CLASSES, SPLIT_DIRS, load_params, label_from_filename, set_seed


def build_split(raw_dir: Path, split: str) -> pd.DataFrame:
    src_dir = raw_dir / SPLIT_DIRS[split]
    rows = []
    for img in sorted(src_dir.glob("*.jpg")):
        label = label_from_filename(img.name)
        rows.append(
            {
                "filepath": img.as_posix(),
                "label": label,
                "class_name": CLASSES[label],
            }
        )
    if not rows:
        raise RuntimeError(f"no images found in {src_dir}")
    return pd.DataFrame(rows)


def main() -> None:
    p = load_params("preprocess")
    set_seed(p["seed"])
    raw_dir = Path(p["raw_dir"])
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "val", "test"):
        df = build_split(raw_dir, split).sample(frac=1.0, random_state=p["seed"])
        df.to_csv(out_dir / f"{split}.csv", index=False)
        summary[split] = {
            "total": int(len(df)),
            "per_class": {CLASSES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        }
        print(f"{split}: {len(df)} images")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
