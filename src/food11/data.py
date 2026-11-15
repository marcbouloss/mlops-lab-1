"""Lab 1: build the ImageFolder datasets from the raw Food-11 images.

    uv run python ./src/food11/data.py

Input (DVC-tracked, flat folders, label = first part of the file name):
    data/food11_raw/{training,validation,evaluation}/<classid>_<idx>.jpg

Output:
    data/food11_processed/{training,validation,evaluation}/<class name>/*.jpg
        every image, shrunk to 128x128
    data/food11_processed_mini/{training,validation,evaluation}/<class name>/*.jpg
        same layout, at most MINI_PER_CLASS images per class and split

This is the layout torchvision.datasets.ImageFolder (and so ResNet training)
expects: one sub-folder per class, the folder name being the label.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]  # repo root, whatever the cwd
DATA = ROOT / "data"
RAW = DATA / "food11_raw"
PROCESSED = DATA / "food11_processed"
MINI = DATA / "food11_processed_mini"

SPLITS = ("training", "validation", "evaluation")
CLASSES = [
    "Bread",            # 0
    "Dairy product",    # 1
    "Dessert",          # 2
    "Egg",              # 3
    "Fried food",       # 4
    "Meat",             # 5
    "Noodles-Pasta",    # 6
    "Rice",             # 7
    "Seafood",          # 8
    "Soup",             # 9
    "Vegetable-Fruit",  # 10
]
SIZE = 128
MINI_PER_CLASS = 100
SEED = 42


def category_from_filename(filename: str) -> str:
    """'3_120.jpg' -> 'Egg'."""
    return CLASSES[int(Path(filename).stem.split("_")[0])]


def resize_image(img: Image.Image, size: int = SIZE) -> Image.Image:
    """RGB image of size x size: center-crop to a square, then resize (keeps the aspect ratio, no stretching)."""
    return ImageOps.fit(img.convert("RGB"), (size, size), Image.Resampling.LANCZOS)


def shrink(job: tuple[Path, Path]) -> None:
    src, dst = job
    with Image.open(src) as im:
        resize_image(im).save(dst, "JPEG", quality=95)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mini-per-class", type=int, default=MINI_PER_CLASS)
    ap.add_argument("--workers", type=int, default=None, help="processes used to resize (default: all cores)")
    args = ap.parse_args()

    if not RAW.is_dir():
        raise SystemExit(f"{RAW} not found - run `dvc pull` first")

    # 1. food11_processed: every image, resized, sorted into class folders.
    # Food-11 ships ~600 byte-identical images in two splits: keep the first copy
    # (training < validation < evaluation) so no test image was seen in training.
    shutil.rmtree(PROCESSED, ignore_errors=True)
    jobs, seen, dropped = [], set(), 0
    for split in SPLITS:
        for src in sorted((RAW / split).glob("*.jpg")):
            h = hashlib.md5(src.read_bytes()).hexdigest()
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            dst = PROCESSED / split / category_from_filename(src.name) / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((src, dst))
    if not jobs:
        raise SystemExit(f"no .jpg files found under {RAW}")
    print(f"dropped {dropped} duplicate images")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(shrink, jobs, chunksize=64), 1):
            if i % 2000 == 0 or i == len(jobs):
                print(f"resized {i}/{len(jobs)}")

    # 2. food11_processed_mini: a fixed random sample of the processed images
    shutil.rmtree(MINI, ignore_errors=True)
    rng = random.Random(SEED)
    for split in SPLITS:
        for cls in CLASSES:
            files = sorted((PROCESSED / split / cls).glob("*.jpg"))
            (MINI / split / cls).mkdir(parents=True, exist_ok=True)
            for f in rng.sample(files, min(args.mini_per_class, len(files))):
                shutil.copy2(f, MINI / split / cls / f.name)

    for name, root in (("food11_processed", PROCESSED), ("food11_processed_mini", MINI)):
        counts = {s: sum(1 for _ in (root / s).rglob("*.jpg")) for s in SPLITS}
        print(f"{name}: " + ", ".join(f"{s}={n}" for s, n in counts.items()))


if __name__ == "__main__":
    main()
