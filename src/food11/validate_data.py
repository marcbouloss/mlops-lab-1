"""Lab 5: check a processed Food-11 dataset against its data contract.

    uv run python src/food11/validate_data.py data/food11_processed_mini
    uv run python src/food11/validate_data.py data/food11_processed --min-per-class 100

Prints one PASS/FAIL line per check and exits with status 1 if any check failed.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run as a script from any cwd
from src.food11.data import CLASSES, SIZE, SPLITS  # noqa: E402


def check(name: str, problems: list[str]) -> bool:
    print(f"{'PASS' if not problems else 'FAIL'}  {name}")
    for p in problems[:10]:
        print(f"      - {p}")
    if len(problems) > 10:
        print(f"      ... and {len(problems) - 10} more")
    return not problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="e.g. data/food11_processed_mini")
    ap.add_argument("--min-per-class", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.05, help="max gap between a class's training and validation share")
    args = ap.parse_args()

    files = {s: {c: sorted((args.root / s / c).glob("*.jpg")) for c in CLASSES} for s in SPLITS}
    results = []

    # 1. exactly the 11 class folders in each split
    problems = []
    for s in SPLITS:
        found = {d.name for d in (args.root / s).iterdir() if d.is_dir()} if (args.root / s).is_dir() else set()
        if found != set(CLASSES):
            problems.append(f"{s}: missing {sorted(set(CLASSES) - found)}, unexpected {sorted(found - set(CLASSES))}")
    results.append(check("each split has exactly the 11 class folders", problems))

    # 2. every image opens, is RGB and SIZE x SIZE
    problems = []
    for s in SPLITS:
        for c in CLASSES:
            for f in files[s][c]:
                try:
                    with Image.open(f) as im:
                        if im.mode != "RGB" or im.size != (SIZE, SIZE):
                            problems.append(f"{f}: {im.mode} {im.size}")
                except OSError as exc:
                    problems.append(f"{f}: cannot open ({exc})")
    results.append(check(f"every image opens, is RGB and {SIZE}x{SIZE}", problems))

    # 3. enough images per class
    problems = [f"{s}/{c}: {len(files[s][c])} images"
                for s in SPLITS for c in CLASSES if len(files[s][c]) < args.min_per_class]
    results.append(check(f"every class has at least {args.min_per_class} images per split", problems))

    # 4. no image in two splits (same bytes = same image, whatever the file name)
    seen: dict[str, Path] = {}
    problems = []
    for s in SPLITS:
        for c in CLASSES:
            for f in files[s][c]:
                h = hashlib.md5(f.read_bytes()).hexdigest()
                if h in seen and seen[h].parts[-3] != s:
                    problems.append(f"{f} duplicates {seen[h]}")
                seen.setdefault(h, f)
    results.append(check("no image appears in two splits", problems))

    # 5. class balance: training share close to validation share
    problems = []
    n_train = sum(len(v) for v in files["training"].values()) or 1
    n_val = sum(len(v) for v in files["validation"].values()) or 1
    for c in CLASSES:
        p_train, p_val = len(files["training"][c]) / n_train, len(files["validation"][c]) / n_val
        if abs(p_train - p_val) > args.tau:
            problems.append(f"{c}: training {p_train:.3f} vs validation {p_val:.3f}")
    results.append(check(f"class shares in training and validation differ by at most {args.tau}", problems))

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
