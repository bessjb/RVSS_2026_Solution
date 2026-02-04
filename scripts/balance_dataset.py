#!/usr/bin/env python3

"""Create a single balanced folder from steering-labeled images.

Simple approach:
1) Copy ALL images into a new folder with suffix "_balanced".
2) Compute class counts in that new folder.
3) For any class with more than the minimum class count, randomly delete extra
    images until all classes match.

Assumptions (matches scripts/steerDS.py and scripts/make_balanced_val_test.py):
- Steering value is encoded in the filename stem after a fixed prefix length.
  Example: "000123-0.25.jpg" with --prefix-len 6 => "-0.25".
- Classes are derived from steering using the same thresholds.

No train/val/test split is performed: output is a single folder.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence


CLASS_LABELS: List[str] = [
    "sharp left",
    "left",
    "straight",
    "right",
    "sharp right",
]


def steering_to_class(steering: float) -> int:
    # Keep identical to scripts/steerDS.py
    if steering <= -0.5:
        return 0
    if steering < 0:
        return 1
    if steering == 0:
        return 2
    if steering <= 0.25:
        return 3
    return 4


def parse_steering_from_path(image_path: Path, prefix_len: int) -> float:
    stem = image_path.stem
    if len(stem) <= prefix_len:
        raise ValueError(
            f"Filename stem too short to parse steering: {image_path.name} "
            f"(stem={stem!r}, prefix_len={prefix_len})"
        )
    raw = stem[prefix_len:]
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse steering from filename {image_path.name}: "
            f"stem[{prefix_len}:]={raw!r}"
        ) from exc


def iter_images(input_dir: Path, exts: Sequence[str]) -> List[Path]:
    exts_lc = {e.lower() for e in exts}
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts_lc]
    )


def count_by_class(paths: Sequence[Path], prefix_len: int) -> Dict[int, List[Path]]:
    by_class: Dict[int, List[Path]] = {i: [] for i in range(len(CLASS_LABELS))}
    for p in paths:
        steering = parse_steering_from_path(p, prefix_len=prefix_len)
        cls = steering_to_class(steering)
        by_class[cls].append(p)
    return by_class


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output folder {path} exists and is not empty. "
                f"Use --overwrite to clear it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a balanced folder by downsampling each steering class to the minimum class count."
    )
    parser.add_argument("--input", required=True, help="Input folder containing images")
    parser.add_argument(
        "--ext",
        default=".jpg",
        help="Image extension(s), comma-separated (default: .jpg). Example: .jpg,.png",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=6,
        help="Number of chars to strip from filename stem before parsing steering (default: 6)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear output folder if it already exists and is non-empty",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_dir}")

    exts = [e.strip() for e in str(args.ext).split(",") if e.strip()]
    if not exts:
        raise SystemExit("No extensions provided via --ext")

    # Always create a sibling output folder with suffix "_balanced".
    # Filenames are NOT modified.
    output_dir = input_dir.parent / f"{input_dir.name}_balanced"
    if output_dir == input_dir:
        raise SystemExit("Output folder resolves to the input folder; refusing to modify the input dataset.")
    print(f"Output folder: {output_dir}")

    paths = iter_images(input_dir=input_dir, exts=exts)
    if not paths:
        raise SystemExit(f"No images found under {input_dir} with extensions: {exts}")

    by_class = count_by_class(paths, prefix_len=args.prefix_len)

    print("Input counts per class:")
    for cid, label in enumerate(CLASS_LABELS):
        print(f"  {cid}: {label:11s} -> {len(by_class[cid])}")

    min_per_class = min(len(v) for v in by_class.values())
    if min_per_class == 0:
        raise SystemExit(
            "At least one class has 0 images, cannot create a balanced folder. "
            "Collect more data for the missing classes."
        )

    rng = random.Random(args.seed)
    ensure_empty_dir(output_dir, overwrite=args.overwrite)

    # 1) Copy everything first (keep same filenames)
    for src in paths:
        shutil.copy2(str(src), str(output_dir / src.name))

    # 2) Recompute class lists for files in output folder
    out_paths = iter_images(input_dir=output_dir, exts=exts)
    out_by_class = count_by_class(out_paths, prefix_len=args.prefix_len)
    min_per_class = min(len(v) for v in out_by_class.values())

    print(f"\nBalancing in-place (output): min_per_class={min_per_class}")

    removed_total = 0
    kept_total = 0
    for cid, label in enumerate(CLASS_LABELS):
        candidates = list(out_by_class[cid])
        rng.shuffle(candidates)

        keep = candidates[:min_per_class]
        remove = candidates[min_per_class:]

        for p in remove:
            p.unlink(missing_ok=True)

        kept_total += len(keep)
        removed_total += len(remove)
        print(f"  class {cid} ({label}): kept={len(keep):4d} removed={len(remove):4d}")

    print(f"\nWrote: {kept_total} images -> {output_dir}")
    print(f"Removed: {removed_total} extra images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
