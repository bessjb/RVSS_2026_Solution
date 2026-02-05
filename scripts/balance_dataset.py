#!/usr/bin/env python3

"""Create a single balanced folder from steering-labeled images.

Optionally, also balance by a scene label (e.g. urban/outback) provided by a CSV
manifest (default: <input>/urban_outback_labels.csv). In that mode, images are
downsampled so each (steering_class, env_label) group has the same count.

Simple approach:
1) Copy ALL images into a new folder with suffix "_balanced".
2) Compute group counts in that new folder.
3) For any group with more than the minimum group count, randomly delete extra
    images until all groups match.

Assumptions (matches scripts/steerDS.py and scripts/make_balanced_val_test.py):
- Steering value is encoded in the filename stem after a fixed prefix length.
  Example: "000123-0.25.jpg" with --prefix-len 6 => "-0.25".
- Classes are derived from steering using the same thresholds.

No train/val/test split is performed: output is a single folder.

If --split-ratio is provided, the script will instead create two balanced
folders (<input>_balanced_train and <input>_balanced_val) by splitting each
group (class or class+env) into train/val with the same per-group counts.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


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


def load_env_labels(csv_path: Path, *, input_dir: Path) -> Dict[str, str]:
    """Load mapping from image relative path -> env label.

    The CSV is expected to have columns: path,label
    where "path" is relative to input_dir.
    """

    labels: Dict[str, str] = {}
    if not csv_path.exists():
        return labels

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return labels

        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"Env CSV must contain columns 'path' and 'label'. Found: {reader.fieldnames!r}"
            )

        for row in reader:
            raw_path = (row.get("path") or "").strip()
            raw_label = (row.get("label") or "").strip()
            if not raw_path:
                continue

            # Normalize path: accept absolute paths too.
            p = Path(raw_path)
            try:
                if p.is_absolute():
                    rel = p.resolve().relative_to(input_dir).as_posix()
                else:
                    rel = Path(raw_path).as_posix().lstrip("./")
            except Exception:
                rel = Path(raw_path).as_posix().lstrip("./")

            labels[rel] = raw_label

    return labels


def env_label_for_path(
    image_path: Path,
    *,
    input_dir: Path,
    env_labels: Dict[str, str],
    allowed: Sequence[str],
    allow_missing: bool,
) -> str:
    rel = image_path.relative_to(input_dir).as_posix() if image_path.is_absolute() else image_path.name

    # Try both (relative path) and (just filename) to be forgiving.
    label = env_labels.get(rel)
    if label is None:
        label = env_labels.get(Path(rel).name)

    if not label:
        if allow_missing:
            return "unlabeled"
        raise ValueError(
            f"Missing env label for image {image_path.name!r}. "
            f"Expected a row in CSV with path={rel!r} (or {Path(rel).name!r})."
        )

    if label not in set(allowed):
        raise ValueError(
            f"Invalid env label {label!r} for image {image_path.name!r}. Allowed: {list(allowed)!r}"
        )

    return label


def count_by_class_and_env(
    paths: Sequence[Path],
    *,
    prefix_len: int,
    input_dir: Path,
    env_labels: Dict[str, str],
    allowed_env: Sequence[str],
    allow_missing_env: bool,
) -> Dict[Tuple[int, str], List[Path]]:
    by_group: Dict[Tuple[int, str], List[Path]] = {}
    for cid in range(len(CLASS_LABELS)):
        for env in list(allowed_env) + (["unlabeled"] if allow_missing_env else []):
            by_group[(cid, env)] = []

    for p in paths:
        steering = parse_steering_from_path(p, prefix_len=prefix_len)
        cid = steering_to_class(steering)
        env = env_label_for_path(
            p,
            input_dir=input_dir,
            env_labels=env_labels,
            allowed=allowed_env,
            allow_missing=allow_missing_env,
        )
        by_group[(cid, env)].append(p)

    return by_group


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
        description=(
            "Create a balanced folder by downsampling groups to the minimum group count. "
            "Groups are either steering class only, or (steering class, env label) if an env CSV is provided."
        )
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
    parser.add_argument(
        "--env-balance",
        choices=["auto", "none", "combined"],
        default="auto",
        help=(
            "How to use env labels from CSV. "
            "auto: use combined mode if CSV exists; none: ignore env labels; combined: balance by (steering_class, env_label)."
        ),
    )
    parser.add_argument(
        "--env-csv",
        default=None,
        help=(
            "CSV with columns path,label for env labels. "
            "Default (when --env-balance=auto): <input>/urban_outback_labels.csv if it exists."
        ),
    )
    parser.add_argument(
        "--env-values",
        default="urban,outback",
        help="Comma-separated allowed env labels (default: urban,outback)",
    )
    parser.add_argument(
        "--allow-missing-env",
        action="store_true",
        help="Allow missing/blank env labels; such images go into an 'unlabeled' bucket (not recommended).",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=None,
        help=(
            "Optional train split ratio in (0,1). Example: 0.8. "
            "If set, writes <input>_balanced_train and <input>_balanced_val with balanced per-group counts."
        ),
    )

    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_dir}")

    exts = [e.strip() for e in str(args.ext).split(",") if e.strip()]
    if not exts:
        raise SystemExit("No extensions provided via --ext")

    allowed_env = [e.strip() for e in str(args.env_values).split(",") if e.strip()]
    if not allowed_env:
        raise SystemExit("No env labels provided via --env-values")

    # Always create a sibling output folder with suffix "_balanced".
    # Filenames are NOT modified.
    output_dir = input_dir.parent / f"{input_dir.name}_balanced"
    if output_dir == input_dir:
        raise SystemExit("Output folder resolves to the input folder; refusing to modify the input dataset.")
    print(f"Output folder: {output_dir}")

    paths = iter_images(input_dir=input_dir, exts=exts)
    if not paths:
        raise SystemExit(f"No images found under {input_dir} with extensions: {exts}")

    # Env CSV selection
    default_env_csv = input_dir / "urban_outback_labels.csv"
    env_csv = Path(args.env_csv).expanduser().resolve() if args.env_csv else default_env_csv
    env_mode = args.env_balance
    if env_mode == "auto":
        env_mode = "combined" if env_csv.exists() else "none"

    env_labels: Dict[str, str] = {}
    if env_mode != "none":
        env_labels = load_env_labels(env_csv, input_dir=input_dir)
        if not env_labels:
            raise SystemExit(
                f"Env balancing requested but no labels loaded from: {env_csv}. "
                "Check the CSV path and that it has 'path' and 'label' columns."
            )

    if env_mode == "none":
        by_class = count_by_class(paths, prefix_len=args.prefix_len)
        print("Input counts per class:")
        for cid, label in enumerate(CLASS_LABELS):
            print(f"  {cid}: {label:11s} -> {len(by_class[cid])}")

        min_per_group = min(len(v) for v in by_class.values())
        if min_per_group == 0:
            raise SystemExit(
                "At least one class has 0 images, cannot create a balanced folder. "
                "Collect more data for the missing classes."
            )
    else:
        print(f"Env CSV: {env_csv}")
        by_group = count_by_class_and_env(
            paths,
            prefix_len=args.prefix_len,
            input_dir=input_dir,
            env_labels=env_labels,
            allowed_env=allowed_env,
            allow_missing_env=bool(args.allow_missing_env),
        )

        print("Input counts per (class, env):")
        for cid, class_name in enumerate(CLASS_LABELS):
            for env in list(allowed_env) + (["unlabeled"] if args.allow_missing_env else []):
                print(f"  class {cid} ({class_name:11s}) env={env:9s} -> {len(by_group[(cid, env)])}")

        min_per_group = min(len(v) for v in by_group.values())
        if min_per_group == 0:
            raise SystemExit(
                "At least one (class, env) group has 0 images, cannot create a balanced folder with env balancing. "
                "Collect more data for the missing groups or run with --env-balance=none."
            )

    rng = random.Random(args.seed)
    ensure_empty_dir(output_dir, overwrite=args.overwrite)

    # 1) Copy everything first (keep same filenames)
    for src in paths:
        shutil.copy2(str(src), str(output_dir / src.name))

    # 2) Recompute class lists for files in output folder
    out_paths = iter_images(input_dir=output_dir, exts=exts)
    removed_total = 0
    kept_total = 0

    if env_mode == "none":
        out_by_class = count_by_class(out_paths, prefix_len=args.prefix_len)
        min_per_group = min(len(v) for v in out_by_class.values())
        print(f"\nBalancing in-place (output): min_per_class={min_per_group}")

        for cid, label in enumerate(CLASS_LABELS):
            candidates = list(out_by_class[cid])
            rng.shuffle(candidates)

            keep = candidates[:min_per_group]
            remove = candidates[min_per_group:]

            for p in remove:
                p.unlink(missing_ok=True)

            kept_total += len(keep)
            removed_total += len(remove)
            print(f"  class {cid} ({label}): kept={len(keep):4d} removed={len(remove):4d}")
    else:
        # Build env mapping for output files. Output folder is a sibling, so relpaths are just filenames.
        out_env_labels = env_labels
        out_by_group = count_by_class_and_env(
            out_paths,
            prefix_len=args.prefix_len,
            input_dir=output_dir,
            env_labels=out_env_labels,
            allowed_env=allowed_env,
            allow_missing_env=bool(args.allow_missing_env),
        )
        min_per_group = min(len(v) for v in out_by_group.values())
        print(f"\nBalancing in-place (output): min_per_group={min_per_group} (by class+env)")

        for cid, class_name in enumerate(CLASS_LABELS):
            for env in list(allowed_env) + (["unlabeled"] if args.allow_missing_env else []):
                candidates = list(out_by_group[(cid, env)])
                rng.shuffle(candidates)

                keep = candidates[:min_per_group]
                remove = candidates[min_per_group:]

                for p in remove:
                    p.unlink(missing_ok=True)

                kept_total += len(keep)
                removed_total += len(remove)
                print(
                    f"  class {cid} ({class_name:11s}) env={env:9s}: kept={len(keep):4d} removed={len(remove):4d}"
                )

    # Optional train/val split (performed after balancing so each split stays balanced per-group)
    if args.split_ratio is not None:
        ratio = float(args.split_ratio)
        if not (0.0 < ratio < 1.0):
            raise SystemExit("--split-ratio must be in (0,1), e.g. 0.8")

        # Recompute groups for the balanced output folder.
        out_paths = iter_images(input_dir=output_dir, exts=exts)
        if env_mode == "none":
            groups: Dict[Tuple[int, str], List[Path]] = {}
            out_by_class = count_by_class(out_paths, prefix_len=args.prefix_len)
            for cid in range(len(CLASS_LABELS)):
                groups[(cid, "__na__")] = list(out_by_class[cid])
        else:
            out_by_group = count_by_class_and_env(
                out_paths,
                prefix_len=args.prefix_len,
                input_dir=output_dir,
                env_labels=env_labels,
                allowed_env=allowed_env,
                allow_missing_env=bool(args.allow_missing_env),
            )
            groups = {k: list(v) for k, v in out_by_group.items()}

        per_group_total = min(len(v) for v in groups.values()) if groups else 0
        train_n = int(per_group_total * ratio)
        val_n = per_group_total - train_n
        if train_n <= 0 or val_n <= 0:
            raise SystemExit(
                f"Split would be empty for at least one split: per_group_total={per_group_total}, "
                f"train_n={train_n}, val_n={val_n}. Adjust --split-ratio or collect more data."
            )

        train_dir = input_dir.parent / f"{input_dir.name}_balanced_train"
        val_dir = input_dir.parent / f"{input_dir.name}_balanced_val"
        ensure_empty_dir(train_dir, overwrite=args.overwrite)
        ensure_empty_dir(val_dir, overwrite=args.overwrite)

        moved_train = 0
        moved_val = 0
        for key, candidates in groups.items():
            # Defensive: groups should already be balanced, but don't assume ordering.
            rng.shuffle(candidates)
            selected = candidates[:per_group_total]
            train_sel = selected[:train_n]
            val_sel = selected[train_n:train_n + val_n]

            for p in train_sel:
                shutil.move(str(p), str(train_dir / p.name))
            for p in val_sel:
                shutil.move(str(p), str(val_dir / p.name))

            moved_train += len(train_sel)
            moved_val += len(val_sel)

        # Remove the intermediate combined balanced folder if it is now empty.
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()

        print(f"\nSplit complete: train_per_group={train_n}, val_per_group={val_n}")
        print(f"Wrote train: {moved_train} images -> {train_dir}")
        print(f"Wrote val:   {moved_val} images -> {val_dir}")
        return 0

    print(f"\nWrote: {kept_total} images -> {output_dir}")
    print(f"Removed: {removed_total} extra images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
