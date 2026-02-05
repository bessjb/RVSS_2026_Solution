#!/usr/bin/env python3

"""Interactive image labeler: urban vs outback.

Shows images from a folder one-by-one and lets you label them quickly.

Keys (while the OpenCV window is focused)
- Enter: label as "urban" and advance
- Space: label as "outback" and advance
- q / Esc: quit (progress is saved)

Output
- Writes a CSV named "urban_outback_labels.csv" in the input folder by default.
- CSV columns: path,label
  - path: image path relative to the input folder
  - label: urban|outback (blank if unlabeled)

Resume
- If the CSV already exists, it is loaded so you can resume labeling.

Usage
  python3 scripts/label_urban_outback.py data/repo1
  python3 scripts/label_urban_outback.py data/repo1 --recursive
  python3 scripts/label_urban_outback.py data/repo1 --out-csv data/repo1/urban_outback_labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "OpenCV (cv2) is required. If you use pixi, ensure the pixi env is active (py-opencv). "
        "Or install via pip: `pip install opencv-python`."
    ) from e


DEFAULT_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_exts(ext_csv: str) -> Tuple[str, ...]:
    parts = [p.strip() for p in ext_csv.split(",") if p.strip()]
    if not parts:
        return DEFAULT_EXTS
    out: List[str] = []
    for p in parts:
        out.append(p.lower() if p.startswith(".") else f".{p.lower()}")
    return tuple(out)


def iter_images(root: Path, *, exts: Sequence[str], recursive: bool) -> List[Path]:
    exts_lc = {e.lower() for e in exts}
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_lc]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts_lc]
    return sorted(paths)


def relpath_posix(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def load_labels(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        return {}

    labels: Dict[str, str] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return labels
        for row in reader:
            rp = (row.get("path") or "").strip()
            lab = (row.get("label") or "").strip()
            if rp:
                labels[rp] = lab
    return labels


def save_labels(csv_path: Path, image_relpaths: Sequence[str], labels: Dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        for rp in image_relpaths:
            writer.writerow({"path": rp, "label": labels.get(rp, "")})


def resize_to_fit(image, *, max_w: int, max_h: int):
    h, w = image.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale >= 1.0:
        return image
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def overlay_text(image, lines: List[str]):
    out = image.copy()
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    rect_w = max(s[0] for s in sizes) + 2 * pad
    rect_h = (len(lines) * (sizes[0][1] + pad)) + pad

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    y = pad + sizes[0][1]
    for line in lines:
        cv2.putText(out, line, (pad, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += sizes[0][1] + pad

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Label images as urban/outback and save to CSV")
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: <folder>/urban_outback_labels.csv)",
    )
    parser.add_argument(
        "--ext",
        default=",".join(DEFAULT_EXTS),
        help="Comma-separated image extensions to include",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=720)
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {root}")

    exts = parse_exts(args.ext)
    csv_path = Path(args.out_csv).expanduser().resolve() if args.out_csv else (root / "urban_outback_labels.csv")

    image_paths = iter_images(root, exts=exts, recursive=args.recursive)
    if not image_paths:
        raise SystemExit(f"No images found in {root} (recursive={args.recursive})")

    image_relpaths = [relpath_posix(root, p) for p in image_paths]
    labels = load_labels(csv_path)

    # Ensure CSV exists so you can see where it is writing.
    save_labels(csv_path, image_relpaths, labels)

    window = "Urban/Outback Labeler"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    idx = 0
    while idx < len(image_paths):
        p = image_paths[idx]
        rp = image_relpaths[idx]

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            idx += 1
            continue

        img = resize_to_fit(img, max_w=args.max_width, max_h=args.max_height)
        cur = labels.get(rp, "")

        disp = overlay_text(
            img,
            [
                f"[{idx+1}/{len(image_paths)}] {rp}",
                f"Current label: {cur if cur else '(unlabeled)'}",
                "Enter=urban | Space=outback | q/Esc=quit",
            ],
        )
        cv2.imshow(window, disp)

        key = int(cv2.waitKey(0))
        key = key & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (13, 10):
            labels[rp] = "urban"
            save_labels(csv_path, image_relpaths, labels)
            idx += 1
            continue
        if key == 32:
            labels[rp] = "outback"
            save_labels(csv_path, image_relpaths, labels)
            idx += 1
            continue

        # Unknown key: ignore and stay on same image

    cv2.destroyAllWindows()
    save_labels(csv_path, image_relpaths, labels)

    labeled_n = sum(1 for rp in image_relpaths if labels.get(rp, "") in ("urban", "outback"))
    print(f"Saved: {csv_path}")
    print(f"Labeled: {labeled_n}/{len(image_relpaths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
