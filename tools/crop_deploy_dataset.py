#!/usr/bin/env python3
"""Crop a single bi-temporal change-detection triplet into LCD-Net layout.

Input:
  T1 image, T2 image, and optional binary/semantic mask.

Output:
  output_root/region_id/
    train/time1/*.png
    train/time2/*.png
    train/label/*.png
    val/time1/*.png
    val/time2/*.png
    val/label/*.png
    test/time1/*.png
    test/time2/*.png
    test/label/*.png

This script is intentionally conservative: it does not invent labels. If no mask
is supplied, it writes empty label tiles so the folder structure can be
inspected, but that output should not be used for supervised training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop deployment images into 256x256 LCD-Net tiles.")
    parser.add_argument("--t1", required=True, help="Path to before-time image.")
    parser.add_argument("--t2", required=True, help="Path to after-time image.")
    parser.add_argument("--mask", default="", help="Path to GT/change mask. Optional.")
    parser.add_argument("--output_root", default="/media/dataset/deploy_uploads")
    parser.add_argument("--region_id", default="", help="Output region id. Defaults to image stem.")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=70.0)
    parser.add_argument("--val_ratio", type=float, default=15.0)
    parser.add_argument("--test_ratio", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask_threshold", type=int, default=0)
    parser.add_argument("--include_empty", action="store_true", help="Keep tiles whose mask is empty.")
    parser.add_argument("--dir_a", default="time1")
    parser.add_argument("--dir_b", default="time2")
    parser.add_argument("--dir_label", default="label")
    return parser.parse_args()


def open_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def open_mask(path: Path | None, size: tuple[int, int], threshold: int) -> Image.Image:
    if not path:
        return Image.new("L", size, 0)
    if not path.exists():
        raise FileNotFoundError(path)
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda px: 255 if px > threshold else 0, mode="L")


def axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def tile_boxes(width: int, height: int, tile_size: int, stride: int) -> Iterable[tuple[int, int, int, int]]:
    xs = axis_positions(width, tile_size, stride)
    ys = axis_positions(height, tile_size, stride)
    for y in ys:
        for x in xs:
            yield (x, y, x + tile_size, y + tile_size)


def padded_crop(image: Image.Image, box: tuple[int, int, int, int], mode: str) -> Image.Image:
    tile = Image.new(mode, (box[2] - box[0], box[3] - box[1]), 0)
    crop = image.crop((box[0], box[1], min(box[2], image.width), min(box[3], image.height)))
    tile.paste(crop, (0, 0))
    return tile


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(round(total * train_ratio / 100.0))
    val_end = train_end + int(round(total * val_ratio / 100.0))
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def ensure_dirs(root: Path, splits: Iterable[str], dir_a: str, dir_b: str, dir_label: str) -> None:
    for split in splits:
        for folder in (dir_a, dir_b, dir_label):
            (root / split / folder).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    t1_path = Path(args.t1).expanduser()
    t2_path = Path(args.t2).expanduser()
    mask_path = Path(args.mask).expanduser() if args.mask else None

    t1 = open_rgb(t1_path)
    t2 = open_rgb(t2_path)
    if t2.size != t1.size:
        t2 = t2.resize(t1.size, Image.Resampling.BILINEAR)
    mask = open_mask(mask_path, t1.size, args.mask_threshold)

    region_id = args.region_id.strip() or t1_path.stem
    out_root = Path(args.output_root).expanduser() / region_id
    ensure_dirs(out_root, ("train", "val", "test"), args.dir_a, args.dir_b, args.dir_label)

    boxes = list(tile_boxes(t1.width, t1.height, args.tile_size, args.stride))
    random.Random(args.seed).shuffle(boxes)
    kept: list[dict[str, object]] = []

    for box in boxes:
        mask_tile = padded_crop(mask, box, "L")
        changed_px = int(sum(1 for px in mask_tile.getdata() if px > 0))
        if changed_px == 0 and (mask_path is not None) and not args.include_empty:
            continue
        kept.append({"box": box, "changed_px": changed_px})

    total = len(kept)
    manifest_rows = []
    for index, item in enumerate(kept):
        split = split_name(index, total, args.train_ratio, args.val_ratio)
        name = f"{region_id}_{index:06d}.png"
        box = tuple(item["box"])  # type: ignore[arg-type]
        t1_tile = padded_crop(t1, box, "RGB")
        t2_tile = padded_crop(t2, box, "RGB")
        mask_tile = padded_crop(mask, box, "L")

        t1_tile.save(out_root / split / args.dir_a / name)
        t2_tile.save(out_root / split / args.dir_b / name)
        mask_tile.save(out_root / split / args.dir_label / name)
        manifest_rows.append(
            {
                "sample": name,
                "split": split,
                "box": list(box),
                "changed_px": item["changed_px"],
                "t1": str(out_root / split / args.dir_a / name),
                "t2": str(out_root / split / args.dir_b / name),
                "label": str(out_root / split / args.dir_label / name),
            }
        )

    manifest_path = out_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_counts = {split: 0 for split in ("train", "val", "test")}
    changed_tiles = 0
    review_tiles = 0
    total_changed_px = 0
    for row in manifest_rows:
        split_counts[str(row["split"])] += 1
        changed_px = int(row.get("changed_px") or 0)
        total_changed_px += changed_px
        if changed_px > 0:
            changed_tiles += 1
        # Lightweight deployment triage: tiny positive tiles are often weak o
        # boundary fragments and are reasonable first-pass review candidates.
        if 0 < changed_px < max(16, args.tile_size * args.tile_size * 0.005):
            review_tiles += 1
    metadata = {
        "region_id": region_id,
        "output_dir": str(out_root),
        "manifest": str(manifest_path),
        "source": {"t1": str(t1_path), "t2": str(t2_path), "mask": str(mask_path) if mask_path else ""},
        "source_size": [t1.width, t1.height],
        "tile_size": args.tile_size,
        "stride": args.stride,
        "include_empty": bool(args.include_empty),
        "total_tiles": total,
        "changed_tiles": changed_tiles,
        "review_tiles": review_tiles,
        "total_changed_px": total_changed_px,
        "split_counts": split_counts,
        "dir_a": args.dir_a,
        "dir_b": args.dir_b,
        "dir_label": args.dir_label,
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
