#!/usr/bin/env python
"""
Create road_v4: a 30,000-image BiSeNetV2 semantic segmentation dataset.

Source:
  ai/scripts/road_v2/c_1280_720_daylight_train_1
  ai/scripts/road_v2/c_1280_720_daylight_train_2
  ai/scripts/road_v2/[label]... matching folders

Output:
  ai/scripts/road_v4/
    train/images, train/labels, train/masks
    val/images,   val/labels,   val/masks
    test/images,  test/labels,  test/masks

Class IDs:
  0 background
  1 lane_white
  2 lane_yellow
  3 lane_blue
  4 crosswalk
  5 stop_line

Lane type is intentionally ignored. Solid and dotted lanes are merged by color.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


CLASS_NAMES = [
    "background",
    "lane_white",
    "lane_yellow",
    "lane_blue",
    "crosswalk",
    "stop_line",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
BALANCE_CLASSES = ["lane_white", "lane_yellow", "lane_blue", "crosswalk", "stop_line"]
LANE_COLOR_TO_CLASS = {
    "white": "lane_white",
    "yellow": "lane_yellow",
    "blue": "lane_blue",
}


@dataclass
class Record:
    uid: str
    image_path: Path
    label_path: Path
    counts: dict[str, int]

    @property
    def present(self) -> tuple[str, ...]:
        return tuple(cls for cls in BALANCE_CLASSES if self.counts.get(cls, 0) > 0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build road_v4 BiSeNetV2 dataset.")
    ap.add_argument("--source-root", type=Path, default=Path("ai/scripts/road_v2"))
    ap.add_argument("--output-root", type=Path, default=Path("ai/scripts/road_v4"))
    ap.add_argument("--target-count", type=int, default=30000)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lane-thickness", type=int, default=12)
    ap.add_argument("--stop-line-thickness", type=int, default=14)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def find_label_dir(source_root: Path, image_dir: Path) -> Path:
    for child in source_root.iterdir():
        if child.is_dir() and child.name.startswith("[") and child.name.endswith(image_dir.name):
            return child
    raise FileNotFoundError(f"Label folder for {image_dir.name} was not found.")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def lane_color(ann: dict[str, Any]) -> str | None:
    for attr in ann.get("attributes", []):
        if attr.get("code") == "lane_color":
            value = attr.get("value")
            return str(value).lower() if value is not None else None
    return None


def annotation_counts(label_path: Path) -> dict[str, int]:
    data = read_json(label_path)
    counts: Counter[str] = Counter()
    for ann in data.get("annotations", []):
        cls = ann.get("class")
        if cls == "traffic_lane":
            mapped = LANE_COLOR_TO_CLASS.get(lane_color(ann) or "")
            if mapped:
                counts[mapped] += 1
        elif cls == "crosswalk":
            counts["crosswalk"] += 1
        elif cls == "stop_line":
            counts["stop_line"] += 1
    return {cls: int(counts.get(cls, 0)) for cls in BALANCE_CLASSES}


def collect_records(source_root: Path) -> list[Record]:
    image_dirs = [
        source_root / "c_1280_720_daylight_train_1",
        source_root / "c_1280_720_daylight_train_2",
    ]
    records: list[Record] = []

    for image_dir in image_dirs:
        label_dir = find_label_dir(source_root, image_dir)
        label_files = sorted(label_dir.glob("*.json"))
        prefix = image_dir.name.replace("c_1280_720_daylight_", "")
        for label_path in tqdm(label_files, desc=f"scan {label_dir.name}"):
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                continue
            counts = annotation_counts(label_path)
            if not any(counts.values()):
                continue
            uid = f"{prefix}_{label_path.stem}"
            records.append(Record(uid=uid, image_path=image_path, label_path=label_path, counts=counts))
    return records


def sum_counts(records: list[Record]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        totals.update(record.counts)
    return {cls: int(totals.get(cls, 0)) for cls in BALANCE_CLASSES}


def presence_counts(records: list[Record]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        for cls in record.present:
            totals[cls] += 1
    return {cls: int(totals.get(cls, 0)) for cls in BALANCE_CLASSES}


def select_balanced_records(records: list[Record], target_count: int, seed: int) -> list[Record]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    buckets: dict[str, list[Record]] = {cls: [] for cls in BALANCE_CLASSES}
    for record in shuffled:
        other_sum = sum(record.counts.values())
        for cls in BALANCE_CLASSES:
            if record.counts.get(cls, 0) > 0:
                buckets[cls].append(record)

    for cls in BALANCE_CLASSES:
        buckets[cls].sort(
            key=lambda r: (
                r.counts.get(cls, 0) / max(1, sum(v for k, v in r.counts.items() if k != cls)),
                r.counts.get(cls, 0),
                rng.random(),
            ),
            reverse=True,
        )

    selected: list[Record] = []
    selected_ids: set[str] = set()
    selected_counts: Counter[str] = Counter()
    bucket_idx = {cls: 0 for cls in BALANCE_CLASSES}
    exhausted: set[str] = set()

    pbar = tqdm(total=target_count, desc="select balanced")
    while len(selected) < target_count and len(exhausted) < len(BALANCE_CLASSES):
        available = [cls for cls in BALANCE_CLASSES if cls not in exhausted]
        cls = min(available, key=lambda name: selected_counts.get(name, 0))
        bucket = buckets[cls]
        idx = bucket_idx[cls]
        chosen: Record | None = None

        while idx < len(bucket):
            candidate = bucket[idx]
            idx += 1
            if candidate.uid in selected_ids:
                continue
            chosen = candidate
            break

        bucket_idx[cls] = idx
        if chosen is None:
            exhausted.add(cls)
            continue

        selected.append(chosen)
        selected_ids.add(chosen.uid)
        selected_counts.update(chosen.counts)
        pbar.update(1)

    if len(selected) < target_count:
        remaining = [record for record in shuffled if record.uid not in selected_ids]
        rng.shuffle(remaining)
        need = target_count - len(selected)
        selected.extend(remaining[:need])
        pbar.update(min(need, len(remaining)))

    pbar.close()
    rng.shuffle(selected)
    return selected[:target_count]


def split_records(
    records: list[Record],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Record]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[Record]] = defaultdict(list)
    for record in records:
        groups[record.present].append(record)

    splits = {"train": [], "val": [], "test": []}
    for group_records in groups.values():
        rng.shuffle(group_records)
        n = len(group_records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(group_records[:n_train])
        splits["val"].extend(group_records[n_train : n_train + n_val])
        splits["test"].extend(group_records[n_train + n_val :])

    for split_records_ in splits.values():
        rng.shuffle(split_records_)

    return splits


def points_array(ann: dict[str, Any]) -> np.ndarray | None:
    pts = []
    for point in ann.get("data", []):
        try:
            pts.append([int(round(float(point["x"]))), int(round(float(point["y"])))])
        except (KeyError, TypeError, ValueError):
            continue
    if len(pts) < 2:
        return None
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def draw_mask(
    label_path: Path,
    image_shape: tuple[int, int],
    lane_thickness: int,
    stop_line_thickness: int,
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    data = read_json(label_path)
    annotations = data.get("annotations", [])

    # Broad filled regions first.
    for ann in annotations:
        if ann.get("class") != "crosswalk":
            continue
        pts = points_array(ann)
        if pts is not None and len(pts) >= 3:
            cv2.fillPoly(mask, [pts], color=CLASS_TO_ID["crosswalk"])

    # Thin lane markings by color only.
    for ann in annotations:
        if ann.get("class") != "traffic_lane":
            continue
        mapped = LANE_COLOR_TO_CLASS.get(lane_color(ann) or "")
        if not mapped:
            continue
        pts = points_array(ann)
        if pts is not None:
            cv2.polylines(mask, [pts], isClosed=False, color=CLASS_TO_ID[mapped], thickness=lane_thickness)

    # Stop line gets the highest priority among road markings.
    for ann in annotations:
        if ann.get("class") != "stop_line":
            continue
        pts = points_array(ann)
        if pts is not None:
            cv2.polylines(mask, [pts], isClosed=False, color=CLASS_TO_ID["stop_line"], thickness=stop_line_thickness)

    return mask


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def materialize_record(
    record: Record,
    split: str,
    output_root: Path,
    copy_mode: str,
    lane_thickness: int,
    stop_line_thickness: int,
) -> str:
    split_root = output_root / split
    image_out = split_root / "images" / f"{record.uid}.jpg"
    label_out = split_root / "labels" / f"{record.uid}.json"
    mask_out = split_root / "masks" / f"{record.uid}.png"

    link_or_copy(record.image_path, image_out, copy_mode)
    shutil.copy2(record.label_path, label_out)

    image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {record.image_path}")
    h, w = image.shape[:2]
    mask = draw_mask(record.label_path, (h, w), lane_thickness, stop_line_thickness)
    ok = cv2.imwrite(str(mask_out), mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Failed to write mask: {mask_out}")
    return record.uid


def ensure_output_dirs(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(output_root)
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels", "masks"]:
            (output_root / split / sub).mkdir(parents=True, exist_ok=True)


def write_split_lists(splits: dict[str, list[Record]], output_root: Path) -> None:
    for split, records in splits.items():
        with (output_root / f"{split}.txt").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(f"{record.uid}\n")


def split_stats(splits: dict[str, list[Record]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split, records in splits.items():
        stats[split] = {
            "images": len(records),
            "annotation_counts": sum_counts(records),
            "image_presence_counts": presence_counts(records),
        }
    stats["total"] = {
        "images": sum(len(records) for records in splits.values()),
        "annotation_counts": sum_counts([r for records in splits.values() for r in records]),
        "image_presence_counts": presence_counts([r for records in splits.values() for r in records]),
    }
    return stats


def write_metadata(
    output_root: Path,
    args: argparse.Namespace,
    all_records: list[Record],
    selected: list[Record],
    splits: dict[str, list[Record]],
) -> None:
    metadata = {
        "dataset": "road_v4_bisenetv2_color",
        "source_root": str(args.source_root.resolve()),
        "target_count": args.target_count,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "class_map": {name: idx for idx, name in enumerate(CLASS_NAMES)},
        "lane_rule": "traffic_lane uses lane_color only; lane_type solid/dotted is ignored",
        "mask_format": "single-channel uint8 PNG, same resolution as source image",
        "draw_priority": ["crosswalk", "lane_color", "stop_line"],
        "line_thickness": {
            "lane": args.lane_thickness,
            "stop_line": args.stop_line_thickness,
        },
        "all_source_stats": {
            "images": len(all_records),
            "annotation_counts": sum_counts(all_records),
            "image_presence_counts": presence_counts(all_records),
        },
        "selected_stats": {
            "images": len(selected),
            "annotation_counts": sum_counts(selected),
            "image_presence_counts": presence_counts(selected),
        },
        "split_stats": split_stats(splits),
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def materialize_dataset(splits: dict[str, list[Record]], args: argparse.Namespace) -> None:
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for split, records in splits.items():
            for record in records:
                futures.append(
                    executor.submit(
                        materialize_record,
                        record,
                        split,
                        args.output_root,
                        args.copy_mode,
                        args.lane_thickness,
                        args.stop_line_thickness,
                    )
                )

        for future in tqdm(as_completed(futures), total=len(futures), desc="write files"):
            future.result()


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    random.seed(args.seed)

    print(f"[source] {args.source_root}")
    print(f"[output] {args.output_root}")
    ensure_output_dirs(args.output_root, args.overwrite)

    all_records = collect_records(args.source_root)
    print(f"[scan] usable images: {len(all_records):,}")
    print(f"[scan] annotation counts: {sum_counts(all_records)}")
    print(f"[scan] image presence counts: {presence_counts(all_records)}")

    selected = select_balanced_records(all_records, args.target_count, args.seed)
    print(f"[select] selected images: {len(selected):,}")
    print(f"[select] annotation counts: {sum_counts(selected)}")
    print(f"[select] image presence counts: {presence_counts(selected)}")

    splits = split_records(selected, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    for split, records in splits.items():
        print(f"[split] {split}: {len(records):,} {sum_counts(records)}")

    materialize_dataset(splits, args)
    write_split_lists(splits, args.output_root)
    write_metadata(args.output_root, args, all_records, selected, splits)
    print("[done] road_v4 dataset is ready.")


if __name__ == "__main__":
    main()
