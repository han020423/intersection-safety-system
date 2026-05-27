#!/usr/bin/env python
"""
Build a color-only UFLDv2-style lane dataset from road_v3 labels.

Class mapping:
  0 = none
  1 = white
  2 = yellow
  3 = blue

The converter intentionally ignores lane_type (solid/dotted). It also merges
compatible same-color lane segments before row-anchor conversion so dotted
lanes are not fragmented into many sparse training targets.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


ORIG_H = 720
ORIG_W = 1280
NUM_ANCHORS = 36
GRID_NUM = 64
MAX_LANES = 8
H_SAMPLES = [int(i * ORIG_H / NUM_ANCHORS) for i in range(NUM_ANCHORS)]

COLOR_TO_CAT = {
    "white": 1,
    "yellow": 2,
    "blue": 3,
}
CAT_NAMES = {
    0: "none",
    1: "white",
    2: "yellow",
    3: "blue",
}


@dataclass
class LaneSegment:
    category: int
    xs: list[int]
    source_points: list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare road_v3 labels as a color-only UFLDv2 dataset."
    )
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "road_v3_ufldv2_color",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--limit-per-folder", type=int, default=0)
    parser.add_argument("--target-count", type=int, default=0)
    parser.add_argument("--balance-colors", action="store_true")
    parser.add_argument("--split-image-folders", action="store_true")
    parser.add_argument("--min-valid-anchors", type=int, default=4)
    parser.add_argument("--extrapolate-px", type=float, default=12.0)
    parser.add_argument("--merge-segments", action="store_true", default=True)
    parser.add_argument("--no-merge-segments", dest="merge_segments", action="store_false")
    parser.add_argument("--merge-overlap-px", type=float, default=18.0)
    parser.add_argument("--merge-gap-px", type=float, default=28.0)
    parser.add_argument("--merge-gap-anchors", type=int, default=4)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_image_dirs(data_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("["):
            continue
        if any(child.glob("*.jpg")) or any(child.glob("*.png")):
            dirs.append(child)
    return dirs


def find_label_dir(data_root: Path, image_dir: Path) -> Path | None:
    exact = data_root / f"[라벨]{image_dir.name}"
    if exact.exists():
        return exact
    for child in data_root.iterdir():
        if child.is_dir() and child.name.startswith("[") and child.name.endswith(image_dir.name):
            return child
    return None


def get_attr(attrs: list[dict[str, Any]], code: str) -> str | None:
    for attr in attrs:
        if attr.get("code") == code:
            value = attr.get("value")
            return str(value).lower() if value is not None else None
    return None


def clean_points(raw_points: list[dict[str, Any]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in raw_points:
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        points.append((float(np.clip(x, 0, ORIG_W - 1)), float(np.clip(y, 0, ORIG_H - 1))))

    if len(points) < 2:
        return []

    points.sort(key=lambda p: (p[1], p[0]))
    merged: list[tuple[float, float]] = []
    cur_y = points[0][1]
    cur_xs = [points[0][0]]
    for x, y in points[1:]:
        if abs(y - cur_y) <= 1e-3:
            cur_xs.append(x)
        else:
            merged.append((float(np.mean(cur_xs)), cur_y))
            cur_y = y
            cur_xs = [x]
    merged.append((float(np.mean(cur_xs)), cur_y))
    return merged if len(merged) >= 2 else []


def polyline_to_anchors(
    points: list[tuple[float, float]],
    h_samples: list[int],
    extrapolate_px: float,
) -> list[int]:
    if len(points) < 2:
        return [-2] * len(h_samples)

    ys = np.array([p[1] for p in points], dtype=np.float32)
    xs = np.array([p[0] for p in points], dtype=np.float32)
    y_min = float(ys.min())
    y_max = float(ys.max())

    anchors: list[int] = []
    for y in h_samples:
        if y < y_min - extrapolate_px or y > y_max + extrapolate_px:
            anchors.append(-2)
            continue

        x = float(np.interp(float(y), ys, xs))
        if x < 0 or x >= ORIG_W:
            anchors.append(-2)
        else:
            anchors.append(int(round(x)))
    return anchors


def valid_indices(xs: list[int]) -> list[int]:
    return [idx for idx, x in enumerate(xs) if x != -2]


def valid_count(xs: list[int]) -> int:
    return len(valid_indices(xs))


def lane_mean_x(xs: list[int]) -> float:
    valid = [x for x in xs if x != -2]
    return float(np.mean(valid)) if valid else -1.0


def can_merge_segments(
    a: LaneSegment,
    b: LaneSegment,
    overlap_px: float,
    gap_px: float,
    gap_anchors: int,
) -> bool:
    if a.category != b.category:
        return False

    a_valid = set(valid_indices(a.xs))
    b_valid = set(valid_indices(b.xs))
    overlap = sorted(a_valid & b_valid)
    if overlap:
        dx = [abs(a.xs[idx] - b.xs[idx]) for idx in overlap]
        return float(np.median(dx)) <= overlap_px

    if not a_valid or not b_valid:
        return False

    best_gap = 10**9
    best_dx = 10**9
    for ia in a_valid:
        for ib in b_valid:
            gap = abs(ia - ib)
            if gap < best_gap:
                best_gap = gap
                best_dx = abs(a.xs[ia] - b.xs[ib])

    return best_gap <= gap_anchors and best_dx <= gap_px


def merge_two_segments(a: LaneSegment, b: LaneSegment) -> LaneSegment:
    merged_xs: list[int] = []
    for ax, bx in zip(a.xs, b.xs):
        if ax != -2 and bx != -2:
            merged_xs.append(int(round((ax + bx) / 2.0)))
        elif ax != -2:
            merged_xs.append(ax)
        else:
            merged_xs.append(bx)
    return LaneSegment(
        category=a.category,
        xs=merged_xs,
        source_points=a.source_points + b.source_points,
    )


def merge_segments(
    segments: list[LaneSegment],
    overlap_px: float,
    gap_px: float,
    gap_anchors: int,
) -> list[LaneSegment]:
    lanes = sorted(segments, key=lambda lane: (lane.category, lane_mean_x(lane.xs)))
    changed = True
    while changed:
        changed = False
        used = [False] * len(lanes)
        next_lanes: list[LaneSegment] = []
        for i, lane in enumerate(lanes):
            if used[i]:
                continue
            current = lane
            used[i] = True
            for j in range(i + 1, len(lanes)):
                if used[j]:
                    continue
                if can_merge_segments(current, lanes[j], overlap_px, gap_px, gap_anchors):
                    current = merge_two_segments(current, lanes[j])
                    used[j] = True
                    changed = True
            next_lanes.append(current)
        lanes = sorted(next_lanes, key=lambda lane: (lane.category, lane_mean_x(lane.xs)))
    return lanes


def parse_label_file(
    label_path: Path,
    min_valid_anchors: int,
    extrapolate_px: float,
    do_merge_segments: bool,
    merge_overlap_px: float,
    merge_gap_px: float,
    merge_gap_anchors: int,
) -> tuple[list[list[int]], list[int], dict[str, int]]:
    data = read_json(label_path)
    stats = {
        "raw_lane_annotations": 0,
        "unknown_color": 0,
        "too_few_points": 0,
        "too_few_anchors": 0,
        "merged_lanes": 0,
    }
    segments: list[LaneSegment] = []

    for ann in data.get("annotations", []):
        if ann.get("class") != "traffic_lane":
            continue
        if ann.get("category") != "polyline":
            continue

        stats["raw_lane_annotations"] += 1
        color = get_attr(ann.get("attributes", []), "lane_color")
        category = COLOR_TO_CAT.get(color or "")
        if category is None:
            stats["unknown_color"] += 1
            continue

        points = clean_points(ann.get("data", []))
        if len(points) < 2:
            stats["too_few_points"] += 1
            continue

        xs = polyline_to_anchors(points, H_SAMPLES, extrapolate_px)
        if valid_count(xs) < 2:
            stats["too_few_anchors"] += 1
            continue

        segments.append(LaneSegment(category=category, xs=xs, source_points=points))

    if do_merge_segments:
        before = len(segments)
        segments = merge_segments(segments, merge_overlap_px, merge_gap_px, merge_gap_anchors)
        stats["merged_lanes"] = max(0, before - len(segments))

    segments = [seg for seg in segments if valid_count(seg.xs) >= min_valid_anchors]
    if not segments:
        return [], [], stats

    segments = sorted(segments, key=lambda lane: lane_mean_x(lane.xs))
    if len(segments) > MAX_LANES:
        segments = sorted(segments, key=lambda lane: valid_count(lane.xs), reverse=True)[:MAX_LANES]
        segments = sorted(segments, key=lambda lane: lane_mean_x(lane.xs))

    lanes = [seg.xs for seg in segments]
    categories = [seg.category for seg in segments]
    return lanes, categories, stats


def collect_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dirs = iter_image_dirs(args.data_root)
    if not image_dirs:
        raise FileNotFoundError(f"No image folders found under {args.data_root}")

    records: list[dict[str, Any]] = []
    totals: dict[str, Any] = {
        "image_dirs": len(image_dirs),
        "images_seen": 0,
        "labels_missing": 0,
        "images_without_lanes": 0,
        "lanes_kept": 0,
        "category_counts": {name: 0 for name in CAT_NAMES.values() if name != "none"},
        "raw_lane_annotations": 0,
        "unknown_color": 0,
        "too_few_points": 0,
        "too_few_anchors": 0,
        "merged_lanes": 0,
    }

    for image_dir in image_dirs:
        label_dir = find_label_dir(args.data_root, image_dir)
        if label_dir is None:
            print(f"[WARN] no label folder for {image_dir.name}")
            continue

        images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        if args.limit_per_folder > 0:
            images = images[: args.limit_per_folder]

        for image_path in tqdm(images, desc=f"scan {image_dir.name}"):
            totals["images_seen"] += 1
            label_path = label_dir / f"{image_path.stem}.json"
            if not label_path.exists():
                totals["labels_missing"] += 1
                continue

            lanes, categories, stats = parse_label_file(
                label_path=label_path,
                min_valid_anchors=args.min_valid_anchors,
                extrapolate_px=args.extrapolate_px,
                do_merge_segments=args.merge_segments,
                merge_overlap_px=args.merge_overlap_px,
                merge_gap_px=args.merge_gap_px,
                merge_gap_anchors=args.merge_gap_anchors,
            )
            for key in [
                "raw_lane_annotations",
                "unknown_color",
                "too_few_points",
                "too_few_anchors",
                "merged_lanes",
            ]:
                totals[key] += stats[key]

            if not lanes:
                totals["images_without_lanes"] += 1
                continue

            for cat in categories:
                totals["category_counts"][CAT_NAMES[cat]] += 1
            totals["lanes_kept"] += len(lanes)

            output_image_name = f"{image_dir.name}_{image_path.name}"
            records.append(
                {
                    "image": f"images/{output_image_name}",
                    "source_image": str(image_path.relative_to(args.data_root)).replace("\\", "/"),
                    "source_label": str(label_path.relative_to(args.data_root)).replace("\\", "/"),
                    "lanes": lanes,
                    "categories": categories,
                }
            )

    return records, totals


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def color_counts(record: dict[str, Any]) -> Counter[int]:
    return Counter(int(cat) for cat in record.get("categories", []) if int(cat) in COLOR_TO_CAT.values())


def primary_color(record: dict[str, Any]) -> int:
    colors = set(color_counts(record).keys())
    if 3 in colors:
        return 3
    if 2 in colors:
        return 2
    if 1 in colors:
        return 1
    return 0


def lane_count_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(color_counts(record))
    return {CAT_NAMES[cat]: int(counts.get(cat, 0)) for cat in sorted(COLOR_TO_CAT.values())}


def select_balanced_records(
    records: list[dict[str, Any]],
    target_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_count <= 0 or target_count >= len(records):
        return records, {
            "enabled": False,
            "requested": target_count,
            "selected": len(records),
            "lane_counts": lane_count_summary(records),
        }

    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for record in records:
        color = primary_color(record)
        if color in buckets:
            buckets[color].append(record)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    target_per_bucket = target_count // 3
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[str, str]] = set()

    for color in [3, 2, 1]:
        take = min(target_per_bucket, len(buckets[color]))
        for record in buckets[color][:take]:
            key = (record["source_image"], record["source_label"])
            selected.append(record)
            selected_ids.add(key)

    remaining = target_count - len(selected)
    if remaining > 0:
        candidates = [
            record
            for record in records
            if (record["source_image"], record["source_label"]) not in selected_ids
        ]
        rng.shuffle(candidates)

        # Prefer records that contain currently underrepresented colors.
        lane_counts = Counter()
        for record in selected:
            lane_counts.update(color_counts(record))

        for _ in range(remaining):
            if not candidates:
                break
            min_count = min(lane_counts.get(cat, 0) for cat in [1, 2, 3])
            rare_colors = {cat for cat in [1, 2, 3] if lane_counts.get(cat, 0) == min_count}
            best_idx = 0
            best_score = -1
            window = min(len(candidates), 512)
            for idx in range(window):
                counts = color_counts(candidates[idx])
                score = sum(counts.get(cat, 0) for cat in rare_colors)
                score -= counts.get(1, 0) * 0.05
                if score > best_score:
                    best_score = score
                    best_idx = idx
            record = candidates.pop(best_idx)
            selected.append(record)
            lane_counts.update(color_counts(record))

    rng.shuffle(selected)
    stats = {
        "enabled": True,
        "requested": target_count,
        "selected": len(selected),
        "primary_color_image_counts": {
            CAT_NAMES[color]: sum(1 for record in selected if primary_color(record) == color)
            for color in [1, 2, 3]
        },
        "lane_counts": lane_count_summary(selected),
    }
    return selected, stats


def copy_images(records: list[dict[str, Any]], data_root: Path, output_dir: Path, mode: str) -> None:
    image_out = output_dir / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    copied: set[str] = set()

    for record in tqdm(records, desc="copy images"):
        dst = output_dir / record["image"]
        if str(dst) in copied:
            continue
        copied.add(str(dst))
        if dst.exists():
            continue
        src = data_root / record["source_image"]
        if mode == "hardlink":
            try:
                dst.hardlink_to(src)
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)


def copy_split_images(
    splits: dict[str, list[dict[str, Any]]],
    data_root: Path,
    output_dir: Path,
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
    updated: dict[str, list[dict[str, Any]]] = {}
    for split, records in splits.items():
        split_dir = output_dir / "images" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        updated[split] = []
        for record in tqdm(records, desc=f"copy {split} images"):
            new_record = dict(record)
            dst_name = Path(record["image"]).name
            new_record["image"] = f"images/{split}/{dst_name}"
            dst = output_dir / new_record["image"]
            src = data_root / record["source_image"]
            if not dst.exists():
                if mode == "hardlink":
                    try:
                        dst.hardlink_to(src)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
            updated[split].append(new_record)
    return updated


def write_dataset(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    totals: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        with (args.output_dir / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

    config = {
        "dataset": "road_v3_ufldv2_color",
        "orig_h": ORIG_H,
        "orig_w": ORIG_W,
        "input_h": 288,
        "input_w": 512,
        "num_anchors": NUM_ANCHORS,
        "h_samples": H_SAMPLES,
        "grid_num": GRID_NUM,
        "max_lanes": MAX_LANES,
        "num_lane_types": len(CAT_NAMES),
        "lane_type_names": CAT_NAMES,
        "class_mapping": COLOR_TO_CAT,
        "ignored_label_attribute": "lane_type",
        "conversion": {
            "min_valid_anchors": args.min_valid_anchors,
            "extrapolate_px": args.extrapolate_px,
            "merge_segments": args.merge_segments,
            "merge_overlap_px": args.merge_overlap_px,
            "merge_gap_px": args.merge_gap_px,
            "merge_gap_anchors": args.merge_gap_anchors,
        },
        "splits": {split: len(items) for split, items in splits.items()},
        "split_image_folders": args.split_image_folders,
        "stats": totals,
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[OK] output: {args.output_dir}")
    print(f"[OK] records: {len(records)}")
    print(f"[OK] splits: {config['splits']}")
    print(f"[OK] category_counts: {totals['category_counts']}")
    print(f"[OK] merged_lanes: {totals['merged_lanes']}")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()

    records, totals = collect_records(args)
    if args.target_count > 0:
        if args.balance_colors:
            records, balance_stats = select_balanced_records(records, args.target_count, args.seed)
        else:
            rng = random.Random(args.seed)
            rng.shuffle(records)
            records = records[: args.target_count]
            balance_stats = {
                "enabled": False,
                "requested": args.target_count,
                "selected": len(records),
                "lane_counts": lane_count_summary(records),
            }
        totals["selected_subset"] = balance_stats

    if args.dry_run:
        print("[DRY-RUN] no files written")
        print(json.dumps(totals, ensure_ascii=False, indent=2))
        print(f"[DRY-RUN] usable_records: {len(records)}")
        return

    if args.output_dir.exists():
        raise FileExistsError(
            f"{args.output_dir} already exists. Move or delete it before rebuilding."
        )

    splits = split_records(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.split_image_folders:
        splits = copy_split_images(splits, args.data_root, args.output_dir, args.copy_mode)
    else:
        copy_images(records, args.data_root, args.output_dir, args.copy_mode)
    write_dataset(args, records, splits, totals)


if __name__ == "__main__":
    main()
