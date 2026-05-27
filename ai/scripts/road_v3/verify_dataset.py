#!/usr/bin/env python
"""
Visual and structural checks for the road_v3 color-only UFLDv2 dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


COLORS_BGR = {
    1: (255, 255, 255),
    2: (0, 220, 255),
    3: (255, 120, 20),
}
NAMES = {
    1: "white",
    2: "yellow",
    3: "blue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify road_v3 UFLDv2 color dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-vis", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--full-stats", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_lane(image: np.ndarray, xs: list[int], category: int, h_samples: list[int]) -> None:
    color = COLORS_BGR.get(category, (128, 128, 128))
    points: list[tuple[int, int]] = []
    for x, y in zip(xs, h_samples):
        if x == -2:
            continue
        points.append((int(x), int(y)))
        cv2.circle(image, (int(x), int(y)), 4, color, -1, lineType=cv2.LINE_AA)

    if len(points) >= 2:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, 3, cv2.LINE_AA)

    if points:
        label = NAMES.get(category, str(category))
        cv2.putText(
            image,
            label,
            points[-1],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def validate_records(records: list[dict[str, Any]], dataset_dir: Path, h_samples: list[int]) -> dict[str, Any]:
    stats = {
        "records": len(records),
        "missing_images": 0,
        "bad_category": 0,
        "bad_anchor_length": 0,
        "bad_x": 0,
        "empty_lane": 0,
        "lane_counts": [],
        "category_counts": {name: 0 for name in NAMES.values()},
        "valid_anchor_counts": [],
    }

    for record in tqdm(records, desc="validate"):
        image_path = dataset_dir / record["image"]
        if not image_path.exists():
            stats["missing_images"] += 1

        lanes = record.get("lanes", [])
        categories = record.get("categories", [])
        stats["lane_counts"].append(len(lanes))

        for xs, cat in zip(lanes, categories):
            if cat not in NAMES:
                stats["bad_category"] += 1
            else:
                stats["category_counts"][NAMES[cat]] += 1

            if len(xs) != len(h_samples):
                stats["bad_anchor_length"] += 1

            valid = [x for x in xs if x != -2]
            stats["valid_anchor_counts"].append(len(valid))
            if not valid:
                stats["empty_lane"] += 1

            if any((x != -2 and (x < 0 or x >= 1280)) for x in xs):
                stats["bad_x"] += 1

    if stats["lane_counts"]:
        stats["avg_lanes_per_image"] = float(np.mean(stats["lane_counts"]))
    else:
        stats["avg_lanes_per_image"] = 0.0

    if stats["valid_anchor_counts"]:
        stats["avg_valid_anchors_per_lane"] = float(np.mean(stats["valid_anchor_counts"]))
        stats["min_valid_anchors_per_lane"] = int(np.min(stats["valid_anchor_counts"]))
    else:
        stats["avg_valid_anchors_per_lane"] = 0.0
        stats["min_valid_anchors_per_lane"] = 0

    return stats


def make_contact_sheet(paths: list[Path], output_path: Path, cols: int = 4) -> None:
    if not paths:
        return
    thumbs: list[np.ndarray] = []
    thumb_w, thumb_h = 320, 180
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return

    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * thumb_h
        x = (idx % cols) * thumb_w
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb

    cv2.imwrite(str(output_path), sheet)


def render_samples(
    records: list[dict[str, Any]],
    dataset_dir: Path,
    h_samples: list[int],
    output_dir: Path,
    num_vis: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sample_records = records[:] if len(records) <= num_vis else rng.sample(records, num_vis)
    rendered: list[Path] = []

    for idx, record in enumerate(tqdm(sample_records, desc="render")):
        image_path = dataset_dir / record["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        overlay = image.copy()
        for xs, cat in zip(record["lanes"], record["categories"]):
            draw_lane(overlay, xs, cat, h_samples)

        out = cv2.addWeighted(overlay, 0.75, image, 0.25, 0)
        cv2.putText(
            out,
            f"{record.get('source_image', record['image'])}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        out_path = output_dir / f"vis_{idx:03d}.jpg"
        cv2.imwrite(str(out_path), out)
        rendered.append(out_path)

    make_contact_sheet(rendered, output_dir / "contact_sheet.jpg")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    config = read_json(dataset_dir / "config.json")
    records = read_json(dataset_dir / f"{args.split}.json")
    h_samples = config["h_samples"]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = dataset_dir / f"verify_{args.split}"
    output_dir = output_dir.resolve()

    stats = validate_records(records, dataset_dir, h_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    printable = stats if args.full_stats else {
        key: value
        for key, value in stats.items()
        if key not in {"lane_counts", "valid_anchor_counts"}
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    render_samples(records, dataset_dir, h_samples, output_dir, args.num_vis, args.seed)
    print(f"[OK] visual samples: {output_dir}")


if __name__ == "__main__":
    main()
