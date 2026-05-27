#!/usr/bin/env python3
"""
Lightweight road-structure perception for intersection safety assistance.

Design goal:
- Reuse the user's custom YOLO detector for semantic objects
- Estimate lane markings and road boundaries with classical CV
- Keep the pipeline lightweight enough for Raspberry Pi CPU execution

Tested assumptions:
- Python 3.10+
- OpenCV 4.x
- Ultralytics package installed when running inference with best.pt

Example:
    python road_structure_assist.py \
        --weights /mnt/data/best.pt \
        --source 0 \
        --width 640 --height 360 \
        --show --save output.mp4
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ----------------------------- Data structures ----------------------------- #


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box: Tuple[int, int, int, int]


@dataclass
class LineModel:
    present: bool = False
    x_bottom: Optional[int] = None
    x_top: Optional[int] = None
    slope: Optional[float] = None
    confidence: float = 0.0
    points: np.ndarray | None = None


@dataclass
class BoundaryModel:
    present: bool = False
    points: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RoadStructureResult:
    mode: str
    center_x: int
    offset_px: float
    heading_deg: float
    confidence: float
    lane_confidence: float
    road_confidence: float
    intersection_likely: bool
    crosswalk_present: bool
    left_lane: LineModel
    right_lane: LineModel
    left_boundary: BoundaryModel
    right_boundary: BoundaryModel
    road_mask: np.ndarray
    lane_mask: np.ndarray
    detections: List[Detection]
    debug: Dict[str, float] = field(default_factory=dict)


# ------------------------------ YOLO wrapper ------------------------------- #


class YoloObjectDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        yolo_interval: int = 2,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install with: pip install ultralytics"
            ) from exc

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.yolo_interval = max(1, yolo_interval)
        self.frame_idx = 0
        self.cached_detections: List[Detection] = []
        self.class_names = self._resolve_class_names()

    def _resolve_class_names(self) -> Dict[int, str]:
        names = getattr(self.model.model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(name) for i, name in enumerate(names)}
        return {}

    def infer(self, frame: np.ndarray) -> List[Detection]:
        self.frame_idx += 1
        if self.frame_idx % self.yolo_interval != 1 and self.cached_detections:
            return self.cached_detections

        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        boxes = getattr(results, "boxes", None)
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(np.int32)
            for box, score, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    Detection(
                        cls_id=int(cls_id),
                        cls_name=self.class_names.get(int(cls_id), str(cls_id)),
                        conf=float(score),
                        box=(x1, y1, x2, y2),
                    )
                )

        self.cached_detections = detections
        return detections


# -------------------------- Geometry/helper methods ------------------------ #


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class EMA:
    def __init__(self, alpha: float, init: Optional[float] = None) -> None:
        self.alpha = alpha
        self.value = init

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


# ----------------------- Road structure core estimator --------------------- #


class RoadStructureEstimator:
    def __init__(self) -> None:
        self.center_ema = EMA(alpha=0.35)
        self.heading_ema = EMA(alpha=0.30)

    def process(self, frame: np.ndarray, detections: List[Detection]) -> RoadStructureResult:
        h, w = frame.shape[:2]

        roi_mask = self._build_driving_roi(h, w)
        exclusion = self._build_exclusion_masks(frame.shape[:2], detections)
        lane_mask, left_lane, right_lane = self._estimate_lane_markings(frame, roi_mask, exclusion)
        road_mask, left_boundary, right_boundary, road_conf = self._estimate_road_region(
            frame, roi_mask, exclusion, lane_mask
        )

        crosswalk_present = any(d.cls_name == "crosswalk" for d in detections)
        lane_conf = float((left_lane.confidence + right_lane.confidence) / 2.0)
        lane_pair_ok = left_lane.present and right_lane.present

        # Mode selection:
        # 1) lane mode: both lane lines are reliable
        # 2) intersection mode: crosswalk exists or lane quality drops, rely more on road region
        # 3) boundary mode: road region works, lane pair is weak
        if lane_pair_ok and lane_conf >= 0.45:
            mode = "lane"
        elif crosswalk_present and lane_conf < 0.45:
            mode = "intersection"
        else:
            mode = "road_boundary"

        center_x, heading_deg = self._fuse_center_and_heading(
            frame.shape[:2], mode, left_lane, right_lane, left_boundary, right_boundary
        )

        offset_px = float(center_x - (w / 2.0))
        confidence = float(clamp(0.6 * lane_conf + 0.4 * road_conf, 0.0, 1.0))
        intersection_likely = bool(crosswalk_present or (lane_conf < 0.35 and road_conf > 0.35))

        return RoadStructureResult(
            mode=mode,
            center_x=center_x,
            offset_px=offset_px,
            heading_deg=heading_deg,
            confidence=confidence,
            lane_confidence=lane_conf,
            road_confidence=road_conf,
            intersection_likely=intersection_likely,
            crosswalk_present=crosswalk_present,
            left_lane=left_lane,
            right_lane=right_lane,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            road_mask=road_mask,
            lane_mask=lane_mask,
            detections=detections,
            debug={
                "lane_confidence": lane_conf,
                "road_confidence": road_conf,
                "offset_px": offset_px,
                "heading_deg": heading_deg,
            },
        )

    def _build_driving_roi(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        polygon = np.array(
            [
                (int(0.05 * w), h - 1),
                (int(0.95 * w), h - 1),
                (int(0.70 * w), int(0.48 * h)),
                (int(0.30 * w), int(0.48 * h)),
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def _build_exclusion_masks(self, shape: Tuple[int, int], detections: List[Detection]) -> Dict[str, np.ndarray]:
        h, w = shape
        obstacle = np.zeros((h, w), dtype=np.uint8)
        crosswalk = np.zeros((h, w), dtype=np.uint8)
        signal = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            x1, y1, x2, y2 = det.box
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)

            if det.cls_name in {"vehicle", "pedestrian"}:
                cv2.rectangle(obstacle, (x1, y1), (x2, y2), 255, -1)
            elif det.cls_name == "crosswalk":
                cv2.rectangle(crosswalk, (x1, y1), (x2, y2), 255, -1)
            elif det.cls_name.startswith("traffic_light"):
                cv2.rectangle(signal, (x1, y1), (x2, y2), 255, -1)

        obstacle = cv2.dilate(obstacle, np.ones((7, 7), np.uint8), iterations=1)
        crosswalk = cv2.dilate(crosswalk, np.ones((11, 11), np.uint8), iterations=1)
        return {"obstacle": obstacle, "crosswalk": crosswalk, "signal": signal}

    def _estimate_lane_markings(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray,
        exclusion: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, LineModel, LineModel]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # White lane markings
        white_mask = cv2.inRange(hls, (0, 165, 0), (180, 255, 120))
        # Yellow lane markings
        yellow_mask = cv2.inRange(hsv, (12, 60, 80), (42, 255, 255))

        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        color_mask = cv2.bitwise_and(color_mask, roi_mask)
        color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(exclusion["crosswalk"]))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 70, 140)
        edges = cv2.bitwise_and(edges, color_mask)
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(exclusion["obstacle"]))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=max(20, int(w * 0.06)),
            maxLineGap=max(15, int(w * 0.025)),
        )

        left_segments: List[Tuple[int, int, int, int]] = []
        right_segments: List[Tuple[int, int, int, int]] = []

        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, line)
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    continue
                length = math.hypot(dx, dy)
                if length < max(18, w * 0.05):
                    continue

                angle_deg = abs(math.degrees(math.atan2(dy, dx)))
                if angle_deg < 22 or angle_deg > 85:
                    continue

                slope = dy / float(dx)
                midx = (x1 + x2) / 2.0
                if slope < 0 and midx < w * 0.72:
                    left_segments.append((x1, y1, x2, y2))
                elif slope > 0 and midx > w * 0.28:
                    right_segments.append((x1, y1, x2, y2))

        left_lane = self._fit_lane_model(left_segments, h, side="left")
        right_lane = self._fit_lane_model(right_segments, h, side="right")
        return color_mask, left_lane, right_lane

    def _fit_lane_model(self, segments: Sequence[Tuple[int, int, int, int]], h: int, side: str) -> LineModel:
        if not segments:
            return LineModel(present=False)

        pts = []
        for x1, y1, x2, y2 in segments:
            pts.append([x1, y1])
            pts.append([x2, y2])
        pts_np = np.array(pts, dtype=np.float32)

        if len(pts_np) < 4:
            return LineModel(present=False)

        vx, vy, x0, y0 = cv2.fitLine(pts_np, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        if abs(vy) < 1e-4:
            return LineModel(present=False)

        def x_at_y(y: float) -> int:
            return int(x0 + (y - y0) * (vx / vy))

        y_bottom = int(h * 0.95)
        y_top = int(h * 0.58)
        x_bottom = x_at_y(y_bottom)
        x_top = x_at_y(y_top)
        slope = float(vy / (vx + 1e-6))

        raw_conf = min(1.0, len(pts_np) / 22.0)
        if side == "left" and x_bottom > x_top:
            raw_conf *= 0.7
        if side == "right" and x_bottom < x_top:
            raw_conf *= 0.7

        return LineModel(
            present=True,
            x_bottom=x_bottom,
            x_top=x_top,
            slope=slope,
            confidence=float(clamp(raw_conf, 0.0, 1.0)),
            points=pts_np,
        )

    def _estimate_road_region(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray,
        exclusion: Dict[str, np.ndarray],
        lane_mask: np.ndarray,
    ) -> Tuple[np.ndarray, BoundaryModel, BoundaryModel, float]:
        h, w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Bottom-center patch approximates current drivable surface.
        patch = self._pick_seed_patch(frame.shape[:2], exclusion)
        px1, py1, px2, py2 = patch
        patch_lab = lab[py1:py2, px1:px2]
        patch_hsv = hsv[py1:py2, px1:px2]

        # Remove lane-mark pixels from the seed statistics.
        patch_lane = lane_mask[py1:py2, px1:px2] > 0
        valid = np.logical_not(patch_lane)
        if valid.sum() < 12:
            valid = np.ones(valid.shape, dtype=bool)

        mean_lab = patch_lab[valid].mean(axis=0)
        std_lab = patch_lab[valid].std(axis=0) + 1.0
        mean_sat = float(patch_hsv[..., 1][valid].mean())

        dL = np.abs(lab[..., 0].astype(np.float32) - mean_lab[0]) / (std_lab[0] + 8.0)
        dA = np.abs(lab[..., 1].astype(np.float32) - mean_lab[1]) / (std_lab[1] + 6.0)
        dB = np.abs(lab[..., 2].astype(np.float32) - mean_lab[2]) / (std_lab[2] + 6.0)
        dist = dL + 0.8 * dA + 0.8 * dB

        low_texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        low_texture = cv2.convertScaleAbs(low_texture)
        texture_mask = cv2.threshold(low_texture, 28, 255, cv2.THRESH_BINARY_INV)[1]

        road_candidate = np.where(dist < 2.6, 255, 0).astype(np.uint8)
        road_candidate = cv2.bitwise_and(road_candidate, roi_mask)
        road_candidate = cv2.bitwise_and(road_candidate, texture_mask)

        # Suppress strong lane markings, crosswalk stripes, and dynamic objects.
        road_candidate = cv2.bitwise_and(road_candidate, cv2.bitwise_not(exclusion["obstacle"]))
        road_candidate = cv2.bitwise_and(road_candidate, cv2.bitwise_not(exclusion["crosswalk"]))
        road_candidate = cv2.bitwise_and(road_candidate, cv2.bitwise_not(lane_mask))

        # Saturated / very bright regions are usually markings rather than asphalt.
        sat_mask = cv2.inRange(hsv[..., 1], 0, int(min(120, mean_sat + 40)))
        road_candidate = cv2.bitwise_and(road_candidate, sat_mask)

        kernel = np.ones((5, 5), np.uint8)
        road_candidate = cv2.morphologyEx(road_candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
        road_candidate = cv2.morphologyEx(road_candidate, cv2.MORPH_OPEN, kernel, iterations=1)

        road_mask = self._select_bottom_connected_component(road_candidate)
        left_boundary, right_boundary = self._extract_boundaries_from_mask(road_mask)

        roi_area = max(1, int(np.count_nonzero(roi_mask)))
        road_area = int(np.count_nonzero(road_mask))
        area_ratio = road_area / roi_area
        boundary_bonus = 0.2 if left_boundary.present and right_boundary.present else 0.0
        road_conf = float(clamp(0.8 * area_ratio + boundary_bonus, 0.0, 1.0))
        return road_mask, left_boundary, right_boundary, road_conf

    def _pick_seed_patch(self, shape: Tuple[int, int], exclusion: Dict[str, np.ndarray]) -> Tuple[int, int, int, int]:
        h, w = shape
        y1 = int(h * 0.84)
        y2 = int(h * 0.96)
        patch_w = int(w * 0.10)
        candidates = [w // 2, int(w * 0.42), int(w * 0.58), int(w * 0.35), int(w * 0.65)]
        for cx in candidates:
            x1 = max(0, cx - patch_w // 2)
            x2 = min(w, cx + patch_w // 2)
            obstacle_ratio = exclusion["obstacle"][y1:y2, x1:x2].mean() / 255.0
            crosswalk_ratio = exclusion["crosswalk"][y1:y2, x1:x2].mean() / 255.0
            if obstacle_ratio < 0.08 and crosswalk_ratio < 0.08:
                return x1, y1, x2, y2
        return max(0, w // 2 - patch_w // 2), y1, min(w, w // 2 + patch_w // 2), y2

    def _select_bottom_connected_component(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return mask

        bottom_band = labels[int(h * 0.88) :, :]
        candidate_ids = np.unique(bottom_band)
        candidate_ids = candidate_ids[candidate_ids != 0]
        if len(candidate_ids) == 0:
            best_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        else:
            best_id = max(candidate_ids, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])

        out = np.zeros_like(mask)
        out[labels == best_id] = 255
        return out

    def _extract_boundaries_from_mask(self, road_mask: np.ndarray) -> Tuple[BoundaryModel, BoundaryModel]:
        h, w = road_mask.shape
        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []

        for y in range(h - 1, int(h * 0.46), -6):
            xs = np.where(road_mask[y] > 0)[0]
            if len(xs) < 25:
                continue
            left_pts.append((int(xs[0]), y))
            right_pts.append((int(xs[-1]), y))

        left_boundary = BoundaryModel(
            present=len(left_pts) >= 8,
            points=left_pts,
            confidence=float(clamp(len(left_pts) / 20.0, 0.0, 1.0)),
        )
        right_boundary = BoundaryModel(
            present=len(right_pts) >= 8,
            points=right_pts,
            confidence=float(clamp(len(right_pts) / 20.0, 0.0, 1.0)),
        )
        return left_boundary, right_boundary

    def _fuse_center_and_heading(
        self,
        shape: Tuple[int, int],
        mode: str,
        left_lane: LineModel,
        right_lane: LineModel,
        left_boundary: BoundaryModel,
        right_boundary: BoundaryModel,
    ) -> Tuple[int, float]:
        h, w = shape
        bottom_y = int(h * 0.95)
        top_y = int(h * 0.58)

        center_bottom: Optional[float] = None
        center_top: Optional[float] = None

        if mode == "lane" and left_lane.present and right_lane.present:
            center_bottom = (left_lane.x_bottom + right_lane.x_bottom) / 2.0
            center_top = (left_lane.x_top + right_lane.x_top) / 2.0
        else:
            lb = self._boundary_x_at_y(left_boundary.points, bottom_y)
            rb = self._boundary_x_at_y(right_boundary.points, bottom_y)
            lt = self._boundary_x_at_y(left_boundary.points, top_y)
            rt = self._boundary_x_at_y(right_boundary.points, top_y)

            if lb is not None and rb is not None:
                center_bottom = (lb + rb) / 2.0
            elif lb is not None:
                center_bottom = lb + w * 0.25
            elif rb is not None:
                center_bottom = rb - w * 0.25

            if lt is not None and rt is not None:
                center_top = (lt + rt) / 2.0
            elif center_bottom is not None:
                center_top = center_bottom

        if center_bottom is None:
            center_bottom = w / 2.0
        if center_top is None:
            center_top = center_bottom

        center_bottom = self.center_ema.update(center_bottom)
        heading = math.degrees(math.atan2(center_bottom - center_top, bottom_y - top_y))
        heading = self.heading_ema.update(heading)

        return int(center_bottom), float(heading)

    def _boundary_x_at_y(self, points: Sequence[Tuple[int, int]], target_y: int) -> Optional[float]:
        if len(points) < 2:
            return None
        pts = sorted(points, key=lambda p: p[1])
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            if (y1 - target_y) * (y2 - target_y) <= 0 and y1 != y2:
                t = (target_y - y1) / float(y2 - y1)
                return x1 + t * (x2 - x1)
        return None


# ------------------------------ Visualization ------------------------------ #


def draw_result(frame: np.ndarray, result: RoadStructureResult) -> np.ndarray:
    vis = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent road mask overlay
    road_overlay = np.zeros_like(frame)
    road_overlay[:, :, 1] = result.road_mask
    vis = cv2.addWeighted(vis, 1.0, road_overlay, 0.22, 0)

    lane_overlay = np.zeros_like(frame)
    lane_overlay[:, :, 2] = result.lane_mask
    vis = cv2.addWeighted(vis, 1.0, lane_overlay, 0.25, 0)

    # Boundaries
    if result.left_boundary.present:
        cv2.polylines(vis, [np.array(result.left_boundary.points, np.int32)], False, (0, 255, 255), 2)
    if result.right_boundary.present:
        cv2.polylines(vis, [np.array(result.right_boundary.points, np.int32)], False, (0, 255, 255), 2)

    # Lane lines
    for lane, color in ((result.left_lane, (255, 100, 0)), (result.right_lane, (255, 100, 0))):
        if lane.present and lane.x_bottom is not None and lane.x_top is not None:
            cv2.line(vis, (lane.x_bottom, int(h * 0.95)), (lane.x_top, int(h * 0.58)), color, 3)

    # Center line
    cv2.line(vis, (w // 2, h - 1), (w // 2, int(h * 0.58)), (120, 120, 120), 1)
    cv2.line(vis, (result.center_x, h - 1), (result.center_x, int(h * 0.58)), (0, 255, 0), 2)
    cv2.circle(vis, (result.center_x, h - 20), 5, (0, 255, 0), -1)

    # Detections
    class_colors = {
        "pedestrian": (0, 200, 255),
        "vehicle": (255, 80, 80),
        "traffic_light_vehicle": (80, 255, 80),
        "traffic_light_pedestrian": (80, 255, 80),
        "crosswalk": (255, 255, 0),
        "left_turn_sign": (255, 0, 255),
    }
    for det in result.detections:
        x1, y1, x2, y2 = det.box
        color = class_colors.get(det.cls_name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.conf:.2f}"
        cv2.putText(vis, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Status panel
    status_lines = [
        f"mode: {result.mode}",
        f"offset_px: {result.offset_px:+.1f}",
        f"heading_deg: {result.heading_deg:+.1f}",
        f"lane_conf: {result.lane_confidence:.2f}",
        f"road_conf: {result.road_confidence:.2f}",
        f"intersection: {result.intersection_likely}",
    ]
    panel_w = 240
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + 22 * len(status_lines) + 10), (0, 0, 0), -1)
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + 22 * len(status_lines) + 10), (80, 80, 80), 1)
    for i, text in enumerate(status_lines):
        cv2.putText(
            vis,
            text,
            (16, 30 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


# ---------------------------------- Main ---------------------------------- #


def open_video_source(source: str | int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight road structure perception")
    parser.add_argument("--weights", type=str, required=True, help="Path to custom YOLO weights")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--yolo-interval", type=int, default=2, help="Run YOLO every N frames")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", type=str, default="", help="Optional output video path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: str | int = int(args.source) if args.source.isdigit() else args.source

    detector = YoloObjectDetector(
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        yolo_interval=args.yolo_interval,
    )
    estimator = RoadStructureEstimator()

    cap = open_video_source(source, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    writer = None
    fps_ema = EMA(alpha=0.15)

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        detections = detector.infer(frame)
        result = estimator.process(frame, detections)
        vis = draw_result(frame, result)

        fps = 1.0 / max(1e-6, time.time() - t0)
        fps = fps_ema.update(fps)
        cv2.putText(vis, f"FPS: {fps:.1f}", (args.width - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if args.save:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, 20.0, (args.width, args.height))
            writer.write(vis)

        if args.show:
            cv2.imshow("road_structure_assist", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
