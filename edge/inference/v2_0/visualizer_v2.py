from __future__ import annotations

from typing import List

import cv2
import numpy as np

from road_v4_segmentor import SEG_COLORS


DECISION_COLORS = {
    "STOP": (0, 0, 255),
    "CAUTION": (0, 200, 255),
    "GO": (0, 255, 0),
}

YOLO_COLORS = {
    "pedestrian": (0, 200, 255),
    "person": (0, 200, 255),
    "vehicle": (255, 80, 80),
    "car": (255, 80, 80),
    "bus": (255, 80, 80),
    "truck": (255, 80, 80),
    "traffic_light_vehicle": (80, 255, 80),
    "traffic_light_pedestrian": (80, 255, 80),
    "crosswalk": (255, 255, 0),
}


def _text(img, text, org, scale=0.45, color=(255, 255, 255), thickness=1):
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray,
                 detections: List,
                 seg_mask: np.ndarray,
                 seg_overlay: np.ndarray,
                 lane_geo,
                 state,
                 decision,
                 ctx,
                 yolo_ms: float,
                 seg_ms: float,
                 total_ms: float,
                 fps: float,
                 debug: bool = False) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    seg_nonzero = seg_mask > 0
    if np.any(seg_nonzero):
        vis[seg_nonzero] = cv2.addWeighted(vis[seg_nonzero], 0.72, seg_overlay[seg_nonzero], 0.28, 0)

    if lane_geo.ego_lane_polygon is not None:
        cv2.polylines(vis, [lane_geo.ego_lane_polygon.reshape(-1, 2)], True, (0, 230, 0), 2, cv2.LINE_AA)

    if lane_geo.path_corridor is not None:
        cv2.polylines(vis, [lane_geo.path_corridor.reshape(-1, 2)], True, (160, 110, 40), 1, cv2.LINE_AA)

    if lane_geo.crosswalk_zone is not None:
        overlay = np.zeros_like(vis)
        cv2.fillPoly(overlay, [lane_geo.crosswalk_zone.reshape(-1, 2)], (0, 150, 255))
        mask = overlay.any(axis=2)
        vis[mask] = cv2.addWeighted(vis[mask], 0.78, overlay[mask], 0.22, 0)
        cv2.drawContours(vis, [lane_geo.crosswalk_zone], -1, (0, 200, 255), 2)

    if lane_geo.centerline is not None and len(lane_geo.centerline) >= 2:
        cv2.polylines(vis, [lane_geo.centerline.reshape(-1, 1, 2)], False, (0, 255, 255), 2, cv2.LINE_AA)

    if lane_geo.left_lane_pts is not None and len(lane_geo.left_lane_pts) >= 2:
        cv2.polylines(vis, [lane_geo.left_lane_pts.reshape(-1, 1, 2)], False, (255, 200, 0), 2, cv2.LINE_AA)
    if lane_geo.right_lane_pts is not None and len(lane_geo.right_lane_pts) >= 2:
        cv2.polylines(vis, [lane_geo.right_lane_pts.reshape(-1, 1, 2)], False, (255, 200, 0), 2, cv2.LINE_AA)

    if lane_geo.stop_line_y is not None:
        cv2.line(vis, (0, lane_geo.stop_line_y), (w, lane_geo.stop_line_y), (0, 0, 255), 2, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        color = YOLO_COLORS.get(det.cls_name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.conf:.2f}"
        if det.distance_m is not None:
            label += f" {det.distance_m:.1f}m"
        _text(vis, label, (x1, max(14, y1 - 4)), 0.42, color, 1)

    decision_text = getattr(decision, "value", str(decision))
    state_text = getattr(state, "value", str(state))
    color = DECISION_COLORS.get(decision_text, (255, 255, 255))
    (tw, th), _ = cv2.getTextSize(decision_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    bx, by = w - tw - 28, 10
    cv2.rectangle(vis, (bx - 10, by - 4), (w - 8, by + th + 16), (0, 0, 0), -1)
    cv2.rectangle(vis, (bx - 10, by - 4), (w - 8, by + th + 16), color, 2)
    _text(vis, decision_text, (bx, by + th + 4), 0.9, color, 2)
    _text(vis, state_text, (max(8, w - 260), by + th + 34), 0.42, (220, 220, 220), 1)

    info = (
        f"FPS:{fps:.1f} Y:{yolo_ms:.0f}ms S:{seg_ms:.0f}ms "
        f"Lane:{lane_geo.lane_confidence:.2f} "
        f"P:{ctx.pedestrian_count} V:{ctx.vehicle_count} "
        f"NV:{ctx.nearest_vehicle_distance_m:.1f}m" if ctx.nearest_vehicle_distance_m is not None else
        f"FPS:{fps:.1f} Y:{yolo_ms:.0f}ms S:{seg_ms:.0f}ms "
        f"Lane:{lane_geo.lane_confidence:.2f} P:{ctx.pedestrian_count} V:{ctx.vehicle_count} NV:-"
    )
    cv2.rectangle(vis, (0, h - 26), (w, h), (0, 0, 0), -1)
    _text(vis, info, (6, h - 8), 0.42, (230, 230, 230), 1)
    _text(vis, f"reason: {ctx.reason}", (6, h - 32), 0.38, color, 1)

    if debug:
        y = 18
        for cls_id, cls_color in SEG_COLORS.items():
            if cls_id == 0:
                continue
            cv2.rectangle(vis, (8, y - 10), (18, y), cls_color, -1)
            _text(vis, f"{cls_id}", (24, y), 0.35, (230, 230, 230), 1)
            y += 16

    return vis
