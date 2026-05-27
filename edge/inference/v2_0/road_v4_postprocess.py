"""
road_v4 클래스 체계에 맞춘 후처리 모듈.

기존 inference 프로그램은 차선을 실선/점선까지 나눠 1~6으로 사용했다.
road_v4는 실선/점선을 합치고 색상만 구분하므로 class id가 달라진다.

기존:
  lane=1..6, crosswalk=7, stop_line=8

road_v4:
  lane=1..3, crosswalk=4, stop_line=5
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import postprocess as base_post  # noqa: E402


LANE_CLASS_MIN = 1
LANE_CLASS_MAX = 3
CROSSWALK_CLASS = 4
STOP_LINE_CLASS = 5

# 기존 postprocess.py의 scan-line 차선 추출 함수는 전역 LANE_CLASS_MIN/MAX를 참조한다.
# 여기서 값을 road_v4 기준으로 덮어써서 기존 로직을 재사용한다.
base_post.LANE_CLASS_MIN = LANE_CLASS_MIN
base_post.LANE_CLASS_MAX = LANE_CLASS_MAX

LaneGeometry = base_post.LaneGeometry
LaneTracker = base_post.LaneTracker


def _split_by_x_jump(pts: np.ndarray, frame_w: int) -> list[np.ndarray]:
    """
    점선 gap 때문에 한쪽 차선 polyline 안에 '진짜 ego 차선'과 '옆 차선'이 섞이면
    y 방향으로 따라가다가 x좌표가 크게 점프한다. 그 점프 지점을 기준으로 구간을 나눈다.
    """
    if pts is None or len(pts) == 0:
        return []

    pts = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
    pts = pts[np.argsort(-pts[:, 1])]
    jump_threshold = max(45, int(frame_w * 0.10))

    segments = []
    current = [pts[0]]
    for prev, cur in zip(pts[:-1], pts[1:]):
        if abs(int(cur[0]) - int(prev[0])) > jump_threshold:
            segments.append(np.asarray(current, dtype=np.int32))
            current = [cur]
        else:
            current.append(cur)
    segments.append(np.asarray(current, dtype=np.int32))
    return segments


def _pick_ego_side_segment(segments: list[np.ndarray], side: str,
                           frame_h: int) -> Optional[np.ndarray]:
    """
    여러 구간 중 ego 차량과 가장 가까운 쪽 차선을 고른다.
    left 차선이면 x가 큰 구간, right 차선이면 x가 작은 구간이 ego에 가깝다.
    """
    usable = []
    min_span = max(35, int(frame_h * 0.10))
    for seg in segments:
        if len(seg) < 3:
            continue
        y_span = int(seg[:, 1].max() - seg[:, 1].min())
        if y_span >= min_span:
            usable.append(seg)

    if not usable:
        return None
    if side == "left":
        return max(usable, key=lambda seg: float(np.median(seg[:, 0])))
    return min(usable, key=lambda seg: float(np.median(seg[:, 0])))


def _straight_fit_is_reliable(seg: np.ndarray, frame_w: int) -> tuple[bool, np.ndarray]:
    """선형 피팅 잔차가 작으면 '거의 1자 차선'으로 판단한다."""
    xs = seg[:, 0].astype(np.float64)
    ys = seg[:, 1].astype(np.float64)
    try:
        coeffs = np.polyfit(ys, xs, 1)
    except (np.linalg.LinAlgError, ValueError):
        return False, np.zeros(2, dtype=np.float64)

    pred = np.poly1d(coeffs)(ys)
    residual = np.abs(pred - xs)
    mean_limit = max(10.0, frame_w * 0.025)
    max_limit = max(22.0, frame_w * 0.055)
    return float(np.mean(residual)) <= mean_limit and float(np.max(residual)) <= max_limit, coeffs


def _extend_straight_lane_to_bottom(pts: Optional[np.ndarray],
                                    side: str,
                                    frame_h: int,
                                    frame_w: int) -> Optional[np.ndarray]:
    """
    화면 하단 점선 gap을 보정한다.

    위쪽에서 충분히 1자로 잡힌 ego 차선이 있으면, 옆 차선으로 튄 하단 구간은 버리고
    그 직선을 화면 바닥까지 연장한다.
    """
    if pts is None or len(pts) < 3:
        return pts

    seg = _pick_ego_side_segment(_split_by_x_jump(pts, frame_w), side, frame_h)
    if seg is None:
        return pts

    reliable, coeffs = _straight_fit_is_reliable(seg, frame_w)
    if not reliable:
        return pts

    bottom_y = frame_h - 1
    seg_bottom_y = int(seg[:, 1].max())
    min_gap = max(22, int(frame_h * 0.07))
    if bottom_y - seg_bottom_y < min_gap:
        return seg[np.argsort(-seg[:, 1])]

    line = np.poly1d(coeffs)
    extra_count = max(3, min(8, int((bottom_y - seg_bottom_y) / max(8, frame_h * 0.025))))
    extra_ys = np.linspace(bottom_y, seg_bottom_y, extra_count, dtype=np.int32)
    extra_xs = np.clip(line(extra_ys.astype(np.float64)), 0, frame_w - 1).astype(np.int32)
    extra = np.column_stack([extra_xs, extra_ys])

    seg_ys = seg[:, 1].astype(np.int32)
    seg_xs = np.clip(line(seg_ys.astype(np.float64)), 0, frame_w - 1).astype(np.int32)
    fitted_seg = np.column_stack([seg_xs, seg_ys])

    repaired = np.vstack([extra, fitted_seg])
    repaired = repaired[np.argsort(-repaired[:, 1])]
    _, unique_indices = np.unique(repaired[:, 1], return_index=True)
    repaired = repaired[np.sort(unique_indices)]
    return repaired.astype(np.int32)


def build_crosswalk_active_zone(seg_mask: np.ndarray,
                                ego_polygon: Optional[np.ndarray],
                                frame_h: int,
                                frame_w: int) -> Tuple[Optional[np.ndarray], int]:
    """횡단보도 전체가 아니라 ego lane과 겹치는 관심 횡단보도 영역만 추출한다."""
    cw_mask = (seg_mask == CROSSWALK_CLASS).astype(np.uint8) * 255
    total_pixels = int(np.count_nonzero(cw_mask))
    if total_pixels < 100:
        return None, 0

    if ego_polygon is not None:
        lane_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(lane_mask, [ego_polygon.reshape(-1, 2)], 255)
        # 횡단보도는 차선 영역보다 넓게 걸치므로 ego lane polygon을 약간 확장한다.
        kernel = np.ones((30, 30), np.uint8)
        lane_mask = cv2.dilate(lane_mask, kernel, iterations=1)
        cw_mask = cv2.bitwise_and(cw_mask, lane_mask)

    pixel_count = int(np.count_nonzero(cw_mask))
    if pixel_count < 50:
        return None, 0

    contours, _ = cv2.findContours(cw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, pixel_count
    return max(contours, key=cv2.contourArea), pixel_count


def detect_stop_line_y(seg_mask: np.ndarray) -> Optional[int]:
    """정지선 class 픽셀들의 평균 y좌표를 정지선 위치로 사용한다."""
    stop_ys = np.where(seg_mask == STOP_LINE_CLASS)[0]
    if len(stop_ys) < 10:
        return None
    return int(np.mean(stop_ys))


def compute_lane_geometry(seg_mask: np.ndarray,
                          frame_h: int,
                          frame_w: int,
                          corridor_half_width: int = 80,
                          detections=None,
                          suppress_vehicle_lanes: bool = True) -> LaneGeometry:
    """segmentation mask를 판단에 필요한 기하 정보로 변환한다."""
    geo = LaneGeometry()

    # 차량 bbox 내부에서 차선으로 잘못 검출된 픽셀은 차선 추출을 흔들 수 있어 제거한다.
    lane_source_mask = (
        base_post.suppress_lane_pixels_in_vehicle_boxes(seg_mask, detections)
        if suppress_vehicle_lanes else seg_mask
    )

    # 1) scan-line 기반으로 ego 차량 좌우 차선 후보를 추출한다.
    left_pts, right_pts, conf = base_post.extract_lane_polylines(lane_source_mask)
    # 점선이 화면 바닥부터 일정 y까지 끊겨 있으면 scan-line이 옆 차선을 ego 차선으로 고를 수 있다.
    # 위쪽에서 1자로 충분히 잡힌 구간이 있으면 그 구간을 바닥까지 연장해 ego lane을 안정화한다.
    left_pts = _extend_straight_lane_to_bottom(left_pts, "left", frame_h, frame_w)
    right_pts = _extend_straight_lane_to_bottom(right_pts, "right", frame_h, frame_w)
    geo.left_lane_pts = left_pts
    geo.right_lane_pts = right_pts
    geo.lane_confidence = conf

    # 2) 차선 후보에서 ego lane, centerline, path corridor를 만든다.
    geo.ego_lane_polygon = base_post.build_ego_lane_polygon(left_pts, right_pts, frame_h, frame_w)
    geo.centerline = base_post.build_lane_centerline(left_pts, right_pts, frame_w, frame_h)
    # 3) 우회전 판단에서 관심을 둘 횡단보도/정지선 정보를 뽑는다.
    geo.crosswalk_zone, geo.crosswalk_pixel_count = build_crosswalk_active_zone(
        seg_mask, geo.ego_lane_polygon, frame_h, frame_w
    )
    geo.path_corridor = base_post.build_path_corridor(geo.centerline, corridor_half_width, frame_w)
    geo.stop_line_y = detect_stop_line_y(seg_mask)
    return geo
