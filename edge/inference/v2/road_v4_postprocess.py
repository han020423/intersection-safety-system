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
