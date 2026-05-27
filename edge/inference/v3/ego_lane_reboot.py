"""
v2_0 ego lane reboot pipeline.

기존 postprocess.py / v2 로직을 재사용하지 않고 새로 만든 주행 차선 추정기다.
흐름은 다음처럼 분리한다.

1. BiSeNet road_v4 mask에서 차선 픽셀만 추출
2. connected component로 작은 잡음을 제거
3. scan band keypoint를 모아 lane candidate polyline 생성
4. 프레임 간 candidate를 temporal track으로 연결
5. ego 차량 중심 기준 좌/우 boundary를 고르고 polygon 생성
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


LANE_CLASS_IDS = (1, 2, 3)  # lane_white, lane_yellow, lane_blue


@dataclass
class LaneCandidate:
    """현재 프레임에서 mask로부터 추출한 차선 후보."""

    keypoints: np.ndarray
    polyline: np.ndarray
    score: float
    bottom_x: float
    y_span: int


@dataclass
class LaneTrack:
    """여러 프레임에 걸쳐 이어진 차선 후보 track."""

    track_id: int
    polyline: np.ndarray
    score: float
    age: int = 1
    hits: int = 1
    missed: int = 0


@dataclass
class LaneRole:
    """현재 ego lane이 주변 차선 배열에서 어떤 위치인지 나타낸다."""

    boundary_count: int = 0
    ego_lane_index_from_left: Optional[int] = None
    ego_lane_index_from_right: Optional[int] = None
    is_leftmost_lane: Optional[bool] = None
    is_rightmost_lane: Optional[bool] = None
    left_adjacent_vehicle: bool = False
    right_adjacent_vehicle: bool = False
    left_boundary_yellow: bool = False
    left_yellow_ratio: float = 0.0
    right_boundary_yellow: bool = False
    right_yellow_ratio: float = 0.0
    leftmost_score: float = 0.0
    rightmost_score: float = 0.0
    confidence: float = 0.0
    reason: str = "not_available"


@dataclass
class EgoLaneResult:
    """한 프레임에서 추정한 현재 주행 차선 정보."""

    left_boundary: Optional[np.ndarray] = None
    right_boundary: Optional[np.ndarray] = None
    centerline: Optional[np.ndarray] = None
    polygon: Optional[np.ndarray] = None
    confidence: float = 0.0
    status: str = "not_found"
    candidate_count: int = 0
    track_count: int = 0
    left_track_id: Optional[int] = None
    right_track_id: Optional[int] = None
    role: Optional[LaneRole] = None


def _build_lane_mask(seg_mask: np.ndarray) -> np.ndarray:
    """road_v4 차선 class만 255로 만들고 얇은 끊김을 작게 메운다."""
    lane_mask = np.isin(seg_mask, LANE_CLASS_IDS).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _filter_components(lane_mask: np.ndarray) -> np.ndarray:
    """
    connected component로 너무 작은 차선 잡음을 제거한다.

    점선 차선은 작은 조각으로 나뉠 수 있으므로 기준을 세게 잡지 않는다. 넓이와
    높이 중 하나라도 의미가 있으면 남기고, 1~2px 먼지 같은 component만 버린다.
    """
    h, w = lane_mask.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lane_mask, connectivity=8)
    filtered = np.zeros_like(lane_mask)
    min_area = max(8, int(h * w * 0.00004))
    min_extent = max(5, int(min(h, w) * 0.012))

    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < min_area:
            continue
        if max(int(bw), int(bh)) < min_extent:
            continue
        filtered[labels == label] = 255
    return filtered


def _cluster_x_positions(xs: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """scan band 안에서 붙어 있는 차선 픽셀들을 x축 구간별로 묶는다."""
    if len(xs) == 0:
        return []

    clusters: list[list[int]] = [[int(xs[0])]]
    for x_value in xs[1:]:
        x = int(x_value)
        if x - clusters[-1][-1] <= max_gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [np.asarray(cluster, dtype=np.int32) for cluster in clusters]


def _scan_keypoints(lane_mask: np.ndarray) -> list[tuple[int, int, int]]:
    """
    여러 y좌표 band에서 차선 cluster 중심을 keypoint로 뽑는다.

    반환값은 (x_center, y, cluster_width)이다. cluster_width는 점수 계산에만 쓰며,
    실제 polyline은 중심점 sequence로 만든다.
    """
    h, w = lane_mask.shape[:2]
    scan_bottom = int(h * 0.98)
    scan_top = int(h * 0.22)
    scan_ys = np.linspace(scan_bottom, scan_top, 56, dtype=np.int32)
    band_half_height = max(2, int(h * 0.006))
    max_gap = max(7, int(w * 0.014))
    min_width = max(2, int(w * 0.004))

    keypoints: list[tuple[int, int, int]] = []
    for y in scan_ys:
        y1 = max(0, int(y) - band_half_height)
        y2 = min(h, int(y) + band_half_height + 1)
        xs = np.where(np.any(lane_mask[y1:y2, :] > 0, axis=0))[0]
        for cluster in _cluster_x_positions(xs, max_gap):
            if len(cluster) < min_width:
                continue
            keypoints.append((int(np.mean(cluster)), int(y), int(len(cluster))))
    return keypoints


def _predict_x(points: list[tuple[int, int]], y: int) -> float:
    """기존 point sequence가 주어진 y에서 가질 x를 짧은 선형 피팅으로 예측한다."""
    if len(points) < 3:
        return float(points[-1][0])

    recent = np.asarray(points[-7:], dtype=np.float64)
    try:
        coeffs = np.polyfit(recent[:, 1], recent[:, 0], 1)
        return float(np.poly1d(coeffs)(y))
    except (np.linalg.LinAlgError, ValueError):
        return float(points[-1][0])


def _group_keypoints(keypoints: list[tuple[int, int, int]], frame_w: int,
                     frame_h: int) -> list[list[tuple[int, int]]]:
    """
    scan keypoint를 차선 후보별 sequence로 묶는다.

    아래에서 위로 올라가며 가까운 track에 붙인다. y 간격과 x 예측 오차를 같이
    제한해서 서로 다른 차선이나 그림자 조각이 한 후보로 섞이는 일을 줄인다.
    """
    if not keypoints:
        return []

    ordered = sorted(keypoints, key=lambda item: (-item[1], item[0]))
    groups: list[list[tuple[int, int]]] = []
    max_y_gap = max(28, int(frame_h * 0.070))
    base_x_gate = max(16, int(frame_w * 0.025))
    max_x_gate = max(32, int(frame_w * 0.060))

    for x, y, _width in ordered:
        best_idx = -1
        best_dist = float("inf")
        for idx, group in enumerate(groups):
            if group[-1][1] == y:
                continue
            y_gap = abs(int(group[-1][1]) - int(y))
            if y_gap > max_y_gap:
                continue
            x_gate = min(max_x_gate, base_x_gate + y_gap * 0.18)
            dist = abs(float(x) - _predict_x(group, y))
            if dist <= x_gate and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx >= 0:
            groups[best_idx].append((x, y))
        else:
            groups.append([(x, y)])

    min_span = max(35, int(frame_h * 0.10))
    return [
        group for group in groups
        if len(group) >= 4 and int(max(p[1] for p in group) - min(p[1] for p in group)) >= min_span
    ]


def _unique_points_by_y(points: list[tuple[int, int]] | np.ndarray) -> Optional[np.ndarray]:
    """같은 y에서 여러 후보가 들어오면 x 평균 하나로 합친다."""
    if points is None or len(points) < 2:
        return None

    grouped: dict[int, list[int]] = {}
    for x, y in np.asarray(points, dtype=np.int32).reshape(-1, 2):
        grouped.setdefault(int(y), []).append(int(x))

    merged = np.asarray([(int(np.mean(xs)), y) for y, xs in grouped.items()], dtype=np.int32)
    if len(merged) < 2:
        return None
    return merged[np.argsort(-merged[:, 1])]


def _fit_polyline(points: list[tuple[int, int]], frame_w: int) -> Optional[LaneCandidate]:
    """keypoint sequence를 y->x 곡선으로 피팅해 LaneCandidate로 만든다."""
    pts = _unique_points_by_y(points)
    if pts is None or len(pts) < 4:
        return None

    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)
    degree = min(2, len(pts) - 1)
    try:
        coeffs = np.polyfit(ys, xs, degree)
    except (np.linalg.LinAlgError, ValueError):
        return None

    fitted = np.poly1d(coeffs)(ys)
    residual = np.abs(fitted - xs)
    residual_limit = max(20.0, frame_w * 0.045)
    if float(np.percentile(residual, 85)) > residual_limit:
        return None

    y_bottom = int(np.max(ys))
    y_top = int(np.min(ys))
    if y_bottom <= y_top:
        return None

    dense_count = max(16, min(42, int((y_bottom - y_top) / 7)))
    dense_ys = np.linspace(y_bottom, y_top, dense_count, dtype=np.int32)
    dense_xs = np.clip(np.poly1d(coeffs)(dense_ys.astype(np.float64)), 0, frame_w - 1).astype(np.int32)
    polyline = np.column_stack([dense_xs, dense_ys]).astype(np.int32)

    y_span = int(y_bottom - y_top)
    span_score = min(1.0, y_span / 220.0)
    point_score = min(1.0, len(pts) / 18.0)
    residual_score = 1.0 - min(1.0, float(np.percentile(residual, 75)) / residual_limit)
    score = float(np.clip(0.42 * span_score + 0.38 * point_score + 0.20 * residual_score, 0.0, 1.0))
    return LaneCandidate(
        keypoints=pts,
        polyline=polyline,
        score=score,
        bottom_x=float(polyline[0, 0]),
        y_span=y_span,
    )


def extract_lane_candidates(seg_mask: np.ndarray) -> list[LaneCandidate]:
    """segmentation mask에서 lane candidate polyline 목록을 추출한다."""
    h, w = seg_mask.shape[:2]
    lane_mask = _filter_components(_build_lane_mask(seg_mask))
    keypoints = _scan_keypoints(lane_mask)
    groups = _group_keypoints(keypoints, w, h)

    candidates = []
    for group in groups:
        candidate = _fit_polyline(group, w)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _sample_polyline_at_ys(polyline: np.ndarray, target_ys: np.ndarray,
                           frame_w: int) -> Optional[np.ndarray]:
    """polyline을 target y좌표들로 재샘플링한다."""
    pts = _unique_points_by_y(polyline)
    if pts is None:
        return None

    degree = min(2, len(pts) - 1)
    try:
        coeffs = np.polyfit(pts[:, 1].astype(np.float64), pts[:, 0].astype(np.float64), degree)
    except (np.linalg.LinAlgError, ValueError):
        return None

    xs = np.clip(np.poly1d(coeffs)(target_ys.astype(np.float64)), 0, frame_w - 1).astype(np.int32)
    return np.column_stack([xs, target_ys.astype(np.int32)]).astype(np.int32)


def _track_distance(track: LaneTrack, candidate: LaneCandidate, frame_w: int) -> float:
    """track과 candidate가 같은 차선인지 판단하기 위한 pixel 거리."""
    y_bottom = max(int(track.polyline[:, 1].max()), int(candidate.polyline[:, 1].max()))
    y_top = min(int(track.polyline[:, 1].min()), int(candidate.polyline[:, 1].min()))
    if y_bottom <= y_top:
        y_bottom = int(candidate.polyline[:, 1].max())
        y_top = int(candidate.polyline[:, 1].min())

    sample_ys = np.linspace(y_bottom, y_top, 5, dtype=np.int32)
    t_pts = _sample_polyline_at_ys(track.polyline, sample_ys, frame_w)
    c_pts = _sample_polyline_at_ys(candidate.polyline, sample_ys, frame_w)
    if t_pts is None or c_pts is None:
        return float("inf")
    return float(np.mean(np.abs(t_pts[:, 0].astype(np.float64) - c_pts[:, 0].astype(np.float64))))


def _blend_polylines(old: np.ndarray, new: np.ndarray, alpha: float,
                     frame_w: int) -> np.ndarray:
    """동일 y좌표로 맞춘 뒤 EMA로 두 polyline을 부드럽게 섞는다."""
    y_bottom = max(int(old[:, 1].max()), int(new[:, 1].max()))
    y_top = min(int(old[:, 1].min()), int(new[:, 1].min()))
    if y_bottom <= y_top:
        y_bottom = int(new[:, 1].max())
        y_top = int(new[:, 1].min())

    target_ys = np.linspace(y_bottom, y_top, max(16, min(36, int((y_bottom - y_top) / 8))), dtype=np.int32)
    old_pts = _sample_polyline_at_ys(old, target_ys, frame_w)
    new_pts = _sample_polyline_at_ys(new, target_ys, frame_w)
    if old_pts is None or new_pts is None:
        return new

    blended_xs = ((1.0 - alpha) * old_pts[:, 0] + alpha * new_pts[:, 0]).astype(np.int32)
    return np.column_stack([np.clip(blended_xs, 0, frame_w - 1), target_ys]).astype(np.int32)


def _estimate_lane_width(left: Optional[np.ndarray], right: Optional[np.ndarray], frame_w: int) -> int:
    """좌/우가 모두 있으면 관측 폭을, 아니면 보수적 기본 폭을 사용한다."""
    if left is not None and right is not None:
        y_bottom = max(int(left[:, 1].max()), int(right[:, 1].max()))
        y_top = min(int(left[:, 1].min()), int(right[:, 1].min()))
        if y_bottom > y_top:
            ys = np.linspace(y_bottom, y_top, 8, dtype=np.int32)
            l_pts = _sample_polyline_at_ys(left, ys, frame_w)
            r_pts = _sample_polyline_at_ys(right, ys, frame_w)
            if l_pts is not None and r_pts is not None:
                widths = r_pts[:, 0] - l_pts[:, 0]
                widths = widths[widths > max(28, int(frame_w * 0.06))]
                if len(widths) > 0:
                    return int(np.clip(np.median(widths), frame_w * 0.16, frame_w * 0.58))
    return int(np.clip(frame_w * 0.34, max(60, frame_w * 0.16), max(120, frame_w * 0.58)))


def _estimate_width_profile(target_ys: np.ndarray,
                            frame_h: int,
                            frame_w: int,
                            left: Optional[np.ndarray],
                            right: Optional[np.ndarray],
                            cached_widths: Optional[np.ndarray]) -> np.ndarray:
    """
    한쪽 차선만 보일 때 사용할 y별 lane width를 만든다.

    고정 x-offset은 원근이 있는 도로 영상에서 polygon을 쉽게 비틀기 때문에 쓰지 않는다.
    최근 양쪽 차선이 모두 보였던 폭 프로파일이 있으면 우선 사용하고, 없으면 화면
    아래쪽은 넓고 위쪽은 좁아지는 간단한 perspective profile로 후퇴한다.
    """
    if cached_widths is not None and len(cached_widths) >= 3:
        old_ys = np.linspace(float(target_ys[0]), float(target_ys[-1]), len(cached_widths))
        interp = np.interp(
            target_ys.astype(np.float64),
            old_ys[::-1],
            cached_widths[::-1].astype(np.float64),
        )
        return np.clip(interp, frame_w * 0.10, frame_w * 0.58).astype(np.int32)

    observed_width = _estimate_lane_width(left, right, frame_w)
    y_ratio = np.clip(target_ys.astype(np.float64) / max(1.0, float(frame_h)), 0.0, 1.0)
    # y가 위로 갈수록 폭을 줄인다. 완전한 카메라 보정은 아니지만, 고정 폭보다
    # 한쪽 추정 polygon의 뒤틀림을 훨씬 덜 만든다.
    perspective_scale = 0.34 + 0.72 * y_ratio
    widths = observed_width * perspective_scale
    return np.clip(widths, frame_w * 0.09, frame_w * 0.58).astype(np.int32)


def _build_polygon(left: np.ndarray, right: np.ndarray, frame_w: int) -> Optional[np.ndarray]:
    """좌/우 경계선을 닫힌 polygon으로 만든다."""
    lane_widths = right[:, 0] - left[:, 0]
    min_width = max(30, int(frame_w * 0.07))
    max_width = max(90, int(frame_w * 0.68))
    valid = (lane_widths > min_width) & (lane_widths < max_width)
    if int(np.count_nonzero(valid)) < 3:
        return None
    if float(np.std(lane_widths[valid])) > frame_w * 0.24:
        return None
    return np.vstack([left[valid], right[valid][::-1]]).astype(np.int32).reshape(-1, 1, 2)


def _is_vehicle_detection(det) -> bool:
    """YOLO detection이 차선 옆 차량 판단에 쓸 차량 class인지 확인한다."""
    name = getattr(det, "cls_name", "").lower()
    return name in {"vehicle", "car", "bus", "truck", "motorcycle"}


def _vehicle_adjacent_flags(detections,
                            result: EgoLaneResult,
                            frame_w: int,
                            frame_h: int) -> tuple[bool, bool]:
    """
    ego lane 좌/우 바깥에 차량 bbox가 있는지 본다.

    차선 boundary가 안 보여도 옆 차량이 있으면 옆 차로가 존재할 가능성이 크다.
    따라서 이 신호는 leftmost/rightmost 판단을 False 쪽으로 보정하는 데만 쓴다.
    """
    if not detections:
        return False, False

    margin = max(10, int(frame_w * 0.025))
    left_vehicle = False
    right_vehicle = False
    for det in detections:
        if not _is_vehicle_detection(det):
            continue
        if getattr(det, "conf", 1.0) < 0.25:
            continue

        x1, y1, x2, y2 = [int(v) for v in det.box]
        cx = int((x1 + x2) / 2)
        foot_y = int(np.clip(y2, 0, frame_h - 1))
        if foot_y < int(frame_h * 0.32):
            continue

        if result.polygon is not None:
            inside = cv2.pointPolygonTest(result.polygon.reshape(-1, 2), (float(cx), float(foot_y)), False)
            if inside >= 0:
                continue

        left_x = None
        right_x = None
        sample_y = np.asarray([foot_y], dtype=np.int32)
        if result.left_boundary is not None:
            sampled_left = _sample_polyline_at_ys(result.left_boundary, sample_y, frame_w)
            if sampled_left is not None:
                left_x = float(sampled_left[0, 0])
        if result.right_boundary is not None:
            sampled_right = _sample_polyline_at_ys(result.right_boundary, sample_y, frame_w)
            if sampled_right is not None:
                right_x = float(sampled_right[0, 0])

        if left_x is not None and cx < left_x - margin:
            left_vehicle = True
        elif right_x is not None and cx > right_x + margin:
            right_vehicle = True
        elif left_x is None or right_x is None:
            ego_x = frame_w / 2.0
            if cx < ego_x - frame_w * 0.18:
                left_vehicle = True
            elif cx > ego_x + frame_w * 0.18:
                right_vehicle = True

    return left_vehicle, right_vehicle


def _boundary_yellow_signal(seg_mask: np.ndarray,
                            boundary: Optional[np.ndarray],
                            frame_w: int,
                            frame_h: int) -> tuple[bool, float, int]:
    """
    ego lane 경계 주변의 yellow lane 비율을 계산한다.

    황색은 중앙선뿐 아니라 도로 경계/차로 경계에도 쓰일 수 있으므로, 여기서는
    색상 신호만 계산한다. leftmost/rightmost 의미 해석은 주변 차선/차량/안정성
    근거와 함께 _estimate_lane_role()에서 처리한다.
    """
    if boundary is None or len(boundary) < 4:
        return False, 0.0, 0

    yellow = 0
    lane = 0
    radius = max(2, int(frame_w * 0.006))
    for x, y in np.asarray(boundary, dtype=np.int32).reshape(-1, 2):
        if int(y) < int(frame_h * 0.22):
            continue
        x1 = max(0, int(x) - radius)
        x2 = min(frame_w, int(x) + radius + 1)
        y1 = max(0, int(y) - radius)
        y2 = min(frame_h, int(y) + radius + 1)
        patch = seg_mask[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        yellow += int(np.count_nonzero(patch == 2))
        lane += int(np.count_nonzero(np.isin(patch, LANE_CLASS_IDS)))

    if lane < max(18, int(len(boundary) * 2.0)):
        return False, 0.0, lane
    ratio = yellow / max(1, lane)
    return ratio >= 0.60, float(ratio), lane


def _left_boundary_yellow_signal(seg_mask: np.ndarray,
                                 left_boundary: Optional[np.ndarray],
                                 frame_w: int,
                                 frame_h: int) -> tuple[bool, float, int]:
    """Backward-compatible wrapper for the ego-lane left boundary."""

    return _boundary_yellow_signal(seg_mask, left_boundary, frame_w, frame_h)


def _make_ego_result(left_track: Optional[LaneTrack], right_track: Optional[LaneTrack],
                     frame_h: int, frame_w: int, candidate_count: int,
                     track_count: int,
                     cached_widths: Optional[np.ndarray] = None) -> EgoLaneResult:
    """선택된 좌/우 track으로 ego lane polygon을 만든다."""
    if left_track is None and right_track is None:
        return EgoLaneResult(status="not_found", candidate_count=candidate_count, track_count=track_count)

    left_poly = left_track.polyline if left_track is not None else None
    right_poly = right_track.polyline if right_track is not None else None
    status = "tracked"

    # 시각화 영역은 하단 중심부를 우선한다. fit으로 약간의 외삽이 가능해서
    # 점선이 끊겨도 polygon이 한두 프레임씩 사라지는 일을 줄인다.
    y_bottom = int(frame_h * 0.96)
    top_candidates = []
    for poly in (left_poly, right_poly):
        if poly is not None:
            top_candidates.append(int(poly[:, 1].min()))
    y_top = min(top_candidates) if top_candidates else int(frame_h * 0.35)
    y_top = max(int(frame_h * 0.22), min(y_top, int(frame_h * 0.68)))
    one_side_without_cache = (left_poly is None) != (right_poly is None) and cached_widths is None
    if one_side_without_cache:
        # 폭 캐시 없이 한쪽만 보이면 멀리까지 면을 채우지 않는다. 원근/곡률 정보가
        # 부족한 상태에서 상단까지 외삽하면 스크린샷처럼 영역이 크게 비틀릴 수 있다.
        y_top = max(y_top, int(frame_h * 0.45))
    if y_bottom <= y_top:
        return EgoLaneResult(status="bad_range", candidate_count=candidate_count, track_count=track_count)

    target_ys = np.linspace(y_bottom, y_top, 30, dtype=np.int32)
    left = _sample_polyline_at_ys(left_poly, target_ys, frame_w) if left_poly is not None else None
    right = _sample_polyline_at_ys(right_poly, target_ys, frame_w) if right_poly is not None else None
    width_profile = _estimate_width_profile(target_ys, frame_h, frame_w, left, right, cached_widths)

    if left is None and right is not None:
        left = right.copy()
        left[:, 0] = np.clip(right[:, 0] - width_profile, 0, frame_w - 1)
        status = "estimated_left"
    elif right is None and left is not None:
        right = left.copy()
        right[:, 0] = np.clip(left[:, 0] + width_profile, 0, frame_w - 1)
        status = "estimated_right"

    if left is None or right is None:
        return EgoLaneResult(status="not_enough_points", candidate_count=candidate_count, track_count=track_count)

    polygon = _build_polygon(left, right, frame_w)
    if polygon is None:
        return EgoLaneResult(left, right, status="bad_polygon", candidate_count=candidate_count, track_count=track_count)

    centerline = np.column_stack([((left[:, 0] + right[:, 0]) / 2.0).astype(np.int32), target_ys]).astype(np.int32)
    bottom_center_error = abs(float(centerline[0, 0]) - frame_w / 2.0)
    if bottom_center_error > frame_w * (0.32 if status.startswith("estimated") else 0.42):
        return EgoLaneResult(left, right, centerline, status="off_center", candidate_count=candidate_count, track_count=track_count)

    lane_widths = right[:, 0] - left[:, 0]
    width_score = 1.0 - min(1.0, float(np.std(lane_widths)) / max(1.0, frame_w * 0.12))
    track_score = np.mean([
        track.score * (0.65 + 0.35 * min(1.0, track.hits / 8.0)) * max(0.0, 1.0 - 0.15 * track.missed)
        for track in (left_track, right_track) if track is not None
    ])
    confidence = float(np.clip(0.72 * track_score + 0.28 * width_score, 0.0, 1.0))
    if status.startswith("estimated"):
        confidence *= 0.72

    return EgoLaneResult(
        left_boundary=left,
        right_boundary=right,
        centerline=centerline,
        polygon=polygon,
        confidence=confidence,
        status=status,
        candidate_count=candidate_count,
        track_count=track_count,
        left_track_id=left_track.track_id if left_track is not None else None,
        right_track_id=right_track.track_id if right_track is not None else None,
    )


@dataclass
class EgoLanePipeline:
    """
    mask -> candidates -> temporal tracks -> ego lane result 파이프라인.

    비디오에서는 이 객체를 한 번 만들고 매 프레임 update()를 호출해야 temporal
    tracking이 유지된다. 단일 이미지에서는 새 객체로 한 번만 호출하면 된다.
    """

    smooth_alpha: float = 0.38
    max_missed: int = 6
    max_tracks: int = 8
    _next_track_id: int = 1
    _tracks: list[LaneTrack] = field(default_factory=list)
    _cached_widths: Optional[np.ndarray] = None
    _cached_width_age: int = 999
    _last_role_key: Optional[tuple] = None
    _role_streak: int = 0
    _left_role_votes: deque = field(default_factory=lambda: deque(maxlen=15))
    _right_role_votes: deque = field(default_factory=lambda: deque(maxlen=15))

    def update(self, seg_mask: np.ndarray, detections=None) -> EgoLaneResult:
        h, w = seg_mask.shape[:2]
        candidates = extract_lane_candidates(seg_mask)
        self._update_tracks(candidates, w)
        left_track, right_track = self._select_ego_tracks(w, h)
        cached_widths = self._cached_widths if self._cached_width_age <= 20 else None
        result = _make_ego_result(
            left_track,
            right_track,
            h,
            w,
            len(candidates),
            len(self._tracks),
            cached_widths,
        )
        self._update_width_cache(result)
        result.role = self._estimate_lane_role(left_track, right_track, result, w, h, seg_mask, detections)
        return result

    def _update_width_cache(self, result: EgoLaneResult) -> None:
        """
        양쪽 차선이 모두 직접 선택된 안정 프레임의 y별 폭을 저장한다.

        한쪽 차선이 잠깐 사라졌을 때 최근 폭 프로파일을 쓰면 단순 x-offset보다
        polygon이 덜 비틀린다. estimated_* 결과는 추정값이므로 캐시에 넣지 않는다.
        """
        self._cached_width_age += 1
        if result.status != "tracked":
            return
        if result.left_boundary is None or result.right_boundary is None:
            return
        widths = result.right_boundary[:, 0] - result.left_boundary[:, 0]
        valid = widths > 0
        if int(np.count_nonzero(valid)) < 3:
            return
        self._cached_widths = widths[valid].astype(np.float64)
        self._cached_width_age = 0

    def _update_tracks(self, candidates: list[LaneCandidate], frame_w: int) -> None:
        matched_tracks: set[int] = set()
        matched_candidates: set[int] = set()
        gate_px = max(32.0, frame_w * 0.085)

        for cand_idx, candidate in enumerate(candidates):
            best_idx = -1
            best_dist = float("inf")
            for track_idx, track in enumerate(self._tracks):
                if track_idx in matched_tracks:
                    continue
                dist = _track_distance(track, candidate, frame_w)
                if dist < best_dist:
                    best_idx = track_idx
                    best_dist = dist
            if best_idx >= 0 and best_dist <= gate_px:
                track = self._tracks[best_idx]
                track.polyline = _blend_polylines(track.polyline, candidate.polyline, self.smooth_alpha, frame_w)
                track.score = float(0.75 * track.score + 0.25 * candidate.score)
                track.age += 1
                track.hits += 1
                track.missed = 0
                matched_tracks.add(best_idx)
                matched_candidates.add(cand_idx)

        for idx, track in enumerate(self._tracks):
            if idx not in matched_tracks:
                track.age += 1
                track.missed += 1

        for cand_idx, candidate in enumerate(candidates):
            if cand_idx in matched_candidates:
                continue
            self._tracks.append(
                LaneTrack(
                    track_id=self._next_track_id,
                    polyline=candidate.polyline,
                    score=candidate.score,
                )
            )
            self._next_track_id += 1

        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]
        self._tracks.sort(key=lambda item: (item.missed, -item.hits, -item.score))
        self._tracks = self._tracks[:self.max_tracks]

    def _track_positions_at_y(self, frame_w: int, frame_h: int,
                              sample_y: int) -> list[tuple[LaneTrack, float]]:
        """
        주변 차선 track을 sample_y에서의 x좌표로 변환한다.

        sample_y가 track 관측 범위에서 너무 멀리 떨어져 있으면 제외한다. lane role은
        차선 순서를 판단하는 값이라, 긴 외삽으로 만든 x좌표를 쓰면 좌/우 끝 차로
        판단이 쉽게 틀어진다.
        """
        positions: list[tuple[LaneTrack, float]] = []
        extra = max(18, int(frame_h * 0.12))
        for track in self._tracks:
            if track.missed > self.max_missed:
                continue
            y_min = int(track.polyline[:, 1].min())
            y_max = int(track.polyline[:, 1].max())
            if sample_y < y_min - extra or sample_y > y_max + extra:
                continue
            sampled = _sample_polyline_at_ys(track.polyline, np.asarray([sample_y], dtype=np.int32), frame_w)
            if sampled is not None:
                positions.append((track, float(sampled[0, 0])))

        positions.sort(key=lambda item: item[1])
        return self._merge_nearby_positions(positions, frame_w)

    @staticmethod
    def _merge_nearby_positions(positions: list[tuple[LaneTrack, float]],
                                frame_w: int) -> list[tuple[LaneTrack, float]]:
        """거의 같은 x에 겹친 중복 track은 더 안정적인 track 하나만 남긴다."""
        if not positions:
            return []

        min_sep = max(18.0, frame_w * 0.035)
        merged: list[tuple[LaneTrack, float]] = []
        for track, x in positions:
            if not merged or abs(x - merged[-1][1]) >= min_sep:
                merged.append((track, x))
                continue
            prev_track, prev_x = merged[-1]
            prev_quality = prev_track.score + 0.04 * min(prev_track.hits, 10) - 0.12 * prev_track.missed
            cur_quality = track.score + 0.04 * min(track.hits, 10) - 0.12 * track.missed
            if cur_quality > prev_quality:
                merged[-1] = (track, x)
        return merged

    def _estimate_lane_role(self,
                            left_track: Optional[LaneTrack],
                            right_track: Optional[LaneTrack],
                            result: EgoLaneResult,
                            frame_w: int,
                            frame_h: int,
                            seg_mask: np.ndarray,
                            detections=None) -> LaneRole:
        """
        주변 차선과 비교해 ego lane의 좌/우 끝 차로 여부를 판단한다.

        출력은 안전판단이 아니라 lane role 정보다. 보이지 않는 차선은 없다고
        확정하지 않는다. 바깥쪽 차선/차량처럼 "옆 차로가 있다"는 근거는 False로
        쓰고, 황색 중앙선처럼 강한 도로 구조 근거만 True로 쓴다.
        """
        if left_track is None or right_track is None:
            self._last_role_key = None
            self._role_streak = 0
            return LaneRole(reason="missing_ego_boundary")

        sample_y = int(frame_h * 0.92)
        positions = self._track_positions_at_y(frame_w, frame_h, sample_y)
        boundary_count = len(positions)
        if boundary_count < 2:
            self._last_role_key = None
            self._role_streak = 0
            return LaneRole(boundary_count=boundary_count, reason="not_enough_boundaries")

        ids = [track.track_id for track, _x in positions]
        if left_track.track_id not in ids or right_track.track_id not in ids:
            self._last_role_key = None
            self._role_streak = 0
            return LaneRole(boundary_count=boundary_count, reason="ego_boundary_not_in_sorted_tracks")

        left_idx = ids.index(left_track.track_id)
        right_idx = ids.index(right_track.track_id)
        if left_idx > right_idx:
            left_idx, right_idx = right_idx, left_idx
        if right_idx - left_idx != 1:
            self._last_role_key = None
            self._role_streak = 0
            return LaneRole(boundary_count=boundary_count, reason="ego_boundaries_not_adjacent")

        lane_count = boundary_count - 1
        lane_index_left = left_idx
        lane_index_right = lane_count - 1 - lane_index_left
        has_visible_left_lane = left_idx > 0
        has_visible_right_lane = right_idx < boundary_count - 1
        visible_left_edge = left_idx == 0
        visible_right_edge = right_idx == boundary_count - 1

        left_vehicle, right_vehicle = _vehicle_adjacent_flags(detections, result, frame_w, frame_h)
        left_yellow, left_yellow_ratio, _left_yellow_samples = _left_boundary_yellow_signal(
            seg_mask, result.left_boundary, frame_w, frame_h
        )
        right_yellow, right_yellow_ratio, _right_yellow_samples = _boundary_yellow_signal(
            seg_mask, result.right_boundary, frame_w, frame_h
        )

        left_score = 0.0
        right_score = 0.0
        reason_parts = []
        if left_yellow:
            left_score += 0.35
            if boundary_count == 2 and not right_vehicle:
                right_score += 0.12
            reason_parts.append(f"left_boundary_yellow_{left_yellow_ratio:.2f}")
        if right_yellow:
            right_score += 0.35
            reason_parts.append(f"right_boundary_yellow_{right_yellow_ratio:.2f}")
        if has_visible_left_lane:
            # If the ego left boundary is yellow, outside-left markings may be
            # centerline-side or opposite-direction markings rather than a
            # usable same-direction adjacent lane.  Down-weight the penalty.
            left_score -= 0.25 if left_yellow else 0.75
            reason_parts.append("visible_left_lane")
        elif visible_left_edge:
            left_score += 0.25 if boundary_count > 2 else 0.08
            reason_parts.append("visible_left_edge")
        if has_visible_right_lane:
            right_score -= 0.25 if right_yellow else 0.75
            reason_parts.append("visible_right_lane")
        elif visible_right_edge:
            right_score += 0.25 if boundary_count > 2 else 0.08
            reason_parts.append("visible_right_edge")
        if left_vehicle:
            # Adjacent vehicles are useful hints, but a vehicle alone can be on
            # a shoulder, side road, opposite-direction lane, or partially
            # outside the ego-road area. Keep it weaker when the boundary is
            # yellow.
            left_score -= 0.20 if left_yellow else 0.35
            reason_parts.append("left_adjacent_vehicle")
        if right_vehicle:
            right_score -= 0.20 if right_yellow else 0.35
            reason_parts.append("right_adjacent_vehicle")

        single_lane_candidate_key = (
            lane_index_left,
            lane_index_right,
            left_vehicle,
            right_vehicle,
            left_yellow,
            right_yellow,
            "single_lane_candidate",
        )
        previous_single_lane_streak = (
            self._role_streak
            if self._last_role_key == single_lane_candidate_key
            else 0
        )
        if boundary_count == 2:
            # 경계가 2개뿐이면 실제 1차로일 수도 있지만, 주변 차선을 놓친 것일 수도 있다.
            # 따라서 황색 중앙선/옆 차량/시간적 안정성 같은 추가 근거가 있을 때만 확정한다.
            reason_parts.append("two_boundaries_guard")
            if not left_yellow:
                is_leftmost = None
            else:
                is_leftmost = True

            if right_vehicle:
                is_rightmost = False
            elif (left_yellow or right_yellow) and previous_single_lane_streak >= 4:
                is_rightmost = True
                reason_parts.append("yellow_boundary_single_lane_stable")
            else:
                is_rightmost = None

        key = single_lane_candidate_key if boundary_count == 2 else (
            lane_index_left,
            lane_index_right,
            has_visible_left_lane,
            has_visible_right_lane,
            visible_left_edge,
            visible_right_edge,
            left_vehicle,
            right_vehicle,
            left_yellow,
            right_yellow,
        )
        if key == self._last_role_key:
            self._role_streak += 1
        else:
            self._last_role_key = key
            self._role_streak = 1

        stability = min(1.0, self._role_streak / 5.0)
        if self._role_streak >= 2:
            if left_score > 0:
                left_score += 0.15 * stability
            elif left_score < 0:
                left_score -= 0.10 * stability
            if right_score > 0:
                right_score += 0.15 * stability
            elif right_score < 0:
                right_score -= 0.10 * stability

        if boundary_count > 2:
            # 주변 경계가 3개 이상이면 정렬된 차선 배열의 끝이라는 근거가 있다.
            if (
                right_yellow
                and visible_right_edge
                and not has_visible_right_lane
                and not right_vehicle
                and self._role_streak >= 2
            ):
                right_score += 0.15
                reason_parts.append("right_yellow_edge_stable")
        elif boundary_count == 2 and not right_vehicle:
            # 경계가 2개만 보이면 차선을 놓친 상황과 실제 가장자리 차로가 섞인다.
            # 황색 경계나 긴 안정성이 있어야 오른쪽끝 점수가 충분히 오른다.
            if (left_yellow or right_yellow) and self._role_streak >= 4:
                right_score += 0.20
                reason_parts.append("yellow_boundary_single_lane_stable")
            elif self._role_streak >= 8:
                right_score += 0.18
                reason_parts.append("two_boundary_long_stable")

        current_left_role = self._score_to_role(left_score)
        current_right_role = self._score_to_role(right_score)
        if self._role_streak < 2:
            # 한 프레임짜리 차로 역할 근거는 차선 mask/차량 bbox 흔들림에 취약하다.
            current_left_role = None
            current_right_role = None
            reason_parts.append("await_role_confirmation")

        self._left_role_votes.append((current_left_role, left_score))
        self._right_role_votes.append((current_right_role, right_score))
        is_leftmost = self._vote_lane_role(self._left_role_votes, left_score)
        is_rightmost = self._vote_lane_role(self._right_role_votes, right_score)

        left_quality = left_track.score * max(0.0, 1.0 - 0.12 * left_track.missed)
        right_quality = right_track.score * max(0.0, 1.0 - 0.12 * right_track.missed)
        boundary_bonus = min(1.0, boundary_count / 4.0)
        confidence = float(np.clip(
            0.45 * result.confidence
            + 0.30 * np.mean([left_quality, right_quality])
            + 0.15 * stability
            + 0.10 * boundary_bonus,
            0.0,
            1.0,
        ))
        reason_parts.insert(0, f"sorted_boundaries_streak_{self._role_streak}")
        reason_parts.append(f"score_L:{left_score:.2f}")
        reason_parts.append(f"score_R:{right_score:.2f}")
        reason = ",".join(reason_parts)

        return LaneRole(
            boundary_count=boundary_count,
            ego_lane_index_from_left=lane_index_left,
            ego_lane_index_from_right=lane_index_right,
            is_leftmost_lane=is_leftmost,
            is_rightmost_lane=is_rightmost,
            left_adjacent_vehicle=left_vehicle,
            right_adjacent_vehicle=right_vehicle,
            left_boundary_yellow=left_yellow,
            left_yellow_ratio=left_yellow_ratio,
            right_boundary_yellow=right_yellow,
            right_yellow_ratio=right_yellow_ratio,
            leftmost_score=float(left_score),
            rightmost_score=float(right_score),
            confidence=confidence,
            reason=reason,
        )

    @staticmethod
    def _score_to_role(score: float) -> Optional[bool]:
        """Convert a one-frame edge score into a tentative tri-state role."""

        if score >= 0.40:
            return True
        if score <= -0.35:
            return False
        return None

    @staticmethod
    def _vote_lane_role(votes: deque, latest_score: float) -> Optional[bool]:
        """
        Turn recent tentative roles into a robust final role.

        False evidence is usually strong because it comes from a visible outside
        lane or adjacent vehicle, so it needs fewer votes. True evidence needs
        more persistence because missing outside boundaries are weaker evidence.
        """

        true_count = sum(1 for role, _score in votes if role is True)
        false_count = sum(1 for role, _score in votes if role is False)
        known_count = true_count + false_count
        if known_count == 0:
            return None

        if latest_score <= -0.70:
            return False
        if false_count >= 2 and false_count >= true_count:
            return False
        if latest_score <= -0.45 and false_count >= 1 and true_count == 0:
            return False
        if latest_score >= 0.40 and true_count >= 4 and false_count == 0:
            return True
        if true_count >= 5 and true_count >= false_count + 2:
            return True
        if latest_score >= 0.70 and true_count >= 4 and true_count >= false_count + 1:
            return True
        return None

    def _select_ego_tracks(self, frame_w: int, frame_h: int) -> tuple[Optional[LaneTrack], Optional[LaneTrack]]:
        """화면 하단 중앙을 기준으로 좌우에서 가장 가까운 안정 track을 선택한다."""
        if not self._tracks:
            return None, None

        ego_x = frame_w / 2.0
        sample_y = int(frame_h * 0.92)
        positions = self._track_positions_at_y(frame_w, frame_h, sample_y)

        left_items = [(track, x) for track, x in positions if x < ego_x - max(6, frame_w * 0.01)]
        right_items = [(track, x) for track, x in positions if x > ego_x + max(6, frame_w * 0.01)]

        best_pair: tuple[Optional[LaneTrack], Optional[LaneTrack], float] = (None, None, float("inf"))
        min_width = max(40, frame_w * 0.09)
        max_width = max(110, frame_w * 0.62)
        for left_track, left_x in left_items:
            for right_track, right_x in right_items:
                width = right_x - left_x
                if not (min_width <= width <= max_width):
                    continue
                center_penalty = abs(((left_x + right_x) / 2.0) - ego_x)
                stability_bonus = 6.0 * (min(left_track.hits, 8) + min(right_track.hits, 8))
                miss_penalty = 18.0 * (left_track.missed + right_track.missed)
                cost = center_penalty + miss_penalty - stability_bonus
                if cost < best_pair[2]:
                    best_pair = (left_track, right_track, cost)

        if best_pair[0] is not None and best_pair[1] is not None:
            return best_pair[0], best_pair[1]

        left_track = max(left_items, key=lambda item: item[1])[0] if left_items else None
        right_track = min(right_items, key=lambda item: item[1])[0] if right_items else None
        return left_track, right_track


def estimate_ego_lane(seg_mask: np.ndarray) -> EgoLaneResult:
    """단일 이미지용 convenience 함수. 비디오는 EgoLanePipeline을 직접 재사용한다."""
    return EgoLanePipeline().update(seg_mask)


def _tri_state_text(value: Optional[bool]) -> str:
    """True/False/Unknown을 짧은 화면 표시 문자열로 바꾼다."""
    if value is True:
        return "T"
    if value is False:
        return "F"
    return "U"


def draw_ego_lane_overlay(
    vis: np.ndarray,
    result: EgoLaneResult,
    alpha: float = 0.24,
    show_text: bool = True,
) -> None:
    """현재 주행 차선 영역은 채우기만 표시하고, 중앙선은 얇게 참고선으로 그린다."""
    if result.polygon is None:
        return

    overlay = vis.copy()
    cv2.fillPoly(overlay, [result.polygon.reshape(-1, 2)], (0, 180, 80))
    mask = np.zeros(vis.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [result.polygon.reshape(-1, 2)], 255)
    vis[mask > 0] = cv2.addWeighted(vis[mask > 0], 1.0 - alpha, overlay[mask > 0], alpha, 0)

    if result.centerline is not None:
        cv2.polylines(vis, [result.centerline.reshape(-1, 1, 2)], False, (255, 80, 40), 1, cv2.LINE_AA)

    if not show_text:
        return

    x, y = 10, 150
    track_text = f"L:{result.left_track_id or '-'} R:{result.right_track_id or '-'}"
    text = (
        f"ego lane {result.status} conf:{result.confidence:.2f} "
        f"cand:{result.candidate_count} trk:{result.track_count} {track_text}"
    )
    cv2.rectangle(vis, (x - 4, y - 16), (x + 365, y + 5), (0, 0, 0), -1)
    cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 180), 1, cv2.LINE_AA)

    if result.role is not None:
        role = result.role
        idx_l = "-" if role.ego_lane_index_from_left is None else str(role.ego_lane_index_from_left)
        idx_r = "-" if role.ego_lane_index_from_right is None else str(role.ego_lane_index_from_right)
        adj = (
            ("L" if role.left_adjacent_vehicle else "")
            + ("R" if role.right_adjacent_vehicle else "")
        ) or "-"
        left_yellow_text = f"LY:{role.left_yellow_ratio:.2f}" if role.left_boundary_yellow else "LY:-"
        right_yellow_text = f"RY:{role.right_yellow_ratio:.2f}" if role.right_boundary_yellow else "RY:-"
        role_text = (
            f"lane role Lidx:{idx_l} Ridx:{idx_r} "
            f"leftmost:{_tri_state_text(role.is_leftmost_lane)} "
            f"rightmost:{_tri_state_text(role.is_rightmost_lane)} "
            f"score:{role.leftmost_score:.2f}/{role.rightmost_score:.2f} "
            f"adjV:{adj} {left_yellow_text} {right_yellow_text} conf:{role.confidence:.2f}"
        )
        cv2.rectangle(vis, (x - 4, y + 6), (x + 510, y + 27), (0, 0, 0), -1)
        cv2.putText(vis, role_text, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (0, 230, 255), 1, cv2.LINE_AA)
