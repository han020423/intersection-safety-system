"""
postprocess.py — BiSeNetV2 결과 후처리

ego lane polygon, lane centerline, crosswalk active zone, path corridor 생성
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class LaneGeometry:
    """후처리 결과를 담는 구조체."""
    # ego lane polygon 꼭짓점 (4~N점)
    ego_lane_polygon: Optional[np.ndarray] = None
    # lane centerline 점 리스트 [(x,y), ...]
    centerline: Optional[np.ndarray] = None
    # 좌측 차선 polyline
    left_lane_pts: Optional[np.ndarray] = None
    # 우측 차선 polyline
    right_lane_pts: Optional[np.ndarray] = None
    # crosswalk active zone polygon
    crosswalk_zone: Optional[np.ndarray] = None
    # crosswalk 원본 마스크 픽셀 수
    crosswalk_pixel_count: int = 0
    # path corridor polygon
    path_corridor: Optional[np.ndarray] = None
    # lane confidence (0~1)
    lane_confidence: float = 0.0
    # stop line 감지 여부 및 y좌표
    stop_line_y: Optional[int] = None


LANE_CLASS_MIN = 1
LANE_CLASS_MAX = 6


def _is_vehicle_detection(det) -> bool:
    name = getattr(det, 'cls_name', '').lower()
    return name == 'vehicle' or name in ('car', 'bus', 'truck', 'motorcycle')


def suppress_lane_pixels_in_vehicle_boxes(seg_mask: np.ndarray,
                                          detections=None,
                                          pad_ratio: float = 0.08) -> np.ndarray:
    """
    YOLO 차량 bbox 안의 lane class 픽셀을 배경으로 지운다.
    차량 가장자리/그림자가 차선처럼 이어져 polygon이 뒤틀리는 것을 줄인다.
    """
    if not detections:
        return seg_mask

    filtered = seg_mask.copy()
    h, w = filtered.shape[:2]
    for det in detections:
        if not _is_vehicle_detection(det):
            continue
        x1, y1, x2, y2 = [int(v) for v in det.box]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        roi = filtered[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        lane_roi = (roi >= LANE_CLASS_MIN) & (roi <= LANE_CLASS_MAX)
        roi[lane_roi] = 0
    return filtered


def _smooth_polyline(pts: np.ndarray, poly_degree: int = 2) -> np.ndarray:
    """
    polyline 점들을 다항식 피팅으로 부드럽게 만든다.

    1) 이상치 제거: x좌표가 median에서 크게 벗어나는 점 제거
    2) y→x 다항식 피팅 (2차)
    3) 원래 y좌표에 대해 피팅된 x좌표를 계산

    Args:
        pts: (N, 2) 배열 — [(x, y), ...]
        poly_degree: 다항식 차수 (기본 2차)

    Returns:
        부드러운 (N, 2) 배열
    """
    if len(pts) < 3:
        return pts

    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)

    # ── 이상치 제거 (Median Absolute Deviation 기반) ──
    # 인접 점 간 x 변화량이 비정상적으로 큰 점을 제거
    median_x = np.median(xs)
    mad = np.median(np.abs(xs - median_x))
    threshold = max(mad * 3.0, 20.0)  # 최소 20px 허용
    inlier_mask = np.abs(xs - median_x) < threshold

    # 추가: 인접 점 간 x 점프가 너무 크면 제거
    for i in range(1, len(xs)):
        if abs(xs[i] - xs[i - 1]) > threshold * 1.5:
            inlier_mask[i] = False

    xs_clean = xs[inlier_mask]
    ys_clean = ys[inlier_mask]

    if len(xs_clean) < 3:
        return pts  # 이상치 제거 후 점이 부족하면 원본 반환

    # ── 다항식 피팅 (y → x) ──
    # y가 독립변수, x가 종속변수 (차선은 세로 방향이므로)
    degree = min(poly_degree, len(xs_clean) - 1)
    try:
        coeffs = np.polyfit(ys_clean, xs_clean, degree)
        poly = np.poly1d(coeffs)
        xs_fitted = poly(ys).astype(np.int32)
        # 원래 이미지 범위로 클리핑
        xs_fitted = np.clip(xs_fitted, 0, max(xs_fitted.max(), int(xs.max())))
        result = np.column_stack([xs_fitted, ys.astype(np.int32)])
        return result
    except (np.linalg.LinAlgError, ValueError):
        return pts  # 피팅 실패 시 원본 반환


def extract_lane_polylines(seg_mask: np.ndarray,
                           scan_top_ratio: float = 0.35,
                           scan_bot_ratio: float = 0.95,
                           num_scan_lines: int = 20
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    세그멘테이션 마스크에서 ego vehicle 기준 좌/우 차선 polyline을 추출한다.

    알고리즘:
      1. 화면 하단~중간 영역을 수평으로 여러 줄(scan line) 스캔
      2. 각 scan line에서 차선 픽셀(class 1~6)의 x좌표 클러스터를 찾음
      3. 화면 하단 중앙(ego position) 기준으로 가장 가까운 좌측/우측 차선 선택
      4. 각 scan line의 좌/우 점을 연결하여 polyline 생성
      5. 2차 다항식 피팅으로 부드러운 곡선 생성 (지그재그 방지)

    Returns:
        (left_pts, right_pts, confidence)
    """
    h, w = seg_mask.shape[:2]
    ego_x = w // 2  # 차량 중심 = 화면 중앙

    # 차선 마스크 (class 1~6)
    lane_mask = ((seg_mask >= LANE_CLASS_MIN) & (seg_mask <= LANE_CLASS_MAX)).astype(np.uint8)

    scan_y_start = int(h * scan_bot_ratio)
    scan_y_end = int(h * scan_top_ratio)
    scan_ys = np.linspace(scan_y_start, scan_y_end, num_scan_lines, dtype=int)

    left_points = []
    right_points = []
    valid_count = 0

    for y in scan_ys:
        if y < 0 or y >= h:
            continue
        row = lane_mask[y]
        lane_xs = np.where(row > 0)[0]
        if len(lane_xs) < 2:
            continue

        # x좌표 클러스터링 (간격 > 15px이면 별도 클러스터)
        clusters = []
        current = [lane_xs[0]]
        for i in range(1, len(lane_xs)):
            if lane_xs[i] - lane_xs[i - 1] <= 15:
                current.append(lane_xs[i])
            else:
                clusters.append(current)
                current = [lane_xs[i]]
        clusters.append(current)

        # 노이즈 제거 (너무 작은 클러스터)
        clusters = [c for c in clusters if len(c) >= 3]
        if not clusters:
            continue

        # 각 클러스터 중심
        centers = [int(np.mean(c)) for c in clusters]
        centers.sort()

        # ego_x 기준 좌측/우측에서 가장 가까운 클러스터
        left_candidates = [cx for cx in centers if cx < ego_x]
        right_candidates = [cx for cx in centers if cx > ego_x]

        if left_candidates:
            left_points.append((left_candidates[-1], int(y)))
        if right_candidates:
            right_points.append((right_candidates[0], int(y)))

        if left_candidates or right_candidates:
            valid_count += 1

    confidence = valid_count / max(1, num_scan_lines)

    # raw 점을 다항식 피팅으로 부드럽게 처리
    left_pts = None
    right_pts = None
    if len(left_points) >= 3:
        left_pts = _smooth_polyline(np.array(left_points, dtype=np.int32))
    elif len(left_points) >= 2:
        left_pts = np.array(left_points, dtype=np.int32)

    if len(right_points) >= 3:
        right_pts = _smooth_polyline(np.array(right_points, dtype=np.int32))
    elif len(right_points) >= 2:
        right_pts = np.array(right_points, dtype=np.int32)

    return left_pts, right_pts, confidence


def _prepare_lane_points(pts: np.ndarray) -> Optional[np.ndarray]:
    """y 내림차순으로 정렬하고 같은 y의 중복 x를 평균낸다."""
    if pts is None or len(pts) < 2:
        return None

    pts = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
    by_y = {}
    for x, y in pts:
        by_y.setdefault(int(y), []).append(int(x))

    merged = np.array(
        [[int(np.mean(xs)), y] for y, xs in by_y.items()],
        dtype=np.int32,
    )
    if len(merged) < 2:
        return None
    return merged[np.argsort(-merged[:, 1])]


def _fit_lane_at_ys(pts: np.ndarray, target_ys: np.ndarray,
                    frame_w: int) -> Optional[np.ndarray]:
    """차선을 y->x 직선/곡선으로 피팅해 target y좌표의 점으로 재샘플링한다."""
    pts = _prepare_lane_points(pts)
    if pts is None:
        return None

    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)
    degree = min(2, len(pts) - 1)
    try:
        coeffs = np.polyfit(ys, xs, degree)
        pred_xs = np.poly1d(coeffs)(target_ys.astype(np.float64))

        # 피팅이 raw point에서 과하게 벗어나면 직선으로 후퇴한다.
        raw_pred = np.poly1d(coeffs)(ys)
        if float(np.max(np.abs(raw_pred - xs))) > frame_w * 0.16:
            coeffs = np.polyfit(ys, xs, 1)
            pred_xs = np.poly1d(coeffs)(target_ys.astype(np.float64))
    except (np.linalg.LinAlgError, ValueError):
        return None

    pred_xs = np.clip(pred_xs, 0, frame_w - 1).astype(np.int32)
    return np.column_stack([pred_xs, target_ys.astype(np.int32)])


def _aligned_lane_pair(left_pts: np.ndarray, right_pts: np.ndarray,
                       frame_h: int, frame_w: int,
                       n_points: int = 16
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    좌/우 차선을 같은 y좌표 기준으로 맞춘다.
    점선 차선이 화면 하단에서 잠깐 끊겨도 짧게 외삽해 polygon 꼬임을 막는다.
    """
    left = _prepare_lane_points(left_pts)
    right = _prepare_lane_points(right_pts)
    if left is None or right is None:
        return None, None

    l_min_y, l_max_y = int(left[:, 1].min()), int(left[:, 1].max())
    r_min_y, r_max_y = int(right[:, 1].min()), int(right[:, 1].max())
    max_extra = max(24, int(frame_h * 0.18))

    # 우선 더 긴 범위를 쓰되, 어느 한쪽이 너무 멀리 외삽되면 공통 범위로 후퇴한다.
    y_bottom = max(l_max_y, r_max_y)
    y_top = min(l_min_y, r_min_y)
    needs_too_much_extra = (
        y_bottom - l_max_y > max_extra or
        y_bottom - r_max_y > max_extra or
        l_min_y - y_top > max_extra or
        r_min_y - y_top > max_extra
    )
    if needs_too_much_extra:
        y_bottom = min(l_max_y, r_max_y)
        y_top = max(l_min_y, r_min_y)

    if y_bottom <= y_top:
        return None, None

    target_ys = np.linspace(y_bottom, y_top, n_points, dtype=np.int32)
    left_aligned = _fit_lane_at_ys(left, target_ys, frame_w)
    right_aligned = _fit_lane_at_ys(right, target_ys, frame_w)
    if left_aligned is None or right_aligned is None:
        return None, None

    # 좌/우가 뒤집히거나 너무 좁아진 샘플은 polygon을 꼬이게 하므로 제거한다.
    lane_widths = right_aligned[:, 0] - left_aligned[:, 0]
    valid = lane_widths > max(24, int(frame_w * 0.06))
    if int(np.count_nonzero(valid)) < 3:
        return None, None

    return left_aligned[valid], right_aligned[valid]


def build_ego_lane_polygon(left_pts: Optional[np.ndarray],
                           right_pts: Optional[np.ndarray],
                           frame_h: int, frame_w: int
                           ) -> Optional[np.ndarray]:
    """
    좌/우 차선 polyline을 이용해 ego lane polygon을 생성한다.
    한쪽만 있으면 다른 쪽을 추정하여 polygon을 만든다.
    """
    if left_pts is None and right_pts is None:
        return None

    ego_x = frame_w // 2
    estimated_lane_width = frame_w // 4  # 추정 차선 폭

    if left_pts is not None and right_pts is not None:
        # 양쪽 다 있으면 같은 y좌표로 맞춘 뒤 좌측 아래→위, 우측 위→아래 순서로 polygon
        left_aligned, right_aligned = _aligned_lane_pair(left_pts, right_pts, frame_h, frame_w)
        if left_aligned is not None and right_aligned is not None:
            polygon = np.vstack([left_aligned, right_aligned[::-1]])
        else:
            polygon = np.vstack([left_pts, right_pts[::-1]])
    elif left_pts is not None:
        # 좌측만 있으면 우측을 추정
        right_estimated = left_pts.copy()
        right_estimated[:, 0] = np.clip(left_pts[:, 0] + estimated_lane_width, 0, frame_w - 1)
        polygon = np.vstack([left_pts, right_estimated[::-1]])
    else:
        # 우측만 있으면 좌측을 추정
        left_estimated = right_pts.copy()
        left_estimated[:, 0] = np.clip(right_pts[:, 0] - estimated_lane_width, 0, frame_w - 1)
        polygon = np.vstack([left_estimated, right_pts[::-1]])

    return polygon.reshape((-1, 1, 2))


def build_lane_centerline(left_pts: Optional[np.ndarray],
                          right_pts: Optional[np.ndarray],
                          frame_w: int,
                          frame_h: Optional[int] = None
                          ) -> Optional[np.ndarray]:
    """좌/우 차선의 중간점을 연결하여 centerline을 생성한다."""
    if left_pts is None and right_pts is None:
        return None

    if left_pts is not None and right_pts is not None:
        if frame_h is not None:
            left_aligned, right_aligned = _aligned_lane_pair(left_pts, right_pts, frame_h, frame_w)
            if left_aligned is not None and right_aligned is not None:
                cx = (left_aligned[:, 0] + right_aligned[:, 0]) // 2
                cy = left_aligned[:, 1]
                return np.column_stack([cx, cy]).astype(np.int32)

        # fallback: 양쪽 점 개수가 다를 수 있으므로 기존 방식 유지
        n = min(len(left_pts), len(right_pts))
        center = []
        for i in range(n):
            cx = (left_pts[i][0] + right_pts[i][0]) // 2
            cy = (left_pts[i][1] + right_pts[i][1]) // 2
            center.append((cx, cy))
        return np.array(center, dtype=np.int32) if center else None
    elif left_pts is not None:
        offset = frame_w // 8
        center = left_pts.copy()
        center[:, 0] = np.clip(center[:, 0] + offset, 0, frame_w - 1)
        return center
    else:
        offset = frame_w // 8
        center = right_pts.copy()
        center[:, 0] = np.clip(center[:, 0] - offset, 0, frame_w - 1)
        return center


def build_crosswalk_active_zone(seg_mask: np.ndarray,
                                ego_polygon: Optional[np.ndarray],
                                frame_h: int, frame_w: int
                                ) -> Tuple[Optional[np.ndarray], int]:
    """
    crosswalk 마스크 중 ego lane / path corridor와 관련 있는 영역만 추출한다.

    Returns:
        (crosswalk_zone_contour, pixel_count)
    """
    # crosswalk 전체 마스크
    cw_mask = (seg_mask == 7).astype(np.uint8) * 255
    total_pixels = int(np.count_nonzero(cw_mask))

    if total_pixels < 100:
        return None, 0

    # ego lane polygon이 있으면 교집합 영역만 취함
    if ego_polygon is not None:
        lane_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        # polygon을 약간 확장 (crosswalk는 차선 밖으로도 걸침)
        cv2.fillPoly(lane_mask, [ego_polygon.reshape(-1, 2)], 255)
        kernel = np.ones((30, 30), np.uint8)
        lane_mask = cv2.dilate(lane_mask, kernel, iterations=1)
        cw_mask = cv2.bitwise_and(cw_mask, lane_mask)

    pixel_count = int(np.count_nonzero(cw_mask))
    if pixel_count < 50:
        return None, 0

    # 가장 큰 contour를 zone으로
    contours, _ = cv2.findContours(cw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, pixel_count

    biggest = max(contours, key=cv2.contourArea)
    return biggest, pixel_count


def build_path_corridor(centerline: Optional[np.ndarray],
                        corridor_half_width: int = 80,
                        frame_w: int = 640
                        ) -> Optional[np.ndarray]:
    """
    centerline 좌우로 corridor_half_width만큼 확장한 polygon을 생성한다.
    직진 방향의 예상 주행 경로.
    """
    if centerline is None or len(centerline) < 2:
        return None

    left_side = centerline.copy()
    right_side = centerline.copy()
    left_side[:, 0] = np.clip(centerline[:, 0] - corridor_half_width, 0, frame_w - 1)
    right_side[:, 0] = np.clip(centerline[:, 0] + corridor_half_width, 0, frame_w - 1)

    polygon = np.vstack([left_side, right_side[::-1]])
    return polygon.reshape((-1, 1, 2))


def detect_stop_line_y(seg_mask: np.ndarray) -> Optional[int]:
    """stop line(class 8)의 평균 y좌표를 반환."""
    stop_ys = np.where(seg_mask == 8)[0]
    if len(stop_ys) < 10:
        return None
    return int(np.mean(stop_ys))


def compute_lane_geometry(seg_mask: np.ndarray,
                          frame_h: int, frame_w: int,
                          corridor_half_width: int = 80,
                          detections=None,
                          suppress_vehicle_lanes: bool = True
                          ) -> LaneGeometry:
    """세그멘테이션 결과로부터 전체 도로 기하 구조를 계산한다."""
    geo = LaneGeometry()

    lane_source_mask = (
        suppress_lane_pixels_in_vehicle_boxes(seg_mask, detections)
        if suppress_vehicle_lanes else seg_mask
    )

    # 1) 차선 polyline 추출
    left_pts, right_pts, conf = extract_lane_polylines(lane_source_mask)
    geo.left_lane_pts = left_pts
    geo.right_lane_pts = right_pts
    geo.lane_confidence = conf

    # 2) ego lane polygon
    geo.ego_lane_polygon = build_ego_lane_polygon(left_pts, right_pts, frame_h, frame_w)

    # 3) lane centerline
    geo.centerline = build_lane_centerline(left_pts, right_pts, frame_w, frame_h)

    # 4) crosswalk active zone
    geo.crosswalk_zone, geo.crosswalk_pixel_count = build_crosswalk_active_zone(
        seg_mask, geo.ego_lane_polygon, frame_h, frame_w
    )

    # 5) path corridor
    geo.path_corridor = build_path_corridor(geo.centerline, corridor_half_width, frame_w)

    # 6) stop line
    geo.stop_line_y = detect_stop_line_y(seg_mask)

    return geo


# ═══════════════════ 시간적 스무딩 (Temporal EMA) ═══════════════════ #


class LaneTracker:
    """
    프레임 간 차선 위치를 시간적으로 스무딩하는 트래커.

    원리:
      - 매 프레임의 좌/우 차선 polyline을 2차 다항식 계수로 변환
      - 이전 프레임 계수와 EMA(지수이동평균)로 블렌딩
      - lane confidence가 낮으면 이전 결과를 더 강하게 유지
      - 연속으로 감지 실패하면 캐시를 포기

    사용법:
        tracker = LaneTracker(alpha=0.4)
        for frame in frames:
            geo = compute_lane_geometry(seg_mask, h, w)
            geo = tracker.smooth(geo, h, w)  # 스무딩 적용
    """

    def __init__(self, alpha: float = 0.4, max_miss: int = 10,
                 model_gate_ratio: float = 0.12, resample_points: int = 18):
        """
        Args:
            alpha: EMA 가중치 (0~1). 클수록 현재 프레임 반영 ↑, 작을수록 안정적
            max_miss: 연속 미감지 허용 프레임 수. 초과하면 캐시 폐기
        """
        self.alpha = alpha
        self.max_miss = max_miss
        self.model_gate_ratio = model_gate_ratio
        self.resample_points = resample_points
        # 이전 프레임의 다항식 계수 캐시 (y→x 2차 다항식)
        self._left_coeffs: Optional[np.ndarray] = None
        self._right_coeffs: Optional[np.ndarray] = None
        # 이전 프레임의 y좌표 범위 (피팅 재생성용)
        self._left_ys: Optional[np.ndarray] = None
        self._right_ys: Optional[np.ndarray] = None
        # 연속 미감지 카운터
        self._left_miss = 0
        self._right_miss = 0

    def smooth(self, geo: LaneGeometry,
               frame_h: int, frame_w: int,
               corridor_half_width: int = 80) -> LaneGeometry:
        """
        LaneGeometry의 좌/우 polyline에 시간적 스무딩을 적용하고,
        ego polygon / centerline / corridor를 재계산한다.
        """
        # ── 좌측 차선 스무딩 ──
        geo.left_lane_pts = self._smooth_side(
            geo.left_lane_pts, 'left', frame_w
        )
        # ── 우측 차선 스무딩 ──
        geo.right_lane_pts = self._smooth_side(
            geo.right_lane_pts, 'right', frame_w
        )

        # ── 스무딩된 polyline으로 파생 구조 재계산 ──
        geo.ego_lane_polygon = build_ego_lane_polygon(
            geo.left_lane_pts, geo.right_lane_pts, frame_h, frame_w
        )
        geo.centerline = build_lane_centerline(
            geo.left_lane_pts, geo.right_lane_pts, frame_w, frame_h
        )
        geo.path_corridor = build_path_corridor(
            geo.centerline, corridor_half_width, frame_w
        )

        return geo

    def _smooth_side(self, pts: Optional[np.ndarray],
                     side: str, frame_w: int) -> Optional[np.ndarray]:
        """한쪽 차선의 시간적 스무딩 처리."""
        if side == 'left':
            prev_coeffs = self._left_coeffs
            prev_ys = self._left_ys
        else:
            prev_coeffs = self._right_coeffs
            prev_ys = self._right_ys

        if pts is not None and len(pts) >= 3:
            # 현재 프레임에서 차선 감지 성공
            xs = pts[:, 0].astype(np.float64)
            ys = pts[:, 1].astype(np.float64)

            # 이전 차선 모델과 너무 멀리 떨어진 후보는 다른 차선/노이즈로 보고 거부한다.
            if prev_coeffs is not None and prev_ys is not None:
                pred_prev = np.poly1d(prev_coeffs)(ys)
                median_diff = float(np.median(np.abs(pred_prev - xs)))
                gate_px = max(35.0, frame_w * self.model_gate_ratio)
                if median_diff > gate_px:
                    if side == 'left':
                        self._left_miss += 1
                    else:
                        self._right_miss += 1
                    return self._coeffs_to_pts(prev_coeffs, prev_ys, frame_w)

            # 2차 다항식 피팅 (y → x)
            degree = min(2, len(xs) - 1)
            try:
                cur_coeffs = np.polyfit(ys, xs, degree)
                # degree가 2 미만이면 0으로 패딩하여 항상 3개 계수 유지
                while len(cur_coeffs) < 3:
                    cur_coeffs = np.insert(cur_coeffs, 0, 0.0)
            except (np.linalg.LinAlgError, ValueError):
                # 피팅 실패 — 이전 캐시 유지
                if prev_coeffs is not None and prev_ys is not None:
                    return self._coeffs_to_pts(prev_coeffs, prev_ys, frame_w)
                return pts

            # EMA 블렌딩
            if prev_coeffs is not None:
                blended = self.alpha * cur_coeffs + (1.0 - self.alpha) * prev_coeffs
            else:
                blended = cur_coeffs

            resampled_ys = self._resample_ys(ys, self.resample_points)

            # 캐시 갱신
            if side == 'left':
                self._left_coeffs = blended
                self._left_ys = resampled_ys
                self._left_miss = 0
            else:
                self._right_coeffs = blended
                self._right_ys = resampled_ys
                self._right_miss = 0

            return self._coeffs_to_pts(blended, resampled_ys, frame_w)

        else:
            # 현재 프레임에서 차선 미감지 — 이전 캐시 사용
            if side == 'left':
                self._left_miss += 1
                if self._left_miss > self.max_miss:
                    self._left_coeffs = None
                    self._left_ys = None
                    return None
            else:
                self._right_miss += 1
                if self._right_miss > self.max_miss:
                    self._right_coeffs = None
                    self._right_ys = None
                    return None

            if prev_coeffs is not None and prev_ys is not None:
                return self._coeffs_to_pts(prev_coeffs, prev_ys, frame_w)
            return pts

    @staticmethod
    def _resample_ys(ys: np.ndarray, n_points: int) -> np.ndarray:
        """점선 gap을 부드럽게 잇기 위해 y 범위를 균일하게 재샘플링한다."""
        if len(ys) == 0:
            return ys
        y_bottom = int(np.max(ys))
        y_top = int(np.min(ys))
        if y_bottom <= y_top or n_points <= 1:
            return ys.astype(np.int32)
        return np.linspace(y_bottom, y_top, n_points, dtype=np.int32)

    @staticmethod
    def _coeffs_to_pts(coeffs: np.ndarray, ys: np.ndarray,
                       frame_w: int) -> np.ndarray:
        """다항식 계수 + y좌표 → (N, 2) 점 배열로 복원."""
        poly = np.poly1d(coeffs)
        xs = poly(ys).astype(np.int32)
        xs = np.clip(xs, 0, frame_w - 1)
        return np.column_stack([xs, ys.astype(np.int32)])
