"""
state_machine.py — 교차로 접근/진입 상태 FSM + STOP/CAUTION/GO 판단

상태:
  LANE_TRACKING        — 정상 차선 추적 중
  CROSSWALK_APPROACH   — 횡단보도 접근 중
  ENTERING_INTERSECTION — 교차로 진입 중
  INTERSECTION_TRACKING — 교차로 내부 주행 중
  RELOCK_LANE          — 교차로 통과 후 차선 재확보

판단:
  STOP     — 즉시 정지 필요 (보행자 위험 등)
  CAUTION  — 주의 필요 (불확실한 상황)
  GO       — 안전하게 진행 가능
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2

VEHICLE_STOP_DISTANCE_M = 7.0
VEHICLE_CAUTION_DISTANCE_M = 15.0
CONFLICTING_VEHICLE_CAUTION_DISTANCE_M = 20.0
LANE_RECOVERY_CONF = 0.30
LANE_STABLE_CONF = 0.50


class DrivingState(Enum):
    LANE_TRACKING = "LANE_TRACKING"
    CROSSWALK_APPROACH = "CROSSWALK_APPROACH"
    ENTERING_INTERSECTION = "ENTERING_INTERSECTION"
    INTERSECTION_TRACKING = "INTERSECTION_TRACKING"
    RELOCK_LANE = "RELOCK_LANE"


class Decision(Enum):
    STOP = "STOP"
    CAUTION = "CAUTION"
    GO = "GO"


class SignalState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


# 판단 결과에 대응하는 색상 (BGR)
DECISION_COLORS = {
    Decision.STOP:    (0, 0, 255),     # 빨강
    Decision.CAUTION: (0, 200, 255),   # 주황/노랑
    Decision.GO:      (0, 255, 0),     # 초록
}


@dataclass
class SceneContext:
    """한 프레임의 장면 컨텍스트. 상태 전이와 판단에 사용."""
    lane_confidence: float = 0.0
    crosswalk_pixel_count: int = 0
    crosswalk_zone_exists: bool = False
    stop_line_visible: bool = False
    stop_line_y_ratio: float = 1.0  # 0=화면 상단, 1=하단
    pedestrian_in_corridor: bool = False
    pedestrian_in_crosswalk: bool = False
    pedestrian_near_crosswalk: bool = False
    pedestrian_count: int = 0
    vehicle_count: int = 0
    nearest_vehicle_distance_m: Optional[float] = None
    nearest_turn_path_vehicle_distance_m: Optional[float] = None
    nearest_conflicting_vehicle_distance_m: Optional[float] = None
    vehicle_in_turn_path: bool = False
    conflicting_vehicle_count: int = 0
    has_traffic_light: bool = False
    vehicle_signal: SignalState = SignalState.UNKNOWN
    pedestrian_signal: SignalState = SignalState.UNKNOWN
    right_turn_signal: SignalState = SignalState.NONE
    red_light_stop_required: bool = False
    stop_completed_on_red: bool = False
    reason: str = "clear"
    frame_h: int = 0
    frame_w: int = 0


class IntersectionFSM:
    """
    Rule-based 유한 상태 머신.
    매 프레임마다 update()를 호출하여 상태를 전이하고 판단을 내린다.
    """

    def __init__(self):
        self.state = DrivingState.LANE_TRACKING
        self.decision = Decision.GO
        self.state_frames = 0  # 현재 상태에서 머문 프레임 수
        self._prev_crosswalk_pixels = 0
        # 히스테리시스를 위한 smoothing 카운터
        self._low_conf_count = 0
        self._high_conf_count = 0

    def update(self, ctx: SceneContext) -> None:
        """매 프레임 호출. 상태 전이 → 판단 수행."""
        self.state_frames += 1
        prev_state = self.state

        # ── 상태 전이 로직 ──
        self._update_state(ctx)

        if self.state != prev_state:
            self.state_frames = 0

        # ── 판단 로직 ──
        self.decision = self._decide(ctx)

        self._prev_crosswalk_pixels = ctx.crosswalk_pixel_count

    def _update_state(self, ctx: SceneContext) -> None:
        """Rule-based 상태 전이."""
        lc = ctx.lane_confidence
        cw_pixels = ctx.crosswalk_pixel_count
        cw_growing = cw_pixels > self._prev_crosswalk_pixels * 1.1
        frame_area = max(1, ctx.frame_h * ctx.frame_w)
        cw_ratio = cw_pixels / frame_area
        no_intersection_marker = not ctx.crosswalk_zone_exists and not ctx.stop_line_visible
        lane_recovered = lc >= LANE_RECOVERY_CONF

        # lane confidence smoothing (히스테리시스)
        if lc < 0.25:
            self._low_conf_count += 1
            self._high_conf_count = 0
        elif lc > LANE_STABLE_CONF:
            self._high_conf_count += 1
            self._low_conf_count = 0
        else:
            self._low_conf_count = max(0, self._low_conf_count - 1)
            self._high_conf_count = max(0, self._high_conf_count - 1)

        if self.state == DrivingState.LANE_TRACKING:
            # 횡단보도가 전방에 나타나고 커지면 → CROSSWALK_APPROACH
            if ctx.crosswalk_zone_exists and cw_growing and cw_ratio > 0.005:
                self.state = DrivingState.CROSSWALK_APPROACH
            # 정지선이 화면 하단에 가까이 오면 → CROSSWALK_APPROACH
            elif ctx.stop_line_visible and ctx.stop_line_y_ratio > 0.6:
                self.state = DrivingState.CROSSWALK_APPROACH
            # lane confidence 급락 → ENTERING_INTERSECTION
            elif self._low_conf_count >= 5:
                self.state = DrivingState.ENTERING_INTERSECTION

        elif self.state == DrivingState.CROSSWALK_APPROACH:
            # 정지선을 지나감 (화면 하단 아래) → ENTERING_INTERSECTION
            if ctx.stop_line_visible and ctx.stop_line_y_ratio > 0.85:
                self.state = DrivingState.ENTERING_INTERSECTION
            # 횡단보도/정지선이 사라지고 차선이 회복되면 정상 추적으로 복귀
            elif no_intersection_marker and lane_recovered and self.state_frames > 12:
                self.state = DrivingState.LANE_TRACKING
            # crosswalk가 사라지고 lane confidence가 오래 낮아짐
            elif no_intersection_marker and self._low_conf_count >= 6:
                self.state = DrivingState.ENTERING_INTERSECTION
            # crosswalk 없이 lane 안정적이면 복귀
            elif no_intersection_marker and lc > LANE_STABLE_CONF and self.state_frames > 8:
                self.state = DrivingState.LANE_TRACKING

        elif self.state == DrivingState.ENTERING_INTERSECTION:
            # 교차로 표지가 사라지고 차선이 다시 보이면 재확보 단계로
            if no_intersection_marker and lane_recovered and self.state_frames > 8:
                self.state = DrivingState.RELOCK_LANE
            # 교차로 내부 (lane 없음)
            elif self._low_conf_count >= 8:
                self.state = DrivingState.INTERSECTION_TRACKING
            # 바로 lane 재확보
            elif self._high_conf_count >= 5:
                self.state = DrivingState.RELOCK_LANE

        elif self.state == DrivingState.INTERSECTION_TRACKING:
            # v4처럼 차선 신뢰도가 0.3대에서 회복되는 경우도 탈출 후보로 본다.
            if no_intersection_marker and lane_recovered and self.state_frames > 10:
                self.state = DrivingState.RELOCK_LANE
            # 새 lane 잡히면 → RELOCK_LANE
            elif self._high_conf_count >= 5:
                self.state = DrivingState.RELOCK_LANE

        elif self.state == DrivingState.RELOCK_LANE:
            # lane이 안정적으로 잡히면 → LANE_TRACKING
            if no_intersection_marker and lane_recovered and self.state_frames > 10:
                self.state = DrivingState.LANE_TRACKING
            # 다시 불안정해지면 교차로로 복귀
            elif self._low_conf_count >= 5:
                self.state = DrivingState.INTERSECTION_TRACKING

    def _decide(self, ctx: SceneContext) -> Decision:
        """우회전 보조장치용 Rule-based 최종 판단."""
        # ── STOP 조건 ──
        # 1) 우회전 전용 신호등이 적색이면 우회전 금지
        if ctx.right_turn_signal == SignalState.RED:
            ctx.reason = "right-turn signal red"
            return Decision.STOP

        # 2) 전방 차량신호 적색이면 우선 정지선/횡단보도/교차로 직전 일시정지
        if ctx.red_light_stop_required and not ctx.stop_completed_on_red:
            ctx.reason = "front vehicle signal red: stop first"
            return Decision.STOP

        # 3) 횡단보도 통행 중/통행하려는 보행자 보호
        if ctx.pedestrian_in_corridor:
            ctx.reason = "pedestrian in turn path"
            return Decision.STOP
        if ctx.pedestrian_in_crosswalk:
            ctx.reason = "pedestrian on crosswalk"
            return Decision.STOP
        if ctx.pedestrian_near_crosswalk:
            ctx.reason = "pedestrian near crosswalk"
            return Decision.STOP
        if self.state == DrivingState.CROSSWALK_APPROACH and ctx.pedestrian_count > 0:
            ctx.reason = "pedestrian while approaching crosswalk"
            return Decision.STOP

        # 4) 우회전 예상 경로 안의 차량은 거리 기반으로 단계화
        if ctx.vehicle_in_turn_path:
            d = ctx.nearest_turn_path_vehicle_distance_m
            if d is None:
                ctx.reason = "vehicle in turn path: distance unknown"
                return Decision.CAUTION
            if d < VEHICLE_STOP_DISTANCE_M:
                ctx.reason = f"vehicle in turn path < {VEHICLE_STOP_DISTANCE_M:.0f}m"
                return Decision.STOP
            if d < VEHICLE_CAUTION_DISTANCE_M:
                ctx.reason = f"vehicle in turn path {d:.1f}m"
                return Decision.CAUTION

        # ── CAUTION 조건 ──
        # 5) 황색 신호는 교차로 진입 전 정지 원칙. 보조장치는 보수적으로 주의/정지 유도
        if ctx.vehicle_signal == SignalState.YELLOW or ctx.right_turn_signal == SignalState.YELLOW:
            ctx.reason = "yellow signal"
            return Decision.CAUTION
        # 6) 적색에서 이미 일시정지한 뒤라도 다른 차마 교통 방해 여부 확인 필요
        if ctx.red_light_stop_required and ctx.stop_completed_on_red:
            ctx.reason = "red after stop: yield and creep"
            return Decision.CAUTION
        # 7) 교차 차량/합류 차량이 있으면 양보 확인
        if ctx.conflicting_vehicle_count > 0:
            d = ctx.nearest_conflicting_vehicle_distance_m
            if d is None or d < CONFLICTING_VEHICLE_CAUTION_DISTANCE_M:
                ctx.reason = "conflicting vehicle" if d is None else f"conflicting vehicle {d:.1f}m"
                return Decision.CAUTION
        # 8) 교차로 진입/내부: 기본적으로 주의
        if self.state in (DrivingState.ENTERING_INTERSECTION,
                          DrivingState.INTERSECTION_TRACKING):
            ctx.reason = "possible intersection / lane uncertain"
            return Decision.CAUTION
        # 9) 횡단보도 접근 중 (보행자 없음)
        if self.state == DrivingState.CROSSWALK_APPROACH:
            ctx.reason = "approaching crosswalk"
            return Decision.CAUTION
        # 10) lane confidence 낮으면 주의
        if ctx.lane_confidence < 0.3:
            ctx.reason = "low lane confidence"
            return Decision.CAUTION
        # 11) RELOCK 상태
        if self.state == DrivingState.RELOCK_LANE:
            ctx.reason = "relocking lane"
            return Decision.CAUTION

        # ── GO ──
        ctx.reason = "clear: slow right turn"
        return Decision.GO


def _parse_signal_state(value, default=SignalState.UNKNOWN) -> SignalState:
    if isinstance(value, SignalState):
        return value
    if value is None:
        return default
    value = str(value).strip().upper()
    if value == "AUTO":
        return default
    return SignalState.__members__.get(value, default)


def _classify_light_crop(frame, box) -> SignalState:
    """HSV 색상 비율로 신호등 crop의 RED/YELLOW/GREEN을 보수적으로 추정."""
    if frame is None:
        return SignalState.UNKNOWN
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return SignalState.UNKNOWN

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return SignalState.UNKNOWN

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    bright = (sat > 70) & (val > 90)
    if int(np.count_nonzero(bright)) < 6:
        return SignalState.UNKNOWN

    hue = hsv[:, :, 0]
    red = bright & ((hue <= 10) | (hue >= 165))
    yellow = bright & (hue >= 15) & (hue <= 38)
    green = bright & (hue >= 40) & (hue <= 95)

    counts = {
        SignalState.RED: int(np.count_nonzero(red)),
        SignalState.YELLOW: int(np.count_nonzero(yellow)),
        SignalState.GREEN: int(np.count_nonzero(green)),
    }
    state, count = max(counts.items(), key=lambda item: item[1])
    if count < max(8, int(np.count_nonzero(bright) * 0.12)):
        return SignalState.UNKNOWN
    return state


def _auto_signal_from_detections(frame, detections, target_name: str) -> SignalState:
    candidates = []
    for det in detections:
        name = det.cls_name.lower()
        if target_name not in name:
            continue
        state = _classify_light_crop(frame, det.box)
        if state == SignalState.UNKNOWN:
            continue
        x1, y1, x2, y2 = det.box
        area = max(1, (x2 - x1) * (y2 - y1))
        candidates.append((float(det.conf) * area, state))
    if not candidates:
        return SignalState.UNKNOWN
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def _point_in_contour(contour, x: float, y: float) -> bool:
    if contour is None:
        return False
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def _point_near_contour(contour, x: float, y: float, margin: int,
                        frame_w: int, frame_h: int) -> bool:
    if contour is None:
        return False
    x, y = int(x), int(y)
    x0, y0, bw, bh = cv2.boundingRect(contour)
    return (
        x0 - margin <= x <= x0 + bw + margin and
        y0 - margin <= y <= y0 + bh + margin and
        0 <= x < frame_w and 0 <= y < frame_h
    )


def build_scene_context(detections, lane_geo, seg_mask, frame_h, frame_w,
                        frame=None,
                        vehicle_signal_override="auto",
                        pedestrian_signal_override="auto",
                        right_turn_signal_override="none",
                        stop_completed_on_red=False) -> SceneContext:
    """
    YOLO 검출 + 후처리 결과를 종합하여 SceneContext를 생성한다.

    Args:
        detections: List[Detection]
        lane_geo: LaneGeometry
        seg_mask: np.ndarray
        frame_h, frame_w: 프레임 크기
    """
    ctx = SceneContext(frame_h=frame_h, frame_w=frame_w)
    ctx.lane_confidence = lane_geo.lane_confidence
    ctx.crosswalk_pixel_count = lane_geo.crosswalk_pixel_count
    ctx.crosswalk_zone_exists = lane_geo.crosswalk_zone is not None
    ctx.stop_line_visible = lane_geo.stop_line_y is not None
    if lane_geo.stop_line_y is not None:
        ctx.stop_line_y_ratio = lane_geo.stop_line_y / max(1, frame_h)

    vehicle_override = str(vehicle_signal_override).lower()
    ped_override = str(pedestrian_signal_override).lower()
    rt_override = str(right_turn_signal_override).lower()

    ctx.vehicle_signal = (
        _auto_signal_from_detections(frame, detections, "traffic_light_vehicle")
        if vehicle_override == "auto"
        else _parse_signal_state(vehicle_signal_override)
    )
    ctx.pedestrian_signal = (
        _auto_signal_from_detections(frame, detections, "traffic_light_pedestrian")
        if ped_override == "auto"
        else _parse_signal_state(pedestrian_signal_override)
    )
    ctx.right_turn_signal = (
        SignalState.NONE if rt_override in ("none", "off", "false")
        else (
            _auto_signal_from_detections(frame, detections, "right_turn")
            if rt_override == "auto"
            else _parse_signal_state(right_turn_signal_override, SignalState.NONE)
        )
    )
    ctx.red_light_stop_required = ctx.vehicle_signal == SignalState.RED
    ctx.stop_completed_on_red = bool(stop_completed_on_red)

    # YOLO 검출 분석
    for det in detections:
        name = det.cls_name.lower()
        if 'traffic_light' in name:
            ctx.has_traffic_light = True
        elif name == 'pedestrian' or name == 'person':
            ctx.pedestrian_count += 1
            # path corridor 안에 있는지 확인
            if lane_geo.path_corridor is not None:
                bx = (det.box[0] + det.box[2]) // 2
                by = det.box[3]  # 발 위치 (bbox 하단)
                if _point_in_contour(lane_geo.path_corridor, bx, by):
                    ctx.pedestrian_in_corridor = True
            # crosswalk zone 안에 있는지 확인
            if lane_geo.crosswalk_zone is not None:
                bx = (det.box[0] + det.box[2]) // 2
                by = det.box[3]
                if _point_in_contour(lane_geo.crosswalk_zone, bx, by):
                    ctx.pedestrian_in_crosswalk = True
                elif _point_near_contour(lane_geo.crosswalk_zone, bx, by, 60,
                                         frame_w, frame_h):
                    ctx.pedestrian_near_crosswalk = True
        elif name == 'vehicle' or name in ('car', 'bus', 'truck', 'motorcycle'):
            ctx.vehicle_count += 1
            if det.distance_m is not None:
                if (ctx.nearest_vehicle_distance_m is None or
                        det.distance_m < ctx.nearest_vehicle_distance_m):
                    ctx.nearest_vehicle_distance_m = det.distance_m
            bx = (det.box[0] + det.box[2]) // 2
            by = det.box[3]
            if _point_in_contour(lane_geo.path_corridor, bx, by):
                ctx.vehicle_in_turn_path = True
                if det.distance_m is not None:
                    if (ctx.nearest_turn_path_vehicle_distance_m is None or
                            det.distance_m < ctx.nearest_turn_path_vehicle_distance_m):
                        ctx.nearest_turn_path_vehicle_distance_m = det.distance_m
            elif by > frame_h * 0.35 and bx > frame_w * 0.35:
                ctx.conflicting_vehicle_count += 1
                if det.distance_m is not None:
                    if (ctx.nearest_conflicting_vehicle_distance_m is None or
                            det.distance_m < ctx.nearest_conflicting_vehicle_distance_m):
                        ctx.nearest_conflicting_vehicle_distance_m = det.distance_m

    return ctx
