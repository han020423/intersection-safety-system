#!/usr/bin/env python3
"""
v3 lightweight BiSeNetV2 recognition visualizer.

이 버전은 판단 로직을 모두 빼고 다음 두 가지 결과만 화면에 표시한다.

1. YOLO 객체 검출 결과
   - 차량, 보행자, 신호등 등 bbox 표시
   - confidence와 차량 거리 표시

2. road_v4 BiSeNet 세그멘테이션 결과
   - lane_white / lane_yellow / lane_blue
   - crosswalk
   - stop_line

목적:
  판단 로직을 고치기 전에 모델 인지 결과 자체가 정상인지 확인한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

V3_DIR = Path(__file__).resolve().parent
PARENT_DIR = V3_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))

from perception import YoloDetector  # noqa: E402
from ego_lane_reboot import EgoLanePipeline, draw_ego_lane_overlay  # noqa: E402
from event_logger_reboot import EventLoggerConfig, LocalEventLogger  # noqa: E402
from intersection_phase_reboot import IntersectionPhaseEstimator  # noqa: E402
from korean_overlay import draw_korean_text  # noqa: E402
from road_structure_reboot import (  # noqa: E402
    PathCorridorPipeline,
    draw_crosswalk_zone,
    draw_path_corridor,
    estimate_crosswalk_zone,
    evaluate_crosswalk_pedestrians,
)
from road_state_reboot import LaneRoleMemory, build_road_structure_state  # noqa: E402
from safety_decision_reboot import (  # noqa: E402
    IntersectionContext,
    SafetyDecisionEngine,
    SignalState,
    draw_safety_decision,
)
from signal_state_reboot import SignalStateTracker, estimate_traffic_signals  # noqa: E402
from stop_state_reboot import StopCompletionConfig, StopCompletionTracker  # noqa: E402
from pedestrian_motion_reboot import PedestrianMotionTracker  # noqa: E402
from voice_alert_reboot import VoiceAlertManager  # noqa: E402
from road_v4_segmentor import RoadV4Segmentor, SEG_CLASS_NAMES, SEG_COLORS  # noqa: E402


DEFAULT_YOLO_WEIGHTS = PARENT_DIR / "yolo.pt"
ALT_YOLO_WEIGHTS = PARENT_DIR.parents[1] / "ai" / "scripts" / "road_v1" / "yolo.pt"
DEFAULT_SEG_WEIGHTS = V3_DIR / "road_v4_best_light.pt"
ALT_SEG_WEIGHTS = PARENT_DIR / "v2" / "road_v4_best_light.pt"

YOLO_COLORS = {
    "pedestrian": (0, 200, 255),
    "person": (0, 200, 255),
    "vehicle": (255, 80, 80),
    "car": (255, 80, 80),
    "bus": (255, 80, 80),
    "truck": (255, 80, 80),
    "motorcycle": (255, 80, 80),
    "traffic_light_vehicle": (80, 255, 80),
    "traffic_light_pedestrian": (80, 255, 80),
    "crosswalk": (255, 255, 0),
}
YOLO_LABEL_KO = {
    "pedestrian": "보행자",
    "person": "보행자",
    "vehicle": "차량",
    "car": "차량",
    "bus": "버스",
    "truck": "트럭",
    "motorcycle": "오토바이",
    "traffic_light_vehicle": "차량신호등",
    "traffic_light_pedestrian": "보행자신호등",
    "crosswalk": "횡단보도",
    "left_turn_sign": "좌회전표지",
}


def find_existing(primary: Path, alt: Path | None, label: str) -> str:
    """기본 weight가 없으면 대체 경로를 찾아준다."""
    if primary.is_file():
        return str(primary)
    if alt is not None and alt.is_file():
        print(f"[{label}] using alternate path: {alt}")
        return str(alt)
    searched = [str(primary)]
    if alt is not None:
        searched.append(str(alt))
    raise FileNotFoundError(f"{label} weights not found. searched={searched}")


def draw_label(vis: np.ndarray, text: str, x: int, y: int, color) -> None:
    """bbox 위에 잘 보이는 라벨을 그린다."""
    font_size = 13
    box_w = min(vis.shape[1] - x - 1, max(54, int(len(text) * font_size * 0.72) + 10))
    box_h = 21
    y0 = max(0, y - box_h)
    cv2.rectangle(vis, (x, y0), (x + box_w, y0 + box_h), color, -1)
    draw_korean_text(vis, text, (x + 4, y0 + 3), font_size=font_size, color=(255, 255, 255))


def draw_yolo(vis: np.ndarray, detections) -> None:
    """YOLO 검출 bbox와 confidence를 표시한다."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        color = YOLO_COLORS.get(det.cls_name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{YOLO_LABEL_KO.get(det.cls_name, det.cls_name)} {det.conf:.2f}"
        if det.distance_m is not None:
            label += f" {det.distance_m:.1f}m"
        draw_label(vis, label, x1, max(0, y1), color)


def draw_segmentation(frame: np.ndarray, seg_mask: np.ndarray, seg_overlay: np.ndarray,
                      alpha: float) -> np.ndarray:
    """BiSeNet 결과를 시각화한다. 차선 class는 면이 아니라 연결된 선으로만 표시한다."""
    vis = frame.copy()
    # 횡단보도/정지선은 면으로 확인하고, 차선은 draw_lane_lines()에서 선으로만 표시한다.
    mask = seg_mask >= 4
    if np.any(mask):
        vis[mask] = cv2.addWeighted(vis[mask], 1.0 - alpha, seg_overlay[mask], alpha, 0)
    draw_lane_lines(vis, seg_mask)
    return vis


def _cluster_row_xs(xs: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """한 scan line 위의 차선 픽셀들을 x좌표 연속 구간별로 묶는다."""
    if len(xs) == 0:
        return []
    clusters = []
    current = [int(xs[0])]
    for x in xs[1:]:
        x = int(x)
        if x - current[-1] <= max_gap:
            current.append(x)
        else:
            clusters.append(np.asarray(current, dtype=np.int32))
            current = [x]
    clusters.append(np.asarray(current, dtype=np.int32))
    return clusters


def _lane_center_samples(seg_mask: np.ndarray, lane_cls: int) -> list[tuple[int, int]]:
    """
    점선 조각을 연결하기 위해 차선 mask를 여러 y좌표에서 가로로 스캔한다.
    각 scan line에서 차선 픽셀 cluster의 중심점을 뽑으면 점선도 하나의 점열로 표현된다.
    """
    h, w = seg_mask.shape[:2]
    lane_mask = seg_mask == lane_cls
    scan_top = int(h * 0.22)
    scan_bottom = int(h * 0.98)
    scan_ys = np.linspace(scan_bottom, scan_top, 84, dtype=np.int32)
    max_cluster_gap = max(6, int(w * 0.012))
    min_cluster_width = max(2, int(w * 0.004))
    samples: list[tuple[int, int]] = []

    for y in scan_ys:
        y1 = max(0, int(y) - 2)
        y2 = min(h, int(y) + 3)
        xs = np.where(np.any(lane_mask[y1:y2, :], axis=0))[0]
        for cluster in _cluster_row_xs(xs, max_cluster_gap):
            if len(cluster) < min_cluster_width:
                continue
            samples.append((int(np.mean(cluster)), int(y)))
    return samples


def _predict_track_x(track: list[tuple[int, int]], y: int) -> float:
    """기존 track의 현재 y에서 예상 x좌표를 계산한다."""
    if len(track) < 3:
        return float(track[-1][0])
    pts = np.asarray(track[-6:], dtype=np.float64)
    try:
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
        return float(np.poly1d(coeffs)(y))
    except (np.linalg.LinAlgError, ValueError):
        return float(track[-1][0])


def _group_lane_tracks(samples: list[tuple[int, int]], frame_w: int, frame_h: int) -> list[np.ndarray]:
    """scan line 중심점들을 여러 개의 차선 track으로 묶는다."""
    if not samples:
        return []

    samples = sorted(samples, key=lambda p: (-p[1], p[0]))
    tracks: list[list[tuple[int, int]]] = []
    # 점선의 가까운 빈칸만 잇는다. y 간격과 x 오차를 같이 제한해야
    # 서로 다른 차선 조각이 한 선으로 억지 연결되는 현상을 줄일 수 있다.
    max_y_gap = max(34, int(frame_h * 0.075))
    base_assign_threshold = max(18, int(frame_w * 0.026))
    max_assign_threshold = max(28, int(frame_w * 0.040))

    for x, y in samples:
        best_idx = -1
        best_dist = float("inf")
        for idx, track in enumerate(tracks):
            if track[-1][1] == y:
                continue
            y_gap = abs(int(track[-1][1]) - int(y))
            if y_gap > max_y_gap:
                continue
            assign_threshold = min(
                max_assign_threshold,
                base_assign_threshold + int(y_gap * 0.16),
            )
            dist = abs(float(x) - _predict_track_x(track, y))
            if dist < best_dist and dist <= assign_threshold:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            tracks[best_idx].append((x, y))
        else:
            tracks.append([(x, y)])

    result = []
    for track in tracks:
        pts = np.asarray(track, dtype=np.int32)
        if len(pts) < 5:
            continue
        if int(pts[:, 1].max() - pts[:, 1].min()) < 35:
            continue
        result.append(pts)
    return result


def _fit_continuous_lane_line(track: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray | None:
    """
    점선 중심점들을 1개 연속 polyline으로 피팅한다.

    관측된 점선 조각 사이 gap은 보간하고, 최하단 관측점에서 화면 바닥까지는
    짧고 안정적인 경우에만 외삽한다. 긴 외삽은 옆 차선/노이즈로 튀기 쉬우므로
    fit 오차, 하단 gap, 기울기를 함께 확인한다.
    """
    if track is None or len(track) < 5:
        return None
    xs = track[:, 0].astype(np.float64)
    ys = track[:, 1].astype(np.float64)
    degree = min(2, len(track) - 1)
    try:
        coeffs = np.polyfit(ys, xs, degree)
    except (np.linalg.LinAlgError, ValueError):
        return None

    # fit 오차가 큰 track은 여러 차선 조각이 섞였을 가능성이 높으므로 그리지 않는다.
    fitted_xs = np.poly1d(coeffs)(ys)
    residual = np.abs(fitted_xs - xs)
    residual_limit = max(20.0, frame_w * 0.025)
    residual_p85 = float(np.percentile(residual, 85))
    if residual_p85 > residual_limit:
        return None

    y_bottom = int(np.max(ys))
    y_top = int(np.min(ys))
    if y_bottom <= y_top:
        return None

    draw_bottom = y_bottom
    image_bottom = int(frame_h * 0.98)
    bottom_gap = image_bottom - y_bottom
    y_span = y_bottom - y_top
    poly = np.poly1d(coeffs)
    slope_at_bottom = abs(float(np.polyder(poly)(y_bottom))) if degree >= 1 else 0.0
    can_extend_to_bottom = (
        bottom_gap > 0
        and bottom_gap <= max(18, int(frame_h * 0.16))
        and y_span >= max(45, int(frame_h * 0.14))
        and residual_p85 <= residual_limit * 0.70
        and slope_at_bottom <= 1.6
    )
    if can_extend_to_bottom:
        draw_bottom = image_bottom

    dense_ys = np.linspace(draw_bottom, y_top, max(20, int((draw_bottom - y_top) / 8)), dtype=np.int32)
    dense_xs = np.clip(poly(dense_ys.astype(np.float64)), 0, frame_w - 1).astype(np.int32)
    return np.column_stack([dense_xs, dense_ys]).astype(np.int32)


def draw_lane_lines(vis: np.ndarray, seg_mask: np.ndarray) -> None:
    """
    lane_white/yellow/blue mask를 그대로 칠하지 않고, 점선 빈 공간을 이어
    하나의 연속된 차선 polyline으로 표시한다.
    """
    h, w = seg_mask.shape[:2]
    for lane_cls in (1, 2, 3):
        samples = _lane_center_samples(seg_mask, lane_cls)
        tracks = _group_lane_tracks(samples, w, h)
        color = SEG_COLORS.get(lane_cls, (255, 255, 255))
        for track in tracks:
            line = _fit_continuous_lane_line(track, w, h)
            if line is None or len(line) < 2:
                continue
            cv2.polylines(vis, [line.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)


def draw_legend(vis: np.ndarray) -> None:
    """road_v4 segmentation class 색상표를 왼쪽 위에 표시한다."""
    x, y = 10, 20
    cv2.rectangle(vis, (4, 4), (190, 128), (0, 0, 0), -1)
    cv2.rectangle(vis, (4, 4), (190, 128), (80, 80, 80), 1)
    cv2.putText(vis, "BiSeNet road_v4", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    y += 20
    for cls_id, name in enumerate(SEG_CLASS_NAMES):
        if cls_id == 0:
            continue
        color = SEG_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(vis, (x, y - 10), (x + 12, y + 2), color, -1)
        cv2.putText(vis, f"{cls_id}: {name}", (x + 18, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
        y += 18


def draw_status(vis: np.ndarray, fps: float, yolo_ms: float, seg_ms: float,
                det_count: int, seg_mask: np.ndarray) -> None:
    """FPS와 모델 추론 시간을 하단에 표시한다."""
    h, w = vis.shape[:2]
    line = f"성능  FPS {fps:.1f} | YOLO {yolo_ms:.0f}ms | BiSeNet {seg_ms:.0f}ms | 객체 {det_count}"
    cv2.rectangle(vis, (0, h - 30), (w, h), (0, 0, 0), -1)
    draw_korean_text(vis, line, (8, h - 24), font_size=15, color=(180, 255, 180))


def draw_runtime_context(vis: np.ndarray, args) -> None:
    """Show one small line only when manual phase/signal testing is active."""
    if args.no_safety_decision:
        return
    if (
        args.phase == "auto"
        and args.vehicle_signal == "auto"
        and args.pedestrian_signal == "auto"
        and args.right_turn_signal == "none"
    ):
        return
    h, w = vis.shape[:2]
    y0 = max(164, h - 58)
    stop_done = int(bool(args.stop_completed_on_red))
    effective_phase = getattr(args, "_effective_phase", args.phase)
    phase_text = _runtime_value_ko(args.phase)
    effective_phase_text = _runtime_value_ko(effective_phase)
    effective_vehicle = getattr(args, "_effective_vehicle_signal", args.vehicle_signal)
    effective_right_turn = getattr(args, "_effective_right_turn_signal", args.right_turn_signal)
    vehicle_signal_text = f"{_runtime_value_ko(args.vehicle_signal)}->{_runtime_value_ko(effective_vehicle)}"
    right_turn_signal_text = f"{_runtime_value_ko(args.right_turn_signal)}->{_runtime_value_ko(effective_right_turn)}"
    stop_text = "완료" if stop_done else "미완료"
    line = (
        f"입력: 단계 {phase_text}->{effective_phase_text} | 차량신호 {vehicle_signal_text} | "
        f"우회전신호 {right_turn_signal_text} | 일시정지 {stop_text}"
    )
    cv2.rectangle(vis, (0, y0), (w, y0 + 24), (0, 0, 0), -1)
    draw_korean_text(vis, line, (8, y0 + 3), font_size=14, color=(220, 220, 220))


def _runtime_value_ko(value: str) -> str:
    return {
        "auto": "자동",
        "approach": "진입 전",
        "in_intersection": "교차로 내부",
        "unknown": "알 수 없음",
        "none": "없음",
        "red": "적색",
        "yellow": "황색",
        "green": "녹색",
        "auto": "자동",
    }.get(str(value), str(value))


def _apply_auto_signals(context: IntersectionContext, args, signal_result) -> None:
    """Override context signal states when CLI uses auto signal mode."""
    if signal_result is None:
        args._effective_vehicle_signal = context.vehicle_signal.value.lower()
        args._effective_pedestrian_signal = context.pedestrian_signal.value.lower()
        args._effective_right_turn_signal = context.right_turn_signal.value.lower()
        args._vehicle_signal_source = context.vehicle_signal_source
        return

    if args.vehicle_signal == "auto":
        context.vehicle_signal = signal_result.vehicle_signal
        context.vehicle_signal_source = getattr(signal_result, "vehicle_signal_source", "auto")
    else:
        context.vehicle_signal_source = "manual"
    if args.pedestrian_signal == "auto":
        context.pedestrian_signal = signal_result.pedestrian_signal
    if args.right_turn_signal == "auto":
        # The current YOLO model has no dedicated right-turn light class. Keep
        # UNKNOWN until a future detector provides one.
        context.right_turn_signal = signal_result.right_turn_signal

    args._effective_vehicle_signal = context.vehicle_signal.value.lower()
    args._effective_pedestrian_signal = context.pedestrian_signal.value.lower()
    args._effective_right_turn_signal = context.right_turn_signal.value.lower()
    args._vehicle_signal_confidence = float(getattr(signal_result, "vehicle_confidence", 0.0))
    args._vehicle_signal_source = context.vehicle_signal_source
    args._pedestrian_signal_confidence = float(getattr(signal_result, "pedestrian_confidence", 0.0))


def print_compliance_events_if_changed(args, safety_decision, context) -> None:
    """Print collected risk events only when a non-empty event set changes."""
    events = getattr(safety_decision, "compliance_events", [])
    signature = tuple(event.code for event in events)
    if signature == getattr(args, "_last_compliance_event_signature", None):
        return
    args._last_compliance_event_signature = signature

    if not events:
        # No compliance-risk event is present.  Keep the console quiet so
        # [RISK] lines always mean an actual risk event was collected.
        return

    print(
        "[RISK] "
        f"count={len(events)} level={safety_decision.level.value} "
        f"reason={safety_decision.reason_code} phase={context.phase.value} "
        f"scenario={context.scenario.value}"
    )
    for idx, event in enumerate(events, start=1):
        basis = ",".join(item.code for item in getattr(event, "legal_basis", [])) or "-"
        evidence = getattr(event, "evidence", {}) or {}
        print(
            f"  - {idx}. code={event.code} severity={event.severity.value} "
            f"title={event.title} basis={basis}"
        )
        print(
            "     evidence="
            f"vehicle_signal:{evidence.get('vehicle_signal')} "
            f"right_turn_signal:{evidence.get('right_turn_signal')} "
            f"stop_completed:{evidence.get('stop_completed_on_red')} "
            f"stop_source:{evidence.get('stop_completed_source')} "
            f"stop_motion:{evidence.get('stop_motion_score')} "
            f"signal_source:{evidence.get('vehicle_signal_source')} "
            f"ped_on:{evidence.get('pedestrian_on_crosswalk')} "
            f"ped_near:{evidence.get('pedestrian_near_crosswalk')} "
            f"ped_approach:{evidence.get('pedestrian_approaching_crosswalk')} "
            f"ped_leave:{evidence.get('pedestrian_leaving_crosswalk')} "
            f"rightmost:{evidence.get('lane_role_rightmost')}"
        )


def handle_runtime_key(args, key: int) -> bool:
    """
    Mutate CLI-style context values while the OpenCV preview is running.

    The safety engine reads args every frame, so these changes affect the next
    frame without restarting the program.  This is only for demo/testing until
    signal-state and phase estimation become automatic.
    """
    if key < 0 or key == 255:
        return False

    changed = True
    if key == ord("1"):
        args.phase = "approach"
    elif key == ord("2"):
        args.phase = "in_intersection"
    elif key == ord("0"):
        args.phase = "unknown"
    elif key == ord("a"):
        args.phase = "auto"
    elif key == ord("r"):
        args.vehicle_signal = "red"
    elif key == ord("y"):
        args.vehicle_signal = "yellow"
    elif key == ord("g"):
        args.vehicle_signal = "green"
    elif key == ord("u"):
        args.vehicle_signal = "unknown"
    elif key == ord("n"):
        args.vehicle_signal = "none"
    elif key == ord("v"):
        args.vehicle_signal = "auto"
    elif key == ord("t"):
        args.right_turn_signal = "none" if args.right_turn_signal == "red" else "red"
    elif key == ord("s"):
        args.stop_completed_on_red = not bool(args.stop_completed_on_red)
    elif key == ord("3"):
        args.scenario = "right_turn"
    elif key == ord("4"):
        args.scenario = "unprotected_left"
    else:
        changed = False

    if changed:
        print(
            "[CTX] "
            f"phase={args.phase} scenario={args.scenario} "
            f"vehicle_signal={args.vehicle_signal} right_turn_signal={args.right_turn_signal} "
            f"stop_completed_on_red={args.stop_completed_on_red}"
        )
    return changed


def process_frame(
    frame,
    yolo,
    seg,
    args,
    ego_lane_pipeline: EgoLanePipeline | None = None,
    path_corridor_pipeline: PathCorridorPipeline | None = None,
    safety_engine: SafetyDecisionEngine | None = None,
    phase_estimator: IntersectionPhaseEstimator | None = None,
    signal_tracker: SignalStateTracker | None = None,
    lane_role_memory: LaneRoleMemory | None = None,
    stop_tracker: StopCompletionTracker | None = None,
    pedestrian_motion_tracker: PedestrianMotionTracker | None = None,
    event_logger: LocalEventLogger | None = None,
    voice_alert_manager: VoiceAlertManager | None = None,
    frame_index: int = 0,
):
    """한 프레임에서 YOLO와 BiSeNet만 수행하고 시각화한다."""
    detections, yolo_ms = ([], 0.0)
    ego_lane = None
    seg_mask = None
    seg_overlay = None
    seg_ms = 0.0
    path_corridor = None
    crosswalk_zone = None
    crosswalk_peds = None
    pedestrian_motion = None
    phase_result = None
    signal_result = None

    if not args.no_yolo:
        detections, yolo_ms = yolo.infer(frame)
        raw_signal_result = estimate_traffic_signals(frame, detections)
        signal_result = signal_tracker.update(raw_signal_result) if signal_tracker is not None else raw_signal_result

    if not args.no_seg:
        seg_mask, seg_ms = seg.infer(frame)
        seg_overlay = seg.build_color_overlay(seg_mask)
        vis = draw_segmentation(frame, seg_mask, seg_overlay, args.seg_alpha)
        if not args.no_ego_lane and ego_lane_pipeline is not None:
            # reboot ego lane pipeline:
            # mask -> lane candidates -> temporal tracks -> ego 좌/우 선택 -> lane polygon.
            # 이 객체를 비디오 전체에서 재사용해야 프레임 간 tracking이 유지된다.
            ego_lane = ego_lane_pipeline.update(seg_mask, detections)
            draw_ego_lane_overlay(vis, ego_lane, args.ego_lane_alpha, show_text=False)

        if path_corridor_pipeline is not None and (not args.no_crosswalk_zone or not args.no_safety_decision):
            # Path corridor는 현재 ego lane을 작게 확장한 영역이며, 차선이 순간적으로 튀면
            # 2~3프레임만 cached_lane으로 유지한다. 오래 없는 경로는 unavailable로 둔다.
            path_corridor = path_corridor_pipeline.update(ego_lane, frame.shape[:2])
            if not args.no_crosswalk_zone:
                draw_path_corridor(vis, path_corridor)
                crosswalk_zone = estimate_crosswalk_zone(seg_mask, path_corridor)
                crosswalk_peds = evaluate_crosswalk_pedestrians(detections, crosswalk_zone, frame.shape[:2])
                draw_crosswalk_zone(vis, crosswalk_zone, crosswalk_peds, args.crosswalk_zone_alpha, show_text=False)
        if pedestrian_motion_tracker is not None:
            pedestrian_motion = pedestrian_motion_tracker.update(detections, seg_mask, frame.shape[:2])
    else:
        seg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        vis = frame.copy()

    if not args.no_safety_decision and safety_engine is not None:
        # Normalize visual perception results into rule-friendly road state.
        # The decision engine keeps legal-risk wording separate from detection.
        road_state = build_road_structure_state(
            ego_lane=ego_lane,
            path_corridor=path_corridor,
            crosswalk_zone=crosswalk_zone,
            crosswalk_pedestrian_status=crosswalk_peds,
            seg_mask=seg_mask,
            detections=detections,
            frame_shape=frame.shape[:2],
            pedestrian_motion_status=pedestrian_motion,
        )
        context = IntersectionContext.from_args(args)
        _apply_auto_signals(context, args, signal_result)
        if args.phase == "auto" and phase_estimator is not None:
            phase_result = phase_estimator.update(road_state, seg_mask, frame.shape[:2])
            context.phase = phase_result.decision_phase
            args._effective_phase = context.phase.value.lower()
        else:
            args._effective_phase = context.phase.value.lower()
        if (
            signal_tracker is not None
            and args.vehicle_signal == "auto"
            and signal_result is not None
        ):
            signal_result = signal_tracker.apply_approach_red_hold(signal_result, context.phase)
            _apply_auto_signals(context, args, signal_result)
        if stop_tracker is not None and not args.no_auto_stop_check:
            stop_state = stop_tracker.update(
                frame,
                context.phase,
                context.vehicle_signal,
                manual_completed=bool(args.stop_completed_on_red),
            )
            context.stop_completed_on_red = bool(stop_state.stop_completed)
            context.stop_completed_source = stop_state.source
            context.stop_motion_score = float(stop_state.motion_score)
            context.stop_stable_frames = int(stop_state.stable_frames)
        else:
            context.stop_completed_source = "manual" if args.stop_completed_on_red else "disabled"
        if lane_role_memory is not None:
            road_state = lane_role_memory.update(road_state, context.phase)
        safety_decision = safety_engine.decide(road_state, context)
        print_compliance_events_if_changed(args, safety_decision, context)
        # 이벤트 증거 사진은 운전자용 판단 패널을 제외한 인지 근거 화면을 저장한다.
        # 저장 시점에 event_logger가 YOLO 객체 박스를 추가로 그린다.
        event_snapshot_frame = vis.copy()
        draw_safety_decision(vis, safety_decision, context, phase_result, road_state)
        if voice_alert_manager is not None:
            voice_alert_manager.update(safety_decision, road_state, context)
        if event_logger is not None:
            event_logger.update(frame_index, event_snapshot_frame, safety_decision, context, detections=detections)

    draw_yolo(vis, detections)
    if args.legend:
        draw_legend(vis)

    return vis, detections, seg_mask, yolo_ms, seg_ms


def parse_args():
    p = argparse.ArgumentParser(description="v3: v2_0 logic + new lightweight BiSeNetV2 visualizer.")
    p.add_argument("--source", default="0", help="video/image path or camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", default="", help="output video/image path")
    p.add_argument("--max-frames", type=int, default=0, help="0 means no limit")

    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--seg-weights", default=None)
    p.add_argument("--device", default="cpu", help="cpu/cuda")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--yolo-interval", type=int, default=1)

    p.add_argument("--seg-input-h", type=int, default=352)
    p.add_argument("--seg-input-w", type=int, default=640)
    p.add_argument("--seg-alpha", type=float, default=0.35, help="segmentation overlay opacity")
    p.add_argument("--ego-lane-alpha", type=float, default=0.24, help="ego lane area overlay opacity")
    p.add_argument("--crosswalk-zone-alpha", type=float, default=0.28, help="crosswalk active zone overlay opacity")
    p.add_argument("--path-cache-frames", type=int, default=3, help="short frame cache TTL for path corridor")

    p.add_argument("--phase", choices=["auto", "approach", "in_intersection", "unknown"], default="auto")
    p.add_argument("--scenario", choices=["right_turn", "unprotected_left", "straight", "unknown"], default="unknown")
    p.add_argument("--vehicle-signal", choices=["auto", "red", "yellow", "green", "unknown", "none"], default="auto")
    p.add_argument("--pedestrian-signal", choices=["auto", "red", "green", "unknown", "none"], default="auto")
    p.add_argument("--right-turn-signal", choices=["auto", "red", "yellow", "green", "unknown", "none"], default="none")
    p.add_argument("--stop-completed-on-red", action="store_true")
    p.add_argument("--no-auto-stop-check", action="store_true", help="disable frame-motion stop-completion estimation")
    p.add_argument("--stop-motion-threshold", type=float, default=0.42, help="motion score threshold for auto stop check")
    p.add_argument("--stop-confirm-frames", type=int, default=5, help="consecutive low-motion frames for stop completion")
    p.add_argument("--no-event-log", action="store_true", help="disable local JSONL risk/intersection event logging")
    p.add_argument("--event-output-dir", default=str(V3_DIR / "event_records"), help="per-intersection event folders")
    p.add_argument("--risk-event-log", default=None, help=argparse.SUPPRESS)
    p.add_argument("--intersection-event-log", default=None, help=argparse.SUPPRESS)
    p.add_argument("--event-snapshot-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--event-device-id", default="edge-001")
    p.add_argument("--event-vehicle-id", default="test-car-01")
    p.add_argument("--no-event-snapshot", action="store_true", help="disable representative JPG snapshots for risk events")
    p.add_argument("--voice-alert", choices=["off", "console", "pyttsx3"], default="off", help="driver voice alert backend")
    p.add_argument("--voice-cooldown-sec", type=float, default=5.0, help="minimum seconds before repeating same voice alert")
    p.add_argument("--voice-rate", type=int, default=185, help="pyttsx3 speech rate")
    p.add_argument("--voice-volume", type=float, default=1.0, help="pyttsx3 speech volume from 0.0 to 1.0")

    p.add_argument("--no-yolo", action="store_true")
    p.add_argument("--no-seg", action="store_true")
    p.add_argument("--no-ego-lane", action="store_true", help="hide reboot ego lane overlay")
    p.add_argument("--no-crosswalk-zone", action="store_true", help="hide ego-path crosswalk zone overlay")
    p.add_argument("--no-safety-decision", action="store_true", help="hide legal/safety decision overlay")
    p.add_argument("--legend", action="store_true", help="show segmentation class legend")

    p.add_argument("--no-distance", action="store_true")
    p.add_argument("--camera-hfov", type=float, default=70.0)
    p.add_argument("--distance-focal-px", type=float, default=0.0)
    p.add_argument("--vehicle-real-width", type=float, default=1.8)
    p.add_argument("--vehicle-real-height", type=float, default=1.5)
    p.add_argument("--distance-scale", type=float, default=0.8)
    return p.parse_args()


def build_event_logger(args) -> LocalEventLogger | None:
    """Create the local JSONL/snapshot logger unless disabled."""
    if args.no_event_log or args.no_safety_decision:
        return None
    return LocalEventLogger(
        EventLoggerConfig(
            event_root_dir=Path(args.event_output_dir),
            device_id=args.event_device_id,
            vehicle_id=args.event_vehicle_id,
            enable_snapshots=not args.no_event_snapshot,
        )
    )


def build_voice_alert_manager(args) -> VoiceAlertManager | None:
    """Create the selected driver guidance backend."""
    if args.no_safety_decision or args.voice_alert == "off":
        return None
    return VoiceAlertManager(
        mode=args.voice_alert,
        cooldown_sec=args.voice_cooldown_sec,
        speech_rate=args.voice_rate,
        speech_volume=args.voice_volume,
    )


def main():
    args = parse_args()

    yolo = None
    seg = None

    if not args.no_yolo:
        yolo_path = args.yolo_weights or find_existing(DEFAULT_YOLO_WEIGHTS, ALT_YOLO_WEIGHTS, "YOLO")
        print(f"[YOLO] {yolo_path}")
        yolo = YoloDetector(
            weights=yolo_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            interval=args.yolo_interval,
            estimate_distance=not args.no_distance,
            camera_hfov_deg=args.camera_hfov,
            distance_focal_px=args.distance_focal_px,
            vehicle_real_width_m=args.vehicle_real_width,
            vehicle_real_height_m=args.vehicle_real_height,
            distance_scale=args.distance_scale,
        )

    if not args.no_seg:
        seg_path = args.seg_weights or find_existing(DEFAULT_SEG_WEIGHTS, ALT_SEG_WEIGHTS, "road_v4_best_light.pt")
        print(f"[BiSeNet] {seg_path}")
        seg = RoadV4Segmentor(
            weights=seg_path,
            input_size=(args.seg_input_h, args.seg_input_w),
            device=args.device,
        )

    source_arg = args.source
    if source_arg.isdigit():
        source = int(source_arg)
        mode = "camera"
    elif source_arg.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        source = source_arg
        mode = "image"
    else:
        source = source_arg
        mode = "video"

    if mode == "image":
        ego_lane_pipeline = None if args.no_seg or args.no_ego_lane else EgoLanePipeline()
        path_corridor_pipeline = None if args.no_seg or args.no_ego_lane else PathCorridorPipeline(args.path_cache_frames)
        safety_engine = None if args.no_safety_decision else SafetyDecisionEngine()
        phase_estimator = None if args.no_safety_decision or args.no_seg else IntersectionPhaseEstimator()
        signal_tracker = None if args.no_yolo else SignalStateTracker()
        lane_role_memory = None if args.no_safety_decision else LaneRoleMemory()
        stop_config = StopCompletionConfig(
            motion_threshold_px=args.stop_motion_threshold,
            confirm_frames=args.stop_confirm_frames,
        )
        stop_tracker = None if args.no_safety_decision else StopCompletionTracker(stop_config)
        pedestrian_motion_tracker = None if args.no_safety_decision or args.no_yolo or args.no_seg else PedestrianMotionTracker()
        event_logger = build_event_logger(args)
        voice_alert_manager = build_voice_alert_manager(args)
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"failed to read image: {source}")
        frame = cv2.resize(frame, (args.width, args.height))
        vis, dets, seg_mask, yolo_ms, seg_ms = process_frame(
            frame, yolo, seg, args, ego_lane_pipeline, path_corridor_pipeline,
            safety_engine, phase_estimator, signal_tracker, lane_role_memory, stop_tracker,
            pedestrian_motion_tracker, event_logger, voice_alert_manager, 1
        )
        if event_logger is not None:
            event_logger.finish(1)
        draw_status(vis, 0.0, yolo_ms, seg_ms, len(dets), seg_mask)
        if args.save:
            cv2.imwrite(args.save, vis)
            print(f"[SAVE] {args.save}")
        if args.show:
            cv2.imshow("v3 Lightweight BiSeNetV2 Visualizer", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or src_fps > 120:
        src_fps = 30.0

    writer = None
    frame_count = 0
    fps_smooth = 0.0
    ego_lane_pipeline = None if args.no_seg or args.no_ego_lane else EgoLanePipeline()
    path_corridor_pipeline = None if args.no_seg or args.no_ego_lane else PathCorridorPipeline(args.path_cache_frames)
    safety_engine = None if args.no_safety_decision else SafetyDecisionEngine()
    phase_estimator = None if args.no_safety_decision or args.no_seg else IntersectionPhaseEstimator()
    signal_tracker = None if args.no_yolo else SignalStateTracker()
    lane_role_memory = None if args.no_safety_decision else LaneRoleMemory()
    stop_config = StopCompletionConfig(
        motion_threshold_px=args.stop_motion_threshold,
        confirm_frames=args.stop_confirm_frames,
    )
    stop_tracker = None if args.no_safety_decision else StopCompletionTracker(stop_config)
    pedestrian_motion_tracker = None if args.no_safety_decision or args.no_yolo or args.no_seg else PedestrianMotionTracker()
    event_logger = build_event_logger(args)
    voice_alert_manager = build_voice_alert_manager(args)

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (args.width, args.height))

            vis, dets, seg_mask, yolo_ms, seg_ms = process_frame(
                frame, yolo, seg, args, ego_lane_pipeline, path_corridor_pipeline,
                safety_engine, phase_estimator, signal_tracker, lane_role_memory, stop_tracker,
                pedestrian_motion_tracker, event_logger, voice_alert_manager, frame_count + 1
            )

            elapsed = time.perf_counter() - t0
            cur_fps = 1.0 / max(1e-6, elapsed)
            fps_smooth = 0.15 * cur_fps + 0.85 * fps_smooth if fps_smooth > 0 else cur_fps
            draw_runtime_context(vis, args)
            draw_status(vis, fps_smooth, yolo_ms, seg_ms, len(dets), seg_mask)

            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f"frame={frame_count:5d} FPS={fps_smooth:5.1f} "
                    f"YOLO={yolo_ms:5.0f}ms BiSeNet={seg_ms:5.0f}ms objects={len(dets)}"
                )

            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, src_fps, (args.width, args.height))
                writer.write(vis)

            if args.show:
                cv2.imshow("v3 Lightweight BiSeNetV2 Visualizer", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                handle_runtime_key(args, key)

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[SAVE] {args.save}")
        if args.show:
            cv2.destroyAllWindows()
        if event_logger is not None:
            event_logger.finish(frame_count)
        print(f"[DONE] frames={frame_count}")


if __name__ == "__main__":
    main()
