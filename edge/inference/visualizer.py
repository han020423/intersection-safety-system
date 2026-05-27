"""
visualizer.py — 데모 화면 오버레이 그리기 (v2 — 가시성 개선)

원본 영상이 잘 보이도록 오버레이를 최소화.
- 반투명 채움 → 윤곽선 위주로 변경
- 패널/배너 → 콤팩트하게 축소
"""

import os
from functools import lru_cache
from typing import List, Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

from perception import Detection, SEG_COLORS, YOLO_COLORS, NUM_SEG_CLASSES, SEG_CLASS_NAMES
from postprocess import LaneGeometry
from state_machine import (
    CONFLICTING_VEHICLE_CAUTION_DISTANCE_M,
    LANE_RECOVERY_CONF,
    SignalState,
    VEHICLE_CAUTION_DISTANCE_M,
    VEHICLE_STOP_DISTANCE_M,
    DrivingState,
    Decision,
    DECISION_COLORS,
    SceneContext,
)


@lru_cache(maxsize=8)
def _get_korean_font(size: int):
    if ImageFont is None:
        return None

    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _put_text(vis: np.ndarray, text: str, org: tuple,
              font_scale: float, color: tuple, thickness: int = 1) -> None:
    if all(ord(ch) < 128 for ch in text) or Image is None:
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        return

    font_size = max(10, int(round(font_scale * 30)))
    font = _get_korean_font(font_size)
    if font is None:
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        return

    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    b, g, r = color
    x, y = org
    draw.text((int(x), int(y - font_size)), text, font=font,
              fill=(int(r), int(g), int(b)))
    vis[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def _fmt_bool(value: bool) -> str:
    return "True" if value else "False"


def _fmt_distance(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}m"


def _format_warning_ko(ctx: SceneContext, state: DrivingState,
                       decision: Decision) -> str:
    if ctx.right_turn_signal == SignalState.RED:
        return "right_turn_signal=RED | 우회전 전용 신호 적색: 정지"

    if ctx.red_light_stop_required and not ctx.stop_completed_on_red:
        return ("red_light_stop_required=True, stop_completed_on_red=False | "
                "전방 차량신호 적색: 우선 일시정지")

    if ctx.pedestrian_in_corridor:
        return "pedestrian_in_corridor=True | 우회전 진행 경로 보행자: 정지"
    if ctx.pedestrian_in_crosswalk:
        return "pedestrian_in_crosswalk=True | 횡단보도 위 보행자: 정지"
    if ctx.pedestrian_near_crosswalk:
        return "pedestrian_near_crosswalk=True | 횡단보도 주변 보행자: 정지"
    if state == DrivingState.CROSSWALK_APPROACH and ctx.pedestrian_count > 0:
        return (f"state=CROSSWALK_APPROACH, pedestrian_count={ctx.pedestrian_count} | "
                "횡단보도 접근 중 보행자 감지: 정지")

    if ctx.vehicle_in_turn_path:
        d = ctx.nearest_turn_path_vehicle_distance_m
        if d is None:
            return ("vehicle_in_turn_path=True, "
                    "nearest_turn_path_vehicle_distance_m=- | "
                    "우회전 경로 차량 거리 불명: 주의")
        if d < VEHICLE_STOP_DISTANCE_M:
            return (f"vehicle_in_turn_path=True, "
                    f"nearest_turn_path_vehicle_distance_m={d:.1f}m < "
                    f"{VEHICLE_STOP_DISTANCE_M:.0f}m | 우회전 경로 차량 근접: 정지")
        if d < VEHICLE_CAUTION_DISTANCE_M:
            return (f"vehicle_in_turn_path=True, "
                    f"nearest_turn_path_vehicle_distance_m={d:.1f}m < "
                    f"{VEHICLE_CAUTION_DISTANCE_M:.0f}m | 우회전 경로 차량: 주의")

    if ctx.vehicle_signal == SignalState.YELLOW or ctx.right_turn_signal == SignalState.YELLOW:
        return (f"vehicle_signal={ctx.vehicle_signal.value}, "
                f"right_turn_signal={ctx.right_turn_signal.value} | 황색 신호: 주의")

    if ctx.red_light_stop_required and ctx.stop_completed_on_red:
        return ("red_light_stop_required=True, stop_completed_on_red=True | "
                "적색 일시정지 후 양보하며 서행")

    if ctx.conflicting_vehicle_count > 0:
        d = ctx.nearest_conflicting_vehicle_distance_m
        if d is None or d < CONFLICTING_VEHICLE_CAUTION_DISTANCE_M:
            return (f"conflicting_vehicle_count={ctx.conflicting_vehicle_count}, "
                    f"nearest_conflicting_vehicle_distance_m={_fmt_distance(d)} | "
                    "교차 차량 가능성: 주의")

    if state in (DrivingState.ENTERING_INTERSECTION,
                 DrivingState.INTERSECTION_TRACKING):
        return (f"state={state.value}, lane_confidence={ctx.lane_confidence:.2f} | "
                "교차로 내부/차선 불확실: 주의")

    if state == DrivingState.CROSSWALK_APPROACH:
        return (f"state=CROSSWALK_APPROACH, "
                f"crosswalk_zone_exists={_fmt_bool(ctx.crosswalk_zone_exists)} | "
                "횡단보도 접근: 주의")

    if ctx.lane_confidence < LANE_RECOVERY_CONF:
        return (f"lane_confidence={ctx.lane_confidence:.2f} < "
                f"{LANE_RECOVERY_CONF:.2f} | 차선 신뢰도 낮음: 주의")

    if state == DrivingState.RELOCK_LANE:
        return "state=RELOCK_LANE | 차선 재확보 중: 주의"

    if decision == Decision.GO:
        return "decision=GO | 위험 조건 없음: 우회전 서행 가능"

    return f"reason={ctx.reason} | 판단 사유 확인"


def _clip_text_to_width(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max(0, max_chars - 3)] + "..."


def draw_demo_overlay(frame: np.ndarray,
                      detections: List[Detection],
                      seg_mask: np.ndarray,
                      seg_overlay: np.ndarray,
                      lane_geo: LaneGeometry,
                      state: DrivingState,
                      decision: Decision,
                      ctx: SceneContext,
                      yolo_ms: float,
                      seg_ms: float,
                      total_ms: float,
                      fps: float,
                      debug: bool = False
                      ) -> np.ndarray:
    """데모 화면에 모든 오버레이를 그린다. 원본 영상 가시성 우선."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── 1) Ego Lane Polygon — 윤곽선만 (채움 없음) ──
    if lane_geo.ego_lane_polygon is not None:
        pts = lane_geo.ego_lane_polygon.reshape(-1, 2)
        cv2.polylines(vis, [pts], True, (0, 220, 0), 2, cv2.LINE_AA)

    # ── 2) Path Corridor — 얇은 점선 윤곽 ──
    if lane_geo.path_corridor is not None:
        pts = lane_geo.path_corridor.reshape(-1, 2)
        # 점선 효과: 짧은 선분으로 그리기
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            if i % 2 == 0:  # 짝수 인덱스만 그려서 점선
                cv2.line(vis, tuple(pts[i]), tuple(pts[j]), (120, 90, 30), 1, cv2.LINE_AA)

    # ── 3) Crosswalk Active Zone — 얇은 윤곽 + 아주 약한 채움 ──
    if lane_geo.crosswalk_zone is not None:
        cw_overlay = np.zeros_like(vis)
        cv2.fillPoly(cw_overlay, [lane_geo.crosswalk_zone.reshape(-1, 2)], (0, 120, 220))
        mask = cw_overlay.any(axis=2)
        vis[mask] = cv2.addWeighted(vis[mask], 0.85, cw_overlay[mask], 0.15, 0)
        cv2.drawContours(vis, [lane_geo.crosswalk_zone], -1, (0, 180, 255), 2)

    # ── 4) Segmentation Color Mask — 차선/횡단보도만 약하게 ──
    seg_nonzero = seg_mask > 0
    if seg_nonzero.any():
        vis[seg_nonzero] = cv2.addWeighted(
            vis[seg_nonzero], 0.75,
            seg_overlay[seg_nonzero], 0.25, 0
        )

    # ── 5) Lane Centerline — 얇은 선 ──
    if lane_geo.centerline is not None and len(lane_geo.centerline) >= 2:
        pts = lane_geo.centerline.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
        p1 = tuple(lane_geo.centerline[-2])
        p2 = tuple(lane_geo.centerline[-1])
        cv2.arrowedLine(vis, p1, p2, (0, 255, 255), 2, tipLength=0.3)

    # ── 6) 좌/우 차선 polyline ──
    if lane_geo.left_lane_pts is not None and len(lane_geo.left_lane_pts) >= 2:
        pts = lane_geo.left_lane_pts.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (255, 200, 0), 1, cv2.LINE_AA)
    if lane_geo.right_lane_pts is not None and len(lane_geo.right_lane_pts) >= 2:
        pts = lane_geo.right_lane_pts.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (255, 200, 0), 1, cv2.LINE_AA)

    # ── 7) Stop Line ──
    if lane_geo.stop_line_y is not None:
        cv2.line(vis, (0, lane_geo.stop_line_y), (w, lane_geo.stop_line_y),
                 (0, 0, 255), 1, cv2.LINE_AA)

    # ── 8) YOLO BBox — 깔끔하게 ──
    for det in detections:
        x1, y1, x2, y2 = det.box
        color = YOLO_COLORS.get(det.cls_name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.conf:.2f}"
        if det.distance_m is not None:
            label += f" {det.distance_m:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # 레이블 배경을 반투명으로
        lbl_y = max(0, y1 - th - 4)
        roi = vis[lbl_y:lbl_y + th + 4, x1:x1 + tw + 4].copy()
        if roi.size > 0:
            bg = np.full_like(roi, color)
            vis[lbl_y:lbl_y + th + 4, x1:x1 + tw + 4] = cv2.addWeighted(roi, 0.4, bg, 0.6, 0)
        cv2.putText(vis, label, (x1 + 2, max(th + 2, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 9) 판단 배너 (우측 상단, 콤팩트) ──
    _draw_decision_badge(vis, decision, state, w)

    # ── 10) 정보 패널 (좌측 하단, 미니멀) ──
    _draw_info_strip(vis, ctx, state, decision, yolo_ms, seg_ms, total_ms, fps, h, w)

    # ── 11) 범례 (debug 모드만) ──
    if debug:
        _draw_legend(vis, seg_mask, detections, w, h)

    return vis


def _draw_decision_badge(vis, decision, state, w):
    """우측 상단에 콤팩트한 판단 뱃지."""
    color = DECISION_COLORS[decision]
    text = decision.value
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 뱃지 크기 계산
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    pad = 8
    bx = w - tw - pad * 2 - 8
    by = 8
    bw = tw + pad * 2
    bh = th + pad * 2

    # 반투명 배경
    roi = vis[by:by + bh, bx:bx + bw].copy()
    if roi.size > 0:
        bg = np.zeros_like(roi)
        vis[by:by + bh, bx:bx + bw] = cv2.addWeighted(roi, 0.4, bg, 0.6, 0)
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)
    cv2.putText(vis, text, (bx + pad, by + th + pad - 2),
                font, 0.8, color, 2, cv2.LINE_AA)

    # 상태 텍스트 (뱃지 아래)
    state_short = state.value.replace("_", " ")
    (sw, sh), _ = cv2.getTextSize(state_short, font, 0.35, 1)
    sx = w - sw - 12
    sy = by + bh + sh + 4
    cv2.putText(vis, state_short, (sx, sy), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_info_strip(vis, ctx, state, decision, yolo_ms, seg_ms, total_ms, fps, h, w):
    """좌측 하단에 한 줄 정보 스트립."""
    sig = f"VS:{ctx.vehicle_signal.value[0]} PS:{ctx.pedestrian_signal.value[0]} RT:{ctx.right_turn_signal.value[0]}"
    nearest = (
        f"NV:{ctx.nearest_vehicle_distance_m:.1f}m"
        if ctx.nearest_vehicle_distance_m is not None else "NV:-"
    )
    info = (f"FPS:{fps:.0f} | Y:{yolo_ms:.0f}ms S:{seg_ms:.0f}ms | "
            f"Lane:{ctx.lane_confidence:.2f} | "
            f"P:{ctx.pedestrian_count} V:{ctx.vehicle_count} {nearest} | {sig}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(info, font, 0.35, 1)
    strip_h = th + 10
    y0 = h - strip_h

    # 반투명 배경 스트립
    roi = vis[y0:h, 0:tw + 16].copy()
    if roi.size > 0:
        bg = np.zeros_like(roi)
        vis[y0:h, 0:tw + 16] = cv2.addWeighted(roi, 0.5, bg, 0.5, 0)
    cv2.putText(vis, info, (8, h - 5), font, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    warning = _format_warning_ko(ctx, state, decision)
    if " | " in warning:
        warning_vars, warning_msg = warning.split(" | ", 1)
    else:
        warning_vars, warning_msg = "", warning
    warning_vars = _clip_text_to_width(warning_vars, 78)
    warning_msg = _clip_text_to_width(warning_msg, 42)

    msg_y = max(28, h - strip_h - 5)
    vars_y = max(14, msg_y - 14)
    warn_w = w - 8
    wy0 = max(0, vars_y - 12)
    roi = vis[wy0:msg_y + 5, 0:warn_w].copy()
    if roi.size > 0:
        bg = np.zeros_like(roi)
        vis[wy0:msg_y + 5, 0:warn_w] = cv2.addWeighted(roi, 0.45, bg, 0.55, 0)

    if warning_vars:
        cv2.putText(vis, warning_vars, (8, vars_y), font, 0.31,
                    (210, 210, 210), 1, cv2.LINE_AA)
    _put_text(vis, warning_msg, (8, msg_y), 0.34, (235, 235, 235), 1)


def _draw_legend(vis, seg_mask, detections, w, h):
    """우측 하단 범례 (debug 모드)."""
    items = [
        ("Ego Lane", (0, 220, 0)),
        ("Corridor", (120, 90, 30)),
        ("CW Zone", (0, 180, 255)),
        ("Centerline", (0, 255, 255)),
    ]
    for cid in range(1, NUM_SEG_CLASSES):
        if int(np.count_nonzero(seg_mask == cid)) > 50:
            items.append((SEG_CLASS_NAMES[cid], SEG_COLORS[cid]))

    if not items:
        return
    lh = 16 * len(items) + 8
    lw = 130
    lx = w - lw - 8
    ly = h - lh - 24  # 하단 info strip 위

    roi = vis[ly:ly + lh, lx:lx + lw].copy()
    if roi.size > 0:
        bg = np.zeros_like(roi)
        vis[ly:ly + lh, lx:lx + lw] = cv2.addWeighted(roi, 0.4, bg, 0.6, 0)
    cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), (60, 60, 60), 1)

    for j, (name, color) in enumerate(items):
        yy = ly + 13 + 16 * j
        cv2.rectangle(vis, (lx + 5, yy - 7), (lx + 15, yy + 1), color, -1)
        cv2.putText(vis, name, (lx + 20, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
