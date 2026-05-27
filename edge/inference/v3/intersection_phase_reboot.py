"""
Crosswalk-based intersection phase estimation for v3.

Stop-line evidence is intentionally not used here.  In the current camera
setup, stop lines can be hidden, worn out, or irrelevant after turn entry.
Instead, this estimator watches the visible crosswalk move toward the lower
part of the image and keeps a small frame-based state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from road_state_reboot import CROSSWALK_CLASS_ID
from safety_decision_reboot import IntersectionPhase


class PhaseInternalState(Enum):
    UNKNOWN = "UNKNOWN"
    APPROACH = "APPROACH"
    NEAR_CROSSWALK = "NEAR_CROSSWALK"
    IN_INTERSECTION = "IN_INTERSECTION"


@dataclass(frozen=True)
class PhaseEstimatorConfig:
    detect_y_ratio: float = 0.35
    near_y_ratio: float = 0.82
    enter_y_ratio: float = 0.97
    # Raw crosswalk pixels can appear on road-edge artifacts.  Phase estimation
    # should only use crosswalk components large enough to affect ego entry.
    min_crosswalk_area_ratio: float = 0.00012
    min_phase_area_ratio: float = 0.0012
    min_phase_width_ratio: float = 0.12
    center_band_left_ratio: float = 0.25
    center_band_right_ratio: float = 0.75
    min_near_frames_before_enter: int = 5
    bottom_enter_confirm_frames: int = 3
    area_drop_ratio_for_enter: float = 0.45
    min_enter_reference_area_ratio: float = 0.01
    lost_enter_confirm_frames: int = 2
    min_in_intersection_frames: int = 20
    exit_stable_frames: int = 8
    exit_path_confidence: float = 0.45


@dataclass
class PhaseEstimateResult:
    internal_state: PhaseInternalState = PhaseInternalState.UNKNOWN
    decision_phase: IntersectionPhase = IntersectionPhase.UNKNOWN
    confidence: float = 0.0
    reason_code: str = "not_initialized"
    marker_source: str = "crosswalk"
    crosswalk_present: bool = False
    crosswalk_bottom_y_ratio: float | None = None
    crosswalk_area_ratio: float = 0.0
    crosswalk_width_ratio: float = 0.0
    crosswalk_center_overlap: bool = False
    age_frames: int = 0
    missing_frames: int = 0
    exit_stable_count: int = 0
    near_age_frames: int = 0
    bottom_confirm_frames: int = 0
    area_drop_ratio: float = 1.0


class IntersectionPhaseEstimator:
    """Frame-based FSM using only crosswalk mask position and persistence."""

    def __init__(self, config: PhaseEstimatorConfig | None = None):
        self.config = config or PhaseEstimatorConfig()
        self.state = PhaseInternalState.UNKNOWN
        self.age_frames = 0
        self.missing_frames = 0
        self.exit_stable_count = 0
        self.last_bottom_y_ratio: float | None = None
        self.last_area_ratio: float = 0.0
        self.near_age_frames = 0
        self.bottom_confirm_frames = 0
        self.near_reference_area_ratio = 0.0

    def update(self, road_state, seg_mask: np.ndarray, frame_shape: tuple[int, int]) -> PhaseEstimateResult:
        marker = _extract_crosswalk_marker(seg_mask, frame_shape, self.config)
        visible = marker["present"]
        bottom = marker["bottom_y_ratio"]
        area_ratio = float(marker["area_ratio"])

        prev_state = self.state
        prev_area_ratio = self.last_area_ratio
        if visible:
            self.missing_frames = 0
            self.last_bottom_y_ratio = bottom
            self.last_area_ratio = area_ratio
        else:
            self.missing_frames += 1

        reason = "hold"

        if self.state == PhaseInternalState.UNKNOWN:
            if visible and bottom is not None and bottom >= self.config.detect_y_ratio:
                self._set_state(PhaseInternalState.APPROACH)
                reason = "crosswalk_detected"
            else:
                reason = "no_reliable_crosswalk"

        elif self.state == PhaseInternalState.APPROACH:
            if visible and bottom is not None and bottom >= self.config.near_y_ratio:
                self._set_state(PhaseInternalState.NEAR_CROSSWALK)
                self.near_reference_area_ratio = max(area_ratio, prev_area_ratio, 1e-6)
                reason = "crosswalk_near_bottom"
            elif not visible and self.missing_frames > self.config.exit_stable_frames:
                self._set_state(PhaseInternalState.UNKNOWN)
                reason = "crosswalk_lost_before_near"
            else:
                reason = "approaching_crosswalk"

        elif self.state == PhaseInternalState.NEAR_CROSSWALK:
            self.near_age_frames += 1
            if visible:
                self.near_reference_area_ratio = max(self.near_reference_area_ratio, area_ratio, 1e-6)
            area_drop_ratio = area_ratio / max(self.near_reference_area_ratio, 1e-6)

            if visible and bottom is not None and bottom >= self.config.enter_y_ratio:
                self.bottom_confirm_frames += 1
            else:
                self.bottom_confirm_frames = 0

            can_enter = self.near_age_frames >= self.config.min_near_frames_before_enter
            area_dropped_after_near = (
                can_enter
                and self.near_reference_area_ratio >= self.config.min_enter_reference_area_ratio
                and visible
                and area_drop_ratio <= self.config.area_drop_ratio_for_enter
                and (bottom is None or bottom >= self.config.near_y_ratio)
            )
            lost_after_near = (
                can_enter
                and self.near_reference_area_ratio >= self.config.min_enter_reference_area_ratio
                and (not visible)
                and self.missing_frames >= self.config.lost_enter_confirm_frames
            )
            bottom_confirmed = can_enter and self.bottom_confirm_frames >= self.config.bottom_enter_confirm_frames

            if lost_after_near:
                self._set_state(PhaseInternalState.IN_INTERSECTION)
                reason = "near_crosswalk_lost_confirmed"
            elif area_dropped_after_near:
                self._set_state(PhaseInternalState.IN_INTERSECTION)
                reason = "crosswalk_area_dropped_after_near"
            elif bottom_confirmed:
                # Seeing the crosswalk touch the lower image is not enough to
                # say the ego vehicle has crossed it. Stay in APPROACH until
                # the first crosswalk begins to disappear or shrink.
                reason = "crosswalk_bottom_seen_waiting_to_cross"
            else:
                reason = "near_crosswalk"

        elif self.state == PhaseInternalState.IN_INTERSECTION:
            path_stable = bool(getattr(road_state, "path_available", False)) and (
                float(getattr(road_state, "path_confidence", 0.0)) >= self.config.exit_path_confidence
            )
            if not visible and path_stable:
                self.exit_stable_count += 1
            else:
                self.exit_stable_count = 0

            if (
                self.age_frames >= self.config.min_in_intersection_frames
                and self.exit_stable_count >= self.config.exit_stable_frames
            ):
                self._set_state(PhaseInternalState.UNKNOWN)
                reason = "intersection_passed"
            else:
                reason = "inside_intersection_hold"

        if self.state == prev_state:
            self.age_frames += 1

        confidence = _phase_confidence(self.state, marker, road_state, self.config)
        return PhaseEstimateResult(
            internal_state=self.state,
            decision_phase=_to_decision_phase(self.state),
            confidence=confidence,
            reason_code=reason,
            crosswalk_present=visible,
            crosswalk_bottom_y_ratio=bottom,
            crosswalk_area_ratio=marker["area_ratio"],
            crosswalk_width_ratio=marker["width_ratio"],
            crosswalk_center_overlap=marker["center_overlap"],
            age_frames=self.age_frames,
            missing_frames=self.missing_frames,
            exit_stable_count=self.exit_stable_count,
            near_age_frames=self.near_age_frames,
            bottom_confirm_frames=self.bottom_confirm_frames,
            area_drop_ratio=(
                marker["area_ratio"] / max(self.near_reference_area_ratio, 1e-6)
                if self.near_reference_area_ratio > 0
                else 1.0
            ),
        )

    def _set_state(self, next_state: PhaseInternalState) -> None:
        if next_state != self.state:
            self.state = next_state
            self.age_frames = 0
            self.exit_stable_count = 0
            self.bottom_confirm_frames = 0
            if next_state != PhaseInternalState.NEAR_CROSSWALK:
                self.near_age_frames = 0
                self.near_reference_area_ratio = 0.0


def _extract_crosswalk_marker(
    seg_mask: np.ndarray,
    frame_shape: tuple[int, int],
    config: PhaseEstimatorConfig,
) -> dict:
    h, w = frame_shape[:2]
    if seg_mask is None or seg_mask.size == 0:
        return {"present": False, "bottom_y_ratio": None, "area_ratio": 0.0, "width_ratio": 0.0, "center_overlap": False}

    mask = (seg_mask == CROSSWALK_CLASS_ID).astype(np.uint8)
    if mask.shape[:2] != (h, w):
        h, w = mask.shape[:2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    raw_min_area = max(30, int(h * w * config.min_crosswalk_area_ratio))
    phase_min_area = max(raw_min_area, int(h * w * config.min_phase_area_ratio))
    best_area = 0
    best_bottom = None
    best_width_ratio = 0.0
    best_center_overlap = False
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        width_ratio = float(width / max(1, w))
        right_ratio = float((left + width - 1) / max(1, w - 1))
        left_ratio = float(left / max(1, w - 1))
        center_overlap = (
            right_ratio >= config.center_band_left_ratio
            and left_ratio <= config.center_band_right_ratio
        )
        if (
            area < phase_min_area
            or width_ratio < config.min_phase_width_ratio
            or not center_overlap
            or area <= best_area
        ):
            continue
        top = int(stats[label, cv2.CC_STAT_TOP])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        best_area = area
        best_bottom = top + height - 1
        best_width_ratio = width_ratio
        best_center_overlap = center_overlap

    if best_bottom is None:
        return {"present": False, "bottom_y_ratio": None, "area_ratio": 0.0, "width_ratio": 0.0, "center_overlap": False}

    return {
        "present": True,
        "bottom_y_ratio": float(best_bottom / max(1, h - 1)),
        "area_ratio": float(best_area / max(1, h * w)),
        "width_ratio": best_width_ratio,
        "center_overlap": best_center_overlap,
    }


def _to_decision_phase(internal_state: PhaseInternalState) -> IntersectionPhase:
    if internal_state in (PhaseInternalState.APPROACH, PhaseInternalState.NEAR_CROSSWALK):
        return IntersectionPhase.APPROACH
    if internal_state == PhaseInternalState.IN_INTERSECTION:
        return IntersectionPhase.IN_INTERSECTION
    return IntersectionPhase.UNKNOWN


def _phase_confidence(state: PhaseInternalState, marker: dict, road_state, config: PhaseEstimatorConfig) -> float:
    if state == PhaseInternalState.UNKNOWN:
        return 0.0
    bottom = marker["bottom_y_ratio"]
    area_ratio = marker["area_ratio"]
    area_score = min(1.0, area_ratio / max(config.min_phase_area_ratio * 3.0, 1e-6))
    path_score = float(getattr(road_state, "path_confidence", 0.0)) if getattr(road_state, "path_available", False) else 0.0
    if bottom is None:
        visible_score = 0.35 if state == PhaseInternalState.IN_INTERSECTION else 0.15
    else:
        visible_score = min(1.0, max(0.0, bottom))
    return float(max(0.0, min(1.0, 0.58 * visible_score + 0.27 * area_score + 0.15 * path_score)))


def draw_phase_estimate(vis: np.ndarray, result: PhaseEstimateResult | None) -> None:
    """Draw compact phase-estimator debug info."""
    if result is None:
        return
    h, w = vis.shape[:2]
    x0 = max(6, w - 368)
    y0 = 126
    bottom = "-" if result.crosswalk_bottom_y_ratio is None else f"{result.crosswalk_bottom_y_ratio:.2f}"
    line = (
        f"phase_auto:{result.internal_state.value} conf:{result.confidence:.2f} "
        f"cwY:{bottom} reason:{result.reason_code}"
    )
    cv2.rectangle(vis, (x0, y0), (min(w - 6, x0 + 362), y0 + 22), (0, 0, 0), -1)
    cv2.putText(vis, line[:58], (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 255, 255), 1, cv2.LINE_AA)
