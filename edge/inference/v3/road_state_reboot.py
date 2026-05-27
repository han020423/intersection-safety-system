"""
v3 road-structure state builder.

This module summarizes recognition outputs into values that a safety-decision
engine can read.  It does not decide STOP/CAUTION/GO.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np


CROSSWALK_CLASS_ID = 4
STOP_LINE_CLASS_ID = 5
PEDESTRIAN_CLASS_NAMES = {"pedestrian", "person"}
VEHICLE_CLASS_NAMES = {"vehicle", "car", "bus", "truck", "motorcycle"}


@dataclass
class RoadStructureState:
    """One-frame road/object state used as input to safety decisions."""

    path_available: bool = False
    path_source: str = "unavailable"
    path_confidence: float = 0.0
    path_age_frames: int = 0

    crosswalk_present: bool = False
    pedestrian_on_crosswalk: bool = False
    pedestrian_near_crosswalk: bool = False
    pedestrian_approaching_crosswalk: bool = False
    pedestrian_leaving_crosswalk: bool = False
    active_pedestrian_count: int = 0
    near_pedestrian_count: int = 0
    approaching_pedestrian_count: int = 0
    leaving_pedestrian_count: int = 0
    stationary_pedestrian_count: int = 0
    unknown_motion_pedestrian_count: int = 0
    nearest_pedestrian_crosswalk_distance_px: Optional[float] = None

    stop_line_present: bool = False
    stop_line_y_ratio: Optional[float] = None

    lane_role_leftmost: Optional[bool] = None
    lane_role_rightmost: Optional[bool] = None
    lane_role_confidence: float = 0.0
    lane_role_source: str = "unavailable"
    lane_role_age_frames: int = 0
    lane_role_left_score: float = 0.0
    lane_role_right_score: float = 0.0
    left_adjacent_vehicle: bool = False
    right_adjacent_vehicle: bool = False

    vehicle_count: int = 0
    pedestrian_count: int = 0


@dataclass(frozen=True)
class LaneRoleMemoryConfig:
    """Frame-based cache settings for lane-role use near intersections."""

    min_store_confidence: float = 0.45
    min_prefer_current_confidence: float = 0.55
    max_cache_age_frames: int = 180


@dataclass
class _CachedLaneRole:
    leftmost: Optional[bool]
    rightmost: Optional[bool]
    confidence: float
    left_score: float
    right_score: float
    left_adjacent_vehicle: bool
    right_adjacent_vehicle: bool
    age_frames: int = 0


class LaneRoleMemory:
    """
    Keep the last reliable lane-role estimate before/around intersection entry.

    Lane masks often become unstable near crosswalks or inside intersections.
    Safety decisions still need the lane role measured just before the structure
    became unstable, especially for right-turn lane checks.  This helper stores
    reliable role estimates and reuses them during APPROACH only when the current
    frame is unknown or weak.
    """

    def __init__(self, config: LaneRoleMemoryConfig | None = None):
        self.config = config or LaneRoleMemoryConfig()
        self._cache: _CachedLaneRole | None = None

    def update(self, state: RoadStructureState, phase) -> RoadStructureState:
        """Return a state whose lane-role fields may be filled from memory."""

        if self._cache is not None:
            self._cache.age_frames += 1
            if self._cache.age_frames > self.config.max_cache_age_frames:
                self._cache = None

        phase_name = str(getattr(phase, "value", phase)).upper()
        reliable_current = self._is_reliable_current(state)

        if phase_name == "IN_INTERSECTION":
            return state

        if phase_name == "APPROACH":
            if self._cache is not None:
                # Do not overwrite pre-intersection lane role with noisy
                # crosswalk/intersection-adjacent markings.  APPROACH should
                # prefer the last reliable role measured before entry.
                return replace(
                    state,
                    lane_role_leftmost=self._cache.leftmost,
                    lane_role_rightmost=self._cache.rightmost,
                    lane_role_confidence=max(state.lane_role_confidence, self._cache.confidence),
                    lane_role_source="cached_before_intersection",
                    lane_role_age_frames=self._cache.age_frames,
                    lane_role_left_score=self._cache.left_score,
                    lane_role_right_score=self._cache.right_score,
                    left_adjacent_vehicle=self._cache.left_adjacent_vehicle,
                    right_adjacent_vehicle=self._cache.right_adjacent_vehicle,
                )
            if reliable_current:
                self._store(state)
                return replace(state, lane_role_source="current", lane_role_age_frames=0)
            return state

        if reliable_current:
            if phase_name != "APPROACH" or state.lane_role_confidence >= self.config.min_prefer_current_confidence:
                self._store(state)
                return replace(state, lane_role_source="current", lane_role_age_frames=0)
            if self._cache is None:
                self._store(state)
                return replace(state, lane_role_source="current", lane_role_age_frames=0)

        if reliable_current:
            self._store(state)
            return replace(state, lane_role_source="current", lane_role_age_frames=0)

        return state

    def _is_reliable_current(self, state: RoadStructureState) -> bool:
        has_role = state.lane_role_leftmost is not None or state.lane_role_rightmost is not None
        return has_role and state.lane_role_confidence >= self.config.min_store_confidence

    def _store(self, state: RoadStructureState) -> None:
        self._cache = _CachedLaneRole(
            leftmost=state.lane_role_leftmost,
            rightmost=state.lane_role_rightmost,
            confidence=float(state.lane_role_confidence),
            left_score=float(state.lane_role_left_score),
            right_score=float(state.lane_role_right_score),
            left_adjacent_vehicle=bool(state.left_adjacent_vehicle),
            right_adjacent_vehicle=bool(state.right_adjacent_vehicle),
            age_frames=0,
        )


def _is_pedestrian(det) -> bool:
    return getattr(det, "cls_name", "").lower() in PEDESTRIAN_CLASS_NAMES


def _is_vehicle(det) -> bool:
    return getattr(det, "cls_name", "").lower() in VEHICLE_CLASS_NAMES


def _foot_point(det, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Use bbox bottom-center as the pedestrian contact point with the road."""

    x1, _y1, x2, y2 = [int(v) for v in getattr(det, "box", (0, 0, 0, 0))]
    x = int((x1 + x2) * 0.5)
    y = int(y2)
    return max(0, min(frame_w - 1, x)), max(0, min(frame_h - 1, y))


def _build_visible_crosswalk_mask(seg_mask: np.ndarray) -> np.ndarray:
    """Extract visible crosswalk pixels for phase-independent pedestrian checks."""

    mask = (seg_mask == CROSSWALK_CLASS_ID).astype(np.uint8) * 255
    if mask.size == 0:
        return mask

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return mask


def _point_hits_mask(mask: np.ndarray, point: tuple[int, int], radius: int = 4) -> bool:
    h, w = mask.shape[:2]
    x, y = point
    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    return bool(np.any(mask[y1:y2, x1:x2] > 0))


def _stop_line_state(seg_mask: np.ndarray) -> tuple[bool, Optional[float]]:
    h, w = seg_mask.shape[:2]
    ys = np.where(seg_mask == STOP_LINE_CLASS_ID)[0]
    min_pixels = max(12, int(h * w * 0.00005))
    if len(ys) < min_pixels:
        return False, None
    return True, float(np.mean(ys) / max(1, h))


def build_road_structure_state(
    ego_lane,
    path_corridor,
    crosswalk_zone,
    crosswalk_pedestrian_status,
    seg_mask: np.ndarray,
    detections,
    frame_shape: tuple[int, int],
    pedestrian_motion_status=None,
) -> RoadStructureState:
    """
    Summarize v3 recognition outputs for safety decisions.

    Safety decisions should be based mainly on visible crosswalk on/near
    pedestrian evidence.  Path corridor remains as auxiliary evidence and
    visualization context.
    """

    h, w = frame_shape[:2]
    state = RoadStructureState()

    if path_corridor is not None:
        state.path_available = bool(getattr(path_corridor, "available", False))
        state.path_source = str(getattr(path_corridor, "source", "unavailable"))
        state.path_confidence = float(getattr(path_corridor, "confidence", 0.0))
        state.path_age_frames = int(getattr(path_corridor, "age_frames", 0))

    crosswalk_mask = _build_visible_crosswalk_mask(seg_mask)
    crosswalk_pixels = int(np.count_nonzero(crosswalk_mask))
    min_crosswalk_pixels = max(30, int(h * w * 0.00008))
    state.crosswalk_present = crosswalk_pixels >= min_crosswalk_pixels

    if state.crosswalk_present:
        near_px = max(18, int(w * 0.055))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_px * 2 + 1, near_px * 2 + 1))
        near_mask = cv2.dilate(crosswalk_mask, kernel, iterations=1)
    else:
        near_mask = np.zeros_like(crosswalk_mask)

    for det in detections or []:
        if _is_pedestrian(det):
            state.pedestrian_count += 1
            foot = _foot_point(det, w, h)
            if state.crosswalk_present and _point_hits_mask(crosswalk_mask, foot):
                state.active_pedestrian_count += 1
            elif state.crosswalk_present and _point_hits_mask(near_mask, foot, radius=6):
                state.near_pedestrian_count += 1
        elif _is_vehicle(det):
            state.vehicle_count += 1

    # Fall back to path-corridor based counts only when visible-crosswalk checks
    # found nothing.  This preserves older debug evidence without making it the
    # primary legal/safety signal.
    if state.active_pedestrian_count == 0 and crosswalk_pedestrian_status is not None:
        state.active_pedestrian_count = int(getattr(crosswalk_pedestrian_status, "active_count", 0))
    if state.near_pedestrian_count == 0 and crosswalk_pedestrian_status is not None:
        state.near_pedestrian_count = int(getattr(crosswalk_pedestrian_status, "near_count", 0))

    state.pedestrian_on_crosswalk = state.active_pedestrian_count > 0
    state.pedestrian_near_crosswalk = state.near_pedestrian_count > 0

    if pedestrian_motion_status is not None:
        state.approaching_pedestrian_count = int(getattr(pedestrian_motion_status, "approaching_count", 0))
        state.leaving_pedestrian_count = int(getattr(pedestrian_motion_status, "leaving_count", 0))
        state.stationary_pedestrian_count = int(getattr(pedestrian_motion_status, "stationary_count", 0))
        state.unknown_motion_pedestrian_count = int(getattr(pedestrian_motion_status, "unknown_count", 0))
        state.nearest_pedestrian_crosswalk_distance_px = getattr(
            pedestrian_motion_status,
            "nearest_distance_px",
            None,
        )
    state.pedestrian_approaching_crosswalk = state.approaching_pedestrian_count > 0
    state.pedestrian_leaving_crosswalk = state.leaving_pedestrian_count > 0

    state.stop_line_present, state.stop_line_y_ratio = _stop_line_state(seg_mask)

    role = getattr(ego_lane, "role", None)
    if role is not None:
        state.lane_role_leftmost = getattr(role, "is_leftmost_lane", None)
        state.lane_role_rightmost = getattr(role, "is_rightmost_lane", None)
        state.lane_role_confidence = float(getattr(role, "confidence", 0.0))
        state.lane_role_left_score = float(getattr(role, "leftmost_score", 0.0))
        state.lane_role_right_score = float(getattr(role, "rightmost_score", 0.0))
        state.left_adjacent_vehicle = bool(getattr(role, "left_adjacent_vehicle", False))
        state.right_adjacent_vehicle = bool(getattr(role, "right_adjacent_vehicle", False))
        if state.lane_role_leftmost is not None or state.lane_role_rightmost is not None:
            state.lane_role_source = "current"

    return state
