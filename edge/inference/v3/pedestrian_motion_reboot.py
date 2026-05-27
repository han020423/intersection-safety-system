"""
Pedestrian motion direction estimation for v3.

The tracker follows pedestrian foot points across frames and compares two
signals:

1. distance to the visible crosswalk mask
2. alignment between the pedestrian movement vector and the vector from the
   pedestrian to the crosswalk center

This is a camera-based movement hint, not a legally definitive intent estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


CROSSWALK_CLASS_ID = 4
PEDESTRIAN_CLASS_NAMES = {"pedestrian", "person"}


@dataclass(frozen=True)
class PedestrianMotionConfig:
    max_match_distance_px: float = 80.0
    max_track_age_frames: int = 8
    min_hits_for_direction: int = 3
    approach_delta_px: float = 6.0
    leave_delta_px: float = 6.0
    min_vector_motion_px: float = 4.0
    approach_cos_threshold: float = 0.50
    leave_cos_threshold: float = -0.30
    near_distance_ratio: float = 0.18


@dataclass
class PedestrianMotionStatus:
    approaching_count: int = 0
    leaving_count: int = 0
    stationary_count: int = 0
    unknown_count: int = 0
    tracked_count: int = 0
    nearest_distance_px: float | None = None
    reason: str = "no_pedestrian"

    @property
    def has_approaching_pedestrian(self) -> bool:
        return self.approaching_count > 0

    @property
    def has_leaving_pedestrian(self) -> bool:
        return self.leaving_count > 0


@dataclass
class _PedTrack:
    track_id: int
    foot: tuple[float, float]
    distance_px: float
    hits: int = 1
    missed: int = 0
    direction: str = "unknown"
    motion_px: float = 0.0
    crosswalk_cos: float | None = None
    direction_reason: str = "new_track"


class PedestrianMotionTracker:
    """Nearest-neighbor foot-point tracker for crosswalk-relative motion."""

    def __init__(self, config: PedestrianMotionConfig | None = None):
        self.config = config or PedestrianMotionConfig()
        self._tracks: list[_PedTrack] = []
        self._next_track_id = 1

    def update(self, detections, seg_mask: np.ndarray, frame_shape: tuple[int, int]) -> PedestrianMotionStatus:
        h, w = frame_shape[:2]
        crosswalk_mask = self._build_crosswalk_mask(seg_mask)
        if int(np.count_nonzero(crosswalk_mask)) < max(30, int(h * w * 0.00008)):
            self._age_unmatched_tracks()
            return PedestrianMotionStatus(reason="no_crosswalk")

        dist_map = self._crosswalk_distance_map(crosswalk_mask)
        crosswalk_center = self._crosswalk_center(crosswalk_mask)
        observations = []
        for det in detections or []:
            if str(getattr(det, "cls_name", "")).lower() not in PEDESTRIAN_CLASS_NAMES:
                continue
            foot = self._foot_point(det, w, h)
            distance_px = float(dist_map[int(foot[1]), int(foot[0])])
            observations.append((foot, distance_px))

        if not observations:
            self._age_unmatched_tracks()
            return PedestrianMotionStatus(reason="no_pedestrian")

        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        observed_track_ids: set[int] = set()
        for obs_idx, (foot, distance_px) in enumerate(observations):
            best_idx = -1
            best_dist = float("inf")
            for track_idx, track in enumerate(self._tracks):
                if track_idx in matched_tracks:
                    continue
                spatial_dist = float(np.hypot(foot[0] - track.foot[0], foot[1] - track.foot[1]))
                if spatial_dist < best_dist and spatial_dist <= self.config.max_match_distance_px:
                    best_dist = spatial_dist
                    best_idx = track_idx
            if best_idx < 0:
                continue

            track = self._tracks[best_idx]
            delta = track.distance_px - distance_px
            motion_vec = np.array([foot[0] - track.foot[0], foot[1] - track.foot[1]], dtype=np.float32)
            to_crosswalk_vec = np.array(
                [crosswalk_center[0] - foot[0], crosswalk_center[1] - foot[1]],
                dtype=np.float32,
            )
            motion_px = float(np.linalg.norm(motion_vec))
            crosswalk_cos = self._cosine_similarity(motion_vec, to_crosswalk_vec)
            if track.hits + 1 >= self.config.min_hits_for_direction:
                # 거리 변화와 방향 정렬을 함께 본다. 한쪽만 순간적으로 튀어도
                # 다른 근거가 맞으면 접근/이탈 힌트를 유지할 수 있다.
                vector_approaching = (
                    motion_px >= self.config.min_vector_motion_px
                    and crosswalk_cos is not None
                    and crosswalk_cos >= self.config.approach_cos_threshold
                )
                vector_leaving = (
                    motion_px >= self.config.min_vector_motion_px
                    and crosswalk_cos is not None
                    and crosswalk_cos <= self.config.leave_cos_threshold
                )
                distance_approaching = delta >= self.config.approach_delta_px
                distance_leaving = delta <= -self.config.leave_delta_px

                if distance_approaching or vector_approaching:
                    track.direction = "approaching"
                    track.direction_reason = self._direction_reason(distance_approaching, vector_approaching)
                elif distance_leaving or vector_leaving:
                    track.direction = "leaving"
                    track.direction_reason = self._direction_reason(distance_leaving, vector_leaving)
                else:
                    track.direction = "stationary"
                    track.direction_reason = "weak_motion"
            else:
                track.direction = "unknown"
                track.direction_reason = "short_track"
            track.motion_px = motion_px
            track.crosswalk_cos = crosswalk_cos
            track.foot = foot
            track.distance_px = distance_px
            track.hits += 1
            track.missed = 0
            matched_tracks.add(best_idx)
            matched_obs.add(obs_idx)
            observed_track_ids.add(track.track_id)

        for idx, (foot, distance_px) in enumerate(observations):
            if idx in matched_obs:
                continue
            track_id = self._next_track_id
            self._tracks.append(
                _PedTrack(
                    track_id=track_id,
                    foot=foot,
                    distance_px=distance_px,
                )
            )
            observed_track_ids.add(track_id)
            self._next_track_id += 1

        for track in self._tracks:
            if track.track_id not in observed_track_ids:
                track.missed += 1

        self._tracks = [track for track in self._tracks if track.missed <= self.config.max_track_age_frames]
        return self._summarize_current_tracks(w)

    def _summarize_current_tracks(self, frame_w: int) -> PedestrianMotionStatus:
        near_limit = max(35.0, frame_w * self.config.near_distance_ratio)
        current = [track for track in self._tracks if track.missed == 0 and track.distance_px <= near_limit]
        if not current:
            return PedestrianMotionStatus(reason="no_near_pedestrian")

        status = PedestrianMotionStatus(
            tracked_count=len(current),
            nearest_distance_px=min(track.distance_px for track in current),
            reason="tracked",
        )
        for track in current:
            if track.direction == "approaching":
                status.approaching_count += 1
            elif track.direction == "leaving":
                status.leaving_count += 1
            elif track.direction == "stationary":
                status.stationary_count += 1
            else:
                status.unknown_count += 1
        return status

    def _age_unmatched_tracks(self) -> None:
        for track in self._tracks:
            track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self.config.max_track_age_frames]

    @staticmethod
    def _build_crosswalk_mask(seg_mask: np.ndarray) -> np.ndarray:
        mask = (seg_mask == CROSSWALK_CLASS_ID).astype(np.uint8) * 255
        if mask.size == 0:
            return mask
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    @staticmethod
    def _crosswalk_distance_map(crosswalk_mask: np.ndarray) -> np.ndarray:
        inverse = np.where(crosswalk_mask > 0, 0, 255).astype(np.uint8)
        return cv2.distanceTransform(inverse, cv2.DIST_L2, 3)

    @staticmethod
    def _crosswalk_center(crosswalk_mask: np.ndarray) -> tuple[float, float]:
        ys, xs = np.where(crosswalk_mask > 0)
        if len(xs) == 0:
            h, w = crosswalk_mask.shape[:2]
            return float(w * 0.5), float(h * 0.5)
        return float(np.mean(xs)), float(np.mean(ys))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-6:
            return None
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _direction_reason(distance_matched: bool, vector_matched: bool) -> str:
        if distance_matched and vector_matched:
            return "distance_and_vector"
        if vector_matched:
            return "vector_to_crosswalk"
        return "distance_to_crosswalk"

    @staticmethod
    def _foot_point(det, frame_w: int, frame_h: int) -> tuple[float, float]:
        x1, _y1, x2, y2 = [int(v) for v in getattr(det, "box", (0, 0, 0, 0))]
        x = float((x1 + x2) * 0.5)
        y = float(y2)
        return max(0.0, min(frame_w - 1.0, x)), max(0.0, min(frame_h - 1.0, y))
