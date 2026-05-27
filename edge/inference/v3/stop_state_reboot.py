"""
Frame-based stop-completion estimation for v3.

This module estimates whether the ego vehicle has completed a brief stop on a
red signal.  It does not use vehicle speed sensors, so the result is only a
camera-based stop-completion hint.  The safety layer should still describe it
as confirmation evidence, not a legal proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from safety_decision_reboot import IntersectionPhase, SignalState


@dataclass(frozen=True)
class StopCompletionConfig:
    """Tunable frame-based stop detector settings."""

    roi_left_ratio: float = 0.18
    roi_right_ratio: float = 0.82
    roi_top_ratio: float = 0.52
    roi_bottom_ratio: float = 0.92
    resize_width: int = 220
    resize_height: int = 120
    motion_threshold_px: float = 0.42
    confirm_frames: int = 5
    max_features: int = 140
    min_features: int = 12


@dataclass
class StopCompletionState:
    """Current stop-completion estimate."""

    stop_completed: bool = False
    stopped_now: bool = False
    motion_score: float = 0.0
    stable_frames: int = 0
    source: str = "unavailable"
    roi: tuple[int, int, int, int] | None = None


class StopCompletionTracker:
    """
    Estimate stop completion from low motion in the lower road image region.

    The tracker is active only for APPROACH + RED.  Once a stop is confirmed, it
    remains completed for the current red-approach episode even if the vehicle
    starts moving again.
    """

    def __init__(self, config: StopCompletionConfig | None = None):
        self.config = config or StopCompletionConfig()
        self._prev_roi_gray: np.ndarray | None = None
        self._stable_frames = 0
        self._completed = False

    def update(self,
               frame: np.ndarray,
               phase: IntersectionPhase,
               vehicle_signal: SignalState,
               manual_completed: bool = False) -> StopCompletionState:
        if manual_completed:
            self._completed = True
            self._stable_frames = max(self._stable_frames, self.config.confirm_frames)
            self._prev_roi_gray, roi = self._extract_roi_gray(frame)
            return StopCompletionState(
                stop_completed=True,
                stopped_now=True,
                motion_score=0.0,
                stable_frames=self._stable_frames,
                source="manual",
                roi=roi,
            )

        roi_gray, roi = self._extract_roi_gray(frame)
        if roi_gray is None:
            return StopCompletionState(source="unavailable")

        active = phase == IntersectionPhase.APPROACH and vehicle_signal == SignalState.RED
        if not active:
            self._prev_roi_gray = roi_gray
            self._stable_frames = 0
            self._completed = False
            return StopCompletionState(
                stop_completed=False,
                motion_score=0.0,
                stable_frames=0,
                source="inactive",
                roi=roi,
            )

        if self._prev_roi_gray is None:
            self._prev_roi_gray = roi_gray
            return StopCompletionState(
                stop_completed=False,
                motion_score=0.0,
                stable_frames=0,
                source="initializing",
                roi=roi,
            )

        motion_score = self._motion_score(self._prev_roi_gray, roi_gray)
        self._prev_roi_gray = roi_gray
        stopped_now = motion_score <= self.config.motion_threshold_px
        if stopped_now:
            self._stable_frames += 1
        else:
            self._stable_frames = 0

        if self._stable_frames >= self.config.confirm_frames:
            self._completed = True

        return StopCompletionState(
            stop_completed=self._completed,
            stopped_now=stopped_now,
            motion_score=float(motion_score),
            stable_frames=self._stable_frames,
            source="auto_motion",
            roi=roi,
        )

    def _extract_roi_gray(self, frame: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        if frame is None or frame.size == 0:
            return None, None

        h, w = frame.shape[:2]
        x1 = int(w * self.config.roi_left_ratio)
        x2 = int(w * self.config.roi_right_ratio)
        y1 = int(h * self.config.roi_top_ratio)
        y2 = int(h * self.config.roi_bottom_ratio)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.config.resize_width, self.config.resize_height), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, (x1, y1, x2, y2)

    def _motion_score(self, prev_gray: np.ndarray, gray: np.ndarray) -> float:
        points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.config.max_features,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=5,
        )
        if points is None or len(points) < self.config.min_features:
            diff = cv2.absdiff(prev_gray, gray)
            return float(np.mean(diff) / 10.0)

        next_points, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
        if next_points is None or status is None:
            diff = cv2.absdiff(prev_gray, gray)
            return float(np.mean(diff) / 10.0)

        valid = status.reshape(-1) == 1
        if int(np.count_nonzero(valid)) < self.config.min_features:
            diff = cv2.absdiff(prev_gray, gray)
            return float(np.mean(diff) / 10.0)

        prev_valid = points.reshape(-1, 2)[valid]
        next_valid = next_points.reshape(-1, 2)[valid]
        flow = np.linalg.norm(next_valid - prev_valid, axis=1)
        return float(np.median(flow))
