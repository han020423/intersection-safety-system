"""
Traffic-signal color estimation for v3.

YOLO detects the traffic-light object box, then this module estimates the light
color from the crop using HSV color ratios.  It is intentionally conservative:
if the crop is too small, too dark, or color evidence is weak, the result stays
UNKNOWN.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from safety_decision_reboot import SignalState


VEHICLE_SIGNAL_CLASS = "traffic_light_vehicle"
PEDESTRIAN_SIGNAL_CLASS = "traffic_light_pedestrian"


@dataclass
class SignalColorEstimate:
    state: SignalState = SignalState.UNKNOWN
    confidence: float = 0.0
    source_class: str = ""
    box: tuple[int, int, int, int] | None = None
    red_score: int = 0
    yellow_score: int = 0
    green_score: int = 0


@dataclass
class SignalRecognitionResult:
    vehicle_signal: SignalState = SignalState.UNKNOWN
    vehicle_confidence: float = 0.0
    vehicle_signal_source: str = "unknown"
    pedestrian_signal: SignalState = SignalState.UNKNOWN
    pedestrian_confidence: float = 0.0
    right_turn_signal: SignalState = SignalState.UNKNOWN
    right_turn_confidence: float = 0.0


@dataclass(frozen=True)
class SignalSmootherConfig:
    window_frames: int = 5
    min_votes: int = 3
    unknown_hold_frames: int = 2
    approach_red_hold_frames: int = 24
    min_confidence: float = 0.18


@dataclass
class _SignalTrack:
    config: SignalSmootherConfig
    history: deque = field(init=False)
    stable_state: SignalState = SignalState.UNKNOWN
    stable_confidence: float = 0.0
    unknown_count: int = 0

    def __post_init__(self) -> None:
        self.history = deque(maxlen=max(1, self.config.window_frames))

    def update(self, raw_state: SignalState, raw_confidence: float) -> tuple[SignalState, float]:
        raw_confidence = float(raw_confidence)
        if raw_confidence < self.config.min_confidence:
            raw_state = SignalState.UNKNOWN
            raw_confidence = 0.0

        self.history.append((raw_state, raw_confidence))

        if raw_state == SignalState.UNKNOWN:
            self.unknown_count += 1
        else:
            self.unknown_count = 0

        votes: dict[SignalState, float] = {}
        counts: dict[SignalState, int] = {}
        for state, confidence in self.history:
            if state in (SignalState.UNKNOWN, SignalState.NONE):
                continue
            votes[state] = votes.get(state, 0.0) + max(0.01, float(confidence))
            counts[state] = counts.get(state, 0) + 1

        if votes:
            best_state = max(votes.keys(), key=lambda state: (counts[state], votes[state]))
            if counts[best_state] >= self.config.min_votes:
                self.stable_state = best_state
                self.stable_confidence = min(1.0, votes[best_state] / max(1, counts[best_state]))
                return self.stable_state, self.stable_confidence

        if self.stable_state != SignalState.UNKNOWN and self.unknown_count <= self.config.unknown_hold_frames:
            return self.stable_state, self.stable_confidence * 0.75

        self.stable_state = SignalState.UNKNOWN
        self.stable_confidence = 0.0
        return self.stable_state, self.stable_confidence


class SignalStateTracker:
    """Temporal smoother for traffic-light color estimates."""

    def __init__(self, config: SignalSmootherConfig | None = None):
        self.config = config or SignalSmootherConfig()
        self.vehicle = _SignalTrack(self.config)
        self.pedestrian = _SignalTrack(self.config)
        self.right_turn = _SignalTrack(self.config)
        self._approach_red_hold_remaining = 0
        self._approach_red_hold_confidence = 0.0

    def update(self, raw: SignalRecognitionResult | None) -> SignalRecognitionResult:
        raw = raw or SignalRecognitionResult()
        vehicle_state, vehicle_conf = self.vehicle.update(raw.vehicle_signal, raw.vehicle_confidence)
        pedestrian_state, pedestrian_conf = self.pedestrian.update(raw.pedestrian_signal, raw.pedestrian_confidence)
        right_turn_state, right_turn_conf = self.right_turn.update(raw.right_turn_signal, raw.right_turn_confidence)
        return SignalRecognitionResult(
            vehicle_signal=vehicle_state,
            vehicle_confidence=vehicle_conf,
            vehicle_signal_source="smoothed",
            pedestrian_signal=pedestrian_state,
            pedestrian_confidence=pedestrian_conf,
            right_turn_signal=right_turn_state,
            right_turn_confidence=right_turn_conf,
        )

    def apply_approach_red_hold(self,
                                result: SignalRecognitionResult | None,
                                phase) -> SignalRecognitionResult:
        """
        Keep a recently stable RED through short UNKNOWN gaps during APPROACH.

        Traffic-light boxes often flicker near intersections because signs,
        gantries, and motion blur disturb the small signal crop.  During the
        approach phase, a recently stable RED should remain RED for a short
        frame window unless a definite GREEN/YELLOW/NONE replaces it.
        """

        result = result or SignalRecognitionResult()
        phase_name = str(getattr(phase, "value", phase)).upper()

        if result.vehicle_signal == SignalState.RED:
            self._approach_red_hold_remaining = self.config.approach_red_hold_frames
            self._approach_red_hold_confidence = max(
                float(result.vehicle_confidence),
                self._approach_red_hold_confidence,
            )
            return result

        if result.vehicle_signal in (SignalState.GREEN, SignalState.YELLOW, SignalState.NONE):
            self._approach_red_hold_remaining = 0
            self._approach_red_hold_confidence = 0.0
            return result

        if phase_name == "APPROACH" and self._approach_red_hold_remaining > 0:
            self._approach_red_hold_remaining -= 1
            return replace(
                result,
                vehicle_signal=SignalState.RED,
                vehicle_confidence=max(0.18, self._approach_red_hold_confidence * 0.70),
                vehicle_signal_source="approach_red_hold",
            )

        if phase_name != "APPROACH":
            self._approach_red_hold_remaining = 0

        return result


def estimate_traffic_signals(frame: np.ndarray, detections) -> SignalRecognitionResult:
    """Estimate vehicle/pedestrian signal states from YOLO traffic-light crops."""
    vehicle = _best_signal_for_class(frame, detections, VEHICLE_SIGNAL_CLASS)
    pedestrian = _best_signal_for_class(frame, detections, PEDESTRIAN_SIGNAL_CLASS)

    # Dedicated right-turn lights are not a separate YOLO class in the current
    # model. Keep this UNKNOWN unless a future detector adds such a class.
    return SignalRecognitionResult(
        vehicle_signal=vehicle.state,
        vehicle_confidence=vehicle.confidence,
        vehicle_signal_source="raw",
        pedestrian_signal=pedestrian.state,
        pedestrian_confidence=pedestrian.confidence,
        right_turn_signal=SignalState.UNKNOWN,
        right_turn_confidence=0.0,
    )


def _best_signal_for_class(frame: np.ndarray, detections, target_class: str) -> SignalColorEstimate:
    candidates: list[tuple[float, SignalColorEstimate]] = []
    for det in detections or []:
        cls_name = str(getattr(det, "cls_name", "")).lower()
        if target_class not in cls_name:
            continue
        estimate = _classify_light_crop(frame, getattr(det, "box", None), cls_name)
        if estimate.state == SignalState.UNKNOWN:
            continue
        x1, y1, x2, y2 = estimate.box or (0, 0, 0, 0)
        area = max(1, (x2 - x1) * (y2 - y1))
        score = float(getattr(det, "conf", 0.0)) * area * estimate.confidence
        candidates.append((score, estimate))

    if not candidates:
        return SignalColorEstimate()
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def _classify_light_crop(frame: np.ndarray, box, source_class: str) -> SignalColorEstimate:
    if frame is None or box is None:
        return SignalColorEstimate(source_class=source_class)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return SignalColorEstimate(source_class=source_class)

    # Slightly shrink the crop so bbox borders, poles, and nearby signs affect
    # color classification less.
    pad_x = max(1, int((x2 - x1) * 0.08))
    pad_y = max(1, int((y2 - y1) * 0.08))
    sx1, sy1 = min(x2 - 1, x1 + pad_x), min(y2 - 1, y1 + pad_y)
    sx2, sy2 = max(sx1 + 1, x2 - pad_x), max(sy1 + 1, y2 - pad_y)
    crop = frame[sy1:sy2, sx1:sx2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return SignalColorEstimate(source_class=source_class, box=(x1, y1, x2, y2))

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    bright = (sat > 65) & (val > 95)
    bright_count = int(np.count_nonzero(bright))
    if bright_count < max(6, int(crop.shape[0] * crop.shape[1] * 0.015)):
        return SignalColorEstimate(source_class=source_class, box=(x1, y1, x2, y2))

    red = bright & ((hue <= 10) | (hue >= 165))
    yellow = bright & (hue >= 15) & (hue <= 38)
    green = bright & (hue >= 40) & (hue <= 95)

    counts = {
        SignalState.RED: int(np.count_nonzero(red)),
        SignalState.YELLOW: int(np.count_nonzero(yellow)),
        SignalState.GREEN: int(np.count_nonzero(green)),
    }
    state, count = max(counts.items(), key=lambda item: item[1])
    second = sorted(counts.values(), reverse=True)[1]
    min_color_pixels = max(8, int(bright_count * 0.12))
    if count < min_color_pixels:
        return SignalColorEstimate(
            source_class=source_class,
            box=(x1, y1, x2, y2),
            red_score=counts[SignalState.RED],
            yellow_score=counts[SignalState.YELLOW],
            green_score=counts[SignalState.GREEN],
        )

    dominance = (count - second) / max(1, count)
    ratio = count / max(1, bright_count)
    confidence = float(max(0.0, min(1.0, 0.55 * ratio + 0.45 * dominance)))
    if confidence < 0.18:
        state = SignalState.UNKNOWN

    return SignalColorEstimate(
        state=state,
        confidence=confidence if state != SignalState.UNKNOWN else 0.0,
        source_class=source_class,
        box=(x1, y1, x2, y2),
        red_score=counts[SignalState.RED],
        yellow_score=counts[SignalState.YELLOW],
        green_score=counts[SignalState.GREEN],
    )
