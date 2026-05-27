"""
v2_0 road structure reboot helpers.

This module keeps road-structure post-processing separate from safety-state
logic.  It only turns segmentation/object detections into interpretable scene
features that a later FSM can use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


CROSSWALK_CLASS_ID = 4
PEDESTRIAN_CLASS_NAMES = {"pedestrian", "person"}


@dataclass
class PathCorridor:
    """
    Current ego path corridor used by road-structure logic.

    source:
        lane         - current frame ego-lane polygon, slightly expanded.
        cached_lane  - very short frame cache for one-to-three-frame dropouts.
        unavailable  - no reliable path corridor.
    """

    mask: Optional[np.ndarray] = None
    source: str = "unavailable"
    confidence: float = 0.0
    age_frames: int = 0

    @property
    def available(self) -> bool:
        return self.mask is not None and self.confidence > 0.0


@dataclass
class CrosswalkZone:
    """
    Crosswalk area related to the current ego path.

    active_mask:
        Crosswalk pixels that overlap the current ego lane/path corridor.
        This is the part the ego vehicle is most likely to enter.
    near_mask:
        Crosswalk pixels around the ego path.  Pedestrians here are not
        necessarily in the direct path yet, but can enter the active zone soon.
    """

    active_mask: Optional[np.ndarray] = None
    near_mask: Optional[np.ndarray] = None
    active_contour: Optional[np.ndarray] = None
    near_contour: Optional[np.ndarray] = None
    active_pixels: int = 0
    near_pixels: int = 0
    total_pixels: int = 0
    overlap_ratio: float = 0.0
    center_y: Optional[int] = None
    bottom_y: Optional[int] = None
    distance_to_bottom_ratio: Optional[float] = None
    confidence: float = 0.0
    status: str = "not_found"
    path_source: str = "unavailable"
    path_confidence: float = 0.0
    path_age_frames: int = 0


@dataclass
class CrosswalkPedestrianStatus:
    """Pedestrian relation to the current crosswalk zone."""

    active_count: int = 0
    near_count: int = 0
    nearest_distance_px: Optional[float] = None
    reason: str = "no_pedestrian"

    @property
    def has_active_pedestrian(self) -> bool:
        return self.active_count > 0

    @property
    def has_near_pedestrian(self) -> bool:
        return self.near_count > 0


def _odd_kernel_size(value: int) -> int:
    size = max(3, int(value))
    return size if size % 2 == 1 else size + 1


def _build_crosswalk_mask(seg_mask: np.ndarray) -> np.ndarray:
    """
    Extract only the road_v4 crosswalk class.

    A small close/dilate step connects tiny segmentation holes, but does not
    invent a full crossing region when the model did not detect one.
    """

    raw = (seg_mask == CROSSWALK_CLASS_ID).astype(np.uint8) * 255
    if raw.size == 0:
        return raw

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(raw, dilate_kernel, iterations=1)


def _largest_contour(mask: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None
    return best


def _lane_width_from_polygon(polygon: np.ndarray, fallback_w: int) -> float:
    """Estimate visible ego-lane width near the bottom of the polygon."""

    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return float(fallback_w)

    bottom_y = float(np.max(points[:, 1]))
    lower_points = points[points[:, 1] >= bottom_y - max(12.0, fallback_w * 0.04)]
    if len(lower_points) < 2:
        lower_points = points
    return float(np.max(lower_points[:, 0]) - np.min(lower_points[:, 0]))


def _expand_mask_by_y_bands(
    mask: np.ndarray,
    max_expand_px: int,
    bottom_min_expand_px: int = 5,
    band_count: int = 12,
) -> np.ndarray:
    """
    Expand the lane corridor mostly in the upper image area.

    Near the bottom of the image, the ego-lane position is usually more stable,
    so only a small minimum margin is added.  Farther ahead, perspective and
    segmentation jitter are larger, so each y-band gets progressively more
    margin.
    """

    if max_expand_px <= 0 or mask.size == 0:
        return mask.copy()

    h, w = mask.shape[:2]
    expanded = mask.copy()
    top_y = 0
    band_count = max(1, int(band_count))
    edges = np.linspace(top_y, h, band_count + 1, dtype=np.int32)
    bottom_min_expand_px = max(0, min(int(bottom_min_expand_px), int(max_expand_px)))

    for y1, y2 in zip(edges[:-1], edges[1:]):
        if y2 <= 0 or y1 >= h:
            continue

        band_center_y = (int(y1) + int(y2)) * 0.5
        upper_ratio = 1.0 - (band_center_y / max(1.0, h))
        expand_px = int(round(bottom_min_expand_px + (max_expand_px - bottom_min_expand_px) * upper_ratio))

        if expand_px <= 0:
            continue

        band = np.zeros_like(mask)
        band[max(0, int(y1)):min(h, int(y2)), :] = mask[max(0, int(y1)):min(h, int(y2)), :]
        kernel_size = expand_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.bitwise_or(expanded, cv2.dilate(band, kernel, iterations=1))

    return expanded


def _path_mask_from_ego(ego_lane, frame_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """
    Build a conservative lane corridor from the ego-lane polygon.

    The corridor keeps the near-field lane mostly unchanged and adds margin
    progressively toward the upper image area.  That absorbs far-field jitter
    without over-expanding the ego vehicle's immediate lower-screen area.
    """

    h, w = frame_shape[:2]
    path_mask = np.zeros((h, w), dtype=np.uint8)

    polygon = getattr(ego_lane, "polygon", None)
    confidence = float(getattr(ego_lane, "confidence", 0.0))
    if polygon is None or len(polygon) < 3 or confidence < 0.20:
        return None

    poly = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(path_mask, [poly], 255)

    lane_width = _lane_width_from_polygon(poly, w)
    max_expand_px = int(np.clip(lane_width * 0.20, 14, min(40, w * 0.065)))
    return _expand_mask_by_y_bands(path_mask, max_expand_px)


class PathCorridorPipeline:
    """
    Build a short-lived path corridor from ego-lane estimates.

    This deliberately uses only current lane geometry plus a frame-count cache.
    No long extrapolation or arbitrary turn curve is added, so unavailable path
    remains unavailable once the short cache expires.
    """

    def __init__(self, ttl_frames: int = 3, min_cached_conf: float = 0.30):
        self.ttl_frames = max(0, int(ttl_frames))
        self.min_cached_conf = float(min_cached_conf)
        self._cached_mask: Optional[np.ndarray] = None
        self._cached_confidence = 0.0
        self._age_frames = 0

    def update(self, ego_lane, frame_shape: tuple[int, int]) -> PathCorridor:
        h, w = frame_shape[:2]
        current_mask = _path_mask_from_ego(ego_lane, (h, w)) if ego_lane is not None else None
        current_conf = float(getattr(ego_lane, "confidence", 0.0)) if ego_lane is not None else 0.0

        if current_mask is not None:
            self._cached_mask = current_mask.copy()
            self._cached_confidence = max(0.0, min(1.0, current_conf))
            self._age_frames = 0
            return PathCorridor(
                mask=current_mask,
                source="lane",
                confidence=self._cached_confidence,
                age_frames=0,
            )

        if self._cached_mask is not None and self._age_frames < self.ttl_frames:
            self._age_frames += 1
            decay = max(0.0, 1.0 - 0.22 * self._age_frames)
            confidence = self._cached_confidence * decay
            if confidence >= self.min_cached_conf:
                return PathCorridor(
                    mask=self._cached_mask.copy(),
                    source="cached_lane",
                    confidence=float(confidence),
                    age_frames=self._age_frames,
                )

        self._cached_mask = None
        self._cached_confidence = 0.0
        self._age_frames = 0
        return PathCorridor(source="unavailable")


def estimate_crosswalk_zone(seg_mask: np.ndarray, path_corridor: PathCorridor) -> CrosswalkZone:
    """
    Estimate crosswalk zones from segmentation and ego path.

    crosswalk_zone(active) = crosswalk_mask intersect path_corridor
    crosswalk_near_zone   = crosswalk_mask near a wider path_corridor,
                            expanded slightly for pedestrian foot-point checks.
    """

    h, w = seg_mask.shape[:2]
    crosswalk_mask = _build_crosswalk_mask(seg_mask)
    total_pixels = int(np.count_nonzero(crosswalk_mask))
    min_pixels = max(30, int(h * w * 0.00008))
    if total_pixels < min_pixels:
        return CrosswalkZone(total_pixels=total_pixels, status="not_found")

    if not path_corridor.available:
        return CrosswalkZone(
            total_pixels=total_pixels,
            status="no_path_corridor",
            path_source=path_corridor.source,
            path_confidence=path_corridor.confidence,
            path_age_frames=path_corridor.age_frames,
        )

    path_mask = path_corridor.mask
    active_dilate = _odd_kernel_size(max(7, int(w * 0.015)))
    near_dilate = _odd_kernel_size(max(45, int(w * 0.14)))
    foot_margin = _odd_kernel_size(max(21, int(w * 0.055)))

    active_path = cv2.dilate(
        path_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (active_dilate, active_dilate)),
        iterations=1,
    )
    near_path = cv2.dilate(
        path_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_dilate, near_dilate)),
        iterations=1,
    )

    active_mask = cv2.bitwise_and(crosswalk_mask, active_path)
    near_base = cv2.bitwise_and(crosswalk_mask, near_path)
    near_mask = cv2.dilate(
        near_base,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (foot_margin, foot_margin)),
        iterations=1,
    )

    active_pixels = int(np.count_nonzero(active_mask))
    near_pixels = int(np.count_nonzero(near_mask))
    overlap_ratio = active_pixels / max(1, total_pixels)
    contour_min_area = max(12.0, h * w * 0.00004)
    active_contour = _largest_contour(active_mask, contour_min_area)
    near_contour = _largest_contour(near_mask, contour_min_area)

    if active_pixels < min_pixels or active_contour is None:
        confidence = min(0.45, near_pixels / max(1.0, total_pixels) * 0.35)
        return CrosswalkZone(
            active_mask=active_mask,
            near_mask=near_mask,
            near_contour=near_contour,
            active_pixels=active_pixels,
            near_pixels=near_pixels,
            total_pixels=total_pixels,
            overlap_ratio=overlap_ratio,
            confidence=float(confidence),
            status="near_only" if near_pixels >= min_pixels else "no_active_overlap",
            path_source=path_corridor.source,
            path_confidence=path_corridor.confidence,
            path_age_frames=path_corridor.age_frames,
        )

    points = active_contour.reshape(-1, 2)
    center_y = int(np.mean(points[:, 1]))
    bottom_y = int(np.max(points[:, 1]))
    distance_to_bottom_ratio = max(0.0, min(1.0, (h - bottom_y) / max(1.0, h)))

    area_score = min(1.0, active_pixels / max(1.0, h * w * 0.012))
    overlap_score = min(1.0, overlap_ratio * 2.0)
    confidence = float(0.25 + 0.45 * area_score + 0.30 * overlap_score)
    confidence *= max(0.0, min(1.0, path_corridor.confidence))

    return CrosswalkZone(
        active_mask=active_mask,
        near_mask=near_mask,
        active_contour=active_contour,
        near_contour=near_contour,
        active_pixels=active_pixels,
        near_pixels=near_pixels,
        total_pixels=total_pixels,
        overlap_ratio=float(overlap_ratio),
        center_y=center_y,
        bottom_y=bottom_y,
        distance_to_bottom_ratio=float(distance_to_bottom_ratio),
        confidence=min(1.0, confidence),
        status="active",
        path_source=path_corridor.source,
        path_confidence=path_corridor.confidence,
        path_age_frames=path_corridor.age_frames,
    )


def _is_pedestrian(det) -> bool:
    name = getattr(det, "cls_name", "").lower()
    return name in PEDESTRIAN_CLASS_NAMES


def _point_hits_mask(mask: Optional[np.ndarray], point: tuple[int, int], radius: int = 4) -> bool:
    if mask is None:
        return False

    h, w = mask.shape[:2]
    x, y = point
    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    return bool(np.any(mask[y1:y2, x1:x2] > 0))


def evaluate_crosswalk_pedestrians(
    detections,
    zone: CrosswalkZone,
    frame_shape: tuple[int, int],
) -> CrosswalkPedestrianStatus:
    """
    Check pedestrian foot-points against the active/near crosswalk zones.

    The bottom-center of a pedestrian bbox is used because it approximates the
    contact point with the road better than the bbox center.
    """

    if zone.status == "not_found":
        return CrosswalkPedestrianStatus(reason="no_crosswalk")

    h, w = frame_shape[:2]
    active_count = 0
    near_count = 0
    nearest: Optional[float] = None

    for det in detections or []:
        if not _is_pedestrian(det):
            continue
        if getattr(det, "conf", 0.0) < 0.25:
            continue

        x1, y1, x2, y2 = [int(v) for v in getattr(det, "box", (0, 0, 0, 0))]
        foot = (int((x1 + x2) * 0.5), int(y2))
        foot = (max(0, min(w - 1, foot[0])), max(0, min(h - 1, foot[1])))

        if _point_hits_mask(zone.active_mask, foot):
            active_count += 1
        elif _point_hits_mask(zone.near_mask, foot, radius=6):
            near_count += 1

        if zone.active_contour is not None:
            signed = cv2.pointPolygonTest(zone.active_contour, foot, True)
            distance = 0.0 if signed >= 0 else abs(float(signed))
            nearest = distance if nearest is None else min(nearest, distance)

    if active_count > 0:
        reason = "active_pedestrian"
    elif near_count > 0:
        reason = "near_pedestrian"
    else:
        reason = "no_pedestrian"

    return CrosswalkPedestrianStatus(
        active_count=active_count,
        near_count=near_count,
        nearest_distance_px=nearest,
        reason=reason,
    )


def _blend_mask(vis: np.ndarray, mask: Optional[np.ndarray], color: tuple[int, int, int], alpha: float) -> None:
    if mask is None or not np.any(mask > 0):
        return

    overlay = vis.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0.0, dst=vis)


def draw_path_corridor(vis: np.ndarray, path_corridor: PathCorridor) -> None:
    """
    Draw the path corridor used by crosswalk-zone calculation.

    It is drawn below crosswalk active/near overlays, but with a visible outline
    so it does not disappear when it overlaps the green ego-lane fill.
    """

    if not path_corridor.available:
        return

    if path_corridor.source == "cached_lane":
        fill_color = (255, 120, 40)
        line_color = (255, 230, 60)
        alpha = 0.26
    else:
        fill_color = (255, 210, 40)
        line_color = (255, 255, 80)
        alpha = 0.22

    _blend_mask(vis, path_corridor.mask, fill_color, alpha)

    contours, _ = cv2.findContours(path_corridor.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(vis, contours, -1, line_color, 1, cv2.LINE_AA)


def draw_crosswalk_zone(
    vis: np.ndarray,
    zone: CrosswalkZone,
    pedestrian_status: CrosswalkPedestrianStatus,
    alpha: float = 0.28,
    show_text: bool = True,
) -> None:
    """Draw crosswalk active/near zones and pedestrian relation text."""

    if zone.status == "not_found":
        return

    _blend_mask(vis, zone.near_mask, (0, 220, 255), min(0.16, alpha * 0.55))
    _blend_mask(vis, zone.active_mask, (0, 120, 255), alpha)

    if zone.active_contour is not None:
        cv2.drawContours(vis, [zone.active_contour], -1, (0, 140, 255), 1, cv2.LINE_AA)
    elif zone.near_contour is not None:
        cv2.drawContours(vis, [zone.near_contour], -1, (0, 220, 255), 1, cv2.LINE_AA)

    if not show_text:
        return

    y = 196
    if vis.shape[0] < 260:
        y = max(70, vis.shape[0] - 78)

    text = (
        f"crosswalk {zone.status} "
        f"actP:{pedestrian_status.active_count} nearP:{pedestrian_status.near_count} "
        f"path:{zone.path_source} pconf:{zone.path_confidence:.2f} age:{zone.path_age_frames} "
        f"conf:{zone.confidence:.2f}"
    )
    cv2.putText(vis, text, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
