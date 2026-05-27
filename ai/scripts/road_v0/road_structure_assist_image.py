#!/usr/bin/env python3
"""
Lightweight road-structure perception for a single image.

Example:
    python road_structure_assist_image.py \
        --weights best.pt \
        --source test.jpg \
        --width 640 --height 360 \
        --show --save output.jpg
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ----------------------------- Data structures ----------------------------- #


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box: Tuple[int, int, int, int]


@dataclass
class LineModel:
    present: bool = False
    x_bottom: Optional[int] = None
    x_top: Optional[int] = None
    slope: Optional[float] = None
    confidence: float = 0.0
    poly_coeffs: np.ndarray | None = None
    curve_points: np.ndarray | None = None


@dataclass
class BoundaryModel:
    present: bool = False
    points: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RoadStructureResult:
    mode: str
    center_x: int
    offset_px: float
    heading_deg: float
    confidence: float
    lane_confidence: float
    road_confidence: float
    intersection_likely: bool
    crosswalk_present: bool
    left_lane: LineModel
    right_lane: LineModel
    left_boundary: BoundaryModel
    right_boundary: BoundaryModel
    road_mask: np.ndarray
    lane_mask: np.ndarray
    detections: List[Detection]
    debug: Dict[str, float] = field(default_factory=dict)


# ------------------------------ YOLO wrapper ------------------------------- #


class YoloObjectDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install with: pip install ultralytics"
            ) from exc

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_names = self._resolve_class_names()

    def _resolve_class_names(self) -> Dict[int, str]:
        names = getattr(self.model.model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(name) for i, name in enumerate(names)}
        return {}

    def infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        boxes = getattr(results, "boxes", None)
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(np.int32)
            for box, score, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    Detection(
                        cls_id=int(cls_id),
                        cls_name=self.class_names.get(int(cls_id), str(cls_id)),
                        conf=float(score),
                        box=(x1, y1, x2, y2),
                    )
                )

        return detections


# -------------------------- Geometry/helper methods ------------------------ #


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class EMA:
    def __init__(self, alpha: float, init: Optional[float] = None) -> None:
        self.alpha = alpha
        self.value = init

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


# ----------------------- Road structure core estimator --------------------- #


class RoadStructureEstimator:
    def __init__(self) -> None:
        # 단일 이미지이므로 이전 프레임의 영향을 받는 EMA (지수 이동 평균)의 알파 값을 1.0으로 설정하여 현재 값을 100% 반영합니다.
        self.center_ema = EMA(alpha=1.0)
        self.heading_ema = EMA(alpha=1.0)
        self._debug_masks: Dict[str, np.ndarray] = {}

    def process(self, frame: np.ndarray, detections: List[Detection]) -> RoadStructureResult:
        h, w = frame.shape[:2]

        roi_mask = self._build_driving_roi(h, w)
        exclusion = self._build_exclusion_masks(frame.shape[:2], detections)
        lane_mask, left_lane, right_lane = self._estimate_lane_markings(frame, roi_mask, exclusion)
        road_mask, left_boundary, right_boundary, road_conf = self._estimate_road_region(
            frame, roi_mask, exclusion, lane_mask
        )

        crosswalk_present = any(d.cls_name == "crosswalk" for d in detections)
        lane_conf = float((left_lane.confidence + right_lane.confidence) / 2.0)
        lane_count = int(left_lane.present) + int(right_lane.present)
        lane_pair_ok = lane_count == 2
        single_lane_ok = lane_count == 1

        if lane_pair_ok and lane_conf >= 0.40:
            mode = "lane"
        elif single_lane_ok and max(left_lane.confidence, right_lane.confidence) >= 0.35:
            mode = "lane"
        elif crosswalk_present and lane_conf < 0.40:
            mode = "intersection"
        else:
            mode = "road_boundary"

        center_x, heading_deg = self._fuse_center_and_heading(
            frame.shape[:2], mode, left_lane, right_lane, left_boundary, right_boundary
        )

        offset_px = float(center_x - (w / 2.0))
        confidence = float(clamp(0.6 * lane_conf + 0.4 * road_conf, 0.0, 1.0))
        intersection_likely = bool(crosswalk_present or (lane_conf < 0.35 and road_conf > 0.35))

        return RoadStructureResult(
            mode=mode,
            center_x=center_x,
            offset_px=offset_px,
            heading_deg=heading_deg,
            confidence=confidence,
            lane_confidence=lane_conf,
            road_confidence=road_conf,
            intersection_likely=intersection_likely,
            crosswalk_present=crosswalk_present,
            left_lane=left_lane,
            right_lane=right_lane,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            road_mask=road_mask,
            lane_mask=lane_mask,
            detections=detections,
            debug={
                "lane_confidence": lane_conf,
                "road_confidence": road_conf,
                "offset_px": offset_px,
                "heading_deg": heading_deg,
            },
        )

    def _build_driving_roi(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        polygon = np.array(
            [
                (int(0.02 * w), h - 1),
                (int(0.98 * w), h - 1),
                (int(0.78 * w), int(0.42 * h)),
                (int(0.22 * w), int(0.42 * h)),
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def _build_exclusion_masks(self, shape: Tuple[int, int], detections: List[Detection]) -> Dict[str, np.ndarray]:
        h, w = shape
        obstacle = np.zeros((h, w), dtype=np.uint8)
        crosswalk = np.zeros((h, w), dtype=np.uint8)
        signal = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            x1, y1, x2, y2 = det.box
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)

            if det.cls_name in {"vehicle", "pedestrian"}:
                cv2.rectangle(obstacle, (x1, y1), (x2, y2), 255, -1)
            elif det.cls_name == "crosswalk":
                cv2.rectangle(crosswalk, (x1, y1), (x2, y2), 255, -1)
            elif det.cls_name.startswith("traffic_light"):
                cv2.rectangle(signal, (x1, y1), (x2, y2), 255, -1)

        obstacle = cv2.dilate(obstacle, np.ones((7, 7), np.uint8), iterations=1)
        crosswalk = cv2.dilate(crosswalk, np.ones((11, 11), np.uint8), iterations=1)
        return {"obstacle": obstacle, "crosswalk": crosswalk, "signal": signal}

    def _get_bev_transforms(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        bev_w, bev_h = 400, 600
        # 지평선 부근의 에러 증폭을 줄이기 위해 원근뷰 꼭대기(top)를 살짝 내림
        src = np.float32([
            [w * 0.06, h * 0.98],
            [w * 0.38, h * 0.66],
            [w * 0.62, h * 0.66],
            [w * 0.94, h * 0.98],
        ])
        dst = np.float32([
            [bev_w * 0.2, bev_h],
            [bev_w * 0.2, 0],
            [bev_w * 0.8, 0],
            [bev_w * 0.8, bev_h],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv, bev_w, bev_h

    def _estimate_lane_markings(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray,
        exclusion: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, LineModel, LineModel]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # 1. Advanced Lane Finding: Sobel X Gradient (수직 엣지 강력 추출)
        # 세로 방향 엣지만 극대화하여 나뭇잎 등 등방성 노이즈 철저히 배제
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= 25) & (scaled_sobel <= 200)] = 255

        # 2. HLS S-Channel (채도) 및 L-Channel (명도)
        # S-Channel 은 빛 번짐/노란색 등을 정말 잘 잡아내며, L 은 뚜렷하게 밝은 흰색을 잡아냅니다.
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 150) & (s_channel <= 255)] = 255
        
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= 210) & (l_channel <= 255)] = 255
        
        # 3. Fusion (융합)
        color_mask = cv2.bitwise_or(sxbinary, cv2.bitwise_or(s_binary, l_binary))
        color_mask = cv2.bitwise_and(color_mask, roi_mask)
        color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(exclusion["crosswalk"]))
        
        # 2. Morphology: 노이즈 제거 및 선형태 보강
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 디버그: color_mask 저장
        self._debug_masks["color_mask"] = color_mask.copy()

        # 3. Canny & BEV Transform
        # color_mask는 차선 후보 강화용, edges/Hough는 구조 보강용으로 분리
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 70, 150)
        edges = cv2.bitwise_and(edges, roi_mask)
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(exclusion["obstacle"]))

        M, Minv, bev_w, bev_h = self._get_bev_transforms(h, w)
        color_bev = cv2.warpPerspective(color_mask, M, (bev_w, bev_h))
        edges_bev = cv2.warpPerspective(edges, M, (bev_w, bev_h))

        # 4. HoughLinesP IN BEV: BEV 공간에서 Hough 변환하여 신뢰성 낮은 반점(blobs) 제거
        lines = cv2.HoughLinesP(
            edges_bev,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=30,
            maxLineGap=25,
        )
        
        filtered_bev_mask = np.zeros_like(color_bev)
        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, line)
                dx = x2 - x1
                dy = y2 - y1
                if dy == 0:
                    continue
                angle = abs(math.degrees(math.atan2(dy, dx)))
                # 세로(주행방향)에 적당히 가까운 선분만 필터링 (가로 차선 배제)
                if 45 <= angle <= 135:
                    cv2.line(filtered_bev_mask, (x1, y1), (x2, y2), 255, 12)
                    
        # Hough는 "검증용" 보강이지 "필수 조건"이 아님 → bitwise_or로 완화
        if np.count_nonzero(filtered_bev_mask) > 0:
            combined_bev = cv2.bitwise_or(color_bev, filtered_bev_mask)
        else:
            combined_bev = color_bev.copy()

        combined_bev = cv2.morphologyEx(combined_bev, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_bev = cv2.morphologyEx(combined_bev, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        # 구역 나누기 (좌/우)
        mid_x = bev_w // 2
        left_bev = combined_bev.copy()
        left_bev[:, mid_x:] = 0
        right_bev = combined_bev.copy()
        right_bev[:, :mid_x] = 0

        # 디버그: BEV 관련 마스크 저장
        self._debug_masks["edges"] = edges.copy()
        self._debug_masks["color_bev"] = color_bev.copy()
        self._debug_masks["hough_bev"] = filtered_bev_mask.copy()
        self._debug_masks["combined_bev"] = combined_bev.copy()

        # 5 & 6. Contour 추출 및 Spline/Poly 피팅
        left_lane = self._fit_spline_model(left_bev, "left", h, Minv, bev_h)
        right_lane = self._fit_spline_model(right_bev, "right", h, Minv, bev_h)

        return color_mask, left_lane, right_lane

    def _fit_spline_model(self, bev_mask: np.ndarray, side: str, h: int, Minv: np.ndarray, bev_h: int) -> LineModel:
        bev_w = bev_mask.shape[1]
        
        # 1. 히스토그램을 사용해 하단(차량 기준 가까운 곳)에서 가장 선명한 차선 뿌리의 x위치를 찾음
        hist = np.sum(bev_mask[int(bev_h * 0.5) :, :], axis=0)
        
        # 중앙선/가로수 편향을 막기 위한 엄격한 시작 축 제약 (Search Margin)
        mid_x = bev_w // 2
        margin_x = int(bev_w * 0.08)
        offset = int(bev_w * 0.05)
        
        if side == "left":
            # 화면 왼쪽 특정 구간(margin_x ~ mid_x - offset)에서만 탐색
            search_area = hist[margin_x : mid_x - offset]
            if len(search_area) == 0:
                return LineModel(present=False)
            base_x = np.argmax(search_area) + margin_x
            max_val = hist[base_x]
        else:
            # 화면 오른쪽 특정 구간(mid_x + offset ~ bev_w - margin_x)에서만 탐색
            search_area = hist[mid_x + offset : bev_w - margin_x]
            if len(search_area) == 0:
                return LineModel(present=False)
            base_x = np.argmax(search_area) + mid_x + offset
            max_val = hist[base_x]
            
        if max_val < 30:
            return LineModel(present=False)
            
        # 2. Sliding Window (슬라이딩 윈도우) 방식으로 잡음(나무, 억측 마스크) 필터링
        n_windows = 15
        window_height = bev_h // n_windows
        margin = 40  # 윈도우 너비(좌우 여유분)
        min_pix = 20 # 중심축을 이동시키기 위한 최소 픽셀 수
        
        nonzero = bev_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        lane_inds = []
        current_x = base_x
        
        for window in range(n_windows):
            win_y_low = bev_h - (window + 1) * window_height
            win_y_high = bev_h - window * window_height
            win_x_low = current_x - margin
            win_x_high = current_x + margin
            
            # 박스 내의 유효 픽셀 수집
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            
            lane_inds.append(good_inds)
            
            # 충분한 픽셀이 확보되면 다음 윈도우 검색 중심을 이번 박스 픽셀의 평균 x로 이동
            if len(good_inds) > min_pix:
                current_x = int(np.mean(nonzerox[good_inds]))
                
        lane_inds = np.concatenate(lane_inds)
        
        if len(lane_inds) < 50:
            return LineModel(present=False)
            
        pts_x = nonzerox[lane_inds]
        pts_y = nonzeroy[lane_inds]
        
        # 픽셀 분포 파악
        min_y = int(np.min(pts_y))
        max_y = int(np.max(pts_y))
        y_range = max_y - min_y
        
        # 길이가 짧거나 휘어짐 연산이 불필요하면 1차식(직선) 피팅, 충분하면 2차식(포물선 스플라인) 피팅
        degree = 2 if y_range > bev_h * 0.35 else 1
        
        try:
            coeffs = np.polyfit(pts_y, pts_x, degree)
        except (np.linalg.LinAlgError, TypeError, ValueError):
            return LineModel(present=False)
            
        # 외삽 방지: 차선 픽셀이 모인 범위를 기준으로만 포인트 생성
        plot_min_y = max(0, min_y - 15)
        plot_max_y = bev_h - 1
        plot_y = np.linspace(plot_min_y, plot_max_y, int(plot_max_y - plot_min_y + 1))
        plot_x = np.polyval(coeffs, plot_y)
        
        # x좌표가 BEV 이미지를 이탈하는 경우 필터링
        valid_idx = (plot_x >= 0) & (plot_x < bev_w)
        plot_x, plot_y = plot_x[valid_idx], plot_y[valid_idx]
        
        if len(plot_y) < 10:
            return LineModel(present=False)
            
        # Inverse Transform
        pts_bev = np.vstack([plot_x, plot_y]).T.reshape(-1, 1, 2).astype(np.float32)
        pts_orig = cv2.perspectiveTransform(pts_bev, Minv).reshape(-1, 2).astype(np.int32)
        
        if len(pts_orig) < 2:
            return LineModel(present=False)
            
        # 좌표 정리
        bottom_idx = np.argmax(pts_orig[:, 1])
        top_idx = np.argmin(pts_orig[:, 1])
        
        x_bottom = int(pts_orig[bottom_idx][0])
        x_top = int(pts_orig[top_idx][0])
        
        dx = x_top - x_bottom
        dy = pts_orig[top_idx][1] - pts_orig[bottom_idx][1]
        slope = float(dy / (dx + 1e-6))
        
        raw_conf = min(1.0, len(pts_y) / (bev_h * 0.45))
        
        return LineModel(
            present=True,
            x_bottom=x_bottom,
            x_top=x_top,
            slope=slope,
            confidence=float(clamp(raw_conf, 0.0, 1.0)),
            poly_coeffs=coeffs,
            curve_points=pts_orig
        )

    def _estimate_road_region(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray,
        exclusion: Dict[str, np.ndarray],
        lane_mask: np.ndarray,
    ) -> Tuple[np.ndarray, BoundaryModel, BoundaryModel, float]:
        h, w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 여러 seed patch의 중앙값으로 도로 색상 추정 (단일 패치 민감도 완화)
        patches = self._pick_seed_patches(frame.shape[:2], exclusion)

        lab_means = []
        lab_stds = []
        sat_means = []

        for px1, py1, px2, py2 in patches:
            patch_lab = lab[py1:py2, px1:px2]
            patch_hsv = hsv[py1:py2, px1:px2]
            patch_lane = lane_mask[py1:py2, px1:px2] > 0
            valid = np.logical_not(patch_lane)
            if valid.sum() < 12:
                continue

            lab_means.append(patch_lab[valid].mean(axis=0))
            lab_stds.append(patch_lab[valid].std(axis=0) + 1.0)
            sat_means.append(float(patch_hsv[..., 1][valid].mean()))

        if len(lab_means) == 0:
            mean_lab = np.array([128, 128, 128], dtype=np.float32)
            std_lab = np.array([20, 10, 10], dtype=np.float32)
            mean_sat = 40.0
        else:
            mean_lab = np.median(np.array(lab_means), axis=0)
            std_lab = np.median(np.array(lab_stds), axis=0)
            mean_sat = float(np.median(np.array(sat_means)))

        dL = np.abs(lab[..., 0].astype(np.float32) - mean_lab[0]) / (std_lab[0] + 8.0)
        dA = np.abs(lab[..., 1].astype(np.float32) - mean_lab[1]) / (std_lab[1] + 6.0)
        dB = np.abs(lab[..., 2].astype(np.float32) - mean_lab[2]) / (std_lab[2] + 6.0)
        dist = dL + 0.8 * dA + 0.8 * dB

        low_texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        low_texture = cv2.convertScaleAbs(low_texture)
        texture_mask = cv2.threshold(low_texture, 28, 255, cv2.THRESH_BINARY_INV)[1]

        road_candidate = np.where(dist < 2.6, 255, 0).astype(np.uint8)
        road_candidate = cv2.bitwise_and(road_candidate, roi_mask)
        road_candidate = cv2.bitwise_and(road_candidate, texture_mask)

        road_candidate = cv2.bitwise_and(road_candidate, cv2.bitwise_not(exclusion["obstacle"]))
        road_candidate = cv2.bitwise_and(road_candidate, cv2.bitwise_not(exclusion["crosswalk"]))
        road_candidate = cv2.bitwise_and(road_candidate, cv2.bitwise_not(lane_mask))

        sat_mask = cv2.inRange(hsv[..., 1], 0, int(min(120, mean_sat + 40)))
        road_candidate = cv2.bitwise_and(road_candidate, sat_mask)

        kernel = np.ones((5, 5), np.uint8)
        road_candidate = cv2.morphologyEx(road_candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
        road_candidate = cv2.morphologyEx(road_candidate, cv2.MORPH_OPEN, kernel, iterations=1)

        # 디버그: road candidate 저장
        self._debug_masks["road_candidate"] = road_candidate.copy()

        road_mask = self._select_bottom_connected_component(road_candidate)
        left_boundary, right_boundary = self._extract_boundaries_from_mask(road_mask)

        # 디버그: road_mask 저장
        self._debug_masks["road_mask"] = road_mask.copy()

        roi_area = max(1, int(np.count_nonzero(roi_mask)))
        road_area = int(np.count_nonzero(road_mask))
        area_ratio = road_area / roi_area
        boundary_bonus = 0.2 if left_boundary.present and right_boundary.present else 0.0
        road_conf = float(clamp(0.8 * area_ratio + boundary_bonus, 0.0, 1.0))
        return road_mask, left_boundary, right_boundary, road_conf

    def _pick_seed_patches(self, shape: Tuple[int, int], exclusion: Dict[str, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """여러 후보 패치를 리턴하여 중앙값 기반 도로 색상 추정에 사용."""
        h, w = shape
        y1 = int(h * 0.84)
        y2 = int(h * 0.96)
        patch_w = int(w * 0.10)
        centers = [w // 2, int(w * 0.42), int(w * 0.58), int(w * 0.35), int(w * 0.65)]
        patches: List[Tuple[int, int, int, int]] = []

        for cx in centers:
            x1 = max(0, cx - patch_w // 2)
            x2 = min(w, cx + patch_w // 2)
            obstacle_ratio = exclusion["obstacle"][y1:y2, x1:x2].mean() / 255.0
            crosswalk_ratio = exclusion["crosswalk"][y1:y2, x1:x2].mean() / 255.0
            if obstacle_ratio < 0.08 and crosswalk_ratio < 0.08:
                patches.append((x1, y1, x2, y2))

        if not patches:
            patches.append((max(0, w // 2 - patch_w // 2), y1, min(w, w // 2 + patch_w // 2), y2))

        return patches

    def _select_bottom_connected_component(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return mask

        bottom_band = labels[int(h * 0.88) :, :]
        candidate_ids = np.unique(bottom_band)
        candidate_ids = candidate_ids[candidate_ids != 0]
        if len(candidate_ids) == 0:
            best_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        else:
            best_id = max(candidate_ids, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])

        out = np.zeros_like(mask)
        out[labels == best_id] = 255
        return out

    def _extract_boundaries_from_mask(self, road_mask: np.ndarray) -> Tuple[BoundaryModel, BoundaryModel]:
        h, w = road_mask.shape
        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []

        ys = list(range(h - 1, int(h * 0.46), -6))
        raw_left = []
        raw_right = []
        raw_y = []

        for y in ys:
            xs = np.where(road_mask[y] > 0)[0]
            if len(xs) < 30:
                continue

            # 극단값 대신 percentile 사용
            left_x = int(np.percentile(xs, 8))
            right_x = int(np.percentile(xs, 92))

            if right_x - left_x < 40:
                continue

            raw_left.append(left_x)
            raw_right.append(right_x)
            raw_y.append(y)

        if len(raw_y) < 8:
            return BoundaryModel(present=False), BoundaryModel(present=False)

        raw_y_arr = np.array(raw_y)
        raw_left_arr = np.array(raw_left)
        raw_right_arr = np.array(raw_right)

        # y에 대한 2차 다항식 스무딩
        left_coef = np.polyfit(raw_y_arr, raw_left_arr, 2 if len(raw_y) >= 10 else 1)
        right_coef = np.polyfit(raw_y_arr, raw_right_arr, 2 if len(raw_y) >= 10 else 1)

        fit_y = np.arange(raw_y_arr.min(), raw_y_arr.max() + 1, 6)
        fit_left = np.polyval(left_coef, fit_y)
        fit_right = np.polyval(right_coef, fit_y)

        for x, y in zip(fit_left, fit_y):
            if 0 <= x < w:
                left_pts.append((int(x), int(y)))

        for x, y in zip(fit_right, fit_y):
            if 0 <= x < w:
                right_pts.append((int(x), int(y)))

        left_boundary = BoundaryModel(
            present=len(left_pts) >= 8,
            points=left_pts,
            confidence=float(clamp(len(left_pts) / 20.0, 0.0, 1.0)),
        )
        right_boundary = BoundaryModel(
            present=len(right_pts) >= 8,
            points=right_pts,
            confidence=float(clamp(len(right_pts) / 20.0, 0.0, 1.0)),
        )
        return left_boundary, right_boundary

    def _fuse_center_and_heading(
        self,
        shape: Tuple[int, int],
        mode: str,
        left_lane: LineModel,
        right_lane: LineModel,
        left_boundary: BoundaryModel,
        right_boundary: BoundaryModel,
    ) -> Tuple[int, float]:
        h, w = shape
        bottom_y = int(h * 0.95)
        top_y = int(h * 0.58)

        center_bottom: Optional[float] = None
        center_top: Optional[float] = None

        if mode == "lane":
            # 양쪽 차선 다 보일 때
            if left_lane.present and right_lane.present:
                center_bottom = (left_lane.x_bottom + right_lane.x_bottom) / 2.0
                center_top = (left_lane.x_top + right_lane.x_top) / 2.0
            # 한쪽 차선만 보일 때: 임시 lane width 추정
            elif left_lane.present:
                lane_width_px = w * 0.42
                center_bottom = left_lane.x_bottom + lane_width_px / 2.0
                center_top = left_lane.x_top + lane_width_px / 2.0
            elif right_lane.present:
                lane_width_px = w * 0.42
                center_bottom = right_lane.x_bottom - lane_width_px / 2.0
                center_top = right_lane.x_top - lane_width_px / 2.0
        else:
            lb = self._boundary_x_at_y(left_boundary.points, bottom_y)
            rb = self._boundary_x_at_y(right_boundary.points, bottom_y)
            lt = self._boundary_x_at_y(left_boundary.points, top_y)
            rt = self._boundary_x_at_y(right_boundary.points, top_y)

            if lb is not None and rb is not None:
                center_bottom = (lb + rb) / 2.0
            elif lb is not None:
                center_bottom = lb + w * 0.25
            elif rb is not None:
                center_bottom = rb - w * 0.25

            if lt is not None and rt is not None:
                center_top = (lt + rt) / 2.0
            elif center_bottom is not None:
                center_top = center_bottom

        if center_bottom is None:
            center_bottom = w / 2.0
        if center_top is None:
            center_top = center_bottom

        center_bottom = self.center_ema.update(center_bottom)
        heading = math.degrees(math.atan2(center_bottom - center_top, bottom_y - top_y))
        heading = self.heading_ema.update(heading)

        return int(center_bottom), float(heading)

    def _boundary_x_at_y(self, points: Sequence[Tuple[int, int]], target_y: int) -> Optional[float]:
        if len(points) < 2:
            return None
        pts = sorted(points, key=lambda p: p[1])
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            if (y1 - target_y) * (y2 - target_y) <= 0 and y1 != y2:
                t = (target_y - y1) / float(y2 - y1)
                return x1 + t * (x2 - x1)
        return None


# ------------------------------ Visualization ------------------------------ #


def draw_result(frame: np.ndarray, result: RoadStructureResult) -> np.ndarray:
    vis = frame.copy()
    h, w = frame.shape[:2]

    road_overlay = np.zeros_like(frame)
    road_overlay[:, :, 1] = result.road_mask
    vis = cv2.addWeighted(vis, 1.0, road_overlay, 0.22, 0)

    lane_overlay = np.zeros_like(frame)
    lane_overlay[:, :, 2] = result.lane_mask
    vis = cv2.addWeighted(vis, 1.0, lane_overlay, 0.25, 0)

    if result.left_boundary.present:
        cv2.polylines(vis, [np.array(result.left_boundary.points, np.int32)], False, (0, 255, 255), 2)
    if result.right_boundary.present:
        cv2.polylines(vis, [np.array(result.right_boundary.points, np.int32)], False, (0, 255, 255), 2)

    for lane, color in ((result.left_lane, (255, 100, 0)), (result.right_lane, (255, 100, 0))):
        if lane.present and lane.curve_points is not None:
            cv2.polylines(vis, [lane.curve_points], False, color, 3)

    cv2.line(vis, (w // 2, h - 1), (w // 2, int(h * 0.58)), (120, 120, 120), 1)
    cv2.line(vis, (result.center_x, h - 1), (result.center_x, int(h * 0.58)), (0, 255, 0), 2)
    cv2.circle(vis, (result.center_x, h - 20), 5, (0, 255, 0), -1)

    class_colors = {
        "pedestrian": (0, 200, 255),
        "vehicle": (255, 80, 80),
        "traffic_light_vehicle": (80, 255, 80),
        "traffic_light_pedestrian": (80, 255, 80),
        "crosswalk": (255, 255, 0),
        "left_turn_sign": (255, 0, 255),
    }
    for det in result.detections:
        x1, y1, x2, y2 = det.box
        color = class_colors.get(det.cls_name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.conf:.2f}"
        cv2.putText(vis, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    status_lines = [
        f"mode: {result.mode}",
        f"offset_px: {result.offset_px:+.1f}",
        f"heading_deg: {result.heading_deg:+.1f}",
        f"lane_conf: {result.lane_confidence:.2f}",
        f"road_conf: {result.road_confidence:.2f}",
        f"intersection: {result.intersection_likely}",
    ]
    panel_w = 240
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + 22 * len(status_lines) + 10), (0, 0, 0), -1)
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + 22 * len(status_lines) + 10), (80, 80, 80), 1)
    for i, text in enumerate(status_lines):
        cv2.putText(
            vis,
            text,
            (16, 30 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


# ---------------------------------- Main ---------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight road structure perception for single image")
    parser.add_argument("--weights", type=str, required=True, help="Path to custom YOLO weights")
    parser.add_argument("--source", type=str, required=True, help="Input image path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show", action="store_true", help="Display the output image")
    parser.add_argument("--save", type=str, default="", help="Optional output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = YoloObjectDetector(
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    estimator = RoadStructureEstimator()

    print(f"Loading image from {args.source}")
    frame = cv2.imread(args.source)
    if frame is None:
        raise RuntimeError(f"Failed to open source image: {args.source}")

    # Resize image if needed
    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

    # Process
    t0 = time.time()
    detections = detector.infer(frame)
    result = estimator.process(frame, detections)
    t_infer = time.time() - t0
    print(f"Inference complete in {t_infer:.3f}s")

    # --- Debug image output (중간 단계별 마스크/결과 저장) ---
    import os
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(args.source)), "debug")
    os.makedirs(debug_dir, exist_ok=True)

    cv2.imwrite(os.path.join(debug_dir, "debug_lane_mask.png"), result.lane_mask)
    cv2.imwrite(os.path.join(debug_dir, "debug_road_mask.png"), result.road_mask)

    # 중간 마스크 저장을 위해 estimator 내부의 디버그 데이터 사용
    if hasattr(estimator, "_debug_masks"):
        for name, mask in estimator._debug_masks.items():
            cv2.imwrite(os.path.join(debug_dir, f"debug_{name}.png"), mask)

    print(f"Debug images saved to {debug_dir}")

    # Draw
    vis = draw_result(frame, result)

    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"Result saved to {args.save}")

    if args.show:
        print("Press any key to close the window...")
        cv2.imshow("road_structure_assist_image", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
