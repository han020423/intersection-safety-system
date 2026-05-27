"""
perception.py — YOLO11n 객체 인식 + BiSeNetV2 도로 구조 세그멘테이션 래퍼
"""

import os
import sys
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

# ──────────────────────── 상수 ──────────────────────── #

# BiSeNetV2 세그멘테이션 클래스
SEG_CLASS_NAMES = [
    'Background',      # 0
    'White_Solid',      # 1
    'White_Dotted',     # 2
    'Yellow_Solid',     # 3
    'Yellow_Dotted',    # 4
    'Blue_Solid',       # 5
    'Blue_Dotted',      # 6
    'Crosswalk',        # 7
    'Stop_Line',        # 8
]
NUM_SEG_CLASSES = len(SEG_CLASS_NAMES)

# 세그멘테이션 컬러맵 (BGR)
SEG_COLORS = {
    0: (0, 0, 0),        # background
    1: (255, 255, 255),  # white solid
    2: (200, 200, 200),  # white dotted
    3: (0, 255, 255),    # yellow solid
    4: (0, 200, 200),    # yellow dotted
    5: (255, 100, 0),    # blue solid
    6: (200, 80, 0),     # blue dotted
    7: (0, 180, 255),    # crosswalk
    8: (0, 0, 255),      # stop line
}

# YOLO 클래스 컬러 (BGR)
YOLO_COLORS = {
    'pedestrian':               (0, 200, 255),
    'vehicle':                  (255, 80, 80),
    'traffic_light_vehicle':    (80, 255, 80),
    'traffic_light_pedestrian': (80, 255, 80),
    'crosswalk':                (255, 255, 0),
    'left_turn_sign':           (255, 0, 255),
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────── 데이터 구조 ──────────────────────── #

@dataclass
class Detection:
    """YOLO 검출 결과 하나."""
    cls_id: int
    cls_name: str
    conf: float
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    distance_m: Optional[float] = None


# ──────────────────────── YOLO 래퍼 ──────────────────────── #

class YoloDetector:
    """Ultralytics YOLO11n 모델 래퍼. 프레임 스킵(interval) 지원."""

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.35,
                 iou: float = 0.45, device: str = 'cpu', interval: int = 1,
                 estimate_distance: bool = True,
                 camera_hfov_deg: float = 70.0,
                 distance_focal_px: float = 0.0,
                 vehicle_real_width_m: float = 1.8,
                 vehicle_real_height_m: float = 1.5,
                 distance_scale: float = 0.8):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.interval = max(1, interval)
        self.estimate_distance = estimate_distance
        self.camera_hfov_deg = camera_hfov_deg
        self.distance_focal_px = distance_focal_px
        self.vehicle_real_width_m = vehicle_real_width_m
        self.vehicle_real_height_m = vehicle_real_height_m
        self.distance_scale = distance_scale
        self._frame_idx = 0
        self._cache: List[Detection] = []
        self.class_names = self._resolve_names()
        print(f"[YOLO] 로드 완료  classes={list(self.class_names.values())}")

    def _resolve_names(self):
        names = getattr(self.model.model, 'names', None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
        return {}

    def _focal_px(self, frame_w: int) -> Optional[float]:
        if self.distance_focal_px and self.distance_focal_px > 0:
            return float(self.distance_focal_px)
        hfov = max(1.0, min(179.0, float(self.camera_hfov_deg)))
        return (frame_w * 0.5) / math.tan(math.radians(hfov) * 0.5)

    def _estimate_distance_m(self, cls_name: str, box, frame_shape) -> Optional[float]:
        """단안 카메라 pinhole 모델 기반 차량 거리 추정."""
        name = cls_name.lower()
        if name not in ('vehicle', 'car', 'bus', 'truck', 'motorcycle'):
            return None

        x1, y1, x2, y2 = [float(v) for v in box]
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        frame_h, frame_w = frame_shape[:2]
        focal_px = self._focal_px(frame_w)
        if focal_px is None:
            return None

        estimates = []
        if self.vehicle_real_width_m > 0 and bbox_w >= 4:
            estimates.append(self.vehicle_real_width_m * focal_px / bbox_w)
        if self.vehicle_real_height_m > 0 and bbox_h >= 4:
            estimates.append(self.vehicle_real_height_m * focal_px / bbox_h)
        estimates = [d for d in estimates if 0.5 <= d <= 200.0]
        if not estimates:
            return None

        # bbox width가 보통 차량 거리에는 더 안정적이므로 우선 반영한다.
        if len(estimates) == 1:
            distance = float(estimates[0])
        else:
            distance = float(0.7 * estimates[0] + 0.3 * estimates[1])
        return max(0.1, distance * max(0.1, float(self.distance_scale)))

    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """프레임에서 객체를 검출한다. (detections, elapsed_ms) 반환."""
        self._frame_idx += 1
        # 프레임 스킵: 이전 캐시 재사용
        if self.interval > 1 and self._frame_idx % self.interval != 0 and self._cache:
            return self._cache, 0.0

        t0 = time.perf_counter()
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf,
                             iou=self.iou, device=self.device, verbose=False)[0]
        detections = []
        boxes = getattr(results, 'boxes', None)
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(np.int32)
            for box, score, cid in zip(xyxy, confs, clss):
                cls_name = self.class_names.get(int(cid), str(cid))
                distance_m = (
                    self._estimate_distance_m(cls_name, box, frame.shape)
                    if self.estimate_distance else None
                )
                detections.append(Detection(
                    cls_id=int(cid),
                    cls_name=cls_name,
                    conf=float(score),
                    box=tuple(box.tolist()),
                    distance_m=distance_m,
                ))
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._cache = detections
        return detections, elapsed


# ──────────────────── BiSeNetV2 래퍼 ──────────────────── #

class RoadSegmentor:
    """BiSeNetV2 도로 구조 세그멘테이션. 별도 입력 해상도 지원."""

    def __init__(self, weights: str, n_classes: int = NUM_SEG_CLASSES,
                 input_size: Tuple[int, int] = (352, 640), device: str = 'cpu'):
        # bisenetv2.py를 같은 디렉터리에서 import
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from bisenetv2 import BiSeNetV2

        self.device_str = device
        self.torch_device = torch.device(device)
        self.input_size = input_size  # (H, W)
        self.model = BiSeNetV2(n_classes=n_classes, aux_mode='eval')
        state = torch.load(weights, map_location=self.torch_device, weights_only=False)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(self.torch_device).eval()
        print(f"[BiSeNetV2] 로드 완료  classes={n_classes}  input={input_size}")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """세그멘테이션 마스크와 소요시간(ms)을 반환."""
        h_orig, w_orig = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        resized = cv2.resize(frame_bgr, (self.input_size[1], self.input_size[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.torch_device)
        logits = self.model(tensor)[0]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if (h_orig, w_orig) != self.input_size:
            pred = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return pred, elapsed

    def build_color_overlay(self, seg_mask: np.ndarray) -> np.ndarray:
        """세그멘테이션 마스크를 BGR 컬러 오버레이로 변환."""
        h, w = seg_mask.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in SEG_COLORS.items():
            if cls_id == 0:
                continue
            overlay[seg_mask == cls_id] = color
        return overlay
