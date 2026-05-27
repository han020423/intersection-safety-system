#!/usr/bin/env python3
"""
Road Perception System v2 — YOLO + BiSeNetV2 + 주행 가능 영역 추정

v1 대비 추가 기능:
  3단계) 차선 사이 영역을 주행 가능 도로로 추정
         → YOLO 검출 객체(차량, 보행자 등) 영역을 제외
         → 최종 "Drivable Area"를 초록색으로 색칠

사용법:
  python road_perception_v2.py --source video.mp4 --show --save output.mp4
  python road_perception_v2.py --source test.jpg --show
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────── 설정 상수 ──────────────────────────── #

SEG_CLASS_NAMES = [
    'Background', 'White_Solid', 'White_Dotted',
    'Yellow_Solid', 'Yellow_Dotted',
    'Blue_Solid', 'Blue_Dotted',
    'Crosswalk', 'Stop_Line',
]

NUM_SEG_CLASSES = len(SEG_CLASS_NAMES)

SEG_COLORS = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (200, 200, 200),
    3: (0, 255, 255),
    4: (0, 200, 200),
    5: (255, 100, 0),
    6: (200, 80, 0),
    7: (0, 180, 255),
    8: (0, 0, 255),
}

YOLO_COLORS = {
    'pedestrian':              (0, 200, 255),
    'vehicle':                 (255, 80, 80),
    'traffic_light_vehicle':   (80, 255, 80),
    'traffic_light_pedestrian':(80, 255, 80),
    'crosswalk':               (255, 255, 0),
    'left_turn_sign':          (255, 0, 255),
}

# 주행 가능 영역에서 제외할 YOLO 클래스 (장애물)
OBSTACLE_CLASSES = {'vehicle', 'pedestrian'}

# 주행 가능 영역 색상 (BGR) — 반투명 초록
DRIVABLE_COLOR = (0, 200, 0)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────── 데이터 구조 ──────────────────────── #


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box: Tuple[int, int, int, int]


@dataclass
class LaneInfo:
    """차선 위치 추정 결과."""
    current_lane: int       # 현재 차선 번호 (1-indexed, 0이면 판별 불가)
    total_lanes: int        # 감지된 총 차선 수
    description: str        # 표시용 문자열 (예: "2/3 차선")
    lane_centers: List[int] # 각 차선 마킹의 x좌표 중심


@dataclass
class PerceptionResult:
    detections: List[Detection]
    seg_mask: np.ndarray
    seg_overlay: np.ndarray
    drivable_mask: np.ndarray        # 주행 가능 영역 마스크
    lane_info: LaneInfo              # ★ 차선 위치 정보
    yolo_ms: float
    seg_ms: float


# ──────────────────────── YOLO 래퍼 ──────────────────────── #


class YoloDetector:
    def __init__(self, weights, imgsz=640, conf=0.35, iou=0.45,
                 device='cpu', interval=1):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("pip install ultralytics") from exc
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device = device
        self.interval = max(1, interval)
        self._frame_idx = 0
        self._cache: List[Detection] = []
        self.class_names = self._resolve_names()
        print(f"[YOLO] 모델 로드 완료  classes={list(self.class_names.values())}")

    def _resolve_names(self):
        names = getattr(self.model.model, 'names', None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
        return {}

    def infer(self, frame):
        self._frame_idx += 1
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
                x1, y1, x2, y2 = box.tolist()
                detections.append(Detection(
                    cls_id=int(cid),
                    cls_name=self.class_names.get(int(cid), str(cid)),
                    conf=float(score), box=(x1, y1, x2, y2),
                ))
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._cache = detections
        return detections, elapsed


# ──────────────────── BiSeNetV2 세그멘테이터 ──────────────────── #


class RoadSegmentor:
    def __init__(self, weights, n_classes=NUM_SEG_CLASSES,
                 input_size=(352, 640), device='cpu'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from bisenetv2 import BiSeNetV2

        self.device = torch.device(device)
        self.input_size = input_size
        self.model = BiSeNetV2(n_classes=n_classes, aux_mode='eval')
        state = torch.load(weights, map_location=self.device)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(self.device).eval()
        print(f"[BiSeNetV2] 모델 로드 완료  classes={n_classes}  input={input_size}")

    @torch.no_grad()
    def infer(self, frame_bgr):
        h_orig, w_orig = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        resized = cv2.resize(frame_bgr, (self.input_size[1], self.input_size[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)[0]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if (h_orig, w_orig) != self.input_size:
            pred = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return pred, elapsed


# ──────────────────── ★ 주행 가능 영역 추정 ──────────────────── #


def estimate_drivable_area(seg_mask: np.ndarray,
                           detections: List[Detection]) -> np.ndarray:
    """
    차선 세그멘테이션 결과로부터 주행 가능 영역을 추정합니다.

    알고리즘:
      1. 각 행(row)에서 차선 픽셀(class 1~6)의 좌측 끝 ~ 우측 끝 사이를 도로로 판정
      2. 횡단보도(7), 정지선(8) 영역도 도로에 포함
      3. YOLO 검출 장애물(차량, 보행자) bbox 영역을 제외
      4. Morphology 정리로 깔끔하게 마무리
    """
    h, w = seg_mask.shape[:2]
    drivable = np.zeros((h, w), dtype=np.uint8)

    # 1. 차선 마스크 (class 1~6)
    lane_mask = (seg_mask >= 1) & (seg_mask <= 6)

    # 2. 행별로 차선 좌끝~우끝 사이를 도로로 채움
    for y in range(h):
        lane_xs = np.where(lane_mask[y])[0]
        if len(lane_xs) >= 2:
            left_x = lane_xs[0]
            right_x = lane_xs[-1]
            drivable[y, left_x:right_x + 1] = 255

    # 3. 횡단보도(7)와 정지선(8) 영역도 도로에 포함
    road_features = (seg_mask == 7) | (seg_mask == 8)
    drivable[road_features] = 255

    # 4. Morphology로 구멍 채우기 및 노이즈 제거
    kernel = np.ones((7, 7), np.uint8)
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, kernel, iterations=2)
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. YOLO 장애물(차량, 보행자) 영역 제외
    for det in detections:
        if det.cls_name in OBSTACLE_CLASSES:
            x1, y1, x2, y2 = det.box
            # bbox에 약간의 패딩 추가
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)
            drivable[y1:y2 + 1, x1:x2 + 1] = 0

    return drivable


# ──────────────────── ★ 차선 위치 추정 ──────────────────── #


def estimate_lane_position(seg_mask: np.ndarray, frame_width: int) -> LaneInfo:
    """
    현재 차량이 몇 차선에 있는지 추정합니다.

    알고리즘:
      1. 화면 하단(75~90%) 영역에서 차선 픽셀을 열(column) 방향으로 집계
      2. 밀도가 높은 x좌표들을 클러스터링하여 개별 차선 마킹으로 그룹화
      3. 차량 중심(화면 중앙)이 몇 번째 차선 마킹 사이에 있는지 판별
    """
    h, w = seg_mask.shape[:2]
    no_info = LaneInfo(0, 0, "N/A", [])

    # 1. 하단 스캔 영역에서 차선(class 1~6) 픽셀의 열별 합산
    scan_y_top = int(h * 0.75)
    scan_y_bot = int(h * 0.90)
    lane_mask = (seg_mask >= 1) & (seg_mask <= 6)
    scan_region = lane_mask[scan_y_top:scan_y_bot, :]
    col_sum = np.sum(scan_region, axis=0)  # 각 x좌표의 차선 픽셀 수

    # 2. 임계값 이상인 x좌표만 차선 후보로 추출
    threshold = (scan_y_bot - scan_y_top) * 0.08
    lane_cols = np.where(col_sum > threshold)[0]

    if len(lane_cols) < 2:
        return no_info

    # 3. 가까운 x좌표를 하나의 차선 마킹으로 클러스터링
    #    (차선 마킹 두께 때문에 여러 x좌표가 연속)
    clusters: List[List[int]] = []
    current_cluster = [int(lane_cols[0])]
    for i in range(1, len(lane_cols)):
        if lane_cols[i] - lane_cols[i - 1] <= 20:  # 20px 이내면 같은 마킹
            current_cluster.append(int(lane_cols[i]))
        else:
            clusters.append(current_cluster)
            current_cluster = [int(lane_cols[i])]
    clusters.append(current_cluster)

    # 너무 작은 클러스터(노이즈) 제거
    clusters = [c for c in clusters if len(c) >= 3]

    if len(clusters) < 2:
        return no_info

    # 4. 각 클러스터의 중심 x좌표
    lane_centers = [int(np.mean(c)) for c in clusters]
    lane_centers.sort()

    # 5. 차선 수 = 마킹 사이의 공간 수
    total_lanes = len(lane_centers) - 1
    if total_lanes < 1:
        return no_info

    # 6. 차량 중심 (화면 중앙 하단) 이 몇 번째 갭에 위치하는지 판별
    ego_x = frame_width // 2
    current_lane = 0

    for i in range(len(lane_centers) - 1):
        if lane_centers[i] <= ego_x <= lane_centers[i + 1]:
            current_lane = i + 1  # 1-indexed
            break

    if current_lane == 0:
        if ego_x < lane_centers[0]:
            return LaneInfo(0, total_lanes, "Outside(L)", lane_centers)
        else:
            return LaneInfo(0, total_lanes, "Outside(R)", lane_centers)

    desc = f"Lane {current_lane}/{total_lanes}"
    return LaneInfo(current_lane, total_lanes, desc, lane_centers)


# ──────────────────── 시각화 유틸 ──────────────────── #


def build_seg_overlay(seg_mask):
    h, w = seg_mask.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in SEG_COLORS.items():
        if cls_id == 0:
            continue
        overlay[seg_mask == cls_id] = color
    return overlay


def draw_perception(frame: np.ndarray, result: PerceptionResult,
                     show_legend: bool = True) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── 1. 주행 가능 영역 (초록 반투명) ──
    drivable_area = result.drivable_mask > 0
    if drivable_area.any():
        green_overlay = np.zeros_like(vis)
        green_overlay[drivable_area] = DRIVABLE_COLOR
        vis[drivable_area] = cv2.addWeighted(
            vis[drivable_area], 0.65,
            green_overlay[drivable_area], 0.35, 0,
        )

    # ── 1.5 차선 경계선 표시 ──  ★ 차선 위치 추정 시각화
    li = result.lane_info
    if li.lane_centers:
        for i, cx in enumerate(li.lane_centers):
            # 차선 마킹 위치에 세로 점선 표시
            color_line = (0, 255, 255)  # 노란색
            for yy in range(int(h * 0.50), h, 8):
                cv2.line(vis, (cx, yy), (cx, min(yy + 4, h - 1)), color_line, 1)

        # 현재 차량 위치 & 차선 번호 하단 중앙에 큰 글씨로 표시
        if li.current_lane > 0:
            lane_text = f"Lane {li.current_lane}/{li.total_lanes}"
            (tw2, th2), _ = cv2.getTextSize(lane_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            tx = (w - tw2) // 2
            ty = h - 20
            cv2.rectangle(vis, (tx - 6, ty - th2 - 6), (tx + tw2 + 6, ty + 6), (0, 0, 0), -1)
            cv2.putText(vis, lane_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # 차량 중심 표시 (화면 하단 중앙)
        ego_x = w // 2
        cv2.circle(vis, (ego_x, h - 40), 6, (0, 255, 0), -1)
        cv2.line(vis, (ego_x, h - 46), (ego_x, h - 34), (0, 255, 0), 2)

    # ── 2. 세그멘테이션 오버레이 (차선, 횡단보도 등) ──
    mask_nonzero = result.seg_mask > 0
    if mask_nonzero.any():
        vis[mask_nonzero] = cv2.addWeighted(
            vis[mask_nonzero], 0.50,
            result.seg_overlay[mask_nonzero], 0.50, 0,
        )

    # ── 3. YOLO 바운딩 박스 ──
    for det in result.detections:
        x1, y1, x2, y2 = det.box
        color = YOLO_COLORS.get(det.cls_name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ── 4. 상태 패널 ──
    total_ms = result.yolo_ms + result.seg_ms
    fps_est = 1000.0 / max(1e-3, total_ms)
    n_det = len(result.detections)

    # 주행 가능 영역 비율
    drivable_pix = int(np.count_nonzero(result.drivable_mask))
    drivable_pct = drivable_pix / (h * w) * 100.0

    # 장애물 수
    n_obstacles = sum(1 for d in result.detections if d.cls_name in OBSTACLE_CLASSES)

    panel_lines = [
        f"FPS: {fps_est:.1f}  ({total_ms:.0f}ms)",
        f"YOLO: {result.yolo_ms:.1f}ms  |  Seg: {result.seg_ms:.1f}ms",
        f"Objects: {n_det}  |  Obstacles: {n_obstacles}",
        f"Drivable: {drivable_pct:.1f}%",
        f"Lane: {result.lane_info.description}",
    ]

    panel_h = 22 * len(panel_lines) + 14
    panel_w = 300
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + panel_h), (100, 100, 100), 1)
    for i, text in enumerate(panel_lines):
        cv2.putText(vis, text, (16, 28 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 5. 범례 ──
    if show_legend:
        legend_items = [('Drivable Area', DRIVABLE_COLOR)]

        seen_yolo = set()
        for det in result.detections:
            if det.cls_name not in seen_yolo:
                seen_yolo.add(det.cls_name)
                legend_items.append((det.cls_name, YOLO_COLORS.get(det.cls_name, (255,255,255))))

        for cid in range(1, NUM_SEG_CLASSES):
            if int(np.count_nonzero(result.seg_mask == cid)) > 100:
                legend_items.append((SEG_CLASS_NAMES[cid], SEG_COLORS[cid]))

        if legend_items:
            lh = 20 * len(legend_items) + 10
            lw = 180
            lx = w - lw - 10
            ly = h - lh - 10
            cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), (0, 0, 0), -1)
            cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), (80, 80, 80), 1)
            for j, (name, color) in enumerate(legend_items):
                yy = ly + 18 + 20 * j
                cv2.rectangle(vis, (lx + 6, yy - 10), (lx + 20, yy), color, -1)
                cv2.putText(vis, name, (lx + 26, yy - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ──────────────────── 메인 루프 ──────────────────── #


def parse_args():
    p = argparse.ArgumentParser(
        description="Road Perception v2: YOLO + BiSeNetV2 + 주행 가능 영역"
    )
    p.add_argument('--source', type=str, default='0')
    p.add_argument('--yolo-weights', type=str, default=None)
    p.add_argument('--seg-weights', type=str, default=None)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=360)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--yolo-interval', type=int, default=1)
    p.add_argument('--show', action='store_true')
    p.add_argument('--save', type=str, default='')
    p.add_argument('--no-legend', action='store_true')
    return p.parse_args()


def is_image_file(path):
    return path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))


def process_frame(frame, yolo, seg):
    """한 프레임을 처리하여 PerceptionResult를 반환."""
    h, w = frame.shape[:2]
    detections, yolo_ms = yolo.infer(frame)
    seg_mask, seg_ms = seg.infer(frame)
    seg_overlay = build_seg_overlay(seg_mask)
    drivable_mask = estimate_drivable_area(seg_mask, detections)
    lane_info = estimate_lane_position(seg_mask, w)

    return PerceptionResult(
        detections=detections,
        seg_mask=seg_mask,
        seg_overlay=seg_overlay,
        drivable_mask=drivable_mask,
        lane_info=lane_info,
        yolo_ms=yolo_ms,
        seg_ms=seg_ms,
    )


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    yolo_w = args.yolo_weights or os.path.join(base_dir, 'yolo.pt')
    seg_w  = args.seg_weights  or os.path.join(base_dir, 'bisenet_custom', 'weights', 'best.pth')

    if not os.path.isfile(yolo_w):
        raise FileNotFoundError(f"YOLO 가중치를 찾을 수 없습니다: {yolo_w}")
    if not os.path.isfile(seg_w):
        raise FileNotFoundError(f"BiSeNetV2 가중치를 찾을 수 없습니다: {seg_w}")

    print("=" * 60)
    print("  Road Perception System v2")
    print("  YOLO + BiSeNetV2 + 주행 가능 영역 추정")
    print("=" * 60)

    yolo = YoloDetector(
        weights=yolo_w, imgsz=args.imgsz, conf=args.conf,
        iou=args.iou, device=args.device, interval=args.yolo_interval,
    )
    seg = RoadSegmentor(weights=seg_w, device=args.device)

    source_str = args.source
    if source_str.isdigit():
        source, mode = int(source_str), 'camera'
    elif is_image_file(source_str):
        source, mode = source_str, 'image'
    else:
        source, mode = source_str, 'video'

    print(f"\n[입력] mode={mode}  source={source}")
    print(f"[해상도] {args.width}x{args.height}  |  device={args.device}\n")

    # ── 단일 이미지 ──
    if mode == 'image':
        img_path = os.path.abspath(source)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {source}")
        frame = cv2.resize(frame, (args.width, args.height))

        result = process_frame(frame, yolo, seg)
        vis = draw_perception(frame, result, show_legend=not args.no_legend)

        total = result.yolo_ms + result.seg_ms
        drivable_pct = np.count_nonzero(result.drivable_mask) / (args.width * args.height) * 100
        print(f"✅ 추론 완료  YOLO={result.yolo_ms:.1f}ms  Seg={result.seg_ms:.1f}ms  Total={total:.1f}ms")
        print(f"   객체 {len(result.detections)}개 검출")
        print(f"   주행 가능 영역: {drivable_pct:.1f}%")
        print(f"   차선 위치: {result.lane_info.description}")

        if args.save:
            cv2.imwrite(args.save, vis)
            print(f"   결과 저장: {args.save}")
        if args.show:
            cv2.imshow('Road Perception v2', vis)
            print("   아무 키를 눌러 창을 닫으세요...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ── 비디오 / 카메라 ──
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"소스를 열 수 없습니다: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or src_fps > 120:
        src_fps = 30.0
    print(f"[원본 FPS] {src_fps:.1f}")

    writer = None
    frame_count = 0
    fps_smooth = 0.0

    print("실행 중... (q 또는 ESC로 종료)")
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (args.width, args.height))

            result = process_frame(frame, yolo, seg)
            vis = draw_perception(frame, result, show_legend=not args.no_legend)

            total_s = time.perf_counter() - t0
            cur_fps = 1.0 / max(1e-6, total_s)
            fps_smooth = 0.15 * cur_fps + 0.85 * fps_smooth if fps_smooth > 0 else cur_fps

            frame_count += 1
            if frame_count % 30 == 0:
                drivable_pct = np.count_nonzero(result.drivable_mask) / (args.width * args.height) * 100
                print(f"  frame={frame_count}  FPS={fps_smooth:.1f}  "
                      f"objs={len(result.detections)}  drivable={drivable_pct:.1f}%  "
                      f"lane={result.lane_info.description}")

            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.save, fourcc, src_fps,
                                             (args.width, args.height))
                writer.write(vis)

            if args.show:
                cv2.imshow('Road Perception v2', vis)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n결과 동영상 저장 완료: {args.save}")
        if args.show:
            cv2.destroyAllWindows()
        print(f"총 {frame_count} 프레임 처리 완료.")


if __name__ == '__main__':
    main()
