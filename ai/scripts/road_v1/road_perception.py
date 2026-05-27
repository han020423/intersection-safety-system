#!/usr/bin/env python3
"""
Road Perception System v1 — YOLO 객체인식 + BiSeNetV2 도로 구조 분할

두 모델을 순차적으로 구동하여 하나의 화면에 객체(차량, 보행자, 신호등 등)와
도로 구조(차선, 횡단보도, 정지선)를 동시에 인식합니다.

파이프라인:
  1단계) YOLO  →  bounding-box 기반 객체 검출
  2단계) BiSeNetV2  →  pixel-level 도로 세그멘테이션

사용법:
  # 웹캠 실시간
  python road_perception.py --source 0 --show

  # 동영상 파일
  python road_perception.py --source video.mp4 --show --save output.mp4

  # 단일 이미지
  python road_perception.py --source test.jpg --show --save result.jpg
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

# BiSeNetV2 세그멘테이션 클래스 (9개)
# 0: Background, 1: White_Solid, 2: White_Dotted,
# 3: Yellow_Solid, 4: Yellow_Dotted,
# 5: Blue_Solid, 6: Blue_Dotted,
# 7: Crosswalk, 8: Stop_Line
SEG_CLASS_NAMES = [
    'Background', 'White_Solid', 'White_Dotted',
    'Yellow_Solid', 'Yellow_Dotted',
    'Blue_Solid', 'Blue_Dotted',
    'Crosswalk', 'Stop_Line',
]

NUM_SEG_CLASSES = len(SEG_CLASS_NAMES)

# 세그멘테이션 클래스별 시각화 색상 (BGR)
SEG_COLORS = {
    0: (0, 0, 0),          # Background — 투명
    1: (255, 255, 255),    # White_Solid — 흰색
    2: (200, 200, 200),    # White_Dotted — 밝은 회색
    3: (0, 255, 255),      # Yellow_Solid — 노란색
    4: (0, 200, 200),      # Yellow_Dotted — 어두운 노란색
    5: (255, 100, 0),      # Blue_Solid — 파란색
    6: (200, 80, 0),       # Blue_Dotted — 어두운 파란색
    7: (0, 180, 255),      # Crosswalk — 주황색
    8: (0, 0, 255),        # Stop_Line — 빨간색
}

# YOLO 객체 클래스별 bbox 색상 (BGR)
YOLO_COLORS = {
    'pedestrian':              (0, 200, 255),
    'vehicle':                 (255, 80, 80),
    'traffic_light_vehicle':   (80, 255, 80),
    'traffic_light_pedestrian':(80, 255, 80),
    'crosswalk':               (255, 255, 0),
    'left_turn_sign':          (255, 0, 255),
}

# ImageNet Normalization (BiSeNetV2 학습에 사용된 값)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────── 데이터 구조 ──────────────────────── #


@dataclass
class Detection:
    """YOLO가 검출한 하나의 객체."""
    cls_id: int
    cls_name: str
    conf: float
    box: Tuple[int, int, int, int]   # (x1, y1, x2, y2)


@dataclass
class PerceptionResult:
    """한 프레임의 전체 인식 결과."""
    detections: List[Detection]
    seg_mask: np.ndarray             # (H, W) class-id map
    seg_overlay: np.ndarray          # (H, W, 3) 컬러 오버레이
    yolo_ms: float                   # YOLO 추론 시간 (ms)
    seg_ms: float                    # BiSeNetV2 추론 시간 (ms)


# ──────────────────────── YOLO 래퍼 ──────────────────────── #


class YoloDetector:
    """Ultralytics YOLO 객체 검출기."""

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = 'cpu',
        interval: int = 1,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics가 설치되지 않았습니다. pip install ultralytics"
            ) from exc

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.interval = max(1, interval)
        self._frame_idx = 0
        self._cache: List[Detection] = []
        self.class_names = self._resolve_names()
        print(f"[YOLO] 모델 로드 완료  classes={list(self.class_names.values())}")

    def _resolve_names(self) -> Dict[int, str]:
        names = getattr(self.model.model, 'names', None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
        return {}

    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """프레임에서 객체를 검출합니다. (검출 결과, 소요 시간 ms)"""
        self._frame_idx += 1
        if self.interval > 1 and self._frame_idx % self.interval != 0 and self._cache:
            return self._cache, 0.0

        t0 = time.perf_counter()
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        boxes = getattr(results, 'boxes', None)
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy().astype(np.int32)
            for box, score, cid in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box.tolist()
                detections.append(Detection(
                    cls_id=int(cid),
                    cls_name=self.class_names.get(int(cid), str(cid)),
                    conf=float(score),
                    box=(x1, y1, x2, y2),
                ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        self._cache = detections
        return detections, elapsed


# ──────────────────── BiSeNetV2 세그멘테이터 ──────────────────── #


class RoadSegmentor:
    """BiSeNetV2 기반 도로 구조 세그멘테이션."""

    def __init__(
        self,
        weights: str,
        n_classes: int = NUM_SEG_CLASSES,
        input_size: Tuple[int, int] = (352, 640),   # (H, W) — 학습 해상도
        device: str = 'cpu',
    ) -> None:
        # lazy import so the file can live alongside bisenetv2.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from bisenetv2 import BiSeNetV2

        self.device = torch.device(device)
        self.input_size = input_size
        self.n_classes = n_classes

        self.model = BiSeNetV2(n_classes=n_classes, aux_mode='eval')
        state = torch.load(weights, map_location=self.device)
        # torch.compile 으로 저장된 weight는 key에 '_orig_mod.' 접두사가 붙음
        cleaned = {}
        for k, v in state.items():
            new_key = k.replace('_orig_mod.', '')
            cleaned[new_key] = v
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"[BiSeNetV2] 모델 로드 완료  classes={n_classes}  input={input_size}")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        BGR 프레임 → (H_orig, W_orig) class-id mask, 소요 시간 ms
        """
        h_orig, w_orig = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        # 1. 전처리: resize → RGB → normalize → tensor
        resized = cv2.resize(frame_bgr, (self.input_size[1], self.input_size[0]),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # 2. 추론
        logits = self.model(tensor)[0]                  # (1, C, H, W)
        pred = logits.argmax(dim=1).squeeze(0).cpu()    # (H, W) int

        # 3. 원본 해상도로 리사이즈
        pred_np = pred.numpy().astype(np.uint8)
        if (h_orig, w_orig) != self.input_size:
            pred_np = cv2.resize(pred_np, (w_orig, h_orig),
                                 interpolation=cv2.INTER_NEAREST)

        elapsed = (time.perf_counter() - t0) * 1000.0
        return pred_np, elapsed


# ──────────────────── 시각화 유틸 ──────────────────── #


def build_seg_overlay(seg_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """class-id mask → (H, W, 3) BGR 컬러 오버레이 생성."""
    h, w = seg_mask.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in SEG_COLORS.items():
        if cls_id == 0:
            continue   # background는 투명
        overlay[seg_mask == cls_id] = color
    return overlay


def draw_perception(frame: np.ndarray, result: PerceptionResult,
                     show_legend: bool = True) -> np.ndarray:
    """최종 시각화 이미지를 생성합니다."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── 1. 세그멘테이션 오버레이 ──
    mask_nonzero = result.seg_mask > 0
    if mask_nonzero.any():
        vis[mask_nonzero] = cv2.addWeighted(
            vis[mask_nonzero], 0.55,
            result.seg_overlay[mask_nonzero], 0.45, 0,
        )

    # ── 2. YOLO 바운딩 박스 ──
    for det in result.detections:
        x1, y1, x2, y2 = det.box
        color = YOLO_COLORS.get(det.cls_name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.conf:.2f}"
        # 배경 사각형
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ── 3. 상태 패널 (좌상단) ──
    total_ms = result.yolo_ms + result.seg_ms
    fps_est = 1000.0 / max(1e-3, total_ms)
    n_det = len(result.detections)

    # 세그멘테이션에서 검출된 도로 구조 요소 카운트
    seg_counts = {}
    for cid in range(1, NUM_SEG_CLASSES):
        cnt = int(np.count_nonzero(result.seg_mask == cid))
        if cnt > 0:
            seg_counts[SEG_CLASS_NAMES[cid]] = cnt

    panel_lines = [
        f"FPS: {fps_est:.1f}  ({total_ms:.0f}ms)",
        f"YOLO: {result.yolo_ms:.1f}ms  |  Seg: {result.seg_ms:.1f}ms",
        f"Objects: {n_det}",
        f"Road features: {len(seg_counts)}",
    ]

    panel_h = 22 * len(panel_lines) + 14
    panel_w = 300
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + panel_h), (100, 100, 100), 1)
    for i, text in enumerate(panel_lines):
        cv2.putText(vis, text, (16, 28 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 4. 범례 (우하단) ──
    if show_legend:
        legend_items = []
        # YOLO 검출 클래스
        seen_yolo = set()
        for det in result.detections:
            if det.cls_name not in seen_yolo:
                seen_yolo.add(det.cls_name)
                c = YOLO_COLORS.get(det.cls_name, (255, 255, 255))
                legend_items.append((det.cls_name, c))

        # 세그멘테이션 활성 클래스
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Road Perception: YOLO 객체인식 + BiSeNetV2 도로 세그멘테이션"
    )
    p.add_argument('--source', type=str, default='0',
                   help='카메라 인덱스(0,1,...) 또는 이미지/동영상 파일 경로')
    p.add_argument('--yolo-weights', type=str, default=None,
                   help='YOLO 가중치 경로 (기본: 같은 폴더의 yolo.pt)')
    p.add_argument('--seg-weights', type=str, default=None,
                   help='BiSeNetV2 가중치 경로 (기본: bisenet_custom/weights/best.pth)')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=360)
    p.add_argument('--imgsz', type=int, default=640, help='YOLO 추론 해상도')
    p.add_argument('--conf', type=float, default=0.35, help='YOLO confidence threshold')
    p.add_argument('--iou', type=float, default=0.45, help='YOLO NMS IoU threshold')
    p.add_argument('--device', type=str, default='cpu',
                   help='cuda 또는 cpu')
    p.add_argument('--yolo-interval', type=int, default=1,
                   help='YOLO를 N 프레임마다 실행 (속도 최적화)')
    p.add_argument('--show', action='store_true', help='결과를 화면에 표시')
    p.add_argument('--save', type=str, default='', help='결과 저장 경로 (이미지 또는 동영상)')
    p.add_argument('--no-legend', action='store_true', help='범례를 비활성화')
    return p.parse_args()


def is_image_file(path: str) -> bool:
    return path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── 가중치 경로 결정 ──
    yolo_w = args.yolo_weights or os.path.join(base_dir, 'yolo.pt')
    seg_w  = args.seg_weights  or os.path.join(base_dir, 'bisenet_custom', 'weights', 'best.pth')

    if not os.path.isfile(yolo_w):
        raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없습니다: {yolo_w}")
    if not os.path.isfile(seg_w):
        raise FileNotFoundError(f"BiSeNetV2 가중치 파일을 찾을 수 없습니다: {seg_w}")

    # ── 모델 초기화 ──
    print("=" * 60)
    print("  Road Perception System v1")
    print("  YOLO (객체인식) + BiSeNetV2 (도로 세그멘테이션)")
    print("=" * 60)

    yolo = YoloDetector(
        weights=yolo_w,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        interval=args.yolo_interval,
    )

    seg = RoadSegmentor(
        weights=seg_w,
        device=args.device,
    )

    # ── 입력 소스 판별 ──
    source_str = args.source
    if source_str.isdigit():
        source = int(source_str)
        mode = 'camera'
    elif is_image_file(source_str):
        mode = 'image'
        source = source_str
    else:
        mode = 'video'
        source = source_str

    print(f"\n[입력] mode={mode}  source={source}")
    print(f"[해상도] {args.width}x{args.height}  |  YOLO imgsz={args.imgsz}")
    print(f"[장치] {args.device}\n")

    # ── 단일 이미지 모드 ──
    if mode == 'image':
        img_path = os.path.abspath(source)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {source}")
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

        detections, yolo_ms = yolo.infer(frame)
        seg_mask, seg_ms = seg.infer(frame)
        seg_overlay = build_seg_overlay(seg_mask)

        result = PerceptionResult(
            detections=detections,
            seg_mask=seg_mask,
            seg_overlay=seg_overlay,
            yolo_ms=yolo_ms,
            seg_ms=seg_ms,
        )
        vis = draw_perception(frame, result, show_legend=not args.no_legend)

        total = yolo_ms + seg_ms
        print(f"✅ 추론 완료  YOLO={yolo_ms:.1f}ms  Seg={seg_ms:.1f}ms  Total={total:.1f}ms")
        print(f"   객체 {len(detections)}개 검출")

        if args.save:
            cv2.imwrite(args.save, vis)
            print(f"   결과 저장: {args.save}")
        if args.show:
            cv2.imshow('Road Perception', vis)
            print("   아무 키를 눌러 창을 닫으세요...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ── 비디오 / 카메라 모드 ──
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"소스를 열 수 없습니다: {source}")

    # 원본 영상의 FPS를 읽어서 출력에 동일하게 적용
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or src_fps > 120:
        src_fps = 30.0  # 카메라 등 FPS를 못 읽는 경우 기본값
    print(f"[원본 FPS] {src_fps:.1f}")

    writer = None
    frame_count = 0
    fps_smooth = 0.0
    alpha = 0.15

    print("실행 중... (q 또는 ESC로 종료)")
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            # 1단계: YOLO 객체 검출
            detections, yolo_ms = yolo.infer(frame)

            # 2단계: BiSeNetV2 도로 세그멘테이션
            seg_mask, seg_ms = seg.infer(frame)
            seg_overlay = build_seg_overlay(seg_mask)

            result = PerceptionResult(
                detections=detections,
                seg_mask=seg_mask,
                seg_overlay=seg_overlay,
                yolo_ms=yolo_ms,
                seg_ms=seg_ms,
            )

            vis = draw_perception(frame, result, show_legend=not args.no_legend)

            # FPS 계산
            total_s = time.perf_counter() - t0
            cur_fps = 1.0 / max(1e-6, total_s)
            fps_smooth = alpha * cur_fps + (1 - alpha) * fps_smooth if fps_smooth > 0 else cur_fps

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  frame={frame_count}  FPS={fps_smooth:.1f}  "
                      f"objs={len(detections)}  YOLO={yolo_ms:.0f}ms  Seg={seg_ms:.0f}ms")

            # 저장
            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.save, fourcc, src_fps,
                                             (args.width, args.height))
                writer.write(vis)

            # 표시
            if args.show:
                cv2.imshow('Road Perception', vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
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
