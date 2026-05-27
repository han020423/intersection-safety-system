"""
road_v4 경량 BiSeNetV2 세그멘테이션 로더.

이 파일은 라즈베리파이 적용을 고려한 추론용 래퍼다.
ResNet-18 같은 외부 대형 백본을 붙이지 않고, 기존 BiSeNetV2 구조와
BiSeNetV2 backbone_v2 기반 학습 결과를 그대로 사용한다.

클래스 번호:
  0 background
  1 lane_white
  2 lane_yellow
  3 lane_blue
  4 crosswalk
  5 stop_line
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch


V2_DIR = Path(__file__).resolve().parent
PARENT_DIR = V2_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from bisenetv2 import BiSeNetV2  # noqa: E402


# road_v4 학습 데이터셋과 반드시 같은 순서를 유지해야 한다.
SEG_CLASS_NAMES = [
    "background",
    "lane_white",
    "lane_yellow",
    "lane_blue",
    "crosswalk",
    "stop_line",
]
NUM_SEG_CLASSES = len(SEG_CLASS_NAMES)

SEG_COLORS = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (0, 220, 255),
    3: (255, 120, 20),
    4: (0, 255, 0),
    5: (0, 0, 255),
}

# 학습 때 사용한 ImageNet 정규화 값이다. 입력 전처리가 학습과 달라지면
# 마스크 품질이 크게 흔들리므로 추론에서도 같은 mean/std를 사용한다.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _extract_state_dict(checkpoint):
    """학습용/추론용 checkpoint 형식 차이를 흡수해서 state_dict만 꺼낸다."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


class RoadV4Segmentor:
    def __init__(self, weights: str, input_size: Tuple[int, int] = (352, 640), device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.input_size = input_size
        # aux_mode='eval'이면 BiSeNetV2가 main logits 하나만 반환한다.
        self.model = BiSeNetV2(n_classes=NUM_SEG_CLASSES, aux_mode="eval")

        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get("input_size"):
            saved_h, saved_w = checkpoint["input_size"]
            # 사용자가 별도 입력 크기를 지정하지 않았으면 checkpoint의 학습 해상도를 따른다.
            if input_size == (352, 640):
                self.input_size = (int(saved_h), int(saved_w))

        state = _extract_state_dict(checkpoint)
        # torch.compile 또는 DataParallel을 거친 checkpoint도 읽을 수 있게 prefix를 제거한다.
        state = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state.items()
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[RoadV4Segmentor] missing keys: {len(missing)}")
        if unexpected:
            print(f"[RoadV4Segmentor] unexpected keys: {len(unexpected)}")

        self.model.to(self.device).eval()
        print(f"[RoadV4Segmentor] lightweight BiSeNetV2 loaded {Path(weights).name}")
        print(f"[RoadV4Segmentor] input={self.input_size} device={device}")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray):
        h_orig, w_orig = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        # 모델 입력 크기로 축소한 뒤 RGB 변환과 정규화를 수행한다.
        resized = cv2.resize(frame_bgr, (self.input_size[1], self.input_size[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # argmax 결과는 class id가 들어 있는 단일 채널 마스크다.
        logits = self.model(tensor)[0]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # 후처리는 원본 프레임 좌표계를 기준으로 하므로 원본 크기로 되돌린다.
        if (h_orig, w_orig) != self.input_size:
            pred = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return pred, elapsed_ms

    def build_color_overlay(self, seg_mask: np.ndarray) -> np.ndarray:
        """디버깅/시각화용 색상 마스크를 만든다. 판단 자체에는 class id 마스크를 사용한다."""
        overlay = np.zeros((*seg_mask.shape[:2], 3), dtype=np.uint8)
        for cls_id, color in SEG_COLORS.items():
            if cls_id == 0:
                continue
            overlay[seg_mask == cls_id] = color
        return overlay
