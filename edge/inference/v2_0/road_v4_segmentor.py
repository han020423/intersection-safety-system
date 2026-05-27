"""
road_v4 ResNet-18 기반 임시 세그멘테이션 로더.

주의:
  이 파일은 경량 BiSeNetV2 재학습이 끝나기 전까지 기존 150MB급
  road_v4_best.pt를 사용하기 위한 v2_0 임시 버전이다.

  라즈베리파이 최종 후보는 edge/inference/v2의 경량 BiSeNetV2이며,
  이 v2_0은 노트북/GPU에서 새 road_v4 데이터셋 성능을 먼저 확인하기 위한 코드다.

클래스 번호:
  0 background
  1 lane_white
  2 lane_yellow
  3 lane_blue
  4 crosswalk
  5 stop_line
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels


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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DetailBranch(nn.Module):
    """BiSeNet 계열의 고해상도 detail branch."""

    def __init__(self) -> None:
        super().__init__()
        self.s1 = nn.Sequential(ConvBNReLU(3, 64, 3, 2, 1), ConvBNReLU(64, 64, 3, 1, 1))
        self.s2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
        )
        self.s3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.s3(self.s2(self.s1(x)))


class ResNet18ContextBranch(nn.Module):
    """road_v4_best.pt 학습 때 사용한 ResNet-18 context branch."""

    def __init__(self) -> None:
        super().__init__()
        # 추론에서는 checkpoint weight를 로드하므로 외부 다운로드가 필요 없다.
        resnet = tvmodels.resnet18(weights=None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.proj8 = ConvBNReLU(128, 128, 1, 1, 0)
        self.proj32 = ConvBNReLU(512, 128, 1, 1, 0)

    def forward(self, x: torch.Tensor):
        x4 = self.stem(x)
        x4 = self.layer1(x4)
        x8 = self.layer2(x4)
        x16 = self.layer3(x8)
        x32 = self.layer4(x16)
        context32 = self.proj32(x32)
        context8 = F.interpolate(context32, size=x8.shape[2:], mode="bilinear", align_corners=False)
        context8 = self.proj8(x8) + context8
        return context8, x8, x16, x32


class BGAFusion(nn.Module):
    """detail branch와 context branch를 합치는 BGA 형태의 fusion block."""

    def __init__(self) -> None:
        super().__init__()
        self.detail_gate = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, bias=False),
        )
        self.context_gate = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.fuse = ConvBNReLU(128, 128, 3, 1, 1)

    def forward(self, detail: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = F.interpolate(context, size=detail.shape[2:], mode="bilinear", align_corners=False)
        left = self.detail_gate(detail) * torch.sigmoid(context)
        right = self.context_gate(context) * torch.sigmoid(detail)
        return self.fuse(left + right)


class SegmentHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, num_classes: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, 3, 1, 1),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_ch, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        x = self.block(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class BiSeNetV2ResNet18(nn.Module):
    """기존 road_v4_best.pt와 같은 구조의 임시 모델."""

    def __init__(self, num_classes: int = NUM_SEG_CLASSES) -> None:
        super().__init__()
        self.detail = DetailBranch()
        self.context = ResNet18ContextBranch()
        self.fusion = BGAFusion()
        self.head = SegmentHead(128, 256, num_classes)
        # checkpoint에는 aux head도 들어 있으므로 구조를 맞춰 둔다.
        self.aux8 = SegmentHead(128, 128, num_classes)
        self.aux16 = SegmentHead(256, 128, num_classes)
        self.aux32 = SegmentHead(512, 128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        detail8 = self.detail(x)
        context8, _, _, _ = self.context(x)
        fused = self.fusion(detail8, context8)
        return self.head(fused, size)


def _extract_state_dict(checkpoint):
    """full checkpoint 또는 infer checkpoint에서 state_dict만 꺼낸다."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


class RoadV4Segmentor:
    def __init__(self, weights: str, input_size: Tuple[int, int] = (512, 928), device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.input_size = input_size
        self.model = BiSeNetV2ResNet18(NUM_SEG_CLASSES)

        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get("input_size"):
            saved_h, saved_w = checkpoint["input_size"]
            if input_size == (512, 928):
                self.input_size = (int(saved_h), int(saved_w))

        state = _extract_state_dict(checkpoint)
        state = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state.items()
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[RoadV4Segmentor v2_0] missing keys: {len(missing)}")
        if unexpected:
            print(f"[RoadV4Segmentor v2_0] unexpected keys: {len(unexpected)}")

        self.model.to(self.device).eval()
        print(f"[RoadV4Segmentor v2_0] ResNet-18 temporary model loaded {Path(weights).name}")
        print(f"[RoadV4Segmentor v2_0] input={self.input_size} device={device}")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray):
        h_orig, w_orig = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        resized = cv2.resize(frame_bgr, (self.input_size[1], self.input_size[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if (h_orig, w_orig) != self.input_size:
            pred = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return pred, elapsed_ms

    def build_color_overlay(self, seg_mask: np.ndarray) -> np.ndarray:
        overlay = np.zeros((*seg_mask.shape[:2], 3), dtype=np.uint8)
        for cls_id, color in SEG_COLORS.items():
            if cls_id == 0:
                continue
            overlay[seg_mask == cls_id] = color
        return overlay
