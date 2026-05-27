#!/usr/bin/env python
"""
Colab training script for road_v4 BiSeNet-style semantic segmentation.

Dataset:
  road_v4/
    train/images/*.jpg, train/masks/*.png
    val/images/*.jpg,   val/masks/*.png
    test/images/*.jpg,  test/masks/*.png
    metadata.json

Classes:
  0 background
  1 lane_white
  2 lane_yellow
  3 lane_blue
  4 crosswalk
  5 stop_line

The model keeps a BiSeNetV2-like two-branch design:
  - Detail branch: high-resolution spatial detail
  - Context branch: imported pretrained ResNet-18 backbone
  - BGA-style fusion + auxiliary supervision

Recommended Colab usage:
  1. Zip ai/scripts/road_v4 as road_v4.zip
  2. Upload it to one of:
     /content/drive/MyDrive/capstone/bisenet/road_v4.zip
     /content/drive/MyDrive/capstone/ufld/road_v4.zip
  3. Run:
     !python colab_train_bisenetv2_road_v4.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def _pip(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


try:
    import albumentations as A
except Exception:
    _pip("albumentations")

try:
    import cv2
except Exception:
    _pip("opencv-python")

try:
    from tqdm import tqdm
except Exception:
    _pip("tqdm")


import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tvmodels
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast

    AMP_API = "torch.amp"
except Exception:
    from torch.cuda.amp import GradScaler, autocast

    AMP_API = "torch.cuda.amp"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

try:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
except Exception:
    print("[INFO] Google Drive is not mounted. Running outside Colab or already mounted.")


DRIVE_CANDIDATES = [
    Path("/content/drive/MyDrive/capstone/bisenet"),
    Path("/content/drive/MyDrive/capstone/ufld"),
    Path("/content/drive/MyDrive/capstone"),
]
DATASET_ZIP_NAME = "road_v4.zip"
WORK_DIR = Path("/content/road_v4")
SAVE_DIR = Path("/content/drive/MyDrive/capstone/bisenet/road_v4_bisenetv2_resnet18_run")


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

SEED = 42
NUM_CLASSES = 6
CLASS_NAMES = ["background", "lane_white", "lane_yellow", "lane_blue", "crosswalk", "stop_line"]

INPUT_H = 512
INPUT_W = 928
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 80
PATIENCE = 18
MIN_DELTA = 1e-4
LR_BACKBONE = 1e-4
LR_HEAD = 8e-4
WEIGHT_DECAY = 1e-4
AMP = True

IGNORE_INDEX = 255
MIXED_DICE_WEIGHT = 0.7
AUX_WEIGHT = 0.35
GRAD_CLIP = 5.0

CLASS_COLORS_BGR = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (0, 220, 255),
    3: (255, 120, 20),
    4: (0, 255, 0),
    5: (0, 0, 255),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_scaler(device: torch.device) -> GradScaler:
    enabled = AMP and device.type == "cuda"
    if AMP_API == "torch.amp":
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


def amp_context(device: torch.device):
    enabled = AMP and device.type == "cuda"
    if AMP_API == "torch.amp":
        return autocast(device_type=device.type, enabled=enabled)
    return autocast(enabled=enabled)


def find_dataset_zip() -> Path:
    for root in DRIVE_CANDIDATES:
        candidate = root / DATASET_ZIP_NAME
        if candidate.exists():
            return candidate
    matches = list(Path("/content/drive/MyDrive").rglob(DATASET_ZIP_NAME))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"{DATASET_ZIP_NAME} not found in Google Drive. "
        "Upload road_v4.zip first."
    )


def prepare_dataset() -> Path:
    if WORK_DIR.exists() and (WORK_DIR / "metadata.json").exists():
        print(f"[DATA] using existing dataset: {WORK_DIR}")
        return WORK_DIR

    zip_path = find_dataset_zip()
    print(f"[DATA] unzip {zip_path} -> /content")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content")

    if (WORK_DIR / "metadata.json").exists():
        return WORK_DIR

    candidates = [
        path.parent
        for path in Path("/content").rglob("metadata.json")
        if path.parent.name == "road_v4"
    ]
    if candidates:
        return candidates[0]
    raise FileNotFoundError("road_v4/metadata.json not found after unzip.")


class RoadV4SegDataset(Dataset):
    def __init__(self, root: Path, split: str, transform: A.Compose | None = None) -> None:
        self.root = root
        self.split = split
        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        self.transform = transform
        self.samples = []

        image_paths = sorted(self.image_dir.glob("*.jpg"))
        for image_path in image_paths:
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                self.samples.append((image_path, mask_path))
        print(f"[DATA] {split}: {len(self.samples):,} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, mask_path = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        return image, mask, str(image_path)


def build_transforms() -> tuple[A.Compose, A.Compose]:
    train_tf = A.Compose(
        [
            A.Resize(INPUT_H, INPUT_W, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                    A.CLAHE(clip_limit=2.0, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 125), p=1.0),
                ],
                p=0.55,
            ),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.35),
            A.MotionBlur(blur_limit=5, p=0.12),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.12),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=4,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
                mask_value=0,
                p=0.45,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    eval_tf = A.Compose(
        [
            A.Resize(INPUT_H, INPUT_W, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return train_tf, eval_tf


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DetailBranch(nn.Module):
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
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            weights = tvmodels.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            try:
                resnet = tvmodels.resnet18(weights=weights)
            except Exception as exc:
                print(f"[WARN] pretrained ResNet-18 load failed: {exc}")
                print("[WARN] falling back to randomly initialized ResNet-18 backbone")
                resnet = tvmodels.resnet18(weights=None)
        except AttributeError:
            try:
                resnet = tvmodels.resnet18(pretrained=pretrained)
            except Exception as exc:
                print(f"[WARN] pretrained ResNet-18 load failed: {exc}")
                print("[WARN] falling back to randomly initialized ResNet-18 backbone")
                resnet = tvmodels.resnet18(pretrained=False)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.proj8 = ConvBNReLU(128, 128, 1, 1, 0)
        self.proj32 = ConvBNReLU(512, 128, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.block(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class BiSeNetV2ResNet18(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.detail = DetailBranch()
        self.context = ResNet18ContextBranch(pretrained=True)
        self.fusion = BGAFusion()
        self.head = SegmentHead(128, 256, num_classes)
        self.aux8 = SegmentHead(128, 128, num_classes)
        self.aux16 = SegmentHead(256, 128, num_classes)
        self.aux32 = SegmentHead(512, 128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        size = x.shape[2:]
        detail8 = self.detail(x)
        context8, feat8, feat16, feat32 = self.context(x)
        fused = self.fusion(detail8, context8)
        out = self.head(fused, size)
        if self.training:
            return (
                out,
                self.aux8(feat8, size),
                self.aux16(feat16, size),
                self.aux32(feat32, size),
            )
        return out


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_target.clamp(0, self.num_classes - 1), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probs = probs * valid
        one_hot = one_hot * valid
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice[1:].mean()


def compute_class_weights(dataset: RoadV4SegDataset, max_samples: int = 1500) -> torch.Tensor:
    rng = random.Random(SEED)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(max_samples, len(indices))]
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    for idx in tqdm(indices, desc="class weights"):
        _, mask_path = dataset.samples[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        hist = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
        counts += hist

    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.log(1.02 + freq)
    weights = weights / weights.mean()
    weights[0] *= 0.35
    weights = np.clip(weights, 0.2, 6.0)
    print("[WEIGHTS]", {CLASS_NAMES[i]: float(weights[i]) for i in range(NUM_CLASSES)})
    return torch.tensor(weights, dtype=torch.float32)


class SegLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.dice = SoftDiceLoss(NUM_CLASSES, ignore_index=IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=IGNORE_INDEX)
        dice = self.dice(logits, target)
        return ce + MIXED_DICE_WEIGHT * dice


@torch.no_grad()
def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, list[float]]:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = (target >= 0) & (target < NUM_CLASSES)
    pred = pred[valid]
    target = target[valid]
    hist = torch.bincount(NUM_CLASSES * target + pred, minlength=NUM_CLASSES * NUM_CLASSES)
    conf = hist.reshape(NUM_CLASSES, NUM_CLASSES).float()
    inter = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    ious = []
    for cls in range(NUM_CLASSES):
        if union[cls] > 0:
            ious.append(float((inter[cls] / union[cls]).cpu()))
        else:
            ious.append(float("nan"))
    miou = float(np.nanmean(ious))
    return miou, ious


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SegLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc="train")
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            outputs = model(images)
            main_loss = criterion(outputs[0], masks)
            aux_loss = sum(criterion(out, masks) for out in outputs[1:])
            loss = main_loss + AUX_WEIGHT * aux_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        pbar.set_postfix(loss=f"{total / max(1, pbar.n + 1):.4f}")
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: SegLoss, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float64, device=device)
    batches = 0

    for images, masks, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp_context(device):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += float(loss.detach().cpu())
        batches += 1
        preds = logits.argmax(dim=1)
        valid = (masks >= 0) & (masks < NUM_CLASSES)
        hist = torch.bincount(
            NUM_CLASSES * masks[valid] + preds[valid],
            minlength=NUM_CLASSES * NUM_CLASSES,
        )
        total_conf += hist.reshape(NUM_CLASSES, NUM_CLASSES).double()

    inter = torch.diag(total_conf)
    union = total_conf.sum(0) + total_conf.sum(1) - inter
    class_iou = []
    for cls in range(NUM_CLASSES):
        if union[cls] > 0:
            class_iou.append(float((inter[cls] / union[cls]).cpu()))
        else:
            class_iou.append(float("nan"))
    miou_all = float(np.nanmean(class_iou))
    miou_fg = float(np.nanmean(class_iou[1:]))
    return {
        "loss": total_loss / max(1, batches),
        "miou_all": miou_all,
        "miou_fg": miou_fg,
        "class_iou": {CLASS_NAMES[i]: class_iou[i] for i in range(NUM_CLASSES)},
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "input_size": [INPUT_H, INPUT_W],
            "metrics": metrics,
        },
        path,
    )


@torch.no_grad()
def render_predictions(model: nn.Module, dataset: RoadV4SegDataset, device: torch.device, out_dir: Path, count: int = 18) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(dataset)), min(count, len(dataset)))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    rendered = []

    for idx, sample_idx in enumerate(indices):
        image_t, mask_t, path = dataset[sample_idx]
        logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        gt = mask_t.numpy().astype(np.uint8)

        image = image_t.permute(1, 2, 0).numpy()
        image = np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        pred_overlay = image.copy()
        gt_overlay = image.copy()
        for cls, color in CLASS_COLORS_BGR.items():
            if cls == 0:
                continue
            pred_overlay[pred == cls] = color
            gt_overlay[gt == cls] = color
        pred_vis = cv2.addWeighted(pred_overlay, 0.50, image, 0.50, 0)
        gt_vis = cv2.addWeighted(gt_overlay, 0.50, image, 0.50, 0)
        combined = np.concatenate([image, gt_vis, pred_vis], axis=1)
        cv2.putText(combined, "image | gt | pred", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(combined, Path(path).name, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_path = out_dir / f"pred_{idx:03d}.jpg"
        cv2.imwrite(str(out_path), combined)
        rendered.append(out_path)

    make_contact_sheet(rendered, out_dir / "contact_sheet.jpg")


def make_contact_sheet(paths: list[Path], output: Path, cols: int = 2) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        thumbs.append(cv2.resize(image, (960, 256), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 256, cols * 960, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * 256
        x = (idx % cols) * 960
        sheet[y : y + 256, x : x + 960] = thumb
    cv2.imwrite(str(output), sheet)


def main() -> None:
    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    data_root = prepare_dataset()
    metadata = json.load(open(data_root / "metadata.json", encoding="utf-8"))
    print("[DATA]", data_root)
    print("[CLASS MAP]", metadata.get("class_map"))

    train_tf, eval_tf = build_transforms()
    train_ds = RoadV4SegDataset(data_root, "train", train_tf)
    val_ds = RoadV4SegDataset(data_root, "val", eval_tf)
    test_ds = RoadV4SegDataset(data_root, "test", eval_tf)

    class_weights = compute_class_weights(train_ds).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = BiSeNetV2ResNet18(NUM_CLASSES).to(device)
    print(f"[MODEL] params={sum(p.numel() for p in model.parameters()):,}")

    backbone_params = list(model.context.parameters())
    head_params = list(model.detail.parameters()) + list(model.fusion.parameters()) + list(model.head.parameters())
    aux_params = list(model.aux8.parameters()) + list(model.aux16.parameters()) + list(model.aux32.parameters())
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params + aux_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = make_scaler(device)
    criterion = SegLoss(class_weights)

    best_score = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[EPOCH {epoch:03d}/{EPOCHS}]")
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        score = val_metrics["miou_fg"]
        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - start, 2),
            "train_loss": train_loss,
            "val": val_metrics,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        with (SAVE_DIR / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"mIoU_all={val_metrics['miou_all']:.4f} "
            f"mIoU_fg={val_metrics['miou_fg']:.4f}"
        )
        print("[IoU]", val_metrics["class_iou"])

        save_checkpoint(SAVE_DIR / "last.pt", model, optimizer, epoch, row)
        if score > best_score + MIN_DELTA:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(SAVE_DIR / "best.pt", model, optimizer, epoch, row)
            print(f"[SAVE] best.pt foreground mIoU={best_score:.4f}")
            if epoch % 5 == 0 or epoch <= 3:
                render_predictions(model, val_ds, device, SAVE_DIR / f"val_predictions_epoch_{epoch:03d}", count=12)
        else:
            bad_epochs += 1
            print(f"[EARLY] no improvement {bad_epochs}/{PATIENCE}")

        if bad_epochs >= PATIENCE:
            print(f"[STOP] best_epoch={best_epoch}, best_fg_mIoU={best_score:.4f}")
            break

    print("\n[TEST] loading best checkpoint")
    ckpt = torch.load(SAVE_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("[TEST]", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    with (SAVE_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    render_predictions(model, test_ds, device, SAVE_DIR / "test_predictions", count=24)
    print(f"[DONE] outputs saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
