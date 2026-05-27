#!/usr/bin/env python
"""
Colab training script for the road_v3 color-only UFLDv2-style lane dataset.

Dataset classes:
  0 = none
  1 = white
  2 = yellow
  3 = blue

Expected dataset structure after unzip:
  road_v3_ufldv2_color_10000/
    images/train/*.jpg
    images/val/*.jpg
    images/test/*.jpg
    train.json
    val.json
    test.json
    config.json

Recommended Colab usage:
  1. Upload road_v3_ufldv2_color_10000.zip to:
     /content/drive/MyDrive/capstone/ufld/road_v3_ufldv2_color_10000.zip
  2. Run:
     !python colab_train_ufldv2_color.py
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path


def _pip(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


try:
    import cv2
except Exception:
    _pip("opencv-python")

try:
    from tqdm import tqdm
except Exception:
    _pip("tqdm")


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tvmodels
import torchvision.transforms as T
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Colab paths
# ---------------------------------------------------------------------------

try:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
except Exception:
    print("[INFO] Google Drive is not mounted. Running outside Colab or already mounted.")


DRIVE_ROOT = Path("/content/drive/MyDrive/capstone/ufld")
DATASET_ZIP = DRIVE_ROOT / "road_v3_ufldv2_color_10000.zip"
WORK_DIR = Path("/content/road_v3_ufldv2_color_10000")
SAVE_DIR = DRIVE_ROOT / "road_v3_color_resnet18_rowhead_run"


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

SEED = 42
INPUT_H = 288
INPUT_W = 512
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 50
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MIN_DELTA = 1e-4
AMP = True

LOC_WEIGHT = 2.0
ROW_WEIGHT = 1.0
TYPE_WEIGHT = 0.6
EXIST_WEIGHT = 0.2
SMOOTH_WEIGHT = 0.05

EXIST_THRESH = 0.45
POINT_THRESH = 0.65
GRID_CONF_THRESH = 0.20


TYPE_NAMES = ["none", "white", "yellow", "blue"]
TYPE_COLORS_BGR = {
    1: (255, 255, 255),
    2: (0, 220, 255),
    3: (255, 120, 20),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def prepare_dataset() -> Path:
    if WORK_DIR.exists() and (WORK_DIR / "config.json").exists():
        print(f"[DATA] using existing dataset: {WORK_DIR}")
        return WORK_DIR

    if not DATASET_ZIP.exists():
        raise FileNotFoundError(
            f"Dataset zip not found: {DATASET_ZIP}\n"
            "Upload road_v3_ufldv2_color_10000.zip to Google Drive first."
        )

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    print(f"[DATA] unzip {DATASET_ZIP} -> /content")
    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        zf.extractall("/content")

    if not (WORK_DIR / "config.json").exists():
        candidates = list(Path("/content").rglob("config.json"))
        candidates = [p for p in candidates if "drive" not in str(p)]
        if not candidates:
            raise FileNotFoundError("config.json not found after unzip.")
        return candidates[0].parent

    return WORK_DIR


def load_config(data_root: Path) -> dict:
    with (data_root / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def mean_valid_x(xs: list[int]) -> float:
    valid = [x for x in xs if x != -2]
    return float(np.mean(valid)) if valid else 1e9


def lanes_to_targets(
    lanes: list[list[int]],
    categories: list[int],
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_lanes = int(cfg["max_lanes"])
    num_anchors = int(cfg["num_anchors"])
    grid_num = int(cfg["grid_num"])
    orig_w = int(cfg["orig_w"])

    # grid_num is the explicit "no point at this row-anchor" class.
    loc = torch.full((max_lanes, num_anchors), grid_num, dtype=torch.long)
    typ = torch.zeros(max_lanes, dtype=torch.long)
    exist = torch.zeros(max_lanes, dtype=torch.float32)

    pairs = sorted(zip(lanes, categories), key=lambda item: mean_valid_x(item[0]))
    for slot, (xs, cat) in enumerate(pairs[:max_lanes]):
        exist[slot] = 1.0
        typ[slot] = int(cat)
        for anchor_idx, x in enumerate(xs[:num_anchors]):
            if x == -2:
                continue
            grid = int(float(x) * grid_num / orig_w)
            grid = max(0, min(grid, grid_num - 1))
            loc[slot, anchor_idx] = grid

    return loc, typ, exist


class UFLDColorDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, data_root: Path, split: str, cfg: dict) -> None:
        self.data_root = data_root
        self.split = split
        self.cfg = cfg
        with (data_root / f"{split}.json").open("r", encoding="utf-8") as f:
            self.records = json.load(f)

        ops: list[nn.Module] = [T.Resize((INPUT_H, INPUT_W))]
        if split == "train":
            ops += [
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
                T.RandomGrayscale(p=0.03),
            ]
        ops += [T.ToTensor(), T.Normalize(self.MEAN, self.STD)]
        self.transform = T.Compose(ops)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]
        image_path = self.data_root / record["image"]
        image = Image.open(image_path).convert("RGB")
        image_t = self.transform(image)

        loc, typ, exist = lanes_to_targets(record["lanes"], record["categories"], self.cfg)
        return {
            "image": image_t,
            "loc": loc,
            "type": typ,
            "exist": exist,
            "path": str(image_path),
        }


class UFLDv2ColorNet(nn.Module):
    """
    UFLDv2-style row-anchor classifier with an imported ResNet-18 backbone.

    loc_logits:   (B, max_lanes, num_anchors, grid_num + 1)
    type_logits:  (B, max_lanes, num_types)
    exist_logits: (B, max_lanes)
    """

    def __init__(self, cfg: dict, pretrained: bool = True) -> None:
        super().__init__()
        self.max_lanes = int(cfg["max_lanes"])
        self.num_anchors = int(cfg["num_anchors"])
        self.grid_num = int(cfg["grid_num"])
        self.loc_classes = self.grid_num + 1
        self.num_types = int(cfg["num_lane_types"])

        try:
            weights = tvmodels.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tvmodels.resnet18(weights=weights)
        except AttributeError:
            backbone = tvmodels.resnet18(pretrained=pretrained)

        self.stage_to_36x64 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        self.context = nn.Sequential(backbone.layer3, backbone.layer4)

        self.loc_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.max_lanes, kernel_size=1),
        )
        self.no_point_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.max_lanes, kernel_size=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.type_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.max_lanes * self.num_types),
        )
        self.exist_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.max_lanes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.stage_to_36x64(x)
        loc = self.loc_head(feat)
        loc = F.interpolate(
            loc,
            size=(self.num_anchors, self.grid_num),
            mode="bilinear",
            align_corners=False,
        )
        no_point = self.no_point_head(feat)
        no_point = F.adaptive_avg_pool2d(no_point, (self.num_anchors, 1))
        loc = torch.cat([loc, no_point], dim=-1)

        ctx = self.context(feat)
        pooled = self.pool(ctx).flatten(1)
        typ = self.type_head(pooled).view(-1, self.max_lanes, self.num_types)
        exist = self.exist_head(pooled)
        return loc, typ, exist


class UFLDv2ColorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        loc_logits: torch.Tensor,
        type_logits: torch.Tensor,
        exist_logits: torch.Tensor,
        loc_target: torch.Tensor,
        type_target: torch.Tensor,
        exist_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loc_logits = loc_logits.permute(0, 1, 2, 3).contiguous()
        no_point_cls = loc_logits.shape[-1] - 1
        grid_logits = loc_logits[..., :no_point_cls]
        no_point_logits = loc_logits[..., no_point_cls]
        valid_lane = exist_target > 0.5
        if valid_lane.any():
            lane_grid_logits = grid_logits[valid_lane]
            lane_no_point_logits = no_point_logits[valid_lane]
            lane_loc_target = loc_target[valid_lane]
            point_target = (lane_loc_target < no_point_cls).float()
            point_logits = torch.logsumexp(lane_grid_logits, dim=-1) - lane_no_point_logits

            pos = point_target.sum().clamp_min(1.0)
            neg = (point_target.numel() - point_target.sum()).clamp_min(1.0)
            pos_weight = torch.clamp(neg / pos, min=1.0, max=5.0)
            loss_row = F.binary_cross_entropy_with_logits(
                point_logits,
                point_target,
                pos_weight=pos_weight,
            )

            point_mask = lane_loc_target < no_point_cls
            if point_mask.any():
                loss_loc = F.cross_entropy(
                    lane_grid_logits[point_mask],
                    lane_loc_target[point_mask],
                )
            else:
                loss_loc = loc_logits.sum() * 0.0
        else:
            loss_loc = loc_logits.sum() * 0.0
            loss_row = loc_logits.sum() * 0.0

        if valid_lane.any():
            loss_type = F.cross_entropy(type_logits[valid_lane], type_target[valid_lane])
        else:
            loss_type = type_logits.sum() * 0.0

        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, exist_target)
        loss_smooth = self.smoothness_loss(loc_logits, loc_target)

        total = (
            LOC_WEIGHT * loss_loc
            + ROW_WEIGHT * loss_row
            + TYPE_WEIGHT * loss_type
            + EXIST_WEIGHT * loss_exist
            + SMOOTH_WEIGHT * loss_smooth
        )
        parts = {
            "loc": float(loss_loc.detach().cpu()),
            "row": float(loss_row.detach().cpu()),
            "type": float(loss_type.detach().cpu()),
            "exist": float(loss_exist.detach().cpu()),
            "smooth": float(loss_smooth.detach().cpu()),
        }
        return total, parts

    @staticmethod
    def smoothness_loss(loc_logits: torch.Tensor, loc_target: torch.Tensor) -> torch.Tensor:
        valid = loc_target < (loc_logits.shape[-1] - 1)
        if valid.sum() < 2:
            return loc_logits.sum() * 0.0

        grid_logits = loc_logits[..., :-1]
        prob = F.softmax(grid_logits, dim=-1)
        grid = torch.arange(grid_logits.shape[-1], device=loc_logits.device, dtype=prob.dtype)
        exp_x = (prob * grid).sum(dim=-1)

        valid_pair = valid[:, :, 1:] & valid[:, :, :-1]
        if not valid_pair.any():
            return loc_logits.sum() * 0.0
        diffs = exp_x[:, :, 1:] - exp_x[:, :, :-1]
        return diffs[valid_pair].abs().mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: UFLDv2ColorLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    loc_correct = 0
    loc_total = 0
    type_correct = 0
    type_total = 0
    exist_correct = 0
    exist_total = 0
    point_tp = 0
    point_fp = 0
    point_fn = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        loc_t = batch["loc"].to(device, non_blocking=True)
        type_t = batch["type"].to(device, non_blocking=True)
        exist_t = batch["exist"].to(device, non_blocking=True)

        loc_p, type_p, exist_p = model(images)
        loss, _ = criterion(loc_p, type_p, exist_p, loc_t, type_t, exist_t)
        total_loss += float(loss.detach().cpu())
        n_batches += 1

        no_point_cls = loc_p.shape[-1] - 1
        grid_logits = loc_p[..., :no_point_cls]
        no_point_logits = loc_p[..., no_point_cls]
        point_logits = torch.logsumexp(grid_logits, dim=-1) - no_point_logits
        point_pred = torch.sigmoid(point_logits) >= POINT_THRESH
        point_target = loc_t < no_point_cls
        valid_lane = exist_t > 0.5

        point_eval_mask = valid_lane.unsqueeze(-1).expand_as(point_target)
        point_tp += int((point_pred & point_target & point_eval_mask).sum().cpu())
        point_fp += int((point_pred & ~point_target & point_eval_mask).sum().cpu())
        point_fn += int((~point_pred & point_target & point_eval_mask).sum().cpu())

        valid_loc = point_target
        loc_pred = grid_logits.argmax(dim=-1)
        loc_correct += int((loc_pred[valid_loc] == loc_t[valid_loc]).sum().cpu())
        loc_total += int(valid_loc.sum().cpu())

        type_pred = type_p.argmax(dim=-1)
        type_correct += int((type_pred[valid_lane] == type_t[valid_lane]).sum().cpu())
        type_total += int(valid_lane.sum().cpu())

        for cls in range(1, len(TYPE_NAMES)):
            mask = valid_lane & (type_t == cls)
            per_class_correct[cls] += int((type_pred[mask] == cls).sum().cpu())
            per_class_total[cls] += int(mask.sum().cpu())

        exist_pred = (torch.sigmoid(exist_p) >= EXIST_THRESH).float()
        exist_correct += int((exist_pred == exist_t).sum().cpu())
        exist_total += int(exist_t.numel())

    per_class_acc = {}
    for cls in range(1, len(TYPE_NAMES)):
        denom = per_class_total[cls]
        per_class_acc[TYPE_NAMES[cls]] = per_class_correct[cls] / denom if denom else float("nan")

    point_precision = point_tp / max(point_tp + point_fp, 1)
    point_recall = point_tp / max(point_tp + point_fn, 1)
    point_f1 = 2 * point_precision * point_recall / max(point_precision + point_recall, 1e-9)

    return {
        "loss": total_loss / max(n_batches, 1),
        "loc_acc": loc_correct / max(loc_total, 1),
        "point_precision": point_precision,
        "point_recall": point_recall,
        "point_f1": point_f1,
        "type_acc": type_correct / max(type_total, 1),
        "exist_acc": exist_correct / max(exist_total, 1),
        "per_class_acc": per_class_acc,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: UFLDv2ColorLoss,
    scaler: GradScaler,
    device: torch.device,
) -> dict:
    model.train()
    running = 0.0
    parts_sum = defaultdict(float)

    pbar = tqdm(loader, desc="train")
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        loc_t = batch["loc"].to(device, non_blocking=True)
        type_t = batch["type"].to(device, non_blocking=True)
        exist_t = batch["exist"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=AMP and device.type == "cuda"):
            loc_p, type_p, exist_p = model(images)
            loss, parts = criterion(loc_p, type_p, exist_p, loc_t, type_t, exist_t)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running += float(loss.detach().cpu())
        for key, value in parts.items():
            parts_sum[key] += value
        avg = running / max(1, pbar.n + 1)
        pbar.set_postfix(loss=f"{avg:.4f}")

    n = max(len(loader), 1)
    return {
        "loss": running / n,
        "parts": {key: value / n for key, value in parts_sum.items()},
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    cfg: dict,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dataset_config": cfg,
            "model_config": {
                "input_h": INPUT_H,
                "input_w": INPUT_W,
                "max_lanes": cfg["max_lanes"],
                "num_anchors": cfg["num_anchors"],
                "grid_num": cfg["grid_num"],
                "loc_classes": int(cfg["grid_num"]) + 1,
                "num_lane_types": cfg["num_lane_types"],
                "type_names": TYPE_NAMES,
            },
            "metrics": metrics,
        },
        path,
    )


def split_lane_segments(
    anchor_points: list[tuple[int, int, int, int]],
    min_points: int = 4,
    max_anchor_gap: int = 2,
    max_grid_jump: int = 10,
) -> list[list[tuple[int, int]]]:
    if not anchor_points:
        return []

    segments: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = [anchor_points[0]]
    for prev, point in zip(anchor_points, anchor_points[1:]):
        prev_anchor, prev_grid, _, _ = prev
        anchor, grid, _, _ = point
        if anchor - prev_anchor <= max_anchor_gap and abs(grid - prev_grid) <= max_grid_jump:
            current.append(point)
        else:
            segments.append(current)
            current = [point]
    segments.append(current)

    clean_segments: list[list[tuple[int, int]]] = []
    for segment in segments:
        if len(segment) < min_points:
            continue
        xs = np.array([p[2] for p in segment], dtype=np.float32)
        ys = np.array([p[3] for p in segment], dtype=np.float32)
        degree = 1 if len(segment) < 6 else 2
        try:
            coeff = np.polyfit(ys, xs, degree)
            xs_smooth = np.polyval(coeff, ys)
            points = [(int(round(x)), int(round(y))) for x, y in zip(xs_smooth, ys)]
        except Exception:
            points = [(p[2], p[3]) for p in segment]
        clean_segments.append(points)
    return clean_segments


@torch.no_grad()
def render_predictions(
    model: nn.Module,
    dataset: UFLDColorDataset,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    count: int = 12,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(dataset)), min(count, len(dataset)))
    mean = np.array(UFLDColorDataset.MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(UFLDColorDataset.STD, dtype=np.float32).reshape(1, 1, 3)

    h_samples = cfg["h_samples"]
    orig_w = int(cfg["orig_w"])
    grid_num = int(cfg["grid_num"])

    for out_idx, idx in enumerate(indices):
        sample = dataset[idx]
        image_t = sample["image"].unsqueeze(0).to(device)
        loc_p, type_p, exist_p = model(image_t)
        loc_p = loc_p[0].cpu()
        type_p = type_p[0].cpu()
        exist_p = torch.sigmoid(exist_p[0]).cpu()

        img = sample["image"].permute(1, 2, 0).cpu().numpy()
        img = np.clip((img * std + mean) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (orig_w, int(cfg["orig_h"])), interpolation=cv2.INTER_LINEAR)

        for lane_idx in range(int(cfg["max_lanes"])):
            if float(exist_p[lane_idx]) < EXIST_THRESH:
                continue
            cls = int(type_p[lane_idx].argmax().item())
            if cls <= 0:
                continue
            color = TYPE_COLORS_BGR.get(cls, (128, 128, 128))
            grid_logits = loc_p[lane_idx, :, :grid_num]
            no_point_logits = loc_p[lane_idx, :, grid_num]
            point_prob = torch.sigmoid(torch.logsumexp(grid_logits, dim=-1) - no_point_logits)
            grid_prob = F.softmax(grid_logits, dim=-1)
            grid_pred = grid_prob.argmax(dim=-1).numpy()
            grid_conf = grid_prob.max(dim=-1).values.numpy()
            point_prob_np = point_prob.numpy()

            anchor_points: list[tuple[int, int, int, int]] = []
            for a_idx, (grid, g_conf, p_conf) in enumerate(zip(grid_pred, grid_conf, point_prob_np)):
                if float(p_conf) < POINT_THRESH:
                    continue
                if float(g_conf) < GRID_CONF_THRESH:
                    continue
                x = int((float(grid) + 0.5) * orig_w / grid_num)
                y = int(h_samples[a_idx])
                anchor_points.append((a_idx, int(grid), x, y))

            segments = split_lane_segments(anchor_points)
            for points in segments:
                cv2.polylines(img, [np.array(points, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
                cv2.putText(
                    img,
                    f"{TYPE_NAMES[cls]} {float(exist_p[lane_idx]):.2f}",
                    points[-1],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.imwrite(str(out_dir / f"pred_{out_idx:03d}.jpg"), img)


def main() -> None:
    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    data_root = prepare_dataset()
    cfg = load_config(data_root)
    cfg["num_lane_types"] = int(cfg.get("num_lane_types", 4))
    assert cfg["num_lane_types"] == 4, f"Expected 4 lane types, got {cfg['num_lane_types']}"

    print("[DATA]", data_root)
    print("[SPLITS]", cfg.get("splits"))
    print("[LANE TYPES]", cfg.get("lane_type_names"))

    train_ds = UFLDColorDataset(data_root, "train", cfg)
    val_ds = UFLDColorDataset(data_root, "val", cfg)
    test_ds = UFLDColorDataset(data_root, "test", cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = UFLDv2ColorNet(cfg, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] UFLDv2ColorNet params={n_params:,}")

    backbone_params = list(model.stage_to_36x64.parameters()) + list(model.context.parameters())
    head_params = (
        list(model.loc_head.parameters())
        + list(model.no_point_head.parameters())
        + list(model.type_head.parameters())
        + list(model.exist_head.parameters())
    )
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = UFLDv2ColorLoss()
    scaler = GradScaler(enabled=AMP and device.type == "cuda")

    best_score = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[EPOCH {epoch:03d}/{EPOCHS}]")
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        score = (
            0.35 * val_metrics["loc_acc"]
            + 0.25 * val_metrics["point_f1"]
            + 0.25 * val_metrics["type_acc"]
            + 0.15 * val_metrics["exist_acc"]
        )
        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - t0, 2),
            "train": train_metrics,
            "val": val_metrics,
            "score": score,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(row)

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"loc={val_metrics['loc_acc']:.4f} "
            f"point_f1={val_metrics['point_f1']:.4f} "
            f"type={val_metrics['type_acc']:.4f} "
            f"exist={val_metrics['exist_acc']:.4f} "
            f"score={score:.4f}"
        )
        print("[per-class]", val_metrics["per_class_acc"])

        save_checkpoint(SAVE_DIR / "last.pt", model, optimizer, epoch, cfg, row)
        with (SAVE_DIR / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if score > best_score + MIN_DELTA:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(SAVE_DIR / "best.pt", model, optimizer, epoch, cfg, row)
            print(f"[SAVE] best.pt score={best_score:.4f}")
        else:
            bad_epochs += 1
            print(f"[EARLY] no improvement {bad_epochs}/{PATIENCE}")

        if bad_epochs >= PATIENCE:
            print(f"[STOP] early stopping. best_epoch={best_epoch} best_score={best_score:.4f}")
            break

    print("\n[TEST] loading best checkpoint")
    ckpt = torch.load(SAVE_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("[TEST]", json.dumps(test_metrics, ensure_ascii=False, indent=2))

    with (SAVE_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    render_predictions(model, val_ds, cfg, device, SAVE_DIR / "val_predictions", count=16)
    print(f"[DONE] outputs saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
