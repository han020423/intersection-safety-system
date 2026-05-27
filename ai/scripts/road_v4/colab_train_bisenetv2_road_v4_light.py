#!/usr/bin/env python3
"""
road_v4용 경량 BiSeNetV2 Colab 학습 스크립트.

라즈베리파이 적용을 고려한 버전이다.
ResNet-18 같은 외부 대형 백본을 붙이지 않고, 기존 BiSeNetV2 구조와
BiSeNetV2 backbone_v2.pth 초기값을 사용한다.

주요 특징:
  - bisenetv2.py의 BiSeNetV2를 그대로 import
  - road_v4 6클래스 학습
  - class imbalance 보정을 위한 class weight 자동 계산
  - CrossEntropy + Dice Loss + aux loss 사용
  - optimizer를 제외한 작은 추론용 checkpoint 저장

데이터셋 구조:
  road_v4/
    train/images/*.jpg, train/masks/*.png
    val/images/*.jpg,   val/masks/*.png
    test/images/*.jpg,  test/masks/*.png
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
import zipfile
import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (
    SCRIPT_DIR,
    Path.cwd(),
    Path("/content/drive/MyDrive/capstone/bisenet"),
    Path("/content/drive/MyDrive/capstone/ufld"),
    Path("/content/drive/MyDrive/capstone"),
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))


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
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def load_bisenetv2_class():
    """
    Colab Drive에서는 실행 위치와 import path가 어긋나는 경우가 있어
    일반 import 실패 시 bisenetv2.py를 파일 경로로 직접 로드한다.
    """
    try:
        from bisenetv2 import BiSeNetV2 as ImportedBiSeNetV2

        return ImportedBiSeNetV2
    except ModuleNotFoundError:
        pass

    candidates = [
        SCRIPT_DIR / "bisenetv2.py",
        Path("/content/drive/MyDrive/capstone/bisenet/bisenetv2.py"),
        Path("/content/drive/MyDrive/capstone/ufld/bisenetv2.py"),
        Path("/content/drive/MyDrive/capstone/bisenet/road_v4/bisenetv2.py"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("bisenetv2", str(candidate))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["bisenetv2"] = module
        spec.loader.exec_module(module)
        print(f"[MODEL] loaded BiSeNetV2 definition from {candidate}")
        return module.BiSeNetV2

    raise ModuleNotFoundError(
        "bisenetv2.py를 찾지 못했습니다. "
        "colab_train_bisenetv2_road_v4_light.py와 같은 Google Drive 폴더에 "
        "ai/scripts/road_v4/bisenetv2.py 파일도 함께 업로드하세요. "
        f"searched={[str(p) for p in candidates]}"
    )


BiSeNetV2 = load_bisenetv2_class()

try:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
except Exception:
    print("[INFO] Google Drive is not mounted. Running outside Colab or already mounted.")


SEED = 42
NUM_CLASSES = 6
# road_v4 데이터셋의 마스크 id와 같은 순서다. 추론 후처리도 이 순서를 기준으로 한다.
CLASS_NAMES = ["background", "lane_white", "lane_yellow", "lane_blue", "crosswalk", "stop_line"]

# 경량 모델과 라즈베리파이 추론을 고려해 기존 BiSeNetV2 학습 해상도를 기본값으로 둔다.
INPUT_H = 352
INPUT_W = 640
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 100
PATIENCE = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
AUX_WEIGHT = 0.4
DICE_WEIGHT = 1.0
HEAD_LR_MULT = 3
GRAD_CLIP_NORM = 2.0
AMP = False
OHEM_THRESH = 0.60
OHEM_MIN_KEPT = 400_000
OHEM_WARMUP_EPOCHS = 999
OHEM_WEIGHT = 0.0

DATASET_ZIP_NAME = "road_v4.zip"
WORK_DIR = Path("/content/road_v4")
DRIVE_CANDIDATES = [
    Path("/content/drive/MyDrive/capstone/bisenet"),
    Path("/content/drive/MyDrive/capstone/ufld"),
    Path("/content/drive/MyDrive/capstone"),
]
SAVE_DIR = Path("/content/drive/MyDrive/capstone/bisenet/road_v4_bisenetv2_light_stable_run")

CLASS_COLORS_BGR = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (0, 220, 255),
    3: (255, 120, 20),
    4: (0, 255, 0),
    5: (0, 0, 255),
}


def set_seed(seed: int) -> None:
    """재현성을 위해 random seed를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def find_dataset_zip() -> Path:
    """Colab Drive 안에서 road_v4.zip을 자동으로 찾는다."""
    for root in DRIVE_CANDIDATES:
        candidate = root / DATASET_ZIP_NAME
        if candidate.exists():
            return candidate
    matches = list(Path("/content/drive/MyDrive").rglob(DATASET_ZIP_NAME))
    if matches:
        return matches[0]
    raise FileNotFoundError("road_v4.zip not found in Google Drive.")


def count_split_images(root: Path) -> dict[str, int]:
    """압축 해제가 정상인지 split별 이미지 개수를 확인한다."""
    return {
        split: len(list((root / split / "images").glob("*.jpg")))
        for split in ("train", "val", "test")
    }


def is_valid_dataset_root(root: Path) -> bool:
    """metadata만 있고 이미지가 없는 깨진 /content/road_v4를 걸러낸다."""
    if not (root / "metadata.json").exists():
        return False
    counts = count_split_images(root)
    return all(counts[split] > 0 for split in ("train", "val", "test"))


def prepare_dataset() -> Path:
    """압축된 road_v4 데이터셋을 /content로 풀고 경로를 반환한다."""
    if is_valid_dataset_root(WORK_DIR):
        print(f"[DATA] using existing dataset: {WORK_DIR} counts={count_split_images(WORK_DIR)}")
        return WORK_DIR
    if WORK_DIR.exists():
        print(f"[DATA] existing dataset is incomplete, removing: {WORK_DIR}")
        shutil.rmtree(WORK_DIR)

    zip_path = find_dataset_zip()
    print(f"[DATA] unzip {zip_path} -> /content")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content")
    if is_valid_dataset_root(WORK_DIR):
        print(f"[DATA] extracted dataset: {WORK_DIR} counts={count_split_images(WORK_DIR)}")
        return WORK_DIR

    candidates = [
        p.parent for p in Path("/content").rglob("metadata.json")
        if p.parent.name == "road_v4" and is_valid_dataset_root(p.parent)
    ]
    if candidates:
        print(f"[DATA] found dataset: {candidates[0]} counts={count_split_images(candidates[0])}")
        return candidates[0]
    raise FileNotFoundError("valid road_v4 dataset not found after unzip.")


class RoadV4Dataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None) -> None:
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []
        image_dir = root / split / "images"
        mask_dir = root / split / "masks"
        for image_path in sorted(image_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                self.samples.append((image_path, mask_path))
        print(f"[DATA] {split}: {len(self.samples):,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # mask는 단일 채널 uint8 PNG이며 픽셀 값이 곧 class id다.
        image_path, mask_path = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"failed to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.uint8)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"].long()
        return image, mask


def build_transforms():
    """이미지와 마스크에 같은 기하 변환을 적용해 라벨 정합성을 유지한다."""
    train_tf = A.Compose([
        A.Resize(INPUT_H, INPUT_W, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.RandomGamma(gamma_limit=(80, 125), p=1.0),
        ], p=0.55),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.35),
        A.Affine(scale=(0.90, 1.10), translate_percent=(-0.04, 0.04), rotate=(-4, 4), p=0.35),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.12),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.10),
        A.RandomShadow(shadow_roi=(0, 0.35, 1, 1), num_shadows_limit=(1, 2), p=0.10),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(INPUT_H, INPUT_W, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, eval_tf


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        # 배경을 제외한 얇은 구조물 차선/정지선 손실을 보강한다.
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target.clamp(0, self.num_classes - 1), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice[1:].mean()


class OhemCrossEntropyLoss(nn.Module):
    """
    일반 CE를 기본으로 두고 어려운 픽셀 CE를 약하게 더한다.
    처음부터 OHEM만 쓰면 배경/차선 균형이 무너질 수 있어 hybrid 방식으로 사용한다.
    """
    def __init__(self, weight=None, ignore_index: int = 255,
                 thresh: float = 0.60, min_kept: int = 400_000,
                 ohem_weight: float = 0.25) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index
        self.thresh_loss = -float(np.log(thresh))
        self.min_kept = min_kept
        self.ohem_weight = ohem_weight

    def forward(self, logits, target):
        losses = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = target != self.ignore_index
        valid_losses = losses[valid]
        if valid_losses.numel() == 0:
            return logits.sum() * 0.0
        normal_ce = valid_losses.mean()
        if self.ohem_weight <= 0:
            return normal_ce

        hard_losses, _ = torch.sort(valid_losses.reshape(-1), descending=True)
        if hard_losses.numel() > self.min_kept:
            threshold = max(self.thresh_loss, float(hard_losses[self.min_kept - 1].detach()))
            hard_losses = hard_losses[hard_losses >= threshold]
        else:
            hard_losses = hard_losses[hard_losses >= self.thresh_loss]
            if hard_losses.numel() == 0:
                hard_losses = valid_losses[:1]
        return normal_ce + self.ohem_weight * hard_losses.mean()


def compute_class_weights(dataset: RoadV4Dataset, max_samples: int = 1500) -> torch.Tensor:
    """희소 클래스인 파랑 차선/정지선/횡단보도가 묻히지 않도록 가중치를 계산한다."""
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
        counts += np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
    freq = counts / max(1.0, counts.sum())
    weights = 1.0 / np.log(1.02 + freq)
    weights = weights / weights.mean()
    weights[0] *= 0.35
    weights = np.clip(weights, 0.2, 6.0)
    print("[WEIGHTS]", {CLASS_NAMES[i]: float(weights[i]) for i in range(NUM_CLASSES)})
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, ce_loss, dice_loss, device):
    """검증/테스트 데이터셋에서 mIoU와 클래스별 IoU를 계산한다."""
    model.eval()
    model.aux_mode = "eval"
    total_loss = 0.0
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float64, device=device)
    batches = 0
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=AMP and device.type == "cuda"):
            logits = model(images)[0]
            loss = ce_loss(logits, masks) + DICE_WEIGHT * dice_loss(logits, masks)
        total_loss += float(loss.detach().cpu())
        batches += 1
        preds = logits.argmax(dim=1)
        valid = (masks >= 0) & (masks < NUM_CLASSES)
        hist = torch.bincount(NUM_CLASSES * masks[valid] + preds[valid], minlength=NUM_CLASSES * NUM_CLASSES)
        conf += hist.reshape(NUM_CLASSES, NUM_CLASSES).double()

    inter = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    ious = []
    for cls in range(NUM_CLASSES):
        ious.append(float((inter[cls] / union[cls]).cpu()) if union[cls] > 0 else float("nan"))
    return {
        "loss": total_loss / max(1, batches),
        "miou_all": float(np.nanmean(ious)),
        "miou_fg": float(np.nanmean(ious[1:])),
        "class_iou": {CLASS_NAMES[i]: ious[i] for i in range(NUM_CLASSES)},
    }


def strip_state_dict(model) -> dict:
    """추론용 checkpoint를 작게 만들기 위해 모델 weight만 CPU tensor로 저장한다."""
    raw = model.state_dict()
    return {k.replace("_orig_mod.", ""): v.detach().cpu() for k, v in raw.items()}


def model_has_nonfinite(model) -> bool:
    """NaN/Inf가 weight에 들어가면 이후 epoch이 전부 무너져서 즉시 중단한다."""
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return True
    return False


@torch.no_grad()
def render_predictions(model, dataset, device, out_dir: Path, count: int = 18):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.aux_mode = "eval"
    indices = random.sample(range(len(dataset)), min(count, len(dataset)))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    thumbs = []
    for i, idx in enumerate(indices):
        image_t, mask_t = dataset[idx]
        logits = model(image_t.unsqueeze(0).to(device))[0]
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        gt = mask_t.numpy().astype(np.uint8)
        image = image_t.permute(1, 2, 0).numpy()
        image = np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        panels = [image]
        for mask in (gt, pred):
            overlay = image.copy()
            for cls, color in CLASS_COLORS_BGR.items():
                if cls:
                    overlay[mask == cls] = color
            panels.append(cv2.addWeighted(image, 0.55, overlay, 0.45, 0))
        combined = np.concatenate(panels, axis=1)
        cv2.putText(combined, "image | gt | pred", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"pred_{i:03d}.jpg"), combined)
        thumbs.append(cv2.resize(combined, (960, 176), interpolation=cv2.INTER_AREA))
    if thumbs:
        cols = 2
        rows = int(np.ceil(len(thumbs) / cols))
        sheet = np.zeros((rows * 176, cols * 960, 3), dtype=np.uint8)
        for i, thumb in enumerate(thumbs):
            y, x = (i // cols) * 176, (i % cols) * 960
            sheet[y:y + 176, x:x + 960] = thumb
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), sheet)


def main():
    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(
        f"[CONFIG] input={INPUT_H}x{INPUT_W} batch={BATCH_SIZE} "
        f"lr={LR} head_lr_mult={HEAD_LR_MULT} amp={AMP} "
        f"loss=CE+Dice ohem_weight={OHEM_WEIGHT} "
        f"save_dir={SAVE_DIR}"
    )

    data_root = prepare_dataset()
    metadata = json.load(open(data_root / "metadata.json", encoding="utf-8"))
    print("[CLASS MAP]", metadata.get("class_map"))

    train_tf, eval_tf = build_transforms()
    train_ds = RoadV4Dataset(data_root, "train", train_tf)
    val_ds = RoadV4Dataset(data_root, "val", eval_tf)
    test_ds = RoadV4Dataset(data_root, "test", eval_tf)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    class_weights = compute_class_weights(train_ds).to(device)
    # 이 모델이 실제 배포 후보인 경량 BiSeNetV2다.
    # bisenetv2.py 내부 init_weights()에서 backbone_v2.pth 로드를 시도한다.
    model = BiSeNetV2(n_classes=NUM_CLASSES, aux_mode="train").to(device)
    print(f"[MODEL] lightweight BiSeNetV2 params={sum(p.numel() for p in model.parameters()):,}")

    # BiSeNetV2 구현체의 get_params()를 사용해 backbone/head 학습률을 분리한다.
    wd, nowd, head_wd, head_nowd = model.get_params()
    optimizer = optim.AdamW([
        {"params": wd, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": nowd, "lr": LR, "weight_decay": 0.0},
        {"params": head_wd, "lr": LR * HEAD_LR_MULT, "weight_decay": WEIGHT_DECAY},
        {"params": head_nowd, "lr": LR * HEAD_LR_MULT, "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and device.type == "cuda")
    ce_loss = OhemCrossEntropyLoss(
        weight=class_weights,
        ignore_index=255,
        thresh=OHEM_THRESH,
        min_kept=OHEM_MIN_KEPT,
        ohem_weight=0.0,
    )
    dice_loss = DiceLoss(NUM_CLASSES)

    history = []
    best_score = -1.0
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[EPOCH {epoch:03d}/{EPOCHS}]")
        model.train()
        model.aux_mode = "train"
        ce_loss.ohem_weight = OHEM_WEIGHT if epoch > OHEM_WARMUP_EPOCHS else 0.0
        print(f"[LOSS] ohem_weight={ce_loss.ohem_weight:.2f}")
        total_loss = 0.0
        valid_batches = 0
        skipped_batches = 0
        start = time.time()
        pbar = tqdm(train_loader, desc="train")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=AMP and device.type == "cuda"):
                outputs = model(images)
                # main head와 aux head를 함께 학습하면 얕은 stage까지 supervision이 들어가 수렴이 안정적이다.
                main = ce_loss(outputs[0], masks) + DICE_WEIGHT * dice_loss(outputs[0], masks)
                aux = sum(ce_loss(out, masks) + DICE_WEIGHT * dice_loss(out, masks) for out in outputs[1:])
                loss = main + AUX_WEIGHT * aux
            if not torch.isfinite(loss):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="skip-nonfinite", skipped=skipped_batches)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM, error_if_nonfinite=True)
            except RuntimeError:
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                pbar.set_postfix(loss="skip-grad", skipped=skipped_batches)
                continue
            scaler.step(optimizer)
            scaler.update()
            if model_has_nonfinite(model):
                raise RuntimeError(
                    "Model weights became NaN/Inf. "
                    "Stop this run and restart from best_light_full.pt with a lower LR."
                )
            total_loss += float(loss.detach().cpu())
            valid_batches += 1
            pbar.set_postfix(loss=f"{total_loss / max(1, valid_batches):.4f}", skipped=skipped_batches)

        train_loss = total_loss / max(1, valid_batches)
        val_metrics = evaluate(model, val_loader, ce_loss, dice_loss, device)
        scheduler.step()
        score = val_metrics["miou_fg"]
        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - start, 2),
            "train_loss": train_loss,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[2]["lr"],
            "skipped_batches": skipped_batches,
        }
        history.append(row)
        with (SAVE_DIR / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"mIoU_all={val_metrics['miou_all']:.4f} mIoU_fg={val_metrics['miou_fg']:.4f}"
        )
        print("[IoU]", val_metrics["class_iou"])

        torch.save({
            "epoch": epoch,
            "state_dict": strip_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "input_size": [INPUT_H, INPUT_W],
            "metrics": row,
            "model_type": "bisenetv2_light",
        }, SAVE_DIR / "last_full.pt")

        if score > best_score + 1e-4:
            best_score = score
            bad_epochs = 0
            # best_light_infer.pt는 optimizer state를 빼서 추론 배포용 크기를 줄인 파일이다.
            infer_ckpt = {
                "state_dict": strip_state_dict(model),
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "input_size": [INPUT_H, INPUT_W],
                "metrics": row,
                "model_type": "bisenetv2_light",
            }
            torch.save(infer_ckpt, SAVE_DIR / "best_light_infer.pt")
            torch.save({**infer_ckpt, "optimizer": optimizer.state_dict()}, SAVE_DIR / "best_light_full.pt")
            print(f"[SAVE] best_light_infer.pt fg_mIoU={best_score:.4f}")
            if epoch <= 3 or epoch % 5 == 0:
                render_predictions(model, val_ds, device, SAVE_DIR / f"val_predictions_epoch_{epoch:03d}", 12)
        else:
            bad_epochs += 1
            print(f"[EARLY] no improvement {bad_epochs}/{PATIENCE}")
            if bad_epochs >= PATIENCE:
                break

    print("\n[TEST] loading best_light_infer.pt")
    ckpt = torch.load(SAVE_DIR / "best_light_infer.pt", map_location=device)
    model = BiSeNetV2(n_classes=NUM_CLASSES, aux_mode="eval").to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    test_metrics = evaluate(model, test_loader, ce_loss, dice_loss, device)
    print("[TEST]", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    with (SAVE_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    render_predictions(model, test_ds, device, SAVE_DIR / "test_predictions", 24)
    print(f"[DONE] outputs saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
