# ╔══════════════════════════════════════════════════════════════════╗
# ║  UFLDv2 차선 분류 학습 스크립트  (Google Colab 전용)             ║
# ║  파일을 Colab에 업로드한 후 아래 순서로 실행하세요.              ║
# ║                                                                  ║
# ║  [권장 실행 방법]                                                ║
# ║  새 셀에서:  !python colab_train_ufldv2.py                       ║
# ║  또는 Runtime → Run all (셀 분리 시)                             ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# ── 아키텍처 요약 ────────────────────────────────────────────────
#  UFLDv2Classifier
#  ├─ Backbone : ResNet-18 (pretrained, ImageNet)
#  │   stem → layer1 → layer2  ← 288×512 입력 기준 36×64 feature map
#  │   layer3  ← type/existence 추가 특징 추출용
#  ├─ Location Head
#  │   (B,128,36,64) → 1×1 conv → (B,MAX_LANES,36,64)
#  │   = per-lane probability over 64 grid cells at each of 36 row-anchors
#  ├─ Type Head  (B,MAX_LANES,NUM_TYPES)
#  └─ Existence Head  (B,MAX_LANES)   sigmoid
#
# ── Lane Category ────────────────────────────────────────────────
#  0=none  1=white-solid  2=white-dotted  3=yellow-solid
#  4=yellow-dotted  5=blue-solid  6=blue-dotted
# ─────────────────────────────────────────────────────────────────

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 0 ▶  패키지 설치
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import subprocess, sys

def _pip(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

_pip('tqdm')
# torch / torchvision은 Colab 기본 설치되어 있음

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 ▶  Import
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os, json, random, time, warnings, zipfile, math
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvmodels
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {DEVICE}')
if torch.cuda.is_available():
    print(f'[GPU]    {torch.cuda.get_device_name(0)}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 ▶  Google Drive 마운트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print('[Drive] 마운트 완료')
except Exception:
    print('[Drive] Colab 환경이 아니거나 이미 마운트됨')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 ▶  ⚙️  설정  (경로/하이퍼파라미터 수정 가능)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 경로 ──────────────────────────────────────────
DRIVE_ZIP  = '/content/drive/MyDrive/capstone/ufld/ufldv2_dataset.zip'   # Drive의 zip 파일 위치
WORK_DIR   = '/content/ufldv2_dataset'                      # 압축 해제 위치
SAVE_DIR   = '/content/drive/MyDrive/capstone/ufld/ufldv2_output'         # 결과 저장 위치

# ── 모델 구조 (prepare_dataset.py 와 일치해야 함) ─
INPUT_H    = 288    # 모델 입력 높이
INPUT_W    = 512    # 모델 입력 너비
NUM_ANCHORS = 36    # row-anchor 수 (= feature map 높이, stride-8)
GRID_NUM   = 64     # x-grid 수   (= feature map 너비, stride-8)
MAX_LANES  = 8      # 슬롯 수 (이미지당 최대 차선, 교차로 기준 양방향 포함)
NUM_TYPES  = 7      # 0=none, 1~6=lane types

# ── 학습 하이퍼파라미터 ────────────────────────────
BATCH_SIZE   = 16
EPOCHS       = 50    # 최대 학습 에포크 (조기먈춤 발동 시 일직 종료)
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LOC_WEIGHT   = 2.0   # 위치 손실 가중치 (지그재그 저감을 위해 증가)
TYPE_WEIGHT  = 0.6   # 타입 손실 가중치
EXIST_WEIGHT = 0.2   # 존재 손실 가중치
SMOOTH_WEIGHT= 0.5   # 부드러움 손실: 인접 앵커 간 x좌표 급변 패널티
NUM_WORKERS  = 2     # DataLoader 워커 수
SEED         = 42
PATIENCE     = 10    # 조기먈춤: val_acc 개선 없는 최대 epoch 수
MIN_DELTA    = 1e-4  # 개선으로 인정하는 최소 향상폭

# ── 클래스 이름 ────────────────────────────────────
TYPE_NAMES = [
    'none', 'white-solid', 'white-dotted',
    'yellow-solid', 'yellow-dotted', 'blue-solid', 'blue-dotted',
]
TYPE_COLORS = [
    '#888888',  # none (gray)
    '#FFFFFF',  # white-solid
    '#DDDDDD',  # white-dotted (light gray)
    '#FFD700',  # yellow-solid
    '#FFA500',  # yellow-dotted
    '#1E90FF',  # blue-solid
    '#87CEEB',  # blue-dotted
]

# ─────────────────────────────────────────────────
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 ▶  데이터셋 압축 해제
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import shutil
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# 항상 깨끗하게 재추출 (mid-run 실패 때 남은 부점 해제 제거)
if Path(WORK_DIR).exists():
    print(f'[unzip] 기존 디렉토리 제거: {WORK_DIR}')
    shutil.rmtree(WORK_DIR)

print(f'[unzip] {DRIVE_ZIP} 압축 해제 중...')
with zipfile.ZipFile(DRIVE_ZIP, 'r') as zf:
    for member in zf.infolist():
        # Windows 백슬래시(\) → 슬래시(/) 변환 (Linux 호환)
        member.filename = member.filename.replace('\\', '/')
        zf.extract(member, '/content')
print('[unzip] 완료')

# config.json 위치 자동 탐색 (zip 내부 구조 자동 상설)
import subprocess
result = subprocess.run(['find', '/content', '-name', 'config.json', '-not', '-path', '*/drive/*'],
                        capture_output=True, text=True)
cfg_paths = [p for p in result.stdout.strip().split('\n') if p]
if not cfg_paths:
    raise FileNotFoundError('[ERROR] config.json을 찾을 수 없습니다. zip 파일을 확인하세요.')
WORK_DIR = str(Path(cfg_paths[0]).parent)  # config.json 있는 폴더와 WORK_DIR 일치
print(f'[unzip] WORK_DIR 확인: {WORK_DIR}')

# config 로드
with open(f'{WORK_DIR}/config.json', encoding='utf-8') as f:
    DS_CFG = json.load(f)
ORIG_H    = DS_CFG['orig_h']
ORIG_W    = DS_CFG['orig_w']
H_SAMPLES = DS_CFG['h_samples']
print(f'[Dataset] train={DS_CFG["splits"]["train"]}  val={DS_CFG["splits"]["val"]}  test={DS_CFG["splits"]["test"]}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 ▶  Dataset & DataLoader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def lanes_to_targets(lanes, categories):
    """
    차선 어노테이션 → 학습 타깃 텐서 변환

    Args:
        lanes      : list of [x0, x1, ..., x35]  (-2 = 없음)
        categories : list of int (카테고리 per lane)

    Returns:
        loc_target  : (MAX_LANES, NUM_ANCHORS) long  — grid cell index, -1=무효
        type_target : (MAX_LANES,)             long  — lane type (0=없음)
        exist_target: (MAX_LANES,)             float — exist=1.0
    """
    loc   = torch.full((MAX_LANES, NUM_ANCHORS), -1, dtype=torch.long)
    typ   = torch.zeros(MAX_LANES, dtype=torch.long)
    exist = torch.zeros(MAX_LANES, dtype=torch.float)

    # 차선을 평균 x 기준 왼→오른쪽 정렬하여 슬롯 할당
    def mean_x(xs):
        valid = [x for x in xs if x != -2]
        return float(np.mean(valid)) if valid else 0.0

    sorted_lanes = sorted(zip(lanes, categories), key=lambda z: mean_x(z[0]))

    for slot, (xs, cat) in enumerate(sorted_lanes):
        if slot >= MAX_LANES:
            break
        exist[slot] = 1.0
        typ[slot]   = cat
        for anchor_idx, x_orig in enumerate(xs):
            if x_orig == -2:
                continue
            # 원본 x → grid 셀 인덱스 [0, GRID_NUM-1]
            grid_cell = min(int(x_orig * GRID_NUM / ORIG_W), GRID_NUM - 1)
            loc[slot, anchor_idx] = grid_cell

    return loc, typ, exist


class UFLDv2Dataset(Dataset):
    """UFLDv2 차선 분류 학습용 데이터셋"""

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, ann_file: str, data_root: str, split: str = 'train'):
        with open(ann_file, encoding='utf-8') as f:
            self.anns = json.load(f)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = self._build_transform(split == 'train')

    def _build_transform(self, augment: bool):
        ops = [T.Resize((INPUT_H, INPUT_W))]
        if augment:
            ops += [
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.RandomGrayscale(p=0.05),
            ]
        ops += [
            T.ToTensor(),
            T.Normalize(self._MEAN, self._STD),
        ]
        return T.Compose(ops)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann  = self.anns[idx]
        img  = Image.open(self.data_root / ann['raw_file']).convert('RGB')
        img  = self.transform(img)
        loc, typ, exist = lanes_to_targets(ann['lanes'], ann['lane_categories'])
        return img, loc, typ, exist


def build_loaders():
    ld = lambda split, shuf: DataLoader(
        UFLDv2Dataset(f'{WORK_DIR}/{split}.json', WORK_DIR, split),
        batch_size=BATCH_SIZE, shuffle=shuf,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=(shuf))
    return ld('train', True), ld('val', False), ld('test', False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 ▶  모델 아키텍처 (UFLDv2Classifier)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConvBNReLU(nn.Sequential):
    def __init__(self, inc, outc, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(inc, outc, k, s, p, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )


class UFLDv2Classifier(nn.Module):
    """
    UFLDv2 스타일 차선 검출 + 분류 네트워크.

    입력  : (B, 3, 288, 512)
    백본  : ResNet-18 (pretrained)
           stem → layer1 → layer2 : (B, 128, 36, 64)   ← stride-8
           layer3                  : (B, 256, 18, 32)   ← type/exist용

    Location Head:
        (B,128,36,64) → reduce(1×1) → (B,64,36,64)
        → cls_conv(1×1) → (B,MAX_LANES,36,64)
        softmax on dim=-1 → 각 슬롯·앵커의 grid cell 확률

    Type Head:
        (B,256,18,32) → GAP → (B,256) → FC → (B,MAX_LANES,NUM_TYPES)

    Existence Head:
        (B,128,36,64) → GAP → (B,128) → FC → (B,MAX_LANES)
    """

    def __init__(self):
        super().__init__()
        bb = tvmodels.resnet18(pretrained=True)

        # ── backbone ──────────────────────────────────
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # (B,  64, 72, 128)
        self.layer2 = bb.layer2   # (B, 128, 36,  64)  ← core feature map
        self.layer3 = bb.layer3   # (B, 256, 18,  32)

        # ── Location head ─────────────────────────────
        self.loc_reduce = ConvBNReLU(128, 64, 1, 1, 0)
        self.loc_cls    = nn.Conv2d(64, MAX_LANES, kernel_size=1)
        # output: (B, MAX_LANES, NUM_ANCHORS=36, GRID_NUM=64)

        # ── Auxiliary attention (row-wise) for location ──
        self.row_att    = nn.Sequential(
            nn.AdaptiveAvgPool2d((NUM_ANCHORS, 1)),   # (B,128,36,1)
            ConvBNReLU(128, 64, 1, 1, 0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        # ── Type head ─────────────────────────────────
        self.type_gap  = nn.AdaptiveAvgPool2d(1)
        self.type_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, MAX_LANES * NUM_TYPES),
        )

        # ── Existence head ────────────────────────────
        self.exist_gap  = nn.AdaptiveAvgPool2d(1)
        self.exist_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, MAX_LANES),
        )

        self._init_heads()

    def _init_heads(self):
        for m in [self.loc_reduce, self.loc_cls,
                  self.type_head, self.exist_head]:
            for layer in (m if isinstance(m, nn.Sequential) else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        B = x.size(0)

        x = self.stem(x)        # (B,  64, 72, 128)
        x = self.layer1(x)      # (B,  64, 72, 128)
        x = self.layer2(x)      # (B, 128, 36,  64)
        loc_feat = x

        x = self.layer3(x)      # (B, 256, 18,  32)

        # ── Location ──────────────────────────────────
        lf    = self.loc_reduce(loc_feat)           # (B,  64, 36, 64)
        # row-wise attention
        att   = self.row_att(loc_feat)              # (B,   1, 36,  1)
        lf    = lf * att                            # broadcast
        loc   = self.loc_cls(lf)                    # (B, MAX_LANES, 36, 64)

        # ── Type ──────────────────────────────────────
        tf    = self.type_gap(x).flatten(1)         # (B, 256)
        typ   = self.type_head(tf)                  # (B, MAX_LANES*NUM_TYPES)
        typ   = typ.view(B, MAX_LANES, NUM_TYPES)   # (B, MAX_LANES, NUM_TYPES)

        # ── Existence ─────────────────────────────────
        ef    = self.exist_gap(loc_feat).flatten(1) # (B, 128)
        exist = self.exist_head(ef)                 # (B, MAX_LANES)

        return loc, typ, exist  # (raw logits — no softmax/sigmoid)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 ▶  손실 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UFLDv2Loss(nn.Module):
    """
    Location CE  +  Smoothness  +  Type CE  +  Existence BCE
    """

    def __init__(self, loc_w=2.0, type_w=0.6, exist_w=0.2, smooth_w=0.5):
        super().__init__()
        self.loc_w    = loc_w
        self.type_w   = type_w
        self.exist_w  = exist_w
        self.smooth_w = smooth_w

    def forward(self, loc_pred, type_pred, exist_pred,
                loc_tgt, type_tgt, exist_tgt):
        B, ML, NA, G = loc_pred.shape

        # ── Location loss (CE, ignore -1) ─────────────
        lp = loc_pred.reshape(B * ML * NA, G)
        lt = loc_tgt.reshape(B * ML * NA)
        loss_loc = F.cross_entropy(lp, lt, ignore_index=-1)

        # ── Smoothness loss (soft-argmax 기반) ─────────
        # 인접 앵커 간 예상 x좌표 차이를 패널티로 부과 → 지그재그 억제
        grid_idx = torch.arange(G, dtype=torch.float, device=loc_pred.device)
        soft_x   = (loc_pred.softmax(-1) * grid_idx).sum(-1)   # (B, ML, NA)
        diffs    = (soft_x[:, :, 1:] - soft_x[:, :, :-1]).abs()  # (B, ML, NA-1)
        e_mask   = (exist_tgt > 0.5).unsqueeze(-1).expand_as(diffs)
        loss_smooth = diffs[e_mask].mean() if e_mask.any() else loc_pred.new_zeros(1).squeeze()

        # ── Type loss (CE, only existing lanes) ────────
        exist_mask = (exist_tgt > 0.5)
        tp_sel = type_pred[exist_mask]
        tt_sel = type_tgt[exist_mask]
        if tt_sel.numel() > 0:
            loss_type = F.cross_entropy(tp_sel, tt_sel)
        else:
            loss_type = loc_pred.new_zeros(1).squeeze()

        # ── Existence loss (BCE) ───────────────────────
        loss_exist = F.binary_cross_entropy_with_logits(exist_pred, exist_tgt)

        # ── Total ──────────────────────────────────────
        loss = (self.loc_w    * loss_loc
                + self.smooth_w * loss_smooth
                + self.type_w   * loss_type
                + self.exist_w  * loss_exist)

        return loss, {
            'loc':    loss_loc.item(),
            'smooth': loss_smooth.item(),
            'type':   loss_type.item(),
            'exist':  loss_exist.item(),
        }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 ▶  메트릭 및 평가 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_type_acc(type_pred, type_tgt, exist_tgt):
    """존재하는 차선에 대한 타입 분류 정확도"""
    mask = (exist_tgt > 0.5)
    if not mask.any():
        return 0.0, 0
    pred = type_pred[mask].argmax(dim=-1)
    tgt  = type_tgt[mask]
    return (pred == tgt).float().mean().item(), tgt.numel()


@torch.no_grad()
def evaluate(model, loader, criterion):
    """전체 손실과 per-class 분류 정확도를 반환합니다."""
    model.eval()
    total_loss = 0.0
    class_correct = [0] * NUM_TYPES
    class_total   = [0] * NUM_TYPES

    for imgs, loc_tgt, type_tgt, exist_tgt in loader:
        imgs, loc_tgt, type_tgt, exist_tgt = (
            imgs.to(DEVICE), loc_tgt.to(DEVICE), type_tgt.to(DEVICE), exist_tgt.to(DEVICE))

        loc_pred, type_pred, exist_pred = model(imgs)
        loss, _ = criterion(loc_pred, type_pred, exist_pred, loc_tgt, type_tgt, exist_tgt)
        total_loss += loss.item()

        mask = (exist_tgt > 0.5)
        if mask.any():
            pred = type_pred[mask].argmax(-1)
            tgt  = type_tgt[mask]
            for t in range(NUM_TYPES):
                m = (tgt == t)
                class_correct[t] += (pred[m] == t).sum().item()
                class_total[t]   += m.sum().item()

    n = len(loader)
    total_correct = sum(class_correct[1:])
    total_total   = sum(class_total[1:])
    overall_acc   = total_correct / total_total if total_total > 0 else 0.0

    per_class_acc = {
        TYPE_NAMES[t]: class_correct[t] / class_total[t] if class_total[t] > 0 else float('nan')
        for t in range(1, NUM_TYPES)
    }

    return total_loss / n, overall_acc, per_class_acc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9 ▶  학습 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss = total_loc = total_type = total_exist = 0.0
    correct = n_lanes = 0

    for imgs, loc_tgt, type_tgt, exist_tgt in tqdm(
            loader, desc=f'  Epoch {epoch+1:02d}', ncols=90, leave=False):

        imgs, loc_tgt, type_tgt, exist_tgt = (
            imgs.to(DEVICE), loc_tgt.to(DEVICE), type_tgt.to(DEVICE), exist_tgt.to(DEVICE))

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loc_pred, type_pred, exist_pred = model(imgs)
            loss, breakdown = criterion(loc_pred, type_pred, exist_pred,
                                        loc_tgt, type_tgt, exist_tgt)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss  += loss.item()
        total_loc   += breakdown['loc']
        total_type  += breakdown['type']
        total_exist += breakdown['exist']

        acc, n = compute_type_acc(type_pred.detach(), type_tgt, exist_tgt)
        correct  += acc * n
        n_lanes  += n

    nb = len(loader)
    type_acc = correct / n_lanes if n_lanes > 0 else 0.0
    return (total_loss / nb, total_loc / nb,
            total_type / nb, total_exist / nb, type_acc)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10 ▶  시각화 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def denormalize(tensor):
    """ImageNet 정규화 역변환 (C,H,W) → (H,W,C) numpy"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def grid_to_pixel(grid_cell):
    """grid 셀 인덱스 → 입력 이미지 내 x 좌표 (픽셀)"""
    return int((grid_cell + 0.5) * INPUT_W / GRID_NUM)


def anchor_to_pixel(anchor_idx):
    """anchor 인덱스 → 입력 이미지 내 y 좌표 (픽셀)"""
    h_in_orig = H_SAMPLES[anchor_idx]
    return int(h_in_orig * INPUT_H / ORIG_H)


@torch.no_grad()
def visualize_predictions(model, dataset, num_samples=6, save_path=None):
    """샘플 예측 결과를 시각화합니다 (신뢰도 필터링 + 다항식 스무딩 적용)."""
    import numpy as np
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    CONF_THRESH = 0.08   # softmax max가 이 값 미만인 앵커는 불확실 → 스킵
    POLY_DEG    = 2      # 다항식 피팅 차수 (2=포물선, 3=3차곡선)

    for ax_i, idx in enumerate(indices):
        img_t, loc_tgt, type_tgt, exist_tgt = dataset[idx]
        img_input = img_t.unsqueeze(0).to(DEVICE)

        loc_pred, type_pred, exist_pred = model(img_input)

        loc_prob   = loc_pred[0].softmax(-1).cpu()          # (ML, NA, G)
        type_pred  = type_pred[0].cpu()                     # (ML, NUM_TYPES)
        exist_pred = torch.sigmoid(exist_pred[0]).cpu()     # (ML,)

        img_np = denormalize(img_t)
        ax = axes[ax_i]
        ax.imshow(img_np)
        ax.set_title(f'Sample #{idx}', fontsize=10)
        ax.axis('off')

        for slot in range(MAX_LANES):
            prob_exist = exist_pred[slot].item()
            if prob_exist < 0.5:
                continue

            pred_type = type_pred[slot].argmax().item()
            color     = TYPE_COLORS[pred_type]

            # 신뢰도 높은 앵커만 사용
            xs_raw, ys_raw = [], []
            for anchor_idx in range(NUM_ANCHORS):
                conf = loc_prob[slot, anchor_idx].max().item()
                if conf < CONF_THRESH:
                    continue
                grid_cell = loc_prob[slot, anchor_idx].argmax().item()
                px = grid_to_pixel(grid_cell)
                py = anchor_to_pixel(anchor_idx)
                xs_raw.append(px)
                ys_raw.append(py)

            if len(xs_raw) < POLY_DEG + 2:
                continue

            # 다항식 피팅으로 부드럽게
            try:
                coeffs  = np.polyfit(ys_raw, xs_raw, POLY_DEG)
                ys_fit  = np.linspace(min(ys_raw), max(ys_raw), 60)
                xs_fit  = np.polyval(coeffs, ys_fit)
                # 이미지 경계 클리핑
                valid   = (xs_fit >= 0) & (xs_fit <= INPUT_W)
                ax.plot(xs_fit[valid], ys_fit[valid],
                        '-', color=color, linewidth=2.5, alpha=0.85)
                ax.text(xs_fit[valid][-1] if valid.any() else xs_raw[-1],
                        ys_fit[valid][-1] if valid.any() else ys_raw[-1],
                        f'{TYPE_NAMES[pred_type]}\n{prob_exist:.2f}',
                        fontsize=5, color='white',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.7))
            except Exception:
                ax.plot(xs_raw, ys_raw, '-', color=color, linewidth=2, alpha=0.8)

    patches = [mpatches.Patch(color=TYPE_COLORS[i], label=TYPE_NAMES[i])
               for i in range(1, NUM_TYPES)]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               fontsize=9, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  시각화 저장: {save_path}')
    plt.show()


def plot_history(history, save_path=None):
    """학습 곡선을 그립니다."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color='#E74C3C')
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='#3498DB')
    axes[0].set_title('Total Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Type Accuracy
    axes[1].plot(epochs, history['train_type_acc'], label='Train', color='#E74C3C')
    axes[1].plot(epochs, history['val_acc'],        label='Val',   color='#3498DB')
    axes[1].set_title('Type Classification Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Sub-losses
    axes[2].plot(epochs, history['train_loc'],   label='Loc',   color='#2ECC71')
    axes[2].plot(epochs, history['train_type'],  label='Type',  color='#F39C12')
    axes[2].plot(epochs, history['train_exist'], label='Exist', color='#9B59B6')
    axes[2].set_title('Train Sub-losses', fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  학습 곡선 저장: {save_path}')
    plt.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 11 ▶  메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── DataLoaders ──────────────────────────────────
    train_ld, val_ld, test_ld = build_loaders()
    train_ds = train_ld.dataset
    test_ds  = test_ld.dataset

    print(f'[DataLoader] train={len(train_ds):,}  val={len(val_ld.dataset):,}  test={len(test_ds):,}')

    # ── Model ────────────────────────────────────────
    model = UFLDv2Classifier().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[Model]  UFLDv2Classifier  파라미터 수: {n_params:,}')

    # ── Optimizer / Scheduler / Loss ─────────────────
    optimizer = optim.AdamW(
        [{'params': list(model.stem.parameters()) +
                    list(model.layer1.parameters()) +
                    list(model.layer2.parameters()) +
                    list(model.layer3.parameters()),
          'lr': LR * 0.1,    # 백본: 낮은 lr
          'name': 'backbone'},
         {'params': list(model.loc_reduce.parameters()) +
                    list(model.loc_cls.parameters()) +
                    list(model.row_att.parameters()) +
                    list(model.type_head.parameters()) +
                    list(model.exist_head.parameters()),
          'lr': LR,           # 헤드: 기본 lr
          'name': 'heads'}],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = UFLDv2Loss(LOC_WEIGHT, TYPE_WEIGHT, EXIST_WEIGHT, SMOOTH_WEIGHT).to(DEVICE)
    scaler    = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # ── Training ─────────────────────────────────────
    history = {k: [] for k in ['train_loss', 'train_loc', 'train_type', 'train_exist',
                                 'train_type_acc', 'val_loss', 'val_acc']}
    best_val_acc   = 0.0
    best_epoch     = 0
    no_improve_cnt = 0          # 조기먈춤 카운터

    print(f'\n{"━"*65}')
    print(f'  학습 시작: 최대 {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  LR={LR}')
    print(f'  조기먈춤: patience={PATIENCE}, min_delta={MIN_DELTA}')
    print(f'{"━"*65}')

    for epoch in range(EPOCHS):
        t0 = time.time()

        t_loss, t_loc, t_type, t_exist, t_acc = train_epoch(
            model, train_ld, optimizer, criterion, scaler, epoch)
        v_loss, v_acc, v_per_class = evaluate(model, val_ld, criterion)
        scheduler.step()

        elapsed = time.time() - t0

        for k, v in zip(['train_loss', 'train_loc', 'train_type', 'train_exist', 'train_type_acc'],
                         [t_loss, t_loc, t_type, t_exist, t_acc]):
            history[k].append(v)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f'Ep {epoch+1:02d}/{EPOCHS} | '
              f'Loss {t_loss:.4f}/{v_loss:.4f} | '
              f'TypeAcc {t_acc:.4f}/{v_acc:.4f} | '
              f'lr={optimizer.param_groups[1]["lr"]:.5f} | '
              f'{elapsed:.0f}s')

        # ── Checkpoint ─────────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch + 1
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc':         v_acc,
                'config': {
                    'input_h': INPUT_H, 'input_w': INPUT_W,
                    'num_anchors': NUM_ANCHORS, 'grid_num': GRID_NUM,
                    'max_lanes': MAX_LANES, 'num_types': NUM_TYPES,
                    'h_samples': H_SAMPLES, 'orig_h': ORIG_H, 'orig_w': ORIG_W,
                    'type_names': TYPE_NAMES,
                },
            }, f'{SAVE_DIR}/best_model.pth')
            print(f'  [★ BEST] Best model saved  (val_acc={v_acc:.4f}, epoch={best_epoch})')
            no_improve_cnt = 0   # 개선 확인 → 카운터 리셋
        else:
            no_improve_cnt += 1
            bar = '#' * no_improve_cnt + '.' * (PATIENCE - no_improve_cnt)
            print(f'  [patience {no_improve_cnt:2d}/{PATIENCE}] [{bar}]  (best={best_val_acc:.4f} @ ep{best_epoch})')
            if no_improve_cnt >= PATIENCE:
                print(f'\n  [Early Stop] {PATIENCE} epoch 동안 val_acc 개선 없음'
                      f' (best={best_val_acc:.4f} @ ep{best_epoch}) -> 조기 종료')
                break

    # ── Test Evaluation ──────────────────────────────
    print(f'\n{"━"*65}')
    print('  테스트 평가')
    print(f'{"━"*65}')

    ckpt = torch.load(f'{SAVE_DIR}/best_model.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f'  Best checkpoint: epoch={best_epoch}, val_acc={best_val_acc:.4f}')

    test_loss, test_acc, test_per_class = evaluate(model, test_ld, criterion)
    print(f'\n  Test Loss: {test_loss:.4f}  |  Test Type Acc: {test_acc:.4f}')
    print('\n  Per-class Accuracy:')
    for name, acc in test_per_class.items():
        bar = '█' * int(acc * 30) + '░' * (30 - int(acc * 30))
        print(f'    {name:20s} [{bar}] {acc:.4f}')

    # ── Results JSON ──────────────────────────────────
    results = {
        'best_epoch':    best_epoch,
        'best_val_acc':  best_val_acc,
        'test_loss':     test_loss,
        'test_acc':      test_acc,
        'per_class_acc': test_per_class,
        'config': {
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
            'total_samples': DS_CFG['splits'],
            'type_names': TYPE_NAMES,
        },
        'history': history,
    }
    with open(f'{SAVE_DIR}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  결과 저장: {SAVE_DIR}/results.json')

    # ── 학습 곡선 ─────────────────────────────────────
    plot_history(history, save_path=f'{SAVE_DIR}/training_curves.png')

    # ── 예측 시각화 ───────────────────────────────────
    print('\n  예측 시각화 생성 중...')
    visualize_predictions(model, test_ds, num_samples=6,
                          save_path=f'{SAVE_DIR}/predictions.png')

    print(f'\n{"━"*65}')
    print(f'  ✅ 학습 완료!')
    print(f'  결과 폴더: {SAVE_DIR}')
    print(f'  ├─ best_model.pth      (최적 모델 가중치)')
    print(f'  ├─ results.json        (수치 결과)')
    print(f'  ├─ training_curves.png (학습 곡선)')
    print(f'  └─ predictions.png     (예측 시각화)')
    print(f'{"━"*65}')


if __name__ == '__main__':
    main()
