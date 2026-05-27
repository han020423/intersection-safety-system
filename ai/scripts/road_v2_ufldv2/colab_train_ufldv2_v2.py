# ╔══════════════════════════════════════════════════════════════════╗
# ║  UFLDv2 차선 분류 학습 스크립트 v2 (Colab 전용)                  ║
# ║  - CULane 사전학습 가중치 로드 지원                              ║
# ║  - NMS(Non-Maximum Suppression) 후처리 적용                      ║
# ╚══════════════════════════════════════════════════════════════════╝

import subprocess, sys

def _pip(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

_pip('tqdm')

import os, json, random, time, warnings, zipfile
from pathlib import Path

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

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print('[Drive] 마운트 완료')
except Exception:
    print('[Drive] Colab 환경이 아니거나 이미 마운트됨')

# ── ⚙️ 설정 ──────────────────────────────────────────
DRIVE_ZIP        = '/content/drive/MyDrive/capstone/ufld/ufldv2_dataset.zip'
WORK_DIR         = '/content/ufldv2_dataset'
SAVE_DIR         = '/content/drive/MyDrive/capstone/ufld/ufldv2_output_v2'

# 공식 CULane 사전학습 모델 가중치 (Drive에 업로드 필요)
# UFLDv2 공식 레포에서 제공하는 resnet18_culane.pth 파일을 아래 경로에 넣어주세요.
PRETRAINED_MODEL = '/content/drive/MyDrive/capstone/ufld/ufldv2_resnet18_culane.pth'

# 모델 구조
INPUT_H    = 288
INPUT_W    = 512
NUM_ANCHORS = 36
GRID_NUM   = 64
MAX_LANES  = 8
NUM_TYPES  = 7

# 학습 하이퍼파라미터
BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LOC_WEIGHT   = 2.0
TYPE_WEIGHT  = 0.6
EXIST_WEIGHT = 0.2
SMOOTH_WEIGHT= 0.5
NUM_WORKERS  = 2
SEED         = 42
PATIENCE     = 10
MIN_DELTA    = 1e-4

TYPE_NAMES = [
    'none', 'white-solid', 'white-dotted',
    'yellow-solid', 'yellow-dotted', 'blue-solid', 'blue-dotted',
]
TYPE_COLORS = [
    '#888888', '#FFFFFF', '#DDDDDD',
    '#FFD700', '#FFA500', '#1E90FF', '#87CEEB',
]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ── 데이터 준비 ────────────────────────────────────────
import shutil
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

if Path(WORK_DIR).exists():
    print(f'[unzip] 기존 디렉토리 제거: {WORK_DIR}')
    shutil.rmtree(WORK_DIR)

print(f'[unzip] {DRIVE_ZIP} 압축 해제 중...')
with zipfile.ZipFile(DRIVE_ZIP, 'r') as zf:
    for member in zf.infolist():
        member.filename = member.filename.replace('\\', '/')
        zf.extract(member, '/content')
print('[unzip] 완료')

result = subprocess.run(['find', '/content', '-name', 'config.json', '-not', '-path', '*/drive/*'],
                        capture_output=True, text=True)
cfg_paths = [p for p in result.stdout.strip().split('\n') if p]
if not cfg_paths:
    raise FileNotFoundError('[ERROR] config.json을 찾을 수 없습니다.')
WORK_DIR = str(Path(cfg_paths[0]).parent)
print(f'[unzip] WORK_DIR 확인: {WORK_DIR}')

with open(f'{WORK_DIR}/config.json', encoding='utf-8') as f:
    DS_CFG = json.load(f)
ORIG_H    = DS_CFG['orig_h']
ORIG_W    = DS_CFG['orig_w']
H_SAMPLES = DS_CFG['h_samples']
print(f'[Dataset] train={DS_CFG["splits"]["train"]}  val={DS_CFG["splits"]["val"]}  test={DS_CFG["splits"]["test"]}')

# ── 데이터셋 ──────────────────────────────────────────
def lanes_to_targets(lanes, categories):
    loc   = torch.full((MAX_LANES, NUM_ANCHORS), -1, dtype=torch.long)
    typ   = torch.zeros(MAX_LANES, dtype=torch.long)
    exist = torch.zeros(MAX_LANES, dtype=torch.float)

    def mean_x(xs):
        valid = [x for x in xs if x != -2]
        return float(np.mean(valid)) if valid else 0.0

    sorted_lanes = sorted(zip(lanes, categories), key=lambda z: mean_x(z[0]))

    for slot, (xs, cat) in enumerate(sorted_lanes):
        if slot >= MAX_LANES: break
        exist[slot] = 1.0
        typ[slot]   = cat
        for anchor_idx, x_orig in enumerate(xs):
            if x_orig == -2: continue
            grid_cell = min(int(x_orig * GRID_NUM / ORIG_W), GRID_NUM - 1)
            loc[slot, anchor_idx] = grid_cell

    return loc, typ, exist

class UFLDv2Dataset(Dataset):
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, ann_file, data_root, split='train'):
        with open(ann_file, encoding='utf-8') as f:
            self.anns = json.load(f)
        self.data_root = Path(data_root)
        self.transform = self._build_transform(split == 'train')

    def _build_transform(self, augment):
        ops = [T.Resize((INPUT_H, INPUT_W))]
        if augment:
            ops += [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)]
        ops += [T.ToTensor(), T.Normalize(self._MEAN, self._STD)]
        return T.Compose(ops)

    def __len__(self): return len(self.anns)
    def __getitem__(self, idx):
        ann = self.anns[idx]
        img = Image.open(self.data_root / ann['raw_file']).convert('RGB')
        img = self.transform(img)
        loc, typ, exist = lanes_to_targets(ann['lanes'], ann['lane_categories'])
        return img, loc, typ, exist

def build_loaders():
    ld = lambda split, shuf: DataLoader(
        UFLDv2Dataset(f'{WORK_DIR}/{split}.json', WORK_DIR, split),
        batch_size=BATCH_SIZE, shuffle=shuf, num_workers=NUM_WORKERS, drop_last=shuf)
    return ld('train', True), ld('val', False), ld('test', False)

# ── 모델 아키텍처 ────────────────────────────────────
class ConvBNReLU(nn.Sequential):
    def __init__(self, inc, outc, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(inc, outc, k, s, p, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )

class UFLDv2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        bb = tvmodels.resnet18(pretrained=True)
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3

        self.loc_reduce = ConvBNReLU(128, 64, 1, 1, 0)
        self.loc_cls    = nn.Conv2d(64, MAX_LANES, kernel_size=1)
        self.row_att    = nn.Sequential(
            nn.AdaptiveAvgPool2d((NUM_ANCHORS, 1)),
            ConvBNReLU(128, 64, 1, 1, 0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.type_gap  = nn.AdaptiveAvgPool2d(1)
        self.type_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, MAX_LANES * NUM_TYPES),
        )

        self.exist_gap  = nn.AdaptiveAvgPool2d(1)
        self.exist_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, MAX_LANES),
        )
        self._init_heads()

    def _init_heads(self):
        for m in [self.loc_reduce, self.loc_cls, self.type_head, self.exist_head]:
            for layer in (m if isinstance(m, nn.Sequential) else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                    if layer.bias is not None: nn.init.zeros_(layer.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        loc_feat = x
        x = self.layer3(x)

        lf  = self.loc_reduce(loc_feat)
        att = self.row_att(loc_feat)
        lf  = lf * att
        loc = self.loc_cls(lf)

        tf  = self.type_gap(x).flatten(1)
        typ = self.type_head(tf).view(B, MAX_LANES, NUM_TYPES)

        ef    = self.exist_gap(loc_feat).flatten(1)
        exist = self.exist_head(ef)

        return loc, typ, exist

class UFLDv2Loss(nn.Module):
    def __init__(self, loc_w=2.0, type_w=0.6, exist_w=0.2, smooth_w=0.5):
        super().__init__()
        self.loc_w, self.type_w, self.exist_w, self.smooth_w = loc_w, type_w, exist_w, smooth_w

    def forward(self, loc_pred, type_pred, exist_pred, loc_tgt, type_tgt, exist_tgt):
        B, ML, NA, G = loc_pred.shape
        lp = loc_pred.reshape(B * ML * NA, G)
        lt = loc_tgt.reshape(B * ML * NA)
        loss_loc = F.cross_entropy(lp, lt, ignore_index=-1)

        grid_idx = torch.arange(G, dtype=torch.float, device=loc_pred.device)
        soft_x   = (loc_pred.softmax(-1) * grid_idx).sum(-1)
        
        # 2nd order difference (곡률 패널티): (x[i-1] + x[i+1] - 2*x[i])
        diffs = (soft_x[:, :, :-2] + soft_x[:, :, 2:] - 2 * soft_x[:, :, 1:-1]).abs()
        
        # 차선이 실제로 존재하는(loc_tgt != -1) 앵커 구간에만 Smoothness 적용
        valid_mask = (loc_tgt[:, :, :-2] != -1) & (loc_tgt[:, :, 1:-1] != -1) & (loc_tgt[:, :, 2:] != -1)
        loss_smooth = diffs[valid_mask].mean() if valid_mask.any() else loc_pred.new_zeros(1).squeeze()

        exist_mask = (exist_tgt > 0.5)
        tp_sel = type_pred[exist_mask]
        tt_sel = type_tgt[exist_mask]
        loss_type = F.cross_entropy(tp_sel, tt_sel) if tt_sel.numel() > 0 else loc_pred.new_zeros(1).squeeze()

        loss_exist = F.binary_cross_entropy_with_logits(exist_pred, exist_tgt)
        loss = self.loc_w * loss_loc + self.smooth_w * loss_smooth + self.type_w * loss_type + self.exist_w * loss_exist

        return loss, {'loc': loss_loc.item(), 'smooth': loss_smooth.item(), 'type': loss_type.item(), 'exist': loss_exist.item()}

# ── NMS 후처리 ────────────────────────────────────────
def apply_nms(loc_prob, exist_prob, type_pred, conf_thresh=0.25, grid_dist_thresh=2.0):
    """중복 감지된 차선 슬롯 중 가장 확률이 높은 것만 남김"""
    keep_indices = []
    # 존재 확률에 따라 내림차순 정렬
    sorted_idx = torch.argsort(exist_prob, descending=True).tolist()

    for idx in sorted_idx:
        if exist_prob[idx] < 0.5:
            continue

        prob_i = loc_prob[idx]                    # (NUM_ANCHORS, GRID_NUM)
        max_prob_i, max_idx_i = prob_i.max(dim=1) # (NUM_ANCHORS)
        valid_i = max_prob_i > conf_thresh

        is_duplicate = False
        for k_idx in keep_indices:
            prob_k = loc_prob[k_idx]
            max_prob_k, max_idx_k = prob_k.max(dim=1)
            valid_k = max_prob_k > conf_thresh

            common_valid = valid_i & valid_k
            if common_valid.sum() >= 5: # 5개 이상 앵커가 겹치면 비교
                diff = (max_idx_i[common_valid].float() - max_idx_k[common_valid].float()).abs().mean()
                if diff <= grid_dist_thresh:
                    is_duplicate = True
                    break

        if not is_duplicate:
            keep_indices.append(idx)

    return keep_indices

# ── 평가 및 시각화 ────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    class_correct = [0] * NUM_TYPES
    class_total   = [0] * NUM_TYPES

    for imgs, loc_tgt, type_tgt, exist_tgt in loader:
        imgs, loc_tgt, type_tgt, exist_tgt = imgs.to(DEVICE), loc_tgt.to(DEVICE), type_tgt.to(DEVICE), exist_tgt.to(DEVICE)
        loc_pred, type_pred, exist_pred = model(imgs)
        loss, _ = criterion(loc_pred, type_pred, exist_pred, loc_tgt, type_tgt, exist_tgt)
        total_loss += loss.item()

        mask = exist_tgt > 0.5
        if mask.any():
            pred = type_pred[mask].argmax(-1)
            tgt  = type_tgt[mask]
            for t in range(NUM_TYPES):
                m = (tgt == t)
                class_correct[t] += (pred[m] == t).sum().item()
                class_total[t]   += m.sum().item()

    total_correct = sum(class_correct[1:])
    total_total   = sum(class_total[1:])
    overall_acc   = total_correct / total_total if total_total > 0 else 0.0
    per_class_acc = {TYPE_NAMES[t]: class_correct[t] / class_total[t] if class_total[t] > 0 else float('nan') for t in range(1, NUM_TYPES)}

    return total_loss / len(loader), overall_acc, per_class_acc

@torch.no_grad()
def visualize_predictions(model, dataset, num_samples=6, save_path=None):
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    POLY_DEG = 2
    CONF_THRESH = 0.25

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for ax_i, idx in enumerate(indices):
        img_t, _, _, _ = dataset[idx]
        img_input = img_t.unsqueeze(0).to(DEVICE)
        loc_pred, type_pred, exist_pred = model(img_input)

        loc_prob   = loc_pred[0].softmax(-1).cpu()
        type_pred  = type_pred[0].cpu()
        exist_prob = torch.sigmoid(exist_pred[0]).cpu()

        img_np = (img_t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax = axes[ax_i]
        ax.imshow(img_np)
        ax.set_title(f'Sample #{idx}', fontsize=10)
        ax.axis('off')

        # NMS 적용
        keep_slots = apply_nms(loc_prob, exist_prob, type_pred, CONF_THRESH, grid_dist_thresh=2.0)

        for slot in keep_slots:
            pred_type = type_pred[slot].argmax().item()
            prob_ex   = exist_prob[slot].item()
            color     = TYPE_COLORS[pred_type]

            xs_raw, ys_raw = [], []
            for anchor_idx in range(NUM_ANCHORS):
                if loc_prob[slot, anchor_idx].max().item() < CONF_THRESH: continue
                gc = loc_prob[slot, anchor_idx].argmax().item()
                xs_raw.append(int((gc + 0.5) * INPUT_W / GRID_NUM))
                ys_raw.append(int(H_SAMPLES[anchor_idx] * INPUT_H / ORIG_H))

            if len(xs_raw) < 2: continue
            
            # 사전학습 모델의 예측은 자연스럽게 부드러우므로 억지 다항식 피팅(polyfit)을 제거합니다.
            ax.plot(xs_raw, ys_raw, '-', color=color, linewidth=2.5, alpha=0.85)
            ax.text(xs_raw[-1], ys_raw[-1], f'{TYPE_NAMES[pred_type]}\n{prob_ex:.2f}',
                    fontsize=5, color='white', bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    patches = [mpatches.Patch(color=TYPE_COLORS[i], label=TYPE_NAMES[i]) for i in range(1, NUM_TYPES)]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=9, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_history(history, save_path=None):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='#E74C3C')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss',   color='#3498DB')
    axes[0].set_title('Total Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_acc'], label='Val Acc', color='#3498DB')
    axes[1].set_title('Type Classification Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ── 학습 ─────────────────────────────────────────────
def main():
    train_ld, val_ld, test_ld = build_loaders()
    model = UFLDv2Classifier().to(DEVICE)

    # 사전학습 가중치 로드
    if Path(PRETRAINED_MODEL).exists():
        print(f'[Model] 사전학습 가중치 로드 중: {PRETRAINED_MODEL}')
        state = torch.load(PRETRAINED_MODEL, map_location='cpu')
        if 'model' in state: state = state['model']
        elif 'state_dict' in state: state = state['state_dict']

        # 백본 및 호환되는 key 매핑
        our_state = model.state_dict()
        matched = 0
        for k, v in state.items():
            # 보통 공식 가중치는 'net.model.layer1...' 형태
            k_mod = k.replace('module.', '').replace('net.', '')
            if k_mod.startswith('model.'):
                k_mod = k_mod[6:]  # 'model.' 제거

            # 기본 ResNet stem을 우리의 stem으로 매핑
            if k_mod.startswith('conv1.'):
                k_mod = k_mod.replace('conv1.', 'stem.0.')
            elif k_mod.startswith('bn1.'):
                k_mod = k_mod.replace('bn1.', 'stem.1.')
            
            # Location Head 매핑 (공식 명칭이 다를 수 있으므로 일치하는 모양만)
            if k_mod == 'pool.weight': k_mod = 'loc_reduce.0.weight'
            elif k_mod == 'pool.bias': k_mod = 'loc_reduce.0.bias'

            if k_mod in our_state and our_state[k_mod].shape == v.shape:
                our_state[k_mod] = v
                matched += 1
                
        model.load_state_dict(our_state, strict=False)
        print(f'[Model] {matched}개 파라미터 성공적으로 로드됨 (나머지 Type/Exist 헤드는 랜덤 초기화 유지)')
    else:
        print(f'[Model] 사전학습 가중치 없음 ({PRETRAINED_MODEL}). ImageNet 백본 가중치로만 시작합니다.')

    optimizer = optim.AdamW(
        [{'params': list(model.stem.parameters()) + list(model.layer1.parameters()) + list(model.layer2.parameters()) + list(model.layer3.parameters()), 'lr': LR * 0.1},
         {'params': list(model.loc_reduce.parameters()) + list(model.loc_cls.parameters()) + list(model.row_att.parameters()) + list(model.type_head.parameters()) + list(model.exist_head.parameters()), 'lr': LR}],
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = UFLDv2Loss(LOC_WEIGHT, TYPE_WEIGHT, EXIST_WEIGHT, SMOOTH_WEIGHT).to(DEVICE)
    scaler    = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_acc, best_epoch, no_improve = 0.0, 0, 0
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0.0
        for imgs, loc_tgt, type_tgt, exist_tgt in tqdm(train_ld, desc=f'Ep {epoch+1}', leave=False):
            imgs, loc_tgt, type_tgt, exist_tgt = imgs.to(DEVICE), loc_tgt.to(DEVICE), type_tgt.to(DEVICE), exist_tgt.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loc_pred, type_pred, exist_pred = model(imgs)
                loss, _ = criterion(loc_pred, type_pred, exist_pred, loc_tgt, type_tgt, exist_tgt)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            t_loss += loss.item()

        t_loss /= len(train_ld)
        v_loss, v_acc, _ = evaluate(model, val_ld, criterion)
        scheduler.step()

        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
        print(f'Ep {epoch+1:02d}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}')

        if v_acc > best_val_acc:
            best_val_acc, best_epoch, no_improve = v_acc, epoch+1, 0
            torch.save(model.state_dict(), f'{SAVE_DIR}/best_model.pth')
            print(f'  [★ BEST] Best model saved (val_acc={v_acc:.4f})')
        else:
            no_improve += 1
            bar = '#' * no_improve + '.' * (PATIENCE - no_improve)
            print(f'  [patience {no_improve:2d}/{PATIENCE}] [{bar}]  (best={best_val_acc:.4f} @ ep{best_epoch})')
            if no_improve >= PATIENCE:
                print(f'\n  [Early Stop] 조기 종료')
                break

    print('\n[테스트 평가 시작]')
    model.load_state_dict(torch.load(f'{SAVE_DIR}/best_model.pth'))
    test_loss, test_acc, test_per_class = evaluate(model, test_ld, criterion)
    print(f'Test Acc: {test_acc:.4f}')
    print('\nPer-class Accuracy:')
    for name, acc in test_per_class.items():
        bar = '█' * int(acc * 30) + '░' * (30 - int(acc * 30))
        print(f'  {name:20s} [{bar}] {acc:.4f}')

    results = {
        'best_epoch': best_epoch, 'best_val_acc': best_val_acc,
        'test_loss': test_loss, 'test_acc': test_acc, 'per_class_acc': test_per_class,
        'config': {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR},
        'history': history,
    }
    with open(f'{SAVE_DIR}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\n[결과 시각화 (NMS 적용됨)]')
    plot_history(history, save_path=f'{SAVE_DIR}/training_curves.png')
    visualize_predictions(model, test_ld.dataset, save_path=f'{SAVE_DIR}/predictions.png')

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
