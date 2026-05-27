import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==============================================================================
# 1. BiSeNetV2 Model Architecture
# ==============================================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2, padding=1),
            ConvBNReLU(128, 128, 3, stride=1, padding=1),
            ConvBNReLU(128, 128, 3, stride=1, padding=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat

class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 32, 3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(48, 16, 3, stride=1, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        left = self.left(feat)
        right = self.right(feat)
        feat = torch.cat([left, right], dim=1)
        return self.fuse(feat)

class CEBlock(nn.Module):
    def __init__(self):
        super(CEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBNReLU(128, 128, 1, stride=1, padding=0)

    def forward(self, x):
        gap = self.gap(x)
        gap = self.conv(gap)
        return x + gap

class GatherExpansionLayer(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6, stride=1):
        super(GatherExpansionLayer, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.stride = stride
        
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1)
        
        if stride == 2:
            self.dwconv1 = nn.Sequential(
                nn.Conv2d(in_chan, mid_chan, 3, stride=stride, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.Conv2d(mid_chan, mid_chan, 3, stride=1, padding=1, groups=mid_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, 3, stride=2, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(in_chan, out_chan, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        else:
            self.dwconv1 = nn.Sequential(
                nn.Conv2d(in_chan, mid_chan, 3, stride=1, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
            )
            self.shortcut = nn.Identity()

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, 3, stride=1, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        return self.relu(feat + shortcut)

class SemanticBranch(nn.Module):
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.stem = StemBlock()
        self.S3 = nn.Sequential(
            GatherExpansionLayer(16, 32, stride=2),
            GatherExpansionLayer(32, 32, stride=1),
        )
        self.S4 = nn.Sequential(
            GatherExpansionLayer(32, 64, stride=2),
            GatherExpansionLayer(64, 64, stride=1),
        )
        self.S5 = nn.Sequential(
            GatherExpansionLayer(64, 128, stride=2),
            GatherExpansionLayer(128, 128, stride=1),
            GatherExpansionLayer(128, 128, stride=1),
            GatherExpansionLayer(128, 128, stride=1),
        )
        self.ce = CEBlock()

    def forward(self, x):
        feat2 = self.stem(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5 = self.S5(feat4)
        feat5 = self.ce(feat5)
        return feat2, feat3, feat4, feat5

class BGA(nn.Module):
    def __init__(self):
        super(BGA, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
        )
        self.conv = ConvBNReLU(128, 128, 3, stride=1, padding=1)

    def forward(self, detail, semantic):
        left1 = self.left1(detail)
        left2 = self.left2(detail)
        right1 = self.right1(semantic)
        right2 = self.right2(semantic)
        
        right1 = F.interpolate(right1, size=left1.shape[2:], mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        
        right2 = F.interpolate(right2, size=left2.shape[2:], mode='bilinear', align_corners=True)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(right, size=left.shape[2:], mode='bilinear', align_corners=True)
        
        out = self.conv(left + right)
        return out

class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1, padding=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, 1, stride=1, padding=0)
        self.up_factor = up_factor

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, scale_factor=self.up_factor, mode='bilinear', align_corners=True)
        return feat

class BiSeNetV2(nn.Module):
    def __init__(self, n_classes):
        super(BiSeNetV2, self).__init__()
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        self.bga = BGA()
        
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8)
        self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
        self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
        self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
        self.aux5 = SegmentHead(128, 128, n_classes, up_factor=32)
        
        self.aux_mode = 'train'

    def forward(self, x):
        detail_feat = self.detail(x)
        feat2, feat3, feat4, feat5 = self.semantic(x)
        out = self.bga(detail_feat, feat5)
        
        out_head = self.head(out)
        
        if self.aux_mode == 'train':
            out_aux2 = self.aux2(feat2)
            out_aux3 = self.aux3(feat3)
            out_aux4 = self.aux4(feat4)
            out_aux5 = self.aux5(feat5)
            return out_head, out_aux2, out_aux3, out_aux4, out_aux5
        else:
            return out_head

# ==============================================================================
# 2. Dataset Definition
# ==============================================================================
class ColabRoadDataset(Dataset):
    def __init__(self, data_dir, target_size=(352, 640), transform=None):
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "labels")
        self.target_size = target_size
        self.transform = transform
        self.line_thickness = 12 # 12px 원본 크기 (리사이즈 후 약 6px). 특징맵 소실 방지와 정밀도의 최적 타협점.
        
        # 클래스 매핑: 0=Background, 1=traffic_lane, 2=crosswalk, 3=stop_line
        self.class_map = {
            "traffic_lane": 1,
            "crosswalk": 2,
            "stop_line": 3
        }

        self.samples = []
        if os.path.exists(self.label_dir):
            for file_name in os.listdir(self.label_dir):
                if file_name.endswith('.json'):
                    j_path = os.path.join(self.label_dir, file_name)
                    img_path = os.path.join(self.image_dir, file_name.replace('.json', '.jpg'))
                    if os.path.exists(img_path):
                        self.samples.append((img_path, j_path))
        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def draw_mask(self, h, w, annotations):
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations:
            cls_name = ann.get('class')
            pts_data = ann.get('data', [])
            if not pts_data: continue
                
            pts = np.array([[pt['x'], pt['y']] for pt in pts_data], np.int32).reshape((-1, 1, 2))
            
            if cls_name == "traffic_lane":
                lane_color = 'white'
                lane_type = 'solid'
                
                attrs = ann.get('attributes', [])
                if isinstance(attrs, list):
                    for attr in attrs:
                        if attr.get('code') == 'lane_color':
                            lane_color = attr.get('value')
                        elif attr.get('code') == 'lane_type':
                            lane_type = attr.get('value')
                            
                color_id = 1
                if lane_color == 'white': color_id = 1
                elif lane_color == 'yellow': color_id = 3
                elif lane_color == 'blue': color_id = 5
                
                type_offset = 0 if lane_type == 'solid' else 1
                final_val = color_id + type_offset
                
                cv2.polylines(mask, [pts], isClosed=False, color=final_val, thickness=self.line_thickness)
                
            elif cls_name == "crosswalk":
                cv2.fillPoly(mask, [pts], color=7)
                
            elif cls_name == "stop_line":
                cv2.polylines(mask, [pts], isClosed=False, color=8, thickness=self.line_thickness)
                
        return mask

    def __getitem__(self, idx):
        img_path, j_path = self.samples[idx]
        
        # 이미지 읽기
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # JSON 라벨 읽기
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 마스크 생성
        mask = self.draw_mask(h, w, data.get('annotations', []))
        
        # 데이터 증강 및 리사이즈 적용
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].long()
            
        return img, mask

# ==============================================================================
# 3. Training & Evaluation Utils
# ==============================================================================
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

def compute_iou(preds, targets, num_classes):
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    valid_idx = (targets >= 0) & (targets < num_classes)
    preds = preds[valid_idx]
    targets = targets[valid_idx]
    
    bincount_2d = torch.bincount(num_classes * targets + preds, minlength=num_classes ** 2)
    conf_matrix = bincount_2d.reshape((num_classes, num_classes)).float()
    
    intersection = torch.diag(conf_matrix)
    pred_sum = conf_matrix.sum(dim=0)
    target_sum = conf_matrix.sum(dim=1)
    union = target_sum + pred_sum - intersection
    
    ious = []
    for i in range(num_classes):
        if target_sum[i] > 0 or pred_sum[i] > 0:
            ious.append((intersection[i] / (union[i] + 1e-10)).item())
        else:
            ious.append(float('nan'))
            
    return np.nanmean(ious), ious

# ==============================================================================
# 4. Main Execution Function for Colab
# ==============================================================================
def run_training(dataset_root="/content/dataset"):
    # 설정값
    NUM_CLASSES = 9 # 0:BG, 1~6:Lanes, 7:crosswalk, 8:stop_line
    BATCH_SIZE = 16
    EPOCHS = 100
    TARGET_SIZE = (352, 640) # BiSeNetV2는 Height/Width가 32의 배수여야 함
    LR = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Augmentations
    train_transform = A.Compose([
        A.Resize(TARGET_SIZE[0], TARGET_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(TARGET_SIZE[0], TARGET_SIZE[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # 데이터로더
    train_ds = ColabRoadDataset(os.path.join(dataset_root, "train"), target_size=TARGET_SIZE, transform=train_transform)
    val_ds = ColabRoadDataset(os.path.join(dataset_root, "val"), target_size=TARGET_SIZE, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 모델 및 손실 함수
    model = BiSeNetV2(n_classes=NUM_CLASSES).to(device)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes=NUM_CLASSES)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_miou = 0.0
    patience = 20
    epochs_no_improve = 0
    os.makedirs("weights", exist_ok=True)
    
    cls_names = ['BG', 'W_Solid', 'W_Dot', 'Y_Solid', 'Y_Dot', 'B_Solid', 'B_Dot', 'Crosswalk', 'StopLine']
    
    # CSV 로그 파일 초기화
    csv_path = "results.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        cls_header = ','.join([f'IoU_{name}' for name in cls_names])
        f.write(f"Epoch,Train_Loss,Val_mIoU,{cls_header}\n")
    
    print("\n--- Training Started ---")
    for epoch in range(EPOCHS):
        # 1. Train
        model.train()
        model.aux_mode = 'train'
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                logits = model(imgs)
                loss_main = ce_loss(logits[0], masks) + dice_loss(logits[0], masks)
                loss_aux = sum([ce_loss(out, masks) + dice_loss(out, masks) for out in logits[1:]])
                loss = loss_main + 0.4 * loss_aux
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # 2. Validation
        model.eval()
        model.aux_mode = 'eval'
        val_mious = []
        val_class_ious_acc = [[] for _ in range(NUM_CLASSES)]
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                    outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                batch_miou, batch_class_ious = compute_iou(preds, masks, NUM_CLASSES)
                val_mious.append(batch_miou)
                for ci, ciou in enumerate(batch_class_ious):
                    if not np.isnan(ciou):
                        val_class_ious_acc[ci].append(ciou)
                
        avg_val_miou = np.nanmean(val_mious)
        avg_class_ious = [np.mean(lst) if lst else 0.0 for lst in val_class_ious_acc]
        
        # CSV에 기록
        cls_iou_csv = ','.join([f'{v:.4f}' for v in avg_class_ious])
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1},{train_loss/len(train_loader):.4f},{avg_val_miou:.4f},{cls_iou_csv}\n")
            
        cls_iou_str = ' | '.join([f'{cls_names[i]}:{avg_class_ious[i]:.3f}' for i in range(NUM_CLASSES)])
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss/len(train_loader):.4f} | Val mIoU: {avg_val_miou:.4f}")
        print(f"    [{cls_iou_str}]")
        
        # Save Best & Early Stopping
        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            epochs_no_improve = 0
            torch.save(model.state_dict(), "weights/bisenetv2_best.pth")
            print("  --> 🌟 Saved Best Model!")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\n🚨 Early stopping triggered after {patience} epochs without improvement!")
            break

    # 3. Final Test Evaluation
    print("\n=======================================================")
    print("🚀 [최종 테스트 데이터 세트 평가 시작]")
    test_ds = ColabRoadDataset(os.path.join(dataset_root, "test"), target_size=TARGET_SIZE, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Load Best Model
    best_path = "weights/bisenetv2_best.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        
    model.eval()
    model.aux_mode = 'eval'
    test_mious = []
    test_class_ious_acc = [[] for _ in range(NUM_CLASSES)]
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            batch_miou, batch_class_ious = compute_iou(preds, masks, NUM_CLASSES)
            test_mious.append(batch_miou)
            for ci, ciou in enumerate(batch_class_ious):
                if not np.isnan(ciou):
                    test_class_ious_acc[ci].append(ciou)
                    
    final_miou = np.nanmean(test_mious)
    final_class_ious = [np.mean(lst) if lst else 0.0 for lst in test_class_ious_acc]
    
    report_text = f"\n[최종 성능 지표 (Final Test Report)]\n"
    report_text += f"⭐ 전체 모델 mIoU: {final_miou:.4f}\n\n"
    for i in range(NUM_CLASSES):
        report_text += f"  - {cls_names[i]}: {final_class_ious[i]:.4f}\n"
    report_text += "=======================================================\n"
    
    print(report_text)
    
    with open("test_performance_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("✅ 최종 리포트가 'test_performance_report.txt' 에 성공적으로 저장되었습니다!")

if __name__ == "__main__":
    # 코랩 환경에서 실행 시 구글 드라이브 마운트 후 
    # 데이터셋 경로를 알맞게 지정해 주세요. 
    # 기본 경로는 '/content/dataset' 으로 설정되어 있습니다.
    
    run_training(dataset_root="/content/dataset")
