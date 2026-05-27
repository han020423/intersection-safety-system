import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from bisenetv2 import BiSeNetV2
from dataset_bisenet import RoadStructureDataset


class DiceLoss(nn.Module):
    """Dice Loss for handling class imbalance in segmentation.
    Particularly effective for thin structures (lanes, stop lines)
    where background dominates 95%+ of pixels."""
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def compute_iou(preds, targets, num_classes):
    """
    Computes Intersection over Union (IoU) per class.
    preds: Model predictions shape (B, H, W)
    targets: Ground truth shape (B, H, W)
    """
    ious = []
    
    # Bincount mapping
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Exclude ignore_index (if any mapping applied, we use 255 typically, but here background is 0)
    valid_idx = (targets >= 0) & (targets < num_classes)
    preds = preds[valid_idx]
    targets = targets[valid_idx]
    
    # Construct Confusion Matrix
    bincount_2d = torch.bincount(num_classes * targets + preds, minlength=num_classes ** 2)
    conf_matrix = bincount_2d.reshape((num_classes, num_classes)).float()
    
    # IoU = intersection / (sum(pred) + sum(target) - intersection)
    intersection = torch.diag(conf_matrix)
    pred_sum = conf_matrix.sum(dim=0)
    target_sum = conf_matrix.sum(dim=1)
    union = target_sum + pred_sum - intersection
    
    # Calculate IoU per class skipping div by zero cases when class doesn't appear
    for i in range(num_classes):
        if target_sum[i] > 0 or pred_sum[i] > 0:
            iou = intersection[i] / (union[i] + 1e-10)
            ious.append(iou.item())
        else:
            ious.append(float('nan'))
            
    return np.nanmean(ious), ious

def test_evaluation(model, test_loader, device, num_classes, output_report_dir):
    """
    Evaluates the model over the final independent test set and writes the test report.
    """
    model.eval()
    model.aux_mode = 'eval'  # Output logits for proper IoU computation
    
    print("\n[테스트 데이터 세트로 최종 검증 시작]")
    
    total_ious = []
    total_class_ious = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)[0]
            preds = outputs.argmax(dim=1)
            
            batch_miou, batch_class_ious = compute_iou(preds, masks, num_classes)
            total_ious.append(batch_miou)
            for i, iou in enumerate(batch_class_ious):
                if not np.isnan(iou):
                    total_class_ious[i].append(iou)

    mean_iou = np.nanmean(total_ious)
    
    # Format the report string similarly to the YOLO logic requested
    cls_names = ['Background', 'White_Solid', 'White_Dotted', 'Yellow_Solid', 'Yellow_Dotted', 'Blue_Solid', 'Blue_Dotted', 'Crosswalk', 'Stop_Line']
    report_text = f"=== BiSeNetV2 모델 최종 테스트 세트 성능 지표 ===\n\n"
    report_text += "[전체 모델 분할 평가 지표 (All Classes)]\n"
    report_text += f"- mIoU (Mean Intersection over Union) : {mean_iou:.4f}\n\n"

    report_text += "[객체(클래스)별 병합 mIoU 성능 지표]\n"
    for i in range(num_classes):
        class_name = cls_names[i] if i < len(cls_names) else f"클래스 {i}"
        class_iou = np.mean(total_class_ious[i]) if total_class_ious[i] else 0.0
        report_text += f"▶ '{class_name}'\n"
        report_text += f"   - IoU    : {class_iou:.4f}\n"

    print("\n" + report_text)
    
    # Output File
    os.makedirs(output_report_dir, exist_ok=True)
    report_path = os.path.join(output_report_dir, "test_performance_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report_text)
        
    print("--------------------------------------------------")
    print(f"✅ 성능 평가지표를 터미널에 출력하였고, '{report_path}' 문서 파일로 저장했습니다!")

def train_model():
    # ── 1. Configuration ───────────────────────────────────────
    EPOCHS = 100
    BATCH_SIZE = 16 
    LEARNING_RATE = 0.0005  # Lowered from 0.001 to prevent AMP gradient explosion
    PATIENCE = 20 # Early Stopping patience
    NUM_CLASSES = 9 # 0: BG, 1~6: Lanes(Color+Type), 7: Crosswalk, 8: Stop_Line
    TARGET_SIZE = (352, 640) # (H, W) matches Albumentations. Height must be multiple of 32 for BiSeNetV2!
    LINE_THICKNESS = 16  # 16px original → 8px at training res → 1px at 1/8 feature map
    
    # Base configuration mimicking YOLO setup formats
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SPLIT_DIR_TRAIN = os.path.join(BASE_DIR, "Split_Data", "train")
    SPLIT_DIR_VAL   = os.path.join(BASE_DIR, "Split_Data", "val")
    SPLIT_DIR_TEST  = os.path.join(BASE_DIR, "Split_Data", "test")
    
    PROJECT_DIR = os.path.join(BASE_DIR, "runs", "bisenet_custom")
    WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms for speed
    print(f"Using device: {device} | Epochs: {EPOCHS} | Patience: {PATIENCE}")
    
    # ── 2. Data Augmentation via Albumentations ─────────────────
    # YOLO-like augmentations (flip, color distortion, small scale translations) synced across image and mask
    train_transform = A.Compose([
        A.Resize(TARGET_SIZE[0], TARGET_SIZE[1], p=1.0),
        A.HorizontalFlip(p=0.5), # 50% left-right flip
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4), # Color jitter
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-5, 5), p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(TARGET_SIZE[0], TARGET_SIZE[1], p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    print("Loading datasets...")
    train_dataset = RoadStructureDataset(SPLIT_DIR_TRAIN, target_size=TARGET_SIZE, line_thickness=LINE_THICKNESS, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    
    val_dataset = RoadStructureDataset(SPLIT_DIR_VAL, target_size=TARGET_SIZE, line_thickness=LINE_THICKNESS, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    test_dataset = RoadStructureDataset(SPLIT_DIR_TEST, target_size=TARGET_SIZE, line_thickness=LINE_THICKNESS, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # ── 3. Model Definition ─────────────────────────────────────
    model = BiSeNetV2(n_classes=NUM_CLASSES)
    model.to(device)
    
    # JIT compile for speed (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Combined loss: CE handles pixel-level classification, Dice handles class imbalance
    ce_criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_criterion = DiceLoss(num_classes=NUM_CLASSES)
    
    # Use model.get_params() for proper parameter grouping:
    # - No weight decay on BatchNorm parameters (prevents training instability)
    # - Higher learning rate on head/aux parameters (new layers need faster convergence)
    wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
    optimizer = optim.AdamW([
        {'params': wd_params, 'lr': LEARNING_RATE, 'weight_decay': 1e-4},
        {'params': nowd_params, 'lr': LEARNING_RATE, 'weight_decay': 0},
        {'params': lr_mul_wd_params, 'lr': LEARNING_RATE * 10, 'weight_decay': 1e-4},
        {'params': lr_mul_nowd_params, 'lr': LEARNING_RATE * 10, 'weight_decay': 0},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Variables for Early Stopping and Tracking
    best_miou = 0.0
    epochs_no_improve = 0
    best_weight_path = os.path.join(WEIGHTS_DIR, "best.pth")

    cls_names = ['BG', 'W_Solid', 'W_Dot', 'Y_Solid', 'Y_Dot', 'B_Solid', 'B_Dot', 'Crosswalk', 'StopLine']
    csv_path = os.path.join(PROJECT_DIR, "results.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8') as f:
            cls_header = ','.join([f'IoU_{name}' for name in cls_names])
            f.write(f"Epoch,Train_Loss,Val_Loss,Val_mIoU,{cls_header}\n")

    # ── 4. Main Training Loop ───────────────────────────────────
    print(f"\n[학습 시작] 프로젝트 경로: {PROJECT_DIR}")
    for epoch in range(EPOCHS):
        model.train()
        model.aux_mode = 'train'  
        epoch_loss = 0.0
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                # Forward pass (BiSeNetV2) Returns 5 logits
                logits_tuple = model(images)
                
                # CE + Dice combined loss for each head
                loss_main = ce_criterion(logits_tuple[0], masks) + dice_criterion(logits_tuple[0], masks)
                loss_aux2 = ce_criterion(logits_tuple[1], masks) + dice_criterion(logits_tuple[1], masks)
                loss_aux3 = ce_criterion(logits_tuple[2], masks) + dice_criterion(logits_tuple[2], masks)
                loss_aux4 = ce_criterion(logits_tuple[3], masks) + dice_criterion(logits_tuple[3], masks)
                loss_aux5 = ce_criterion(logits_tuple[4], masks) + dice_criterion(logits_tuple[4], masks)
                
                loss = loss_main + 0.4 * (loss_aux2 + loss_aux3 + loss_aux4 + loss_aux5)
                
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], T_Loss: {loss.item():.4f}")
                
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        
        # ── 5. Validation Loop ────────────────────────────────────
        model.eval()
        model.aux_mode = 'eval' # Outputs only main head
        val_loss = 0.0
        val_mious = []
        val_class_ious_acc = [[] for _ in range(NUM_CLASSES)]
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(images)[0]
                    batch_loss = ce_criterion(outputs, masks) + dice_criterion(outputs, masks)
                
                val_loss += batch_loss.item()
                
                preds = outputs.argmax(dim=1)
                batch_miou, batch_class_ious = compute_iou(preds, masks, NUM_CLASSES)
                val_mious.append(batch_miou)
                for ci, ciou in enumerate(batch_class_ious):
                    if not np.isnan(ciou):
                        val_class_ious_acc[ci].append(ciou)
                
        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_miou = np.nanmean(val_mious)
        avg_class_ious = [np.mean(lst) if lst else 0.0 for lst in val_class_ious_acc]
        
        # Step LR scheduler tracking validation mIoU map
        scheduler.step(avg_val_miou)
        
        cls_iou_csv = ','.join([f'{v:.4f}' for v in avg_class_ious])
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_val_miou:.4f},{cls_iou_csv}\n")
        
        cls_iou_str = ' | '.join([f'{cls_names[i]}:{avg_class_ious[i]:.3f}' for i in range(NUM_CLASSES)])
        print(f"==> Epoch {epoch+1} Completed | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f}")
        print(f"    [{cls_iou_str}]")
        
        # ── 6. Checkpoint & Early Stopping ─────────────────────────
        if avg_val_miou > best_miou:
            print(f"    ⭐ Validation mIoU improved from {best_miou:.4f} to {avg_val_miou:.4f}. Saving best.pth !")
            best_miou = avg_val_miou
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weight_path)
        else:
            epochs_no_improve += 1
            print(f"    - No improvements for {epochs_no_improve} epoch(s).")
            
        last_weight_path = os.path.join(WEIGHTS_DIR, "last.pth")
        torch.save(model.state_dict(), last_weight_path)
        
        if epochs_no_improve >= PATIENCE:
            print(f"\n🚨 Early stopping triggered! Patience limit ({PATIENCE} epochs) successfully halted overfitting training.")
            break

    # ── 7. Final Test Set Evaluation ─────────────────────────────
    print("\n[학습 종료]")
    print(f"최적 가중치 저장 위치: {best_weight_path}")
    
    # Load optimal model for testing execution reporting
    print("Executing best.pth context for evaluation...")
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path, map_location=device))
        
    test_evaluation(model, test_loader, device, NUM_CLASSES, output_report_dir=PROJECT_DIR)

if __name__ == "__main__":
    train_model()
