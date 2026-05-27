# ╔══════════════════════════════════════════════════════════════════╗
# ║  UFLDv2 테스트 추론 스크립트 (이미지/비디오)                     ║
# ║  - 학습된 best_model.pth를 이용해 새로운 사진/영상에 차선 시각화 ║
# ╚══════════════════════════════════════════════════════════════════╝

import subprocess, sys

def _pip(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

_pip('opencv-python')

import os, json
import numpy as np
import cv2
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# ── ⚙️ 설정 ──────────────────────────────────────────
# 추론할 파일 경로 (이미지 또는 비디오)
INPUT_PATH  = '/content/drive/MyDrive/capstone/ufld/test_video.mp4'  # 테스트할 영상/사진 경로로 변경하세요!
OUTPUT_PATH = '/content/drive/MyDrive/capstone/ufld/output_video.mp4'

# 학습된 모델 가중치 경로
MODEL_WEIGHTS = '/content/drive/MyDrive/capstone/ufld/ufldv2_output_v2/best_model.pth'

# 학습 시 사용했던 설정 (변경 금지)
INPUT_H    = 288
INPUT_W    = 512
NUM_ANCHORS = 36
GRID_NUM   = 64
MAX_LANES  = 8
NUM_TYPES  = 7
H_SAMPLES  = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510] # 원본 h_samples (가정)
ORIG_H     = 720
ORIG_W     = 1280

TYPE_NAMES = [
    'none', 'white-solid', 'white-dotted',
    'yellow-solid', 'yellow-dotted', 'blue-solid', 'blue-dotted',
]

# OpenCV BGR 색상 코드 (Blue, Green, Red)
TYPE_COLORS_BGR = [
    (136, 136, 136), # none
    (255, 255, 255), # white-solid
    (221, 221, 221), # white-dotted
    (0, 215, 255),   # yellow-solid
    (0, 165, 255),   # yellow-dotted
    (255, 144, 30),  # blue-solid
    (235, 206, 135)  # blue-dotted
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 모델 아키텍처 (학습 코드와 동일) ─────────────────────────
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
        bb = tvmodels.resnet18(pretrained=False)
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

# ── NMS 후처리 ────────────────────────────────────────
def apply_nms(loc_prob, exist_prob, type_pred, conf_thresh=0.08, grid_dist_thresh=2.0):
    keep_indices = []
    sorted_idx = torch.argsort(exist_prob, descending=True).tolist()

    for idx in sorted_idx:
        if exist_prob[idx] < 0.5:
            continue

        prob_i = loc_prob[idx]
        max_prob_i, max_idx_i = prob_i.max(dim=1)
        valid_i = max_prob_i > conf_thresh

        is_duplicate = False
        for k_idx in keep_indices:
            prob_k = loc_prob[k_idx]
            max_prob_k, max_idx_k = prob_k.max(dim=1)
            valid_k = max_prob_k > conf_thresh

            common_valid = valid_i & valid_k
            if common_valid.sum() >= 5:
                diff = (max_idx_i[common_valid].float() - max_idx_k[common_valid].float()).abs().mean()
                if diff <= grid_dist_thresh:
                    is_duplicate = True
                    break

        if not is_duplicate:
            keep_indices.append(idx)

    return keep_indices

# ── 추론 파이프라인 ──────────────────────────────────
def init_model():
    print(f'[Init] 모델 로드 중: {MODEL_WEIGHTS}')
    model = UFLDv2Classifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    return model

def process_frame(model, frame_bgr):
    # 1. 전처리
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    transform = T.Compose([
        T.Resize((INPUT_H, INPUT_W)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    # 2. 모델 예측
    with torch.no_grad():
        loc_pred, type_pred, exist_pred = model(img_tensor)

    loc_prob   = loc_pred[0].softmax(-1).cpu()
    type_pred  = type_pred[0].cpu()
    exist_prob = torch.sigmoid(exist_pred[0]).cpu()

    # 3. NMS 적용
    CONF_THRESH = 0.08
    keep_slots = apply_nms(loc_prob, exist_prob, type_pred, CONF_THRESH, grid_dist_thresh=2.0)

    # 4. 프레임에 그리기 (원본 이미지 사이즈 기준)
    draw_frame = frame_bgr.copy()
    h, w = draw_frame.shape[:2]

    for slot in keep_slots:
        pred_type = type_pred[slot].argmax().item()
        prob_ex   = exist_prob[slot].item()
        color_bgr = TYPE_COLORS_BGR[pred_type]

        xs_raw, ys_raw = [], []
        for anchor_idx in range(NUM_ANCHORS):
            if loc_prob[slot, anchor_idx].max().item() < CONF_THRESH: continue
            gc = loc_prob[slot, anchor_idx].argmax().item()
            
            # 예측된 그리드 셀을 원본 픽셀 좌표로 역산
            x_pixel = int((gc + 0.5) * w / GRID_NUM)
            y_pixel = int(H_SAMPLES[anchor_idx] * h / ORIG_H)
            
            xs_raw.append(x_pixel)
            ys_raw.append(y_pixel)

        if len(xs_raw) < 2: continue

        # 선 그리기
        pts = np.array(list(zip(xs_raw, ys_raw)), np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(draw_frame, [pts], isClosed=False, color=color_bgr, thickness=8)

        # 텍스트 박스 추가
        label = f"{TYPE_NAMES[pred_type]} {prob_ex:.2f}"
        cv2.putText(draw_frame, label, (xs_raw[-1] + 10, ys_raw[-1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        
    return draw_frame

def main():
    model = init_model()

    ext = Path(INPUT_PATH).suffix.lower()
    
    if ext in ['.jpg', '.png', '.jpeg']:
        # 이미지 테스트
        print(f'[Test] 이미지 추론: {INPUT_PATH}')
        img = cv2.imread(INPUT_PATH)
        res_img = process_frame(model, img)
        cv2.imwrite(OUTPUT_PATH, res_img)
        print(f'[Done] 결과 저장 완료: {OUTPUT_PATH}')
        
        # Colab 화면에 띄우기
        res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 7))
        plt.imshow(res_rgb)
        plt.axis('off')
        plt.show()

    elif ext in ['.mp4', '.avi']:
        # 비디오 테스트
        print(f'[Test] 비디오 추론: {INPUT_PATH}')
        cap = cv2.VideoCapture(INPUT_PATH)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

        frame_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            res_frame = process_frame(model, frame)
            out.write(res_frame)
            
            frame_cnt += 1
            if frame_cnt % 30 == 0:
                print(f"  처리 중... {frame_cnt}/{total_frames} 프레임")

        cap.release()
        out.release()
        print(f'[Done] 비디오 결과 저장 완료: {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
