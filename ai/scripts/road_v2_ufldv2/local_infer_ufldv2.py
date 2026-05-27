import argparse
import os
import sys
import time
import math
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchvision.transforms as T
from PIL import Image

# ── ⚙️ 설정 ──────────────────────────────────────────
# 학습 시 사용했던 설정 (변경 금지)
INPUT_H    = 288
INPUT_W    = 512
NUM_ANCHORS = 36
GRID_NUM   = 64
MAX_LANES  = 8
NUM_TYPES  = 7
H_SAMPLES  = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510]
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

# ── 래퍼 클래스 (road_perception 스타일) ──────────────────────
class UFLDv2LaneDetector:
    """UFLDv2 기반 차선 탐지 및 분류기"""
    def __init__(self, weights_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"[Info] 로컬 모델 초기화 중... Device: {self.device}")
        
        self.model = UFLDv2Classifier().to(self.device)
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"가중치 파일이 존재하지 않습니다: {weights_path}")
            
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize((INPUT_H, INPUT_W)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _apply_nms(self, loc_prob, exist_prob, type_pred, conf_thresh=0.25, grid_dist_thresh=2.0):
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

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """프레임에서 차선을 검출합니다. (검출 결과 리스트, 소요 시간 ms)"""
        t0 = time.perf_counter()
        
        # 전처리
        orig_h, orig_w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # 모델 추론
        loc_pred, type_pred, exist_pred = self.model(img_tensor)

        loc_prob   = loc_pred[0].softmax(-1).cpu()
        type_pred  = type_pred[0].cpu()
        exist_prob = torch.sigmoid(exist_pred[0]).cpu()

        # NMS 후처리
        CONF_THRESH = 0.25
        keep_slots = self._apply_nms(loc_prob, exist_prob, type_pred, CONF_THRESH, grid_dist_thresh=2.0)

        # 결과 파싱
        lanes = []
        for slot in keep_slots:
            pred_type = type_pred[slot].argmax().item()
            prob_ex   = exist_prob[slot].item()

            xs_raw, ys_raw = [], []
            for anchor_idx in range(NUM_ANCHORS):
                if loc_prob[slot, anchor_idx].max().item() < CONF_THRESH: 
                    continue
                gc = loc_prob[slot, anchor_idx].argmax().item()
                
                # 예측된 그리드 셀을 원본 프레임 크기에 맞춰 역투영
                x_pixel = int((gc + 0.5) * orig_w / GRID_NUM)
                y_pixel = int(H_SAMPLES[anchor_idx] * orig_h / ORIG_H)
                
                xs_raw.append(x_pixel)
                ys_raw.append(y_pixel)

            if len(xs_raw) >= 2:
                lanes.append({
                    'type_id': pred_type,
                    'type_name': TYPE_NAMES[pred_type],
                    'prob': prob_ex,
                    'points': list(zip(xs_raw, ys_raw)),
                    'color': TYPE_COLORS_BGR[pred_type]
                })

        elapsed = (time.perf_counter() - t0) * 1000.0
        return lanes, elapsed


# ── 메인 파이프라인 ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UFLDv2 Local Lane Detection Inference")
    parser.add_argument('--source', type=str, required=True, help="입력 영상 파일 또는 이미지 경로")
    parser.add_argument('--weights', type=str, required=True, help="학습된 best_model.pth 경로")
    parser.add_argument('--output', type=str, default=None, help="출력 결과 저장 경로 (지정하지 않으면 저장 안 함)")
    parser.add_argument('--device', type=str, default='cuda', help="cuda 또는 cpu")
    parser.add_argument('--no_show', action='store_true', help="화면 출력을 끌 경우 사용")
    args = parser.parse_args()

    detector = UFLDv2LaneDetector(weights_path=args.weights, device=args.device)

    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[Error] 입력 소스를 열 수 없습니다: {args.source}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or math.isnan(fps): fps = 30.0

    print(f"[Info] 입력 소스 해상도: {w}x{h}, FPS: {fps:.2f}, 총 프레임: {total_frames}")

    # 비디오 라이터 초기화 (선택 사항)
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"[Info] 결과가 다음 경로에 저장됩니다: {args.output}")

    frame_count = 0
    total_time_ms = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 1. 차선 추론 (Infer)
        lanes, elapsed_ms = detector.infer(frame)
        total_time_ms += elapsed_ms

        # 2. 결과 시각화
        vis_frame = frame.copy()
        
        for lane in lanes:
            # 점들 이어 그리기
            pts = np.array(lane['points'], np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_frame, [pts], isClosed=False, color=lane['color'], thickness=8)

            # 라벨 텍스트 표시
            label = f"{lane['type_name']} {lane['prob']:.2f}"
            last_x, last_y = lane['points'][-1]
            cv2.putText(vis_frame, label, (last_x + 10, last_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # 시스템 상태 표시 (FPS 등)
        avg_infer_ms = total_time_ms / frame_count
        status_text = f"Frame: {frame_count} | Infer: {elapsed_ms:.1f}ms (Avg: {avg_infer_ms:.1f}ms) | Lanes: {len(lanes)}"
        cv2.putText(vis_frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 3. 저장 및 화면 출력
        if out:
            out.write(vis_frame)

        if not args.no_show:
            # 화면에 맞게 창 크기 축소 표시 (선택)
            disp_frame = cv2.resize(vis_frame, (1280, 720)) if w > 1280 else vis_frame
            cv2.imshow("UFLDv2 Inference", disp_frame)
            # 이미지 파일 1장인 경우 화면 유지(무한대기), 영상이면 1ms 대기
            delay = 0 if total_frames == 1 else 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                print("[Info] 사용자 종료 (q 입력됨)")
                break

        if frame_count % 30 == 0:
            print(f"진행: {frame_count}/{total_frames} 프레임 처리 완료 (평균 추론 시간: {avg_infer_ms:.2f}ms)")

    # 리소스 해제
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    if args.output:
        print(f"[Done] 테스트 완료. 저장 경로: {args.output}")
    else:
        print("[Done] 실시간 테스트 완료.")

if __name__ == '__main__':
    main()
