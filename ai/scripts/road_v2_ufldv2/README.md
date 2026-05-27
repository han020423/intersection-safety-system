# UFLDv2 차선 분류 학습 파이프라인

이 디렉토리는 `road_v2` 데이터셋을 기반으로  
**UFLDv2 (Ultra Fast Lane Detection v2)** 모델을 사용하여  
차선 색상·유형을 분류하는 학습 파이프라인을 포함합니다.

---

## 🗂️ 파일 구성

| 파일 | 설명 | 실행 환경 |
|---|---|---|
| `prepare_dataset.py` | road_v2 → UFLDv2 포맷 변환 + 10,000개 균등 샘플링 | **로컬 (Windows)** |
| `colab_train_ufldv2.py` | 모델 정의 + 학습 + 평가 + 시각화 | **Google Colab** |

---

## 🏷️ Lane Category 매핑

| ID | 이름 | 설명 |
|---|---|---|
| 0 | none | 배경 (차선 없음) |
| 1 | white-solid | 흰색 실선 |
| 2 | white-dotted | 흰색 점선 |
| 3 | yellow-solid | 노란색 실선 |
| 4 | yellow-dotted | 노란색 점선 |
| 5 | blue-solid | 파란색 실선 |
| 6 | blue-dotted | 파란색 점선 |

> 횡단보도(`crosswalk`)와 정지선(`stop_line`)은 처리하지 않습니다.

---

## ⚙️ 아키텍처 요약

```
입력 이미지  (B, 3, 288, 512)
      │
  ResNet-18 Backbone (pretrained, ImageNet)
      ├─ stem + layer1 + layer2 → (B, 128, 36, 64)  ← stride-8, 핵심 feature
      └─ layer3               → (B, 256, 18, 32)   ← type/exist 보조 feature
      │
      ├─ Location Head
      │   1×1 conv → (B, MAX_LANES=6, 36, 64)
      │   Softmax over dim=-1 → 각 슬롯·앵커에서의 x-grid 확률
      │
      ├─ Type Head
      │   GlobalAvgPool → FC → (B, 6, 7)
      │   7 = none + 6 lane types
      │
      └─ Existence Head
          GlobalAvgPool → FC → Sigmoid → (B, 6)
```

**학습 손실** = `loc_loss × 1.0` + `type_loss × 0.6` + `exist_loss × 0.2`

---

## 🚀 STEP 1: 로컬 데이터 전처리

```bash
# 의존성 설치
pip install tqdm numpy

# 실행
cd ai/scripts/road_v2_ufldv2
python prepare_dataset.py \
    --data_root    ../road_v2 \
    --output_dir   ./ufldv2_dataset \
    --num_samples  10000 \
    --train_ratio  0.8 \
    --val_ratio    0.1

# PowerShell로 압축
Compress-Archive -Path ufldv2_dataset -DestinationPath ufldv2_dataset.zip
```

출력 구조:
```
ufldv2_dataset/
├── images/train/  (8,000장)
├── images/val/    (1,000장)
├── images/test/   (1,000장)
├── train.json
├── val.json
├── test.json
└── config.json
```

균등 샘플링 기준 (dominant lane 카테고리):
- 각 카테고리마다 `10,000 / 카테고리수` 장 할당
- 데이터 부족 시 과샘플링(oversampling) 자동 적용

---

## 🚀 STEP 2: Google Drive 업로드

1. `ufldv2_dataset.zip` → Google Drive 루트에 업로드
2. `colab_train_ufldv2.py` → Google Drive 또는 Colab에 업로드

---

## 🚀 STEP 3: Colab 학습

### 파일 업로드 및 실행

```python
# Colab 셀에서:
!python colab_train_ufldv2.py
```

### 경로 설정 (`colab_train_ufldv2.py` SECTION 3)
```python
DRIVE_ZIP = '/content/drive/MyDrive/ufldv2_dataset.zip'  # zip 위치
WORK_DIR  = '/content/ufldv2_dataset'                    # 압축 해제 위치
SAVE_DIR  = '/content/drive/MyDrive/ufldv2_output'       # 결과 저장 위치
```

### 하이퍼파라미터 (수정 가능)
```python
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 1e-3
```

---

## 📊 결과물

학습 완료 후 `SAVE_DIR`에 저장됩니다:

| 파일 | 내용 |
|---|---|
| `best_model.pth` | 최적 모델 가중치 + config |
| `results.json` | 손실·정확도·per-class 지표 |
| `training_curves.png` | 학습/검증 곡선 |
| `predictions.png` | 테스트셋 예측 시각화 (6샘플) |

---

## 📐 구현 세부사항

| 항목 | 값 |
|---|---|
| 입력 해상도 | 288 × 512 (H × W) |
| Row Anchors | 36개 (원본 y=[0, 20, 40, ..., 700]) |
| X Grid 셀 | 64개 (정밀도 ≈ 20px/셀) |
| 최대 차선 수 | 6 슬롯 |
| Backbone | ResNet-18 (pretrained) |
| Optimizer | AdamW (백본 LR × 0.1) |
| Scheduler | CosineAnnealing |
| AMP | ✅ (Colab GPU 자동 활성화) |

---

## ⚠️ 주의사항

- `road_v2` 데이터에 파란색(`blue`) 차선이 없거나 매우 적을 경우  
  해당 카테고리는 자동으로 과샘플링 처리됩니다.  
  카테고리 자체가 0개이면 균등 분배에서 제외됩니다.

- 횡단보도·정지선은 `prepare_dataset.py`에서 완전히 제외됩니다.

- Colab T4 GPU 기준 예상 학습 시간: **약 1~2시간** (30 epochs, batch=16)
