"""
YOLO11n 커스텀 학습 스크립트

실행 순서:
    1. python prepare_dataset.py   # 데이터 분할 (최초 1회)
    2. python train_yolo.py        # 학습

학습 결과는 runs/detect/train* 폴더에 저장됩니다.
"""

from pathlib import Path
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────
DATA_YAML   = Path(__file__).parent / "dataset" / "data.yaml"
MODEL       = "yolo11n.pt"      # 사전학습 가중치 (없으면 자동 다운로드)

EPOCHS      = 100               # 에포크 수 (데이터가 적으므로 100 권장)
IMGSZ       = 640               # 입력 이미지 크기
BATCH       = 16                # 배치 크기 (GPU VRAM에 맞게 조절)
PATIENCE    = 20                # Early stopping: val mAP가 20 epoch 개선 없으면 중단
DEVICE      = ""                # "" → 자동(GPU 있으면 GPU, 없으면 CPU)
WORKERS     = 4                 # dataloader worker 수
PROJECT     = "runs/detect"     # 결과 저장 폴더
NAME        = "yolo11n_custom"  # 실험 이름
# ─────────────────────────────────────────────────────────


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml 이 없습니다: {DATA_YAML}\n"
            "먼저 'python prepare_dataset.py' 를 실행하세요."
        )

    print(f"[학습 시작] 모델: {MODEL} | 에포크: {EPOCHS} | 배치: {BATCH} | 이미지크기: {IMGSZ}")
    print(f"데이터: {DATA_YAML}\n")

    model = YOLO(MODEL)

    results = model.train(
        data      = str(DATA_YAML),
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        patience  = PATIENCE,
        device    = DEVICE,
        workers   = WORKERS,
        project   = PROJECT,
        name      = NAME,
        exist_ok  = True,
        # 데이터가 적을 때 도움이 되는 augmentation
        augment   = True,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 5.0,
        translate = 0.1,
        scale     = 0.5,
        flipud    = 0.0,
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.1,
    )

    print("\n[학습 완료]")
    print(f"최적 가중치: {Path(PROJECT) / NAME / 'weights' / 'best.pt'}")

    # ── 테스트 세트 평가 ────────────────────────────────
    print("\n[테스트 세트 평가]")
    best_model = YOLO(str(Path(PROJECT) / NAME / "weights" / "best.pt"))
    metrics = best_model.val(
        data   = str(DATA_YAML),
        split  = "test",
        imgsz  = IMGSZ,
        device = DEVICE,
    )
    print(f"mAP50   : {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
