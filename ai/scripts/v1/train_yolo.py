"""
YOLO11n 커스텀 학습 및 성능 문서화 스크립트

실행 순서:
    1. 데이터 준비 스크립트 실행 (organize_dataset.py)
    2. python train_yolo.py        # 본 스크립트로 학습

학습 결과는 runs/detect/yolo11n_custom 폴더에,
최종 성능 평가 문서는 파일과 같은 폴더의 test_performance_report.txt 에 저장됩니다.
"""

from pathlib import Path
from ultralytics import YOLO

# ── 1. 하이퍼파라미터 및 경로 설정 ─────────────────────────
CURRENT_DIR = Path(__file__).parent
DATA_YAML   = CURRENT_DIR / "yolo_dataset" / "data.yaml" # 위 경로에 맞춰 수정됨
MODEL       = "yolo11n.pt"      # 사전학습 가중치 (없으면 자동 다운로드)

EPOCHS      = 100               # 에포크 수 (과적합 방지를 위해 100 이상 권장, patience 작동함)
IMGSZ       = 640               # 입력 이미지 크기
BATCH       = 16                # 배치 크기 
PATIENCE    = 20                # Early stopping: 검증단계(val) mAP가 20 epoch 동안 개선 안되면 자동 종료
DEVICE      = ""                # "" → 자동(GPU 있으면 우선 사용, 없으면 CPU로 자동 전환되어 에러 방지)
WORKERS     = 4                 # dataloader worker 수 (CPU 멀티프로세싱 자원)
PROJECT     = str(CURRENT_DIR / "runs" / "detect") # 결과 저장 폴더
NAME        = "yolo11n_custom"  # 실험 이름
# ─────────────────────────────────────────────────────────

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\n❌ data.yaml 을 찾을 수 없습니다: {DATA_YAML}\n"
            "먼저 데이터셋 정리(organize_dataset.py) 과정을 완료해주세요."
        )

    print(f"\n🚀 [학습 시작] 모델: {MODEL} | 에포크: {EPOCHS} | 배치: {BATCH} | 이미지크기: {IMGSZ}")
    print(f"📁 데이터 위치: {DATA_YAML}\n")

    # [1] 모델 로드
    model = YOLO(MODEL)

    # [2] 모델 학습 진행
    # 향상된 데이터 증강(Augmentation) 파라미터가 포함되어, 데이터 수 부족 문제를 보완합니다.
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
        exist_ok  = True,       # 덮어쓰기 허용
        
        # --- Data Augmentation ---
        augment   = True,       
        hsv_h     = 0.015,      # 색조 변형
        hsv_s     = 0.7,        # 채도 변형
        hsv_v     = 0.4,        # 명도 변형
        degrees   = 5.0,        # 약간의 회전 회전
        translate = 0.1,        # 이동
        scale     = 0.5,        # 크기 조절
        flipud    = 0.0,        # 상하반전 안함 (도로주행 특징 유지)
        fliplr    = 0.5,        # 50% 확률로 좌우반전
        mosaic    = 1.0,        # 모자이크 기법 (작은 객체 탐지에 매우 좋음)
        mixup     = 0.1,        # 투명하게 겹치는 기법
    )

    print("\n[학습 완료]")
    best_weight_path = Path(PROJECT) / NAME / 'weights' / 'best.pt'
    print(f"최적 가중치 저장 위치: {best_weight_path}")

    # ── 3. 테스트 세트 성능 평가 ────────────────────────────────
    print("\n[테스트 데이터 세트로 최종 검증 시작]")
    
    # 확실성을 위해 저장된 최고 성능(best.pt)의 모델을 새로 불러옵니다.
    best_model = YOLO(str(best_weight_path))
    metrics = best_model.val(
        data   = str(DATA_YAML),
        split  = "test",    # 평가용(test) 데이터셋 대상으로 평가
        imgsz  = IMGSZ,
        device = DEVICE,
    )
    
    # ── 4. 성능 지표 추출 및 리포트 파일 기록 ─────────────────────
    mean_p = metrics.box.mp
    mean_r = metrics.box.mr
    map50 = metrics.box.map50
    map95 = metrics.box.map

    report_text = f"=== YOLO 모델 ({NAME}) 테스트 세트 성능 지표 ===\n\n"
    report_text += "[전체 모델 평가 지표 (All Classes)]\n"
    report_text += f"- mAP50     : {map50:.4f}\n"
    report_text += f"- mAP50-95  : {map95:.4f}\n"
    report_text += f"- Precision : {mean_p:.4f}\n"
    report_text += f"- Recall    : {mean_r:.4f}\n\n"

    report_text += "[객체(클래스)별 성능 지표]\n"
    try:
        class_indices = metrics.box.ap_class_index
        class_ap50 = metrics.box.ap50
        class_ap95 = metrics.box.ap
        
        for idx, ap50, ap95 in zip(class_indices, class_ap50, class_ap95):
            class_name = best_model.names.get(idx, f"클래스 {idx}")
            report_text += f"▶ '{class_name}'\n"
            report_text += f"   - mAP50    : {ap50:.4f}\n"
            report_text += f"   - mAP50-95 : {ap95:.4f}\n"
    except Exception as e:
        report_text += f"\n* 상세 클래스별 메트릭을 불러오는 중 오류 발생: {e}\n"

    print("\n" + report_text)
    
    # 리포트 파일로 출력
    report_path = CURRENT_DIR / "test_performance_report.txt"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report_text)
        
    print("--------------------------------------------------")
    print(f"✅ 성능 평가지표를 터미널에 출력하였고, '{report_path}' 문서 파일로 저장했습니다!")


if __name__ == "__main__":
    main()
