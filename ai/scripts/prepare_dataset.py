"""
v0 폴더의 이미지/라벨을 train(70%) / val(20%) / test(10%) 로 분할하여
YOLO 학습에 맞는 폴더 구조를 만들고 dataset.yaml을 생성합니다.

실행:
    python prepare_dataset.py
"""

import random
import shutil
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10          # 나머지

V0_DIR      = Path(__file__).parent / "v0"
IMG_SRC     = V0_DIR / "image"
LBL_SRC     = V0_DIR / "labels" / "train"

DATASET_DIR = Path(__file__).parent / "dataset"  # 출력 루트
# ─────────────────────────────────────────────────────────


def split_and_copy():
    # 이미지 파일 목록 (jpg)
    images = sorted(IMG_SRC.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"이미지가 없습니다: {IMG_SRC}")

    random.seed(SEED)
    random.shuffle(images)

    n        = len(images)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)
    # 나머지 = test
    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    print(f"전체: {n}장  →  train:{len(splits['train'])}  val:{len(splits['val'])}  test:{len(splits['test'])}")

    for split, imgs in splits.items():
        img_dst = DATASET_DIR / "images" / split
        lbl_dst = DATASET_DIR / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            # 이미지 복사
            shutil.copy2(img_path, img_dst / img_path.name)

            # 라벨(txt) 복사 – 없는 경우 빈 파일 생성 (background image)
            lbl_path = LBL_SRC / (img_path.stem + ".txt")
            dst_lbl  = lbl_dst / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.touch()  # 라벨 없는 이미지 → 빈 파일

    # data.yaml 생성
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_content = f"""\
path: {DATASET_DIR.resolve().as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 7
names:
  0: pedestrian
  1: vehicle
  2: traffic_light_vehicle
  3: traffic_light_pedestrian
  4: crosswalk
  5: stop_line
  6: left_turn_sign
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\ndata.yaml 저장: {yaml_path}")
    print("데이터셋 준비 완료!")


if __name__ == "__main__":
    split_and_copy()
