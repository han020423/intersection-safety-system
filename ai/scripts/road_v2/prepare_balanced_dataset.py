"""
road_v2 데이터셋 균형 선별 스크립트
====================================
- 차선(traffic_lane), 횡단보도(crosswalk), 정지선(stop_line) 3개 클래스
- 총 20,000개 데이터를 클래스별 균등 분포로 선별
- train(70%) / val(15%) / test(15%) 분할
- road_v2/dataset/ 폴더에 저장 (이미지 + JSON 라벨)
"""

import json
import os
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path

# ─── 설정 ───────────────────────────────────────────────────────
SEED = 42
TOTAL_SELECT = 20000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BASE_DIR = Path(r"c:\Users\han02\Documents\SMU\4grade\capstone\intersection-safety-system\ai\scripts\road_v2")

IMAGE_DIRS = [
    BASE_DIR / "c_1280_720_daylight_train_1",
    BASE_DIR / "c_1280_720_daylight_train_2",
]
LABEL_DIRS = [
    BASE_DIR / "[라벨]c_1280_720_daylight_train_1",
    BASE_DIR / "[라벨]c_1280_720_daylight_train_2",
]

OUTPUT_DIR = BASE_DIR / "dataset"
CLASSES = ["traffic_lane", "crosswalk", "stop_line"]

random.seed(SEED)


def get_file_classes(label_path: Path) -> set:
    """JSON 라벨에서 포함된 클래스 집합 반환"""
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = set()
    for ann in data.get("annotations", []):
        cls = ann.get("class", "")
        if cls in CLASSES:
            classes.add(cls)
    return classes


def main():
    print("=" * 60)
    print("road_v2 균형 데이터셋 준비 스크립트")
    print("=" * 60)

    # ─── 1단계: 모든 파일 스캔 & 클래스별 분류 ───────────────
    print("\n[1/5] 라벨 파일 스캔 중...")

    # file_id -> (image_dir, label_dir) 매핑
    all_files = {}  # file_id -> { "img": path, "lbl": path, "classes": set }

    for img_dir, lbl_dir in zip(IMAGE_DIRS, LABEL_DIRS):
        for lbl_file in os.listdir(lbl_dir):
            if not lbl_file.endswith(".json"):
                continue
            file_id = lbl_file.replace(".json", "")
            img_file = file_id + ".jpg"
            img_path = img_dir / img_file
            lbl_path = lbl_dir / lbl_file

            if not img_path.exists():
                continue

            classes = get_file_classes(lbl_path)
            if not classes:  # 클래스 없는 파일 제외
                continue

            all_files[file_id] = {
                "img": img_path,
                "lbl": lbl_path,
                "classes": classes,
            }

    print(f"  총 유효 파일: {len(all_files):,}개")

    # ─── 2단계: 클래스별 파일 리스트 생성 ────────────────────
    print("\n[2/5] 클래스별 분류 중...")

    # 각 클래스를 포함하는 파일 ID 리스트
    class_to_files = defaultdict(set)
    for fid, info in all_files.items():
        for cls in info["classes"]:
            class_to_files[cls].add(fid)

    for cls in CLASSES:
        print(f"  {cls}: {len(class_to_files[cls]):,}개")

    # ─── 3단계: 균등 선별 ────────────────────────────────────
    # 전략:
    # 1) 희소 클래스(stop_line, crosswalk) 포함 파일 우선 선택
    # 2) 나머지는 traffic_lane만 포함 파일에서 채움
    print("\n[3/5] 클래스 균등 분포 선별 중...")

    selected = set()

    # 희소 클래스 우선 순위: stop_line > crosswalk > traffic_lane
    # (a) stop_line 포함 파일: 모두 선택 (4,967개)
    stopline_files = class_to_files["stop_line"]
    selected.update(stopline_files)
    print(f"  stop_line 포함 파일 전체 선택: {len(stopline_files):,}개")

    # (b) crosswalk 포함이지만 stop_line 미포함 파일에서 추가
    crosswalk_only = class_to_files["crosswalk"] - selected
    crosswalk_only_list = sorted(crosswalk_only)
    random.shuffle(crosswalk_only_list)

    # crosswalk 총 개수를 stop_line과 비슷하게 맞춤
    # 현재 selected 중 crosswalk 포함 개수
    selected_crosswalk = sum(1 for fid in selected if "crosswalk" in all_files[fid]["classes"])
    # stop_line 전체 = 4967, crosswalk 현재 = selected_crosswalk
    # crosswalk만 있는 파일에서 추가로 선택할 수 있는 만큼 선택
    crosswalk_target = len(stopline_files)  # stop_line과 동일하게
    crosswalk_need = max(0, crosswalk_target - selected_crosswalk)
    crosswalk_add = crosswalk_only_list[:crosswalk_need]
    selected.update(crosswalk_add)
    print(f"  crosswalk 추가 선택: {len(crosswalk_add):,}개 (이미 {selected_crosswalk:,}개 포함)")

    # (c) 나머지를 traffic_lane-only 파일에서 채움
    remaining = TOTAL_SELECT - len(selected)
    traffic_only = set()
    for fid in class_to_files["traffic_lane"]:
        if fid not in selected:
            traffic_only.add(fid)

    traffic_only_list = sorted(traffic_only)
    random.shuffle(traffic_only_list)
    traffic_add = traffic_only_list[:remaining]
    selected.update(traffic_add)
    print(f"  traffic_lane 추가 선택: {len(traffic_add):,}개")
    print(f"  최종 선택: {len(selected):,}개")

    # 선택된 데이터 클래스 분포 확인
    final_class_count = Counter()
    for fid in selected:
        for cls in all_files[fid]["classes"]:
            final_class_count[cls] += 1

    print("\n  [선택된 데이터 클래스 분포]")
    for cls in CLASSES:
        print(f"    {cls}: {final_class_count[cls]:,}개")

    # ─── 4단계: train/val/test 분할 ──────────────────────────
    print(f"\n[4/5] train/val/test 분할 중 ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")

    selected_list = sorted(selected)
    random.shuffle(selected_list)

    # Stratified split: 클래스 조합별로 분할
    combo_groups = defaultdict(list)
    for fid in selected_list:
        combo = tuple(sorted(all_files[fid]["classes"]))
        combo_groups[combo].append(fid)

    train_ids, val_ids, test_ids = [], [], []

    for combo, fids in combo_groups.items():
        random.shuffle(fids)
        n = len(fids)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # 나머지는 test로
        train_ids.extend(fids[:n_train])
        val_ids.extend(fids[n_train : n_train + n_val])
        test_ids.extend(fids[n_train + n_val :])

    print(f"  Train: {len(train_ids):,}개")
    print(f"  Val:   {len(val_ids):,}개")
    print(f"  Test:  {len(test_ids):,}개")
    print(f"  Total: {len(train_ids) + len(val_ids) + len(test_ids):,}개")

    # 각 split별 클래스 분포
    for split_name, split_ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        split_class_count = Counter()
        for fid in split_ids:
            for cls in all_files[fid]["classes"]:
                split_class_count[cls] += 1
        print(f"\n  [{split_name} 클래스 분포]")
        for cls in CLASSES:
            print(f"    {cls}: {split_class_count[cls]:,}개")

    # ─── 5단계: 파일 복사 ────────────────────────────────────
    print(f"\n[5/5] 파일 복사 중...")

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    for split_name, split_ids in splits.items():
        img_out = OUTPUT_DIR / split_name / "images"
        lbl_out = OUTPUT_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for i, fid in enumerate(split_ids):
            info = all_files[fid]
            shutil.copy2(info["img"], img_out / (fid + ".jpg"))
            shutil.copy2(info["lbl"], lbl_out / (fid + ".json"))

            if (i + 1) % 2000 == 0 or (i + 1) == len(split_ids):
                print(f"  {split_name}: {i+1:,}/{len(split_ids):,} 완료")

    # ─── 최종 요약 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("완료!")
    print(f"데이터셋 경로: {OUTPUT_DIR}")
    print(f"  train/images: {len(train_ids):,}개")
    print(f"  train/labels: {len(train_ids):,}개")
    print(f"  val/images:   {len(val_ids):,}개")
    print(f"  val/labels:   {len(val_ids):,}개")
    print(f"  test/images:  {len(test_ids):,}개")
    print(f"  test/labels:  {len(test_ids):,}개")
    print("=" * 60)


if __name__ == "__main__":
    main()
