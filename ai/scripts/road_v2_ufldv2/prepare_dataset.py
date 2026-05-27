#!/usr/bin/env python3
"""
UFLDv2 학습용 road_v2 데이터셋 전처리 스크립트
=============================================
로컬 환경(Windows)에서 실행하여 ufldv2_dataset/ 폴더를 생성합니다.

사용법:
    pip install tqdm numpy
    python prepare_dataset.py
    python prepare_dataset.py --data_root ./road_v2 --output_dir ./ufldv2_dataset --num_samples 10000

출력 구조:
    ufldv2_dataset/
    ├── images/
    │   ├── train/   (8,000 장)
    │   ├── val/     (1,000 장)
    │   └── test/    (1,000 장)
    ├── train.json
    ├── val.json
    ├── test.json
    └── config.json

학습 어노테이션 포맷 (one record per image):
    {
        "raw_file":        "images/train/000001.jpg",
        "h_samples":       [0, 20, 40, ..., 700],   # 36 row-anchor positions (720px 기준)
        "lanes":           [[x0,x1,...,x35], ...],   # x-좌표 (없으면 -2), 최대 MAX_LANES개
        "lane_categories": [3, 1, ...]               # 각 lane의 카테고리 (아래 참조)
    }

Lane category 매핑:
    0 = none (배경)
    1 = white-solid  (흰색 실선)
    2 = white-dotted (흰색 점선)
    3 = yellow-solid (노란색 실선)
    4 = yellow-dotted(노란색 점선)
    5 = blue-solid   (파란색 실선)
    6 = blue-dotted  (파란색 점선)
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

# ═══════════════════════ 설정 (수정 가능) ═══════════════════════

ORIG_H    = 720
ORIG_W    = 1280
NUM_ANCHORS = 36                                        # feature map 높이 (stride-8 기준)
H_SAMPLES   = [int(i * ORIG_H / NUM_ANCHORS)           # [0, 20, 40, ..., 700]
               for i in range(NUM_ANCHORS)]
GRID_NUM    = 64                                        # feature map 너비 (512/8)
MAX_LANES   = 8                                         # 이미지당 최대 차선 수 (교차로 기준 양방향 포함)

LANE_CATEGORIES = {
    ('white',  'solid'):  1,
    ('white',  'dotted'): 2,
    ('yellow', 'solid'):  3,
    ('yellow', 'dotted'): 4,
    ('blue',   'solid'):  5,
    ('blue',   'dotted'): 6,
}
CAT_NAMES = {
    0: 'none',
    1: 'white-solid',
    2: 'white-dotted',
    3: 'yellow-solid',
    4: 'yellow-dotted',
    5: 'blue-solid',
    6: 'blue-dotted',
}

# ═══════════════════════════════════════════════════════════════


def get_category(attributes: list) -> int:
    """어노테이션 속성에서 lane category를 추출합니다."""
    color, lane_type = None, None
    for attr in attributes:
        code = attr.get('code', '')
        val  = attr.get('value', '').lower().strip()
        if code == 'lane_color':
            color = val
        elif code == 'lane_type':
            lane_type = val
    return LANE_CATEGORIES.get((color, lane_type), 0)


def polyline_to_anchors(points: list, h_samples: list) -> list:
    """
    Polyline 점들을 각 row-anchor에서의 x좌표로 변환합니다.
    anchor가 polyline 범위 밖이면 -2를 반환합니다.
    """
    if len(points) < 2:
        return [-2] * len(h_samples)

    pts = sorted(points, key=lambda p: p['y'])
    ys  = np.array([p['y'] for p in pts], dtype=float)
    xs  = np.array([p['x'] for p in pts], dtype=float)
    y_min, y_max = float(ys[0]), float(ys[-1])

    result = []
    for h in h_samples:
        if h < y_min - 1 or h > y_max + 1:
            result.append(-2)
        else:
            x = float(np.interp(h, ys, xs))
            x = max(0.0, min(ORIG_W - 1, x))
            result.append(round(x))
    return result


def parse_label_file(json_path: Path) -> tuple:
    """
    road_v2 JSON 라벨 파일을 파싱하여 lanes, categories를 반환합니다.
    traffic_lane (polyline) 만 처리하며 crosswalk/stop_line은 무시합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lanes, cats = [], []
    for ann in data.get('annotations', []):
        if ann.get('class') != 'traffic_lane':
            continue
        if ann.get('category') != 'polyline':
            continue

        cat = get_category(ann.get('attributes', []))
        if cat == 0:
            continue  # 색/타입 정보가 없으면 스킵

        pts = ann.get('data', [])
        if len(pts) < 2:
            continue

        xs = polyline_to_anchors(pts, H_SAMPLES)
        valid_cnt = sum(1 for x in xs if x != -2)
        if valid_cnt < 3:
            continue  # 유효한 앵커가 너무 적으면 스킵

        lanes.append(xs)
        cats.append(cat)

    return lanes, cats


def scan_dataset(data_root: Path) -> list:
    """
    road_v2 디렉토리를 스캔하여 유효한 (이미지, 라벨) 쌍을 모두 수집합니다.
    """
    records = []
    scanned_dirs = 0

    for img_dir in sorted(data_root.iterdir()):
        if not img_dir.is_dir():
            continue
        if img_dir.name.startswith('[라벨]'):
            continue

        label_dir = data_root / f'[라벨]{img_dir.name}'
        if not label_dir.exists():
            print(f"  [WARN] label dir not found: {img_dir.name}")
            continue

        print(f"  > {img_dir.name}")
        scanned_dirs += 1
        found = 0

        for img_path in sorted(img_dir.glob('*.jpg')):
            lbl_path = label_dir / (img_path.stem + '.json')
            if not lbl_path.exists():
                continue

            try:
                lanes, cats = parse_label_file(lbl_path)
            except Exception as e:
                continue

            if not lanes:
                continue

            dominant = Counter(cats).most_common(1)[0][0]
            records.append({
                'img':      img_path,
                'lanes':    lanes,
                'cats':     cats,
                'dominant': dominant,
            })
            found += 1

        print(f"     +-- {found:,} valid samples")

    print(f"\n  총 {scanned_dirs}개 디렉토리, {len(records):,}개 유효 레코드")
    return records


def balance_and_sample(records: list, total: int, seed: int) -> list:
    """
    dominant lane category 기준으로 균등 샘플링합니다.
    (존재하는 카테고리만 대상으로 함)
    """
    random.seed(seed)

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r['dominant']].append(r)

    present = sorted(by_cat.keys())
    quota   = total // len(present)
    remainder = total - quota * len(present)

    print(f"\n  카테고리별 할당량: {quota} (나머지 {remainder}개 첫 카테고리에 추가)")

    selected = []
    for i, cat in enumerate(present):
        pool = by_cat[cat]
        cat_quota = quota + (remainder if i == 0 else 0)

        if len(pool) >= cat_quota:
            chosen = random.sample(pool, cat_quota)
            method = "샘플"
        else:
            chosen = random.choices(pool, k=cat_quota)  # 과샘플링
            method = "과샘플"

        print(f"  {CAT_NAMES.get(cat, cat):20s}: {len(pool):6,d}개 → {len(chosen):5,d}개 선택 ({method})")
        selected.extend(chosen)

    random.shuffle(selected)
    return selected[:total]


def save_split(samples: list, split_name: str, out_dir: Path, idx_offset: int = 0) -> list:
    """이미지를 복사하고 어노테이션 리스트를 반환합니다."""
    img_out = out_dir / 'images' / split_name
    img_out.mkdir(parents=True, exist_ok=True)

    annotations = []
    for i, s in enumerate(tqdm(samples, desc=f"  {split_name:5s}", ncols=80)):
        new_name = f'{idx_offset + i:07d}.jpg'
        shutil.copy2(s['img'], img_out / new_name)

        annotations.append({
            'raw_file':        f'images/{split_name}/{new_name}',
            'h_samples':       H_SAMPLES,
            'lanes':           s['lanes'],
            'lane_categories': s['cats'],
        })
    return annotations


def print_split_stats(samples: list, name: str):
    """Split 내 카테고리 분포를 출력합니다."""
    cat_cnt = Counter()
    for s in samples:
        for c in s['cats']:
            cat_cnt[c] += 1
    total = sum(cat_cnt.values())
    print(f"  [{name}] 총 {len(samples):,}장, 차선 인스턴스 {total:,}개")
    for cat in sorted(cat_cnt):
        print(f"    {CAT_NAMES.get(cat,'?'):20s}: {cat_cnt[cat]:5,}개 ({100*cat_cnt[cat]/total:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description='road_v2 -> UFLDv2 dataset')
    ap.add_argument('--data_root',   default='./road_v2',        help='road_v2 루트 경로')
    ap.add_argument('--output_dir',  default='./ufldv2_dataset', help='출력 디렉토리')
    ap.add_argument('--num_samples', type=int,   default=10000,  help='총 샘플 수')
    ap.add_argument('--train_ratio', type=float, default=0.8,    help='학습 비율')
    ap.add_argument('--val_ratio',   type=float, default=0.1,    help='검증 비율')
    ap.add_argument('--seed',        type=int,   default=42,     help='랜덤 시드')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  UFLDv2 데이터셋 준비 스크립트')
    print('=' * 60)
    print(f'  입력: {data_root.absolute()}')
    print(f'  출력: {out_dir.absolute()}')
    print(f'  목표: {args.num_samples:,}개 | Train/Val/Test = {args.train_ratio:.0%}/{args.val_ratio:.0%}/{1-args.train_ratio-args.val_ratio:.0%}')
    print(f'  앵커: {NUM_ANCHORS}개 @ {H_SAMPLES}')
    print(f'  그리드: {GRID_NUM}  |  최대 차선: {MAX_LANES}')

    # ── 1. 스캔 ──
    print('\n[1/4] 데이터셋 스캔 중...')
    records = scan_dataset(data_root)
    if not records:
        print('  [ERROR] No valid records. Check --data_root path.')
        return

    # ── 2. 균등 샘플링 ──
    print('\n[2/4] 균등 샘플링 중...')
    selected = balance_and_sample(records, args.num_samples, args.seed)
    print(f'  총 선택: {len(selected):,}개')

    # ── 3. 분할 ──
    print('\n[3/4] Train / Val / Test 분할 중...')
    n      = len(selected)
    n_tr   = int(n * args.train_ratio)
    n_val  = int(n * args.val_ratio)
    splits = {
        'train': selected[:n_tr],
        'val':   selected[n_tr:n_tr + n_val],
        'test':  selected[n_tr + n_val:],
    }
    for name, samples in splits.items():
        print_split_stats(samples, name)

    # ── 4. 저장 ──
    print('\n[4/4] 이미지 복사 및 어노테이션 저장 중...')
    offset = 0
    for split_name, samples in splits.items():
        anns = save_split(samples, split_name, out_dir, offset)
        offset += len(samples)
        ann_path = out_dir / f'{split_name}.json'
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(anns, f, ensure_ascii=False)
        print(f'  [SAVED] {split_name}.json ({len(anns):,} records)')

    # ── config.json ──
    config = {
        'orig_h':        ORIG_H,
        'orig_w':        ORIG_W,
        'input_h':       288,
        'input_w':       512,
        'h_samples':     H_SAMPLES,
        'num_anchors':   NUM_ANCHORS,
        'grid_num':      GRID_NUM,
        'max_lanes':     MAX_LANES,
        'num_lane_types': max(LANE_CATEGORIES.values()) + 1,   # 7 (0~6)
        'lane_type_names': CAT_NAMES,
        'splits': {k: len(v) for k, v in splits.items()},
        'seed':          args.seed,
    }
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print('  [OK] config.json 저장')

    # ── 완료 안내 ──
    print('\n' + '=' * 60)
    print(f'  [DONE] 완료! -> {out_dir.absolute()}')
    print()
    print('  다음 단계:')
    print('  1. 폴더 압축 (PowerShell):')
    print('       Compress-Archive -Path ufldv2_dataset -DestinationPath ufldv2_dataset.zip')
    print('  2. ufldv2_dataset.zip 을 Google Drive에 업로드')
    print('  3. colab_train_ufldv2.py 를 Colab에 업로드하여 실행')
    print('=' * 60)


if __name__ == '__main__':
    main()
