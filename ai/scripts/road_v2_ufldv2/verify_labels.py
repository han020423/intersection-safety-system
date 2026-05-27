#!/usr/bin/env python3
"""
UFLDv2 라벨링 검증 스크립트
==========================
ufldv2_dataset/ 폴더의 이미지와 JSON 파일을 분석하여
라벨링이 올바르게 되었는지 검증합니다.

검증 항목:
  1. JSON 구조 무결성 (필수 키, 타입)
  2. 이미지 파일 존재 여부
  3. h_samples 일관성
  4. lanes x-좌표 범위 검증 (0~1279 또는 -2)
  5. lane_categories 유효성 (1~6)
  6. lanes 수와 lane_categories 수 일치 여부
  7. 차선 연속성 (인접 앵커 간 x 좌표 급변 감지)
  8. 비정상 차선 비율 (유효 앵커 3개 미만)
  9. 시각적 검증용 이미지 생성 (랜덤 샘플)

사용법:
  cd road_v2_ufldv2
  python verify_labels.py
  python verify_labels.py --num_vis 20  # 시각화 개수 변경
"""

import json
import os
import sys
import random
import argparse
from pathlib import Path
from collections import Counter

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ──────── 설정 ────────
ORIG_H = 720
ORIG_W = 1280
EXPECTED_H_SAMPLES = [int(i * ORIG_H / 36) for i in range(36)]
VALID_CATEGORIES = {1, 2, 3, 4, 5, 6}
MAX_CONSECUTIVE_JUMP = 200  # 인접 앵커 간 x 좌표 최대 허용 차이 (px)

CAT_NAMES = {
    0: 'none', 1: 'white-solid', 2: 'white-dotted',
    3: 'yellow-solid', 4: 'yellow-dotted',
    5: 'blue-solid', 6: 'blue-dotted',
}
CAT_COLORS = {
    1: (255, 255, 255),  # white-solid
    2: (200, 200, 200),  # white-dotted
    3: (0, 255, 255),    # yellow-solid (BGR)
    4: (0, 200, 200),    # yellow-dotted
    5: (255, 0, 0),      # blue-solid
    6: (200, 0, 0),      # blue-dotted
}


def load_json(path: Path) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def verify_split(split_name: str, annotations: list, dataset_dir: Path) -> dict:
    """단일 split에 대한 검증을 수행합니다."""
    stats = {
        'total': len(annotations),
        'missing_images': 0,
        'wrong_h_samples': 0,
        'mismatched_lanes_cats': 0,
        'invalid_categories': [],
        'out_of_bound_x': 0,
        'short_lanes': 0,       # 유효 앵커 < 3
        'big_jump_lanes': 0,    # 연속 앵커 간 급변
        'lane_count_dist': Counter(),
        'cat_dist': Counter(),
        'total_lane_instances': 0,
        'issues': [],
    }

    for idx, ann in enumerate(annotations):
        img_rel = ann.get('raw_file', '')
        h_samples = ann.get('h_samples', [])
        lanes = ann.get('lanes', [])
        cats = ann.get('lane_categories', [])

        # 1) 이미지 존재 확인
        img_path = dataset_dir / img_rel
        if not img_path.exists():
            stats['missing_images'] += 1
            stats['issues'].append(f"[{split_name}#{idx}] 이미지 없음: {img_rel}")

        # 2) h_samples 일관성
        if h_samples != EXPECTED_H_SAMPLES:
            stats['wrong_h_samples'] += 1
            if stats['wrong_h_samples'] <= 3:
                stats['issues'].append(f"[{split_name}#{idx}] h_samples 불일치")

        # 3) lanes 수와 categories 수 일치
        if len(lanes) != len(cats):
            stats['mismatched_lanes_cats'] += 1
            stats['issues'].append(
                f"[{split_name}#{idx}] lanes({len(lanes)})≠cats({len(cats)}): {img_rel}")

        # 4) 차선별 검증
        stats['lane_count_dist'][len(lanes)] += 1
        stats['total_lane_instances'] += len(lanes)

        for lane_idx, lane in enumerate(lanes):
            # 길이 일치
            if len(lane) != len(h_samples):
                stats['issues'].append(
                    f"[{split_name}#{idx}] lane[{lane_idx}] len={len(lane)} ≠ h_samples={len(h_samples)}")
                continue

            valid_xs = [(i, x) for i, x in enumerate(lane) if x != -2]
            valid_count = len(valid_xs)

            # 유효 앵커 부족
            if valid_count < 3:
                stats['short_lanes'] += 1

            # x 범위 체크
            for i, x in valid_xs:
                if x < 0 or x >= ORIG_W:
                    stats['out_of_bound_x'] += 1
                    if stats['out_of_bound_x'] <= 5:
                        stats['issues'].append(
                            f"[{split_name}#{idx}] lane[{lane_idx}] x={x} out of [0,{ORIG_W-1}] at anchor {i}")
                    break

            # 연속성 체크 (인접 유효 앵커 간 x 급변)
            has_jump = False
            for j in range(1, len(valid_xs)):
                prev_i, prev_x = valid_xs[j - 1]
                curr_i, curr_x = valid_xs[j]
                if curr_i == prev_i + 1:  # 연속 앵커
                    dx = abs(curr_x - prev_x)
                    if dx > MAX_CONSECUTIVE_JUMP:
                        has_jump = True
                        break
            if has_jump:
                stats['big_jump_lanes'] += 1

        # 5) 카테고리 유효성
        for c_idx, c in enumerate(cats):
            stats['cat_dist'][c] += 1
            if c not in VALID_CATEGORIES:
                stats['invalid_categories'].append(
                    f"[{split_name}#{idx}] cat={c} for lane[{c_idx}] in {img_rel}")

    return stats


def print_report(split_name: str, stats: dict):
    """검증 결과를 출력합니다."""
    total = stats['total']
    print(f"\n{'─' * 60}")
    print(f"  [{split_name.upper()}] 검증 결과  ({total:,}개 레코드)")
    print(f"{'─' * 60}")

    # 기본 통계
    print(f"  총 차선 인스턴스: {stats['total_lane_instances']:,}")
    print(f"  이미지당 차선 수 분포:")
    for k in sorted(stats['lane_count_dist'].keys()):
        cnt = stats['lane_count_dist'][k]
        print(f"    {k}개: {cnt:>5,}장 ({100 * cnt / total:.1f}%)")

    # 카테고리 분포
    print(f"\n  카테고리 분포:")
    total_lanes = stats['total_lane_instances']
    for c in sorted(stats['cat_dist'].keys()):
        cnt = stats['cat_dist'][c]
        name = CAT_NAMES.get(c, f'unknown({c})')
        print(f"    {name:20s}: {cnt:>5,} ({100 * cnt / max(total_lanes, 1):.1f}%)")

    # 이슈 요약
    issues = {
        '누락 이미지': stats['missing_images'],
        'h_samples 불일치': stats['wrong_h_samples'],
        'lanes/cats 수 불일치': stats['mismatched_lanes_cats'],
        '유효하지 않은 카테고리': len(stats['invalid_categories']),
        'x 범위 초과': stats['out_of_bound_x'],
        '짧은 차선 (앵커<3)': stats['short_lanes'],
        '급변 차선 (Δx>200)': stats['big_jump_lanes'],
    }
    print(f"\n  이슈 요약:")
    all_pass = True
    for name, cnt in issues.items():
        status = "✓" if cnt == 0 else "✗"
        if cnt > 0:
            all_pass = False
        print(f"    {status} {name:25s}: {cnt:>5,}")

    if all_pass:
        print(f"\n  ✅ [{split_name}] 모든 검증 통과!")
    else:
        print(f"\n  ⚠️  [{split_name}] 일부 이슈 발견")

    # 상세 이슈 (최대 10개)
    if stats['issues']:
        print(f"\n  상세 이슈 (최대 10개):")
        for issue in stats['issues'][:10]:
            print(f"    • {issue}")
        if len(stats['issues']) > 10:
            print(f"    ... 외 {len(stats['issues']) - 10}개")


def visualize_samples(annotations: list, dataset_dir: Path, output_dir: Path,
                      num_vis: int = 10, split_name: str = 'train'):
    """랜덤 샘플을 시각화하여 저장합니다."""
    if not HAS_CV2:
        print("  ⚠️  cv2 없음 → 시각화 생략")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    indices = random.sample(range(len(annotations)), min(num_vis, len(annotations)))

    for idx in indices:
        ann = annotations[idx]
        img_path = dataset_dir / ann['raw_file']
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h_samples = ann['h_samples']
        lanes = ann['lanes']
        cats = ann.get('lane_categories', [0] * len(lanes))

        for lane_idx, (lane, cat) in enumerate(zip(lanes, cats)):
            color = CAT_COLORS.get(cat, (0, 255, 0))
            cat_name = CAT_NAMES.get(cat, '?')

            # 유효한 점만 추출
            pts = [(x, h_samples[i]) for i, x in enumerate(lane) if x != -2]
            if len(pts) < 2:
                continue

            # 차선 그리기
            for j in range(1, len(pts)):
                cv2.line(img, pts[j - 1], pts[j], color, 2, cv2.LINE_AA)

            # 점 그리기
            for px, py in pts:
                cv2.circle(img, (px, py), 3, color, -1)

            # 라벨 텍스트
            mid = pts[len(pts) // 2]
            cv2.putText(img, f"L{lane_idx}:{cat_name}",
                        (mid[0] + 5, mid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # h_samples 가이드 라인
        for h in h_samples[::4]:  # 매 4번째만
            cv2.line(img, (0, h), (ORIG_W, h), (50, 50, 50), 1)

        # 정보 텍스트
        info = f"{split_name}/{ann['raw_file'].split('/')[-1]} | {len(lanes)} lanes"
        cv2.putText(img, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        save_path = output_dir / f"verify_{split_name}_{idx:05d}.jpg"
        cv2.imwrite(str(save_path), img)

    print(f"  📸 {len(indices)}개 시각화 이미지 저장 → {output_dir}")


def check_duplicate_lanes(annotations: list, split_name: str) -> int:
    """동일 이미지 내 완전히 동일한 차선이 있는지 확인합니다."""
    dup_count = 0
    for idx, ann in enumerate(annotations):
        lanes = ann['lanes']
        seen = set()
        for lane in lanes:
            key = tuple(lane)
            if key in seen:
                dup_count += 1
                break
            seen.add(key)
    return dup_count


def main():
    parser = argparse.ArgumentParser(description='UFLDv2 라벨 검증')
    parser.add_argument('--dataset_dir', default='./ufldv2_dataset',
                        help='ufldv2_dataset 경로')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='시각화 샘플 수 (각 split)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_dir = Path(args.dataset_dir)

    print('=' * 60)
    print('  UFLDv2 라벨링 검증 스크립트')
    print('=' * 60)
    print(f'  데이터셋: {dataset_dir.absolute()}\n')

    splits = ['train', 'val', 'test']
    all_stats = {}

    for split in splits:
        json_path = dataset_dir / f'{split}.json'
        if not json_path.exists():
            print(f"  [SKIP] {split}.json 없음")
            continue

        print(f"  [{split}] 로딩 중...")
        annotations = load_json(json_path)
        print(f"  [{split}] {len(annotations):,}개 레코드 로드")

        # 구조 검증
        stats = verify_split(split, annotations, dataset_dir)
        all_stats[split] = stats
        print_report(split, stats)

        # 중복 차선 검사
        dup = check_duplicate_lanes(annotations, split)
        if dup > 0:
            print(f"  ⚠️  [{split}] 동일 이미지 내 중복 차선: {dup}건")
        else:
            print(f"  ✓ [{split}] 중복 차선 없음")

        # 시각화
        if args.num_vis > 0:
            vis_dir = dataset_dir / 'verify_vis'
            visualize_samples(annotations, dataset_dir, vis_dir,
                              args.num_vis, split)

    # 최종 요약
    print(f"\n{'=' * 60}")
    print(f"  최종 요약")
    print(f"{'=' * 60}")

    total_issues = 0
    for split, stats in all_stats.items():
        n_issues = (stats['missing_images'] + stats['wrong_h_samples'] +
                    stats['mismatched_lanes_cats'] + len(stats['invalid_categories']) +
                    stats['out_of_bound_x'])
        total_issues += n_issues
        status = "✅" if n_issues == 0 else "⚠️"
        print(f"  {status} {split:5s}: {stats['total']:>5,}장, "
              f"{stats['total_lane_instances']:>6,} 차선, "
              f"심각한 이슈 {n_issues}건")

    # 경고성 이슈 (치명적이지 않음)
    total_warn = sum(s['short_lanes'] + s['big_jump_lanes'] for s in all_stats.values())
    if total_warn > 0:
        print(f"\n  ⚠️  경고성 이슈 (학습 가능, 주의 필요): {total_warn}건")
        for split, stats in all_stats.items():
            if stats['short_lanes'] + stats['big_jump_lanes'] > 0:
                print(f"    {split}: 짧은 차선 {stats['short_lanes']}건, "
                      f"급변 차선 {stats['big_jump_lanes']}건")

    if total_issues == 0:
        print(f"\n  🎉 모든 심각한 검증 통과! 라벨링 구조가 정상입니다.")
    else:
        print(f"\n  ❌ {total_issues}건의 심각한 이슈 발견. 위 상세 내용을 확인하세요.")

    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
