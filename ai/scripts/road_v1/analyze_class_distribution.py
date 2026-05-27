"""
클래스별 분포 분석 스크립트
- 각 클래스가 전체 데이터셋에서 몇 개 이미지에 등장하는지
- 각 클래스의 픽셀 비율이 얼마인지
"""
import os
import json
import cv2
import numpy as np
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(BASE_DIR, "Split_Data", "train")

LABEL_DIR = os.path.join(SPLIT_DIR, "labels")
IMAGE_DIR = os.path.join(SPLIT_DIR, "images")

CLS_NAMES = ['Background', 'White_Solid', 'White_Dotted', 'Yellow_Solid', 
             'Yellow_Dotted', 'Blue_Solid', 'Blue_Dotted', 'Crosswalk', 'Stop_Line']

LINE_THICKNESS = 16

def get_class_id(ann):
    cls = ann.get('class')
    if cls == 'traffic_lane':
        lane_color = 'white'
        lane_type = 'solid'
        attrs = ann.get('attributes', [])
        if isinstance(attrs, list):
            for attr in attrs:
                if attr.get('code') == 'lane_color':
                    lane_color = attr.get('value')
                elif attr.get('code') == 'lane_type':
                    lane_type = attr.get('value')
        color_id = 1
        if lane_color == 'yellow': color_id = 3
        elif lane_color == 'blue': color_id = 5
        type_offset = 0 if lane_type == 'solid' else 1
        return color_id + type_offset
    elif cls == 'crosswalk':
        return 7
    elif cls == 'stop_line':
        return 8
    return None

def main():
    json_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith('.json')])
    
    # Sample a subset for speed (pixel analysis is slow on 24k images)
    sample_size = min(2000, len(json_files))
    np.random.seed(42)
    sampled = np.random.choice(json_files, sample_size, replace=False)
    
    print(f"총 Train Label 파일 수: {len(json_files)}")
    print(f"픽셀 분석 샘플 수: {sample_size}")
    print()
    
    # 1. Annotation-level: how many images contain each class?
    class_image_count = Counter()
    class_instance_count = Counter()
    total_files = 0
    empty_files = 0
    
    for jf in json_files:
        with open(os.path.join(LABEL_DIR, jf), 'r', encoding='utf-8') as f:
            data = json.load(f)
        anns = data.get('annotations', [])
        total_files += 1
        if not anns:
            empty_files += 1
            continue
        
        classes_in_img = set()
        for ann in anns:
            cid = get_class_id(ann)
            if cid is not None:
                classes_in_img.add(cid)
                class_instance_count[cid] += 1
        for cid in classes_in_img:
            class_image_count[cid] += 1
    
    print("=" * 60)
    print("1. 클래스별 이미지 출현 횟수 (전체 Train 데이터)")
    print("=" * 60)
    print(f"{'Class':<20} {'Images':>10} {'Instances':>10} {'% of Dataset':>12}")
    print("-" * 60)
    for cid in range(1, 9):
        name = CLS_NAMES[cid]
        img_cnt = class_image_count.get(cid, 0)
        inst_cnt = class_instance_count.get(cid, 0)
        pct = 100.0 * img_cnt / total_files
        print(f"{name:<20} {img_cnt:>10} {inst_cnt:>10} {pct:>11.1f}%")
    print(f"\n빈 Annotation 파일 (Background only): {empty_files}/{total_files} ({100*empty_files/total_files:.1f}%)")
    
    # 2. Pixel-level: what fraction of pixels belong to each class?
    print(f"\n{'=' * 60}")
    print("2. 클래스별 픽셀 비율 분석 (샘플 {sample_size}개)")
    print("=" * 60)
    
    pixel_counts = np.zeros(9, dtype=np.int64)
    
    for i, jf in enumerate(sampled):
        with open(os.path.join(LABEL_DIR, jf), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_name = data.get('image', {}).get('file_name')
        if not img_name:
            img_name = jf.replace('.json', '.jpg')
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        anns = data.get('annotations', [])
        for ann in anns:
            cid = get_class_id(ann)
            if cid is None:
                continue
            pts_data = ann.get('data', [])
            if not pts_data:
                continue
            pts = np.array([[pt['x'], pt['y']] for pt in pts_data], np.int32).reshape((-1, 1, 2))
            
            cls = ann.get('class')
            if cls == 'crosswalk':
                cv2.fillPoly(mask, [pts], color=cid)
            else:
                cv2.polylines(mask, [pts], isClosed=False, color=cid, thickness=LINE_THICKNESS)
        
        for cid in range(9):
            pixel_counts[cid] += np.sum(mask == cid)
        
        if (i + 1) % 200 == 0:
            print(f"  진행: {i+1}/{sample_size}")
    
    total_pixels = pixel_counts.sum()
    print(f"\n{'Class':<20} {'Pixels':>15} {'% of Total':>12} {'Ratio vs BG':>12}")
    print("-" * 60)
    bg_pixels = pixel_counts[0]
    for cid in range(9):
        name = CLS_NAMES[cid]
        px = pixel_counts[cid]
        pct = 100.0 * px / total_pixels
        ratio = px / bg_pixels if cid > 0 else 1.0
        print(f"{name:<20} {px:>15,} {pct:>11.4f}% {ratio:>11.6f}")

if __name__ == "__main__":
    main()
