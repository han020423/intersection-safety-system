import os
import glob
import json
import random
import shutil
import multiprocessing
from functools import partial

# Configuration
BASE_DIR = r"c:\Users\han02\Documents\SMU\4grade\capstone\intersection-safety-system\ai\scripts\road_v1"
SRC_JSON_DIR = os.path.join(BASE_DIR, "[라벨]c_1280_720_daylight_train_1")
SRC_IMG_DIR = os.path.join(BASE_DIR, "c_1280_720_daylight_train_1")

OUTPUT_DIR = os.path.join(BASE_DIR, "Split_Data")
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

def copy_worker(item, out_base):
    json_path, img_path, set_name = item
    
    dst_img_dir = os.path.join(out_base, set_name, "images")
    dst_json_dir = os.path.join(out_base, set_name, "labels")
    
    # Check if paths exist in source
    if not os.path.exists(json_path) or not os.path.exists(img_path):
        return False
        
    try:
        shutil.copy2(img_path, os.path.join(dst_img_dir, os.path.basename(img_path)))
        shutil.copy2(json_path, os.path.join(dst_json_dir, os.path.basename(json_path)))
        return True
    except Exception:
        return False

def main():
    print("Finding JSONs...")
    json_files = []
    for f in os.listdir(SRC_JSON_DIR):
        if f.endswith('.json'):
            json_files.append(os.path.join(SRC_JSON_DIR, f))
            
    print(f"Total potential JSON files: {len(json_files)}")
    
    # Extract pairs
    pairs = []
    print("Matching pairs...")
    for j_path in json_files:
        try:
            with open(j_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            img_name = data.get('image', {}).get('file_name', None)
            if img_name:
                img_path = os.path.join(SRC_IMG_DIR, img_name)
                if os.path.exists(img_path):
                    pairs.append((j_path, img_path))
        except Exception:
            pass
            
    print(f"Total valid pairs: {len(pairs)}")
    
    # Shuffle and split
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    print(f"Splits - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    # Create Dirs
    for split in SPLITS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    # Assign sets
    tasks = []
    for p in train_pairs: tasks.append((p[0], p[1], "train"))
    for p in val_pairs: tasks.append((p[0], p[1], "val"))
    for p in test_pairs: tasks.append((p[0], p[1], "test"))
    
    # Execute batch copy
    print(f"Dispatching multiprocessing copy over {len(tasks)} items... This may take a few minutes depending on SSD/HDD speed.")
    
    func = partial(copy_worker, out_base=OUTPUT_DIR)
    
    # Multiprocessing pool
    copied = 0
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(func, tasks)
        copied = sum(1 for r in results if r)
        
    print(f"Completed! Successfully copied {copied}/{len(tasks)} file pairs.")

if __name__ == "__main__":
    main()
