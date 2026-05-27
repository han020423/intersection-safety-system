import os
import shutil
import random
from glob import glob

def get_all_txt_files(directory):
    """지정된 디렉토리 하위의 모든 txt 파일을 재귀적으로 찾아서 (파일명: 절대경로) 딕셔너리로 반환합니다."""
    txt_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            # 설정 파일명(train.txt 등) 제외하고 라벨 파일만 찾기
            if file.endswith(".txt") and file not in ['train.txt', 'val.txt', 'test.txt', 'classes.txt']:
                txt_files[file] = os.path.join(root, file)
    return txt_files

def main():
    base_dir = r"C:\Users\han02\Documents\SMU\4grade\capstone\intersection-safety-system\ai\scripts\v1"
    img_dir = os.path.join(base_dir, "img")
    label1_dir = os.path.join(base_dir, "label", "1")
    label2_dir = os.path.join(base_dir, "label", "2")
    
    dataset_dir = os.path.join(base_dir, "yolo_dataset")
    merged_label_dir = os.path.join(base_dir, "merged_labels")
    os.makedirs(merged_label_dir, exist_ok=True)
    
    print("1. 라벨 병합 시작...")
    
    # 하위 폴더(labels/train 등)에 있는 모든 txt 파일 수집
    label1_files = get_all_txt_files(label1_dir)
    label2_files = get_all_txt_files(label2_dir)
    
    all_label_filenames = set(label1_files.keys()).union(set(label2_files.keys()))

    for file_name in all_label_filenames:
        merged_lines = []
        
        # 1번 폴더에 파일이 있으면 합침
        if file_name in label1_files:
            file1_path = label1_files[file_name]
            with open(file1_path, 'r', encoding='utf-8') as f:
                merged_lines.extend(f.readlines())
                
        # 2번 폴더에 파일이 있으면 합침
        if file_name in label2_files:
            file2_path = label2_files[file_name]
            with open(file2_path, 'r', encoding='utf-8') as f:
                merged_lines.extend(f.readlines())
                
        # 빈 줄 제거
        merged_lines = [line.strip() for line in merged_lines if line.strip()]
        
        # 병합된 파일 저장
        with open(os.path.join(merged_label_dir, file_name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_lines) + '\n')
            
    print(f"   -> 총 {len(all_label_filenames)}개의 라벨(박스) 파일 통합 및 복사 완료.")

    print("2. 데이터셋 분할 (Train/Val/Test) 시작...")
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

    # 이미지 목록 (jpg, png)
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(glob(os.path.join(img_dir, ext)))
        
    random.seed(42)
    random.shuffle(image_files)
    
    num_images = len(image_files)
    if num_images == 0:
        print("   -> 이미지가 없습니다.")
        return

    train_end = int(num_images * 0.8)
    val_end = int(num_images * 0.9)

    split_dict = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    def copy_files(file_list, split_name):
        count = 0
        img_count = 0
        for img_path in file_list:
            file_name = os.path.basename(img_path)
            name_only = os.path.splitext(file_name)[0]
            
            # 이미지 복사
            dest_img = os.path.join(dataset_dir, 'images', split_name, file_name)
            shutil.copy(img_path, dest_img)
            img_count += 1
            
            # 매칭되는 라벨 파일 찾기
            label_path = os.path.join(merged_label_dir, name_only + '.txt')
            if os.path.exists(label_path):
                dest_label = os.path.join(dataset_dir, 'labels', split_name, name_only + '.txt')
                shutil.copy(label_path, dest_label)
                count += 1
                
        return img_count, count

    for split in splits:
        img_cnt, lbl_cnt = copy_files(split_dict[split], split)
        print(f"      - {split}: 이미지 {img_cnt}장 / 라벨 {lbl_cnt}개 복사 완료.")

    print("3. YOLO용 data.yaml 파일 생성...")
    # path 속성을 지우거나 상대 경로를 유지하여 Colab 이동 시 경로 에러(FileNotFoundError) 방지
    # 데이터셋 내 라벨에 최대 5번(총 6종류) 클래스가 나오므로 nc: 6으로 세팅하여 데이터 드랍(corrupt) 에러 방지
    yaml_content = """train: images/train
val: images/val
test: images/test

nc: 6
names: ['Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5']
"""
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"   -> data.yaml 생성 완료: {yaml_path}")
    print("데이터셋 정리가 모두 완료되었습니다! 이제 학습 코드를 실행할 수 있습니다.")

if __name__ == "__main__":
    main()
