import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadStructureDataset(Dataset):
    def __init__(self, split_dir, target_size=(360, 640), line_thickness=5, transform=None):
        """
        split_dir: Directory containing 'images' and 'labels' subfolders (e.g. Split_Data/train)
        target_size: (HEIGHT, WIDTH) for resizing. Notice the order for Albumentations is (H, W).
        transform: Albumentations Compose object
        """
        self.image_dir = os.path.join(split_dir, "images")
        self.label_dir = os.path.join(split_dir, "labels")
        self.target_size = target_size
        self.line_thickness = line_thickness
        
        # If no transform is provided, just Resize & ToTensor
        self.transform = transform if transform else A.Compose([
            A.Resize(height=target_size[0], width=target_size[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.samples = []
        if os.path.exists(self.label_dir):
            for file_name in os.listdir(self.label_dir):
                if file_name.endswith('.json'):
                    j_path = os.path.join(self.label_dir, file_name)
                    try:
                        with open(j_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        img_name = data.get('image', {}).get('file_name')
                        if not img_name:
                            img_name = file_name.replace('.json', '.jpg')
                            
                        img_path = os.path.join(self.image_dir, img_name)
                        if os.path.exists(img_path):
                            self.samples.append({
                                'json_path': j_path,
                                'img_path': img_path,
                                'annotations': data.get('annotations', [])
                            })
                    except Exception:
                        pass
                        
        # Count samples with empty annotations
        empty_count = sum(1 for s in self.samples if not s['annotations'])
        print(f"Loaded {len(self.samples)} valid samples from {split_dir}")
        if empty_count > 0:
            print(f"  ⚠️ Warning: {empty_count}/{len(self.samples)} samples have empty annotations (background only)")

    def __len__(self):
        return len(self.samples)

    def draw_mask(self, original_h, original_w, annotations):
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        for ann in annotations:
            cls = ann.get('class')
            pts_data = ann.get('data', [])
            if not pts_data: continue
                
            pts = np.array([[pt['x'], pt['y']] for pt in pts_data], np.int32).reshape((-1, 1, 2))
            
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
                
                # Determine Class ID based on Color and Type
                # White Solid: 1, White Dotted: 2
                # Yellow Solid: 3, Yellow Dotted: 4
                # Blue Solid: 5, Blue Dotted: 6
                color_id = 1
                if lane_color == 'white': color_id = 1
                elif lane_color == 'yellow': color_id = 3
                elif lane_color == 'blue': color_id = 5
                
                type_offset = 0 if lane_type == 'solid' else 1
                final_val = color_id + type_offset
                
                cv2.polylines(mask, [pts], isClosed=False, color=final_val, thickness=self.line_thickness)
                
            elif cls == 'crosswalk':
                # 첨부된 표 기준 횡단보도는 폴리곤(Polygon)
                cv2.fillPoly(mask, [pts], color=7)
                
            elif cls == 'stop_line':
                # 첨부된 표 기준 정지선은 폴리라인(Polyline)
                cv2.polylines(mask, [pts], isClosed=False, color=8, thickness=self.line_thickness)
                
        return mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        annotations = sample['annotations']
        
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # 1. Create mask in its pure original coordinate space
        mask = self.draw_mask(h, w, annotations)
        
        # 2. Let Albumentations strictly handle resizing AND geometric augmentations (flipping etc) 
        # to ensure image and mask are warped perfectly identically!
        augmented = self.transform(image=img, mask=mask)
        
        img_tensor = augmented['image']
        mask_tensor = augmented['mask'].long()
            
        return img_tensor, mask_tensor
