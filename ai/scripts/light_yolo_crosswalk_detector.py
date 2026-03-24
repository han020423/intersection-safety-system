import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


YOLO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}

BOX_COLORS = {
    "person": (0, 255, 0),
    "vehicle": (255, 0, 0),
    "traffic light": (0, 255, 255),
    "crosswalk": (255, 255, 255),
}


def classify_label(class_name: str) -> str:
    if class_name == "person":
        return "person"
    if class_name in {"car", "bus", "truck", "motorcycle", "bicycle"}:
        return "vehicle"
    if class_name == "traffic light":
        return "traffic light"
    return class_name


class RoadSceneDetector:
    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz

    def detect_yolo(self, image: np.ndarray):
        result = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, cls_id, score in zip(boxes, clss, confs):
            # 모델이 내보내는 실제 클래스 이름 가져오기 (커스텀 모델 대응)
            class_name = result.names.get(cls_id, str(cls_id)).lower()
            
            detections.append(
                {
                    "label": classify_label(class_name),
                    "raw_label": class_name,
                    "score": float(score),
                    "box": tuple(box.tolist()),
                }
            )
        return detections

    def detect_crosswalk(self, image: np.ndarray):
        """
        Heuristic crosswalk detector:
        - looks only at the lower part of the image
        - extracts bright road markings
        - searches for 3+ horizontal stripe bands

        This is a practical fallback because crosswalk is not a default COCO class.
        """
        h, w = image.shape[:2]
        y0 = int(h * 0.45)
        roi = image[y0:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # bright road paint extraction with dynamic thresholding to handle different lighting
        mean_val = np.mean(blur)
        std_val = np.std(blur)
        thresh_val = max(120, min(220, int(mean_val + 1.5 * std_val)))
        _, white = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

        # remove small noise and connect broken stripes
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, open_kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, close_kernel)

        # 윤곽선(Contours)을 찾아 횡단보도의 특징을 가진 흰색 무늬들을 필터링합니다
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_boxes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            # 노이즈나 너무 거대한 차선 덩어리 화살표 제외
            if area < (w * h * 0.0005) or area > (w * h * 0.1):
                continue
            
            aspect_ratio = float(cw) / float(ch) if ch > 0 else 0
            if 0.2 < aspect_ratio < 5.0:  
                valid_boxes.append((x, y, cw, ch))

        # 횡단보도는 여러 개의 흰색 줄이 동일한 선상(비슷한 Y좌표)에 무리지어 나타나는 특징이 있습니다.
        # 바닥에 있는 화살표나 옆 차선(거대한 세로줄)까지 박스가 넓게 묶이는 것을 방지하기 위해 
        # 무늬들을 Y좌표 기준으로 그룹핑(Clustering)합니다.
        clusters = []
        for box in valid_boxes:
            bx, by, bw, bh = box
            by_center = by + bh / 2
            
            added = False
            for cluster in clusters:
                # 클러스터 무리의 평균 Y 중심점 계산
                cluster_y_center = sum(b[1] + b[3]/2 for b in cluster) / len(cluster)
                
                # 동일 횡단보도로 판별할 Y축 오차 범위: 
                # 원근감이나 도로 기울기(Slant)를 허용하되, 높이가 완전히 다른 물체(하단 직진 화살표 등)는 묶지 않음
                if abs(by_center - cluster_y_center) < max(bh * 1.5, h * 0.15):
                    cluster.append(box)
                    added = True
                    break
            
            if not added:
                clusters.append([box])

        # 가장 많은 무늬(Stripe)를 뭉쳐놓은 클러스터를 '횡단보도'라고 판단합니다 (최대 밴드 덩어리, 최소 4줄 이상)
        best_cluster = None
        for cluster in clusters:
            if len(cluster) >= 4:
                if best_cluster is None or len(cluster) > len(best_cluster):
                    best_cluster = cluster

        if best_cluster is None:
            return None, white

        # 선택된 클러스터만으로 아주 타이트한 Bounding Box를 계산합니다
        x_min = min(b[0] for b in best_cluster)
        y_min = min(b[1] for b in best_cluster)
        x_max = max(b[0] + b[2] for b in best_cluster)
        y_max = max(b[1] + b[3] for b in best_cluster)

        cw, ch = x_max - x_min, y_max - y_min
        if cw < int(w * 0.1) or ch < int(h * 0.02):
            return None, white

        return {
            "label": "crosswalk",
            "score": 0.0,
            "box": (x_min, y_min + y0, x_max, y_max + y0),
            "band_count": len(best_cluster),
        }, white

    def draw_detections(self, image: np.ndarray, detections, crosswalk=None):
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            raw_label = det["raw_label"]
            score = det["score"]
            color = BOX_COLORS.get(label, (0, 0, 255))

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            caption = f"{raw_label} {score:.2f}"
            cv2.putText(output, caption, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if crosswalk is not None:
            x1, y1, x2, y2 = crosswalk["box"]
            color = BOX_COLORS["crosswalk"]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            caption = f"crosswalk bands={crosswalk['band_count']}"
            cv2.putText(output, caption, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output

    def run(self, image_path: str, output_path: str = "result.jpg", save_mask: bool = False):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

        detections = self.detect_yolo(image)
        crosswalk, crosswalk_mask = self.detect_crosswalk(image)
        result_image = self.draw_detections(image, detections, crosswalk)

        cv2.imwrite(output_path, result_image)

        if save_mask:
            mask_path = str(Path(output_path).with_name(Path(output_path).stem + "_crosswalk_mask.png"))
            cv2.imwrite(mask_path, crosswalk_mask)
            print(f"횡단보도 마스크 저장: {mask_path}")

        print("=== 감지 결과 ===")
        for det in detections:
            print(f"- {det['raw_label']:<13} score={det['score']:.2f} box={det['box']}")
        if crosswalk is not None:
            print(f"- crosswalk      bands={crosswalk['band_count']} box={crosswalk['box']}")
        else:
            print("- crosswalk      not detected")
        print(f"결과 이미지 저장: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight YOLO road-scene detector")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--output", default="result.jpg", help="출력 이미지 경로")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO 모델 파일명")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--save-mask", action="store_true", help="횡단보도 이진 마스크도 저장")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = RoadSceneDetector(model_name=args.model, conf=args.conf, imgsz=args.imgsz)
    detector.run(args.image, args.output, args.save_mask)
