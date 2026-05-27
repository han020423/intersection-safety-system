"""
light_yolo_crosswalk_detector2.py  – v2
변경 사항:
- 횡단보도(crosswalk) 인식을 커스텀 학습 모델(best.pt)로 수행
- OpenCV 휴리스틱 횡단보도 검출 제거
- 보행자, 차량, 신호등, 좌회전표지판 등도 동일 모델로 탐지

실행 예시:
    python light_yolo_crosswalk_detector2.py --image test.jpg --output result2.jpg
    python light_yolo_crosswalk_detector2.py --image test.jpg --model runs/detect/yolo11n_custom/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── 커스텀 모델 클래스 이름(학습 시 data.yaml 기준) ──────────────────
CUSTOM_CLASS_NAMES = {
    0: "pedestrian",
    1: "vehicle",
    2: "traffic_light_vehicle",
    3: "traffic_light_pedestrian",
    4: "crosswalk",
    5: "stop_line",
    6: "left_turn_sign",
}

BOX_COLORS = {
    "pedestrian":              (0, 255, 0),       # 초록
    "vehicle":                 (255, 80, 80),      # 파랑
    "traffic_light_vehicle":   (0, 220, 255),      # 노랑
    "traffic_light_pedestrian":(0, 180, 255),      # 주황
    "crosswalk":               (255, 255, 255),    # 흰색
    "stop_line":               (180, 180, 180),    # 회색
    "left_turn_sign":          (255, 100, 200),    # 분홍
}
DEFAULT_COLOR = (0, 0, 255)


class RoadSceneDetectorV2:
    def __init__(
        self,
        model_path: str = "runs/detect/yolo11n_custom/weights/best.pt",
        conf: float = 0.25,
        imgsz: int = 640,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image: np.ndarray) -> list[dict]:
        """커스텀 모델 단일 추론으로 모든 클래스 탐지"""
        result = self.model.predict(
            image, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]

        detections = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        clss  = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, cls_id, score in zip(boxes, clss, confs):
            class_name = result.names.get(cls_id, str(cls_id)).lower()
            detections.append(
                {
                    "label": class_name,
                    "score": float(score),
                    "box":   tuple(box.tolist()),
                }
            )
        return detections

    def draw_detections(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        output = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            score = det["score"]
            color = BOX_COLORS.get(label, DEFAULT_COLOR)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            caption = f"{label} {score:.2f}"
            cv2.putText(
                output, caption,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
        return output

    def run(self, image_path: str, output_path: str = "result2.jpg"):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

        detections = self.detect(image)
        result_image = self.draw_detections(image, detections)
        cv2.imwrite(output_path, result_image)

        print("=== 감지 결과 (v2) ===")
        for det in detections:
            print(f"- {det['label']:<26} score={det['score']:.2f}  box={det['box']}")
        if not any(d["label"] == "crosswalk" for d in detections):
            print("- crosswalk               not detected")
        print(f"결과 이미지 저장: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Road Scene Detector v2 (custom model)")
    parser.add_argument("--image",  required=True,  help="입력 이미지 경로")
    parser.add_argument("--output", default="result2.jpg", help="출력 이미지 경로")
    parser.add_argument(
        "--model",
        default="runs/detect/yolo11n_custom/weights/best.pt",
        help="커스텀 YOLO 가중치 경로",
    )
    parser.add_argument("--conf",   type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz",  type=int,   default=640,  help="inference image size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = RoadSceneDetectorV2(
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    detector.run(args.image, args.output)
