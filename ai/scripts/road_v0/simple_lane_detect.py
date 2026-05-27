#!/usr/bin/env python3
"""
Simple lane detection — no BEV, no YOLO.
Just color mask + Hough lines + left/right grouping.

Usage:
    python simple_lane_detect.py --source test.jpg --show
"""

import argparse
import numpy as np
import cv2


def build_roi(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([
        [int(0.02 * w), h - 1],
        [int(0.98 * w), h - 1],
        [int(0.60 * w), int(0.50 * h)],
        [int(0.40 * w), int(0.50 * h)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def get_lane_mask(frame: np.ndarray, roi: np.ndarray) -> np.ndarray:
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    l_ch = hls[:, :, 1]
    s_ch = hls[:, :, 2]

    white = cv2.inRange(l_ch, 200, 255)
    yellow = cv2.inRange(s_ch, 100, 255)

    sobelx = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.uint8(255 * np.abs(sobelx) / (np.max(np.abs(sobelx)) + 1e-6))
    sobel_mask = cv2.inRange(abs_sobel, 20, 200)

    combined = cv2.bitwise_or(white, cv2.bitwise_or(yellow, sobel_mask))
    combined = cv2.bitwise_and(combined, roi)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return combined


def detect_lanes(frame: np.ndarray):
    h, w = frame.shape[:2]
    roi = build_roi(h, w)
    mask = get_lane_mask(frame, roi)

    # Canny + Hough
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=30, maxLineGap=20)

    left_lines = []
    right_lines = []
    mid_x = w // 2

    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # 너무 수평인 선 제거
            if abs(slope) < 0.3:
                continue

            # 기울기와 위치로 좌/우 분류
            if slope < 0 and x1 < mid_x and x2 < mid_x:
                left_lines.append(line)
            elif slope > 0 and x1 > mid_x and x2 > mid_x:
                right_lines.append(line)

    left_pts = _fit_line(left_lines, h) if left_lines else None
    right_pts = _fit_line(right_lines, h) if right_lines else None

    return left_pts, right_pts, mask


def _fit_line(lines, h: int):
    """여러 Hough 선분의 점들을 모아 1차 직선 피팅 → 두 점 반환."""
    xs, ys = [], []
    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    if len(xs) < 4:
        return None

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    try:
        coeffs = np.polyfit(ys, xs, 1)  # x = f(y)
    except Exception:
        return None

    y_bottom = h - 1
    y_top = int(h * 0.55)
    x_bottom = int(np.polyval(coeffs, y_bottom))
    x_top = int(np.polyval(coeffs, y_top))

    return np.array([[x_top, y_top], [x_bottom, y_bottom]], dtype=np.int32)


def draw(frame: np.ndarray, left_pts, right_pts) -> np.ndarray:
    vis = frame.copy()
    h, w = frame.shape[:2]

    # 차선 사이 영역 채우기
    if left_pts is not None and right_pts is not None:
        fill = np.zeros_like(frame)
        poly = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]])
        cv2.fillPoly(fill, [poly], (0, 180, 0))
        vis = cv2.addWeighted(vis, 1.0, fill, 0.3, 0)

    # 차선 그리기
    if left_pts is not None:
        cv2.line(vis, tuple(left_pts[0]), tuple(left_pts[1]), (255, 80, 0), 3)
    if right_pts is not None:
        cv2.line(vis, tuple(right_pts[0]), tuple(right_pts[1]), (255, 80, 0), 3)

    # 중심 표시
    if left_pts is not None and right_pts is not None:
        cx = (left_pts[1][0] + right_pts[1][0]) // 2
        cv2.circle(vis, (cx, h - 20), 6, (0, 255, 0), -1)
        cv2.line(vis, (w // 2, h - 1), (w // 2, h - 40), (120, 120, 120), 1)
        cv2.line(vis, (cx, h - 1), (cx, h - 40), (0, 255, 0), 2)
        label = f"offset: {cx - w // 2:+d}px"
    elif left_pts is not None:
        label = "Left lane only"
    elif right_pts is not None:
        label = "Right lane only"
    else:
        label = "No lane detected"

    cv2.rectangle(vis, (8, 8), (280, 40), (0, 0, 0), -1)
    cv2.putText(vis, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Simple lane detection (no BEV)")
    parser.add_argument("--source", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    frame = cv2.imread(args.source)
    if frame is None:
        raise RuntimeError(f"Cannot open: {args.source}")

    frame = cv2.resize(frame, (args.width, args.height))
    left, right, mask = detect_lanes(frame)
    vis = draw(frame, left, right)

    print(f"Left:  {'detected' if left is not None else 'not found'}")
    print(f"Right: {'detected' if right is not None else 'not found'}")

    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"Saved to {args.save}")

    if args.show:
        cv2.imshow("Lane Detection", vis)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
