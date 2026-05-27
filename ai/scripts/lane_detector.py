"""
lane_detector.py – 강건한 차선 인식 프로그램
==============================================
파이프라인
  1. 전처리   : 그레이스케일 → 가우시안 블러 → Canny 엣지
  2. ROI 마스크: 이미지 하단 사다리꼴 영역만 사용
  3. Hough 변환: 선분 추출 → 기울기 필터로 좌/우 차선 분리 → 가중 평균 피팅
  4. 슬라이딩 윈도우: 버드아이뷰(Bird's-eye view) 변환 후 다항식 피팅 (곡선 차선 대응)
  5. 시각화   : 차선 영역 반투명 오버레이 + 라인 그리기

실행:
    python lane_detector.py --image test.jpg
    python lane_detector.py --image test.jpg --output lane_result.jpg --debug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────── 설정 ────────────────────────────
# ROI 꼭짓점 (이미지 너비/높이 비율로 지정)
ROI_VERTICES_RATIO = [
    (0.0,  1.0),   # 좌하단
    (0.42, 0.58),  # 좌상단 (소실점 근처)
    (0.58, 0.58),  # 우상단
    (1.0,  1.0),   # 우하단
]

# Canny 임계값
CANNY_LOW  = 50
CANNY_HIGH = 150

# Hough 파라미터
HOUGH_RHO       = 1
HOUGH_THETA     = np.pi / 180
HOUGH_THRESHOLD = 30
HOUGH_MIN_LEN   = 30
HOUGH_MAX_GAP   = 100

# 기울기 필터 (수평에 가까운 선 제거)
MIN_SLOPE = 0.3

# Bird's-eye view 변환 크기
BEV_W, BEV_H = 400, 600
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# 1. Hough 기반 직선 차선 검출 파이프라인
# ══════════════════════════════════════════════════════════════

def preprocess(img: np.ndarray) -> np.ndarray:
    """그레이스케일 → CLAHE(명암 정규화) → 가우시안 블러 → Canny"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return edges


def roi_mask(edges: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(edges, mask)


def get_roi_vertices(h: int, w: int) -> np.ndarray:
    pts = [(int(x * w), int(y * h)) for x, y in ROI_VERTICES_RATIO]
    return np.array(pts, dtype=np.int32)


def fit_line_from_segments(segments, h: int):
    """선분들을 끝점 좌표로 분리하고 기울기·절편의 가중 평균으로 하나의 라인 반환."""
    xs, ys, ws = [], [], []
    for x1, y1, x2, y2 in segments:
        length = np.hypot(x2 - x1, y2 - y1)
        xs += [x1, x2]
        ys += [y1, y2]
        ws += [length, length]
    if len(xs) < 2:
        return None
    try:
        coeffs = np.polyfit(ys, xs, 1, w=ws)  # x = a*y + b
    except np.linalg.LinAlgError:
        return None
    a, b = coeffs
    y_bot = h
    y_top = int(h * 0.58)
    x_bot = int(a * y_bot + b)
    x_top = int(a * y_top + b)
    return (x_bot, y_bot, x_top, y_top)


def hough_lanes(masked_edges: np.ndarray, h: int, w: int):
    """Hough 변환 → 좌/우 차선 라인 반환"""
    lines = cv2.HoughLinesP(
        masked_edges,
        HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
    )
    if lines is None:
        return None, None

    left_segs, right_segs = [], []
    cx = w / 2
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_SLOPE:
            continue
        if slope < 0 and x1 < cx and x2 < cx:
            left_segs.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > cx and x2 > cx:
            right_segs.append((x1, y1, x2, y2))

    left_line  = fit_line_from_segments(left_segs, h)
    right_line = fit_line_from_segments(right_segs, h)
    return left_line, right_line


# ══════════════════════════════════════════════════════════════
# 2. Bird's-eye view 슬라이딩 윈도우 (곡선 대응)
# ══════════════════════════════════════════════════════════════

def get_bev_transform(h: int, w: int):
    """Bird's-eye view 원근 변환 행렬 계산"""
    src_pts = np.float32([
        [w * 0.15, h * 1.0],
        [w * 0.44, h * 0.60],
        [w * 0.56, h * 0.60],
        [w * 0.85, h * 1.0],
    ])
    dst_pts = np.float32([
        [BEV_W * 0.2,  BEV_H],
        [BEV_W * 0.2,  0],
        [BEV_W * 0.8,  0],
        [BEV_W * 0.8,  BEV_H],
    ])
    M    = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    return M, Minv


def sliding_window_fit(binary_bev: np.ndarray):
    """슬라이딩 윈도우로 좌/우 차선 픽셀 찾아 2차 다항식 피팅"""
    hist = np.sum(binary_bev[BEV_H // 2:], axis=0)
    mid  = BEV_W // 2
    left_x  = np.argmax(hist[:mid])
    right_x = np.argmax(hist[mid:]) + mid

    n_windows = 10
    win_h = BEV_H // n_windows
    margin = 40
    min_pix = 30

    nz_y, nz_x = binary_bev.nonzero()
    left_pix, right_pix = [], []

    for win in range(n_windows):
        y_lo = BEV_H - (win + 1) * win_h
        y_hi = BEV_H - win * win_h
        xl_lo, xl_hi = left_x  - margin, left_x  + margin
        xr_lo, xr_hi = right_x - margin, right_x + margin

        good_left  = ((nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= xl_lo) & (nz_x < xl_hi)).nonzero()[0]
        good_right = ((nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= xr_lo) & (nz_x < xr_hi)).nonzero()[0]

        left_pix.append(good_left)
        right_pix.append(good_right)

        if len(good_left)  > min_pix: left_x  = int(np.mean(nz_x[good_left]))
        if len(good_right) > min_pix: right_x = int(np.mean(nz_x[good_right]))

    left_idx  = np.concatenate(left_pix)
    right_idx = np.concatenate(right_pix)

    ys = np.linspace(0, BEV_H - 1, BEV_H)
    left_fit = right_fit = None

    if len(left_idx) > 50:
        left_fit = np.polyfit(nz_y[left_idx], nz_x[left_idx], 2)
    if len(right_idx) > 50:
        right_fit = np.polyfit(nz_y[right_idx], nz_x[right_idx], 2)

    return left_fit, right_fit, ys


def draw_bev_lane(binary_bev: np.ndarray, left_fit, right_fit, ys, Minv, orig_shape):
    """BEV 좌표 차선 다항식을 원본 이미지 좌표로 역변환해 오버레이"""
    h, w = orig_shape[:2]
    lane_img = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

    if left_fit is not None and right_fit is not None:
        left_x  = np.polyval(left_fit,  ys)
        right_x = np.polyval(right_fit, ys)

        left_pts  = np.array([np.stack([left_x,  ys], axis=1)], dtype=np.int32)
        right_pts = np.array([np.stack([right_x, ys], axis=1)[::-1]], dtype=np.int32)
        poly_pts  = np.hstack([left_pts, right_pts])

        cv2.fillPoly(lane_img, poly_pts, (0, 200, 80))          # 초록 채움
        cv2.polylines(lane_img, left_pts,  False, (255, 200, 0), 6)  # 노랑
        cv2.polylines(lane_img, right_pts[:, ::-1], False, (255, 200, 0), 6)

    # 역원근 변환 → 원본 해상도
    lane_orig = cv2.warpPerspective(lane_img, Minv, (w, h))
    return lane_orig


# ══════════════════════════════════════════════════════════════
# 3. 최종 시각화
# ══════════════════════════════════════════════════════════════

def overlay_hough_lines(img: np.ndarray, left_line, right_line) -> np.ndarray:
    line_img = np.zeros_like(img)
    for line in [left_line, right_line]:
        if line is None:
            continue
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 200, 255), 6)

    # 차선 사이 채우기
    if left_line and right_line:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [left_line[2],  left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]],
        ], dtype=np.int32)
        cv2.fillPoly(line_img, [pts], (0, 120, 40))

    return cv2.addWeighted(img, 1.0, line_img, 0.45, 0)


def put_status(img: np.ndarray, method: str, left: bool, right: bool) -> np.ndarray:
    out = img.copy()
    color = (0, 255, 128)
    cv2.putText(out, f"Method : {method}",       (15, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(out, f"Left   : {'OK' if left  else 'X'}", (15, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(out, f"Right  : {'OK' if right else 'X'}", (15, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


# ══════════════════════════════════════════════════════════════
# 4. 메인 파이프라인
# ══════════════════════════════════════════════════════════════

def detect_lanes(
    image_path: str,
    output_path: str = "lane_result.jpg",
    debug: bool = False,
) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    h, w = img.shape[:2]

    # ── 전처리 ──────────────────────────────────────────────
    edges    = preprocess(img)
    vertices = get_roi_vertices(h, w)
    masked   = roi_mask(edges, vertices)

    # ── [A] Hough 직선 검출 ─────────────────────────────────
    left_line, right_line = hough_lanes(masked, h, w)
    hough_ok = left_line is not None or right_line is not None

    # ── [B] Bird's-eye view 슬라이딩 윈도우 ─────────────────
    M, Minv  = get_bev_transform(h, w)
    bev_bin  = cv2.warpPerspective(masked, M, (BEV_W, BEV_H))
    left_fit, right_fit, ys = sliding_window_fit(bev_bin)
    bev_ok   = left_fit is not None or right_fit is not None

    # ── 결과 합성 ────────────────────────────────────────────
    result = img.copy()

    # BEV 차선 오버레이 (우선)
    if bev_ok:
        lane_overlay = draw_bev_lane(bev_bin, left_fit, right_fit, ys, Minv, img.shape)
        result = cv2.addWeighted(result, 1.0, lane_overlay, 0.5, 0)
        method = "Sliding Window (BEV)"
    elif hough_ok:
        result = overlay_hough_lines(result, left_line, right_line)
        method = "Hough Transform"
    else:
        method = "Not Detected"

    # Hough 라인도 함께 표시 (얇게)
    if hough_ok:
        for line in [left_line, right_line]:
            if line:
                x1, y1, x2, y2 = line
                cv2.line(result, (x1, y1), (x2, y2), (0, 200, 255), 3)

    # ROI 경계 표시 (디버그)
    if debug:
        cv2.polylines(result, [vertices], True, (255, 0, 200), 1)

    # 상태 정보
    result = put_status(result, method, left_line is not None, right_line is not None)

    cv2.imwrite(output_path, result)
    print(f"[lane_detector] method={method} | 좌={left_line is not None} 우={right_line is not None}")
    print(f"결과 저장: {output_path}")

    # ── 디버그 이미지 ─────────────────────────────────────
    if debug:
        dbg_dir = Path(output_path).parent / "lane_debug"
        dbg_dir.mkdir(exist_ok=True)
        stem = Path(output_path).stem
        cv2.imwrite(str(dbg_dir / f"{stem}_edges.jpg"),  edges)
        cv2.imwrite(str(dbg_dir / f"{stem}_masked.jpg"), masked)
        bev_color = cv2.cvtColor(bev_bin, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(dbg_dir / f"{stem}_bev.jpg"),    bev_color)
        print(f"디버그 이미지 저장: {dbg_dir}/")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Lane Detector – 강건한 차선 인식")
    parser.add_argument("--image",  required=True,               help="입력 이미지 경로")
    parser.add_argument("--output", default="lane_result.jpg",   help="출력 이미지 경로")
    parser.add_argument("--debug",  action="store_true",         help="디버그 이미지 저장")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_lanes(args.image, args.output, args.debug)
