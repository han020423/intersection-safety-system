"""
lane_detector_v0.py – 교차로 특화 강건 차선 인식 프로그램 (v0)
================================================================
지원 차선:
  - 흰색 실선 / 흰색 점선
  - 노란색 실선 / 노란색 점선

파이프라인:
  1. 전처리   : BGR → HSV + LAB 색상 분리로 흰색/노란색 마스크 추출
  2. 엣지 검출: CLAHE 명암 정규화 → 가우시안 블러 → Canny
  3. ROI 마스크: 동적 사다리꼴 ROI (이미지별 자동 조정)
  4. Hough 변환: 선분 추출 → 기울기 클러스터링으로 좌/우 차선 분리
  5. Bird's-eye view: 원근 변환 + 슬라이딩 윈도우 + 2차 다항식 피팅
  6. 이상치 제거: 이전 프레임 기반 temporal smoothing (LaneTracker)
  7. 시각화   : 반투명 오버레이 + 색상 구분 + 상태 정보 HUD

실행:
    # 단일 이미지
    python lane_detector_v0.py --image v0/image/08_103700_220615_01.jpg

    # 폴더 일괄 처리
    python lane_detector_v0.py --folder v0/image --output_dir v0/results

    # # ── 색상 마스크 HSV 범위 ─────────────────────────────────────────
# 흰색 차선: 채도 0~50, 밝기 160~255
# (ROI 적용으로 하늘 영역 제외되미로 다소 넓게 설정)
WHITE_HSV_LOW  = np.array([0,   0,  160], dtype=np.uint8)
WHITE_HSV_HIGH = np.array([180, 50, 255], dtype=np.uint8)

# 노란색 차선: 한국 도로 마킹 (H:15~38, 채도 중상, 밝기 중상)
YELLOW_HSV_LOW  = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HSV_HIGH = np.array([38, 255, 255], dtype=np.uint8)

# ── ROI 비율 ─────────────────────────────────────────────────────
# 이미지 하단 60%에서만 차선 탐색 (ROI가 색상마스크에 적용되므로 하늘 제외됨)
ROI_TOP_RATIO    = 0.58   # 상단 y (소실점 직전)
ROI_TOP_W_RATIO  = 0.10   # 상단폭 널직하게
ROI_BOT_RATIO    = 1.00   # 하단 y
ROI_BOT_W_RATIO  = 0.50   # 하단폭�────────────────────────
# 흰색: 채도 매우 낮고(0~35), 밝기 높음(200~255)
# 하늘/건물 제외: 도로 위 흰색 차선은 채도가 거의 0에 가깝고 매우 밝음
WHITE_HSV_LOW  = np.array([0,   0,  200], dtype=np.uint8)
WHITE_HSV_HIGH = np.array([180, 35, 255], dtype=np.uint8)

# 노란색: 한국 도로 노란 차선 (H: 18~32, 채도 높음, 밝기 중간 이상)
# 나무/풀 제외: 채도를 100 이상으로 높임
YELLOW_HSV_LOW  = np.array([18, 100, 100], dtype=np.uint8)
YELLOW_HSV_HIGH = np.array([32, 255, 255], dtype=np.uint8)

# ── ROI 비율 ─────────────────────────────────────────────────────
# 하단 사다리꼴: 도로가 보이는 이미지 하단 영역에 집중
ROI_TOP_RATIO    = 0.60   # 상단 y 시작 (이미지 높이 대비)
ROI_TOP_W_RATIO  = 0.08   # 상단폭 (중심에서 좌우) – 협소하게
ROI_BOT_RATIO    = 1.00   # 하단 y (이미지 맨 아래)
ROI_BOT_W_RATIO  = 0.52   # 하단폭 (중심에서 좌우)

# ── Canny ────────────────────────────────────────────────────────
CANNY_LOW  = 40
CANNY_HIGH = 120

# ── HoughLinesP ─────────────────────────────────────────────────
HOUGH_RHO       = 1
HOUGH_THETA     = np.pi / 180
HOUGH_THRESHOLD = 25
HOUGH_MIN_LEN   = 20
HOUGH_MAX_GAP   = 80

# ── 기울기 필터 ──────────────────────────────────────────────────
MIN_SLOPE = 0.25        # 수평에 가까운 선 제거
MAX_SLOPE = 10.0        # 수직에 가까운 선 제거 (교차로 정지선 등)

# ── Bird's-eye view ──────────────────────────────────────────────
BEV_W, BEV_H = 480, 640

# ── Temporal Smoothing ───────────────────────────────────────────
SMOOTH_N = 5   # 평균에 사용할 이전 프레임 수


# ═══════════════════════════════════════════════════════════════════
# 1. 색상 마스크 추출 (흰색 + 노란색)
# ═══════════════════════════════════════════════════════════════════

def extract_color_mask(img: np.ndarray,
                       roi_vertices: np.ndarray = None) -> tuple:
    """
    HSV 색공간에서 흰색과 노란색 차선 픽셀을 추출.
    roi_vertices를 전달하면 ROI 내부만 분석해 하늘/건물 오검출을 방지.
    Returns: (white_mask, yellow_mask, combined_mask)
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── ROI 마스크 생성 ──────────────────────────────────
    roi_mask_base = np.zeros((h, w), dtype=np.uint8)
    if roi_vertices is not None:
        cv2.fillPoly(roi_mask_base, [roi_vertices], 255)
    else:
        roi_mask_base[:] = 255

    # ── 흰색 마스크 ─────────────────────────────────────
    # HSV: 채도 낮고(0~35) 밝기 높음(200~255) – 하늘은 채도가 있어 제외됨
    white_mask = cv2.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)
    # ROI 내부만 유효
    white_mask = cv2.bitwise_and(white_mask, roi_mask_base)

    # ── 노란색 마스크 ────────────────────────────────────
    yellow_mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)
    # ROI 내부만 유효
    yellow_mask = cv2.bitwise_and(yellow_mask, roi_mask_base)

    # ── 합성 ────────────────────────────────────────────
    combined = cv2.bitwise_or(white_mask, yellow_mask)

    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    white_clean  = cv2.bitwise_and(white_mask,  combined)
    yellow_clean = cv2.bitwise_and(yellow_mask, combined)

    return white_clean, yellow_clean, combined


# ═══════════════════════════════════════════════════════════════════
# 2. 전처리 (엣지 검출)
# ═══════════════════════════════════════════════════════════════════

def preprocess(img: np.ndarray, color_mask: np.ndarray) -> np.ndarray:
    """
    CLAHE + Canny 엣지 검출.
    Hough fallback용으로만 사용; BEV는 color_mask를 직접 사용.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (7, 7), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return edges


# ═══════════════════════════════════════════════════════════════════
# 3. ROI 마스크
# ═══════════════════════════════════════════════════════════════════

def get_roi_vertices(h: int, w: int) -> np.ndarray:
    """동적 사다리꼴 ROI 꼭짓점 계산."""
    cx = w / 2
    top_y  = int(h * ROI_TOP_RATIO)
    bot_y  = int(h * ROI_BOT_RATIO)
    top_w  = int(w * ROI_TOP_W_RATIO)
    bot_w  = int(w * ROI_BOT_W_RATIO)

    pts = np.array([
        (int(cx - bot_w), bot_y),   # 좌하단
        (int(cx - top_w), top_y),   # 좌상단
        (int(cx + top_w), top_y),   # 우상단
        (int(cx + bot_w), bot_y),   # 우하단
    ], dtype=np.int32)
    return pts


def apply_roi_mask(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)


# ═══════════════════════════════════════════════════════════════════
# 4. Hough 차선 검출 + 클러스터링
# ═══════════════════════════════════════════════════════════════════

def cluster_lines_by_slope(lines, cx: float, slope_tol: float = 0.15):
    """
    기울기 기반 클러스터링으로 좌/우 차선 선분 분리.
    교차로에서 여러 차선이 혼재할 때 주요 선분만 선택.
    """
    left_segs, right_segs = [], []

    if lines is None:
        return left_segs, right_segs

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_SLOPE or abs(slope) > MAX_SLOPE:
            continue

        length = np.hypot(x2 - x1, y2 - y1)
        mid_x  = (x1 + x2) / 2

        if slope < 0 and mid_x < cx:
            left_segs.append((x1, y1, x2, y2, length))
        elif slope > 0 and mid_x > cx:
            right_segs.append((x1, y1, x2, y2, length))

    return left_segs, right_segs


def weighted_line_fit(segments, h: int, top_y_ratio: float = 0.56):
    """
    선분들을 길이 가중 다항식 피팅으로 하나의 직선으로 통합.
    이상치(RANSAC 유사 필터링) 제거 포함.
    """
    if not segments:
        return None

    xs, ys, ws = [], [], []
    for x1, y1, x2, y2, length in segments:
        xs += [x1, x2]
        ys += [y1, y2]
        ws += [length, length]

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    ws = np.array(ws, dtype=np.float32)

    try:
        # RANSAC-like: 중간값에서 크게 벗어나는 점 제거
        coeffs_all = np.polyfit(ys, xs, 1, w=ws)
        xs_pred = np.polyval(coeffs_all, ys)
        residuals = np.abs(xs - xs_pred)
        threshold = np.median(residuals) * 2.0 + 5
        inlier_mask = residuals < threshold
        if np.sum(inlier_mask) < 2:
            inlier_mask = np.ones_like(inlier_mask, dtype=bool)

        coeffs = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1,
                            w=ws[inlier_mask])
    except (np.linalg.LinAlgError, ValueError):
        return None

    a, b = coeffs
    y_bot = h
    y_top = int(h * top_y_ratio)
    x_bot = int(a * y_bot + b)
    x_top = int(a * y_top + b)
    return (x_bot, y_bot, x_top, y_top)


def hough_lane_detect(masked_edges: np.ndarray, h: int, w: int):
    """HoughLinesP → 클러스터링 → 피팅."""
    lines = cv2.HoughLinesP(
        masked_edges,
        HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
    )
    left_segs, right_segs = cluster_lines_by_slope(lines, w / 2)

    left_line  = weighted_line_fit(left_segs,  h)
    right_line = weighted_line_fit(right_segs, h)
    return left_line, right_line


# ═══════════════════════════════════════════════════════════════════
# 5. Bird's-eye view 슬라이딩 윈도우 (곡선 차선 대응)
# ═══════════════════════════════════════════════════════════════════

def get_bev_transform(h: int, w: int):
    """원근 변환 행렬 계산."""
    src = np.float32([
        [w * (0.5 - ROI_BOT_W_RATIO), h * ROI_BOT_RATIO],
        [w * (0.5 - ROI_TOP_W_RATIO), h * ROI_TOP_RATIO],
        [w * (0.5 + ROI_TOP_W_RATIO), h * ROI_TOP_RATIO],
        [w * (0.5 + ROI_BOT_W_RATIO), h * ROI_BOT_RATIO],
    ])
    dst = np.float32([
        [BEV_W * 0.15, BEV_H],
        [BEV_W * 0.15, 0],
        [BEV_W * 0.85, 0],
        [BEV_W * 0.85, BEV_H],
    ])
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def sliding_window_fit(binary_bev: np.ndarray, n_windows: int = 12,
                       margin: int = 35, min_pix: int = 30):
    """
    슬라이딩 윈도우 → 좌/우 차선 픽셀 수집 → 2차 다항식 피팅.
    히스토그램 피크를 초기 탐색점으로 사용.
    """
    hist = np.sum(binary_bev[BEV_H // 2:], axis=0).astype(np.float32)
    # 스무딩
    hist = np.convolve(hist, np.ones(15) / 15, mode='same')

    mid = BEV_W // 2
    left_x  = int(np.argmax(hist[:mid]))
    right_x = int(np.argmax(hist[mid:]) + mid)

    win_h = BEV_H // n_windows
    nz_y, nz_x = binary_bev.nonzero()
    left_pix, right_pix = [], []

    for win in range(n_windows):
        y_lo = BEV_H - (win + 1) * win_h
        y_hi = BEV_H - win * win_h
        xl_lo, xl_hi = left_x  - margin, left_x  + margin
        xr_lo, xr_hi = right_x - margin, right_x + margin

        good_left  = np.where(
            (nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= xl_lo) & (nz_x < xl_hi)
        )[0]
        good_right = np.where(
            (nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= xr_lo) & (nz_x < xr_hi)
        )[0]

        left_pix.append(good_left)
        right_pix.append(good_right)

        if len(good_left)  > min_pix: left_x  = int(np.mean(nz_x[good_left]))
        if len(good_right) > min_pix: right_x = int(np.mean(nz_x[good_right]))

    left_idx  = np.concatenate(left_pix)
    right_idx = np.concatenate(right_pix)

    ys = np.linspace(0, BEV_H - 1, BEV_H)
    left_fit = right_fit = None

    if len(left_idx) > 60:
        try:
            left_fit = np.polyfit(nz_y[left_idx], nz_x[left_idx], 2)
        except (np.linalg.LinAlgError, ValueError):
            pass
    if len(right_idx) > 60:
        try:
            right_fit = np.polyfit(nz_y[right_idx], nz_x[right_idx], 2)
        except (np.linalg.LinAlgError, ValueError):
            pass

    return left_fit, right_fit, ys


# ═══════════════════════════════════════════════════════════════════
# 6. Temporal Smoothing (LaneTracker)
# ═══════════════════════════════════════════════════════════════════

class LaneTracker:
    """이전 프레임 피팅 계수를 저장해 급격한 변화를 완화."""

    def __init__(self, n: int = SMOOTH_N):
        self.left_fits  = deque(maxlen=n)
        self.right_fits = deque(maxlen=n)

    def update(self, left_fit, right_fit):
        if left_fit  is not None: self.left_fits.append(left_fit)
        if right_fit is not None: self.right_fits.append(right_fit)

    def smooth_left(self):
        if not self.left_fits: return None
        return np.mean(self.left_fits, axis=0)

    def smooth_right(self):
        if not self.right_fits: return None
        return np.mean(self.right_fits, axis=0)


_tracker = LaneTracker()   # 모듈 레벨 전역 트래커


# ═══════════════════════════════════════════════════════════════════
# 7. 시각화
# ═══════════════════════════════════════════════════════════════════

def draw_bev_lane_overlay(left_fit, right_fit, ys, Minv, orig_shape):
    """BEV 다항식 → 역변환 → 원본 이미지에 오버레이할 레이어 반환."""
    h, w = orig_shape[:2]
    lane_img = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

    pts_list = []
    if left_fit is not None:
        lx = np.polyval(left_fit, ys).clip(0, BEV_W - 1)
        pts_list.append(('left', lx, ys))
    if right_fit is not None:
        rx = np.polyval(right_fit, ys).clip(0, BEV_W - 1)
        pts_list.append(('right', rx, ys))

    # 채움 영역
    if left_fit is not None and right_fit is not None:
        lx = np.polyval(left_fit,  ys).clip(0, BEV_W - 1)
        rx = np.polyval(right_fit, ys).clip(0, BEV_W - 1)
        left_pts  = np.stack([lx, ys], axis=1).astype(np.int32)
        right_pts = np.stack([rx, ys], axis=1).astype(np.int32)[::-1]
        polygon   = np.vstack([left_pts, right_pts])
        cv2.fillPoly(lane_img, [polygon], (0, 180, 60))   # 초록

    # 차선 선
    for side, xs, ys_ in pts_list:
        color = (255, 220, 0) if side == 'left' else (0, 200, 255)  # 노랑/하늘
        pts = np.stack([xs, ys_], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(lane_img, [pts], False, color, 8, cv2.LINE_AA)

    # 역원근 변환
    lane_orig = cv2.warpPerspective(lane_img, Minv, (w, h))
    return lane_orig


def overlay_hough_lines(img: np.ndarray, left_line, right_line) -> np.ndarray:
    """Hough 직선 차선을 원본 이미지에 오버레이."""
    line_img = np.zeros_like(img)

    for line, color in [(left_line, (255, 220, 0)), (right_line, (0, 200, 255))]:
        if line is None:
            continue
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), color, 8, cv2.LINE_AA)

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


def draw_color_debug(img: np.ndarray, white_mask: np.ndarray,
                     yellow_mask: np.ndarray) -> np.ndarray:
    """흰색/노란색 마스크를 원본 이미지에 시각화."""
    vis = img.copy()
    vis[white_mask  > 0] = (220, 220, 255)   # 흰색 → 연파랑
    vis[yellow_mask > 0] = (0,   200, 255)   # 노란색 → 노랑
    return vis


def draw_hud(img: np.ndarray, method: str,
             left_ok: bool, right_ok: bool,
             left_color: str = "?", right_color: str = "?") -> np.ndarray:
    """상태 정보 HUD (반투명 박스 + 텍스트)."""
    out = img.copy()
    h, w = out.shape[:2]

    # 반투명 패널
    panel_h, panel_w = 105, 340
    panel = out[10:10 + panel_h, 10:10 + panel_w].copy()
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.65, out[10:10 + panel_h, 10:10 + panel_w], 0.35, 0,
                    out[10:10 + panel_h, 10:10 + panel_w])

    # 텍스트
    font  = cv2.FONT_HERSHEY_SIMPLEX
    green = (80, 255, 120)
    red   = (80, 100, 255)
    white = (230, 230, 230)
    cyan  = (255, 220, 80)

    cv2.putText(out, f"Method : {method}",            (20, 35),  font, 0.6, cyan,  2, cv2.LINE_AA)
    cv2.putText(out, f"Left   : {'OK' if left_ok  else 'MISS'}",
                (20, 65), font, 0.6, green if left_ok  else red, 2, cv2.LINE_AA)
    cv2.putText(out, f"Right  : {'OK' if right_ok else 'MISS'}",
                (20, 95), font, 0.6, green if right_ok else red, 2, cv2.LINE_AA)

    return out


# ═══════════════════════════════════════════════════════════════════
# 8. 메인 파이프라인
# ═══════════════════════════════════════════════════════════════════

def detect_lanes(
    image_path: str,
    output_path: str = "lane_result.jpg",
    debug: bool = False,
    use_temporal: bool = True,
) -> np.ndarray:
    """
    단일 이미지에 차선 인식을 수행하고 결과를 저장.

    Returns:
        result_img: 차선이 시각화된 이미지 (numpy array)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    h, w = img.shape[:2]

    # ── ① ROI 계산 ───────────────────────────────────────────────
    vertices = get_roi_vertices(h, w)

    # ── ② 색상 마스크 (색상 필터도 ROI 적용됨) ───────────────────
    white_mask, yellow_mask, full_color_mask = extract_color_mask(img, vertices)

    # ── ③ Canny 엣지 (Hough fallback용) ──────────────────────────
    edges      = preprocess(img, full_color_mask)
    masked_edges = apply_roi_mask(edges, vertices)

    # ── ④ Bird's-eye view: 색상 마스크만 사용 (엣지 제외) ──────────
    # 노이즈가 많은 Canny 엣지를 BEV에 넘기면 배경절/건물 엣지가 차선으로 오인됨
    M, Minv  = get_bev_transform(h, w)
    bev_bin  = cv2.warpPerspective(full_color_mask, M, (BEV_W, BEV_H))

    left_fit,  right_fit,  ys = sliding_window_fit(bev_bin)

    # ── ⑤ Temporal smoothing ────────────────────────────────────
    if use_temporal:
        _tracker.update(left_fit, right_fit)
        if left_fit  is None: left_fit  = _tracker.smooth_left()
        if right_fit is None: right_fit = _tracker.smooth_right()

    bev_ok = (left_fit is not None) or (right_fit is not None)

    # ── ⑥ Hough (BEV 실패 시 fallback) ─────────────────────────
    left_line, right_line = hough_lane_detect(masked_edges, h, w)
    hough_ok = (left_line is not None) or (right_line is not None)

    # ── ⑦ 결과 합성 ─────────────────────────────────────────────
    result = img.copy()

    if bev_ok:
        lane_overlay = draw_bev_lane_overlay(left_fit, right_fit, ys, Minv, img.shape)
        result = cv2.addWeighted(result, 1.0, lane_overlay, 0.5, 0)
        method = "BEV Sliding Window"
    elif hough_ok:
        result = overlay_hough_lines(result, left_line, right_line)
        method = "Hough Transform"
    else:
        method = "Not Detected"

    # Hough 라인도 얇게 보조 표시
    if hough_ok and bev_ok:
        for line, color in [(left_line, (255, 220, 0)), (right_line, (0, 200, 255))]:
            if line:
                x1, y1, x2, y2 = line
                cv2.line(result, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    # ROI 경계 (디버그)
    if debug:
        cv2.polylines(result, [vertices], True, (255, 0, 200), 2)

    # HUD
    result = draw_hud(
        result, method,
        left_ok  = (left_fit  is not None) or (left_line  is not None),
        right_ok = (right_fit is not None) or (right_line is not None),
    )

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"[lane_detector_v0] {method} | "
          f"좌={'OK' if left_fit is not None else 'X'} "
          f"우={'OK' if right_fit is not None else 'X'} "
          f"→ {output_path}")

    # ── ⑧ 디버그 이미지 저장 ────────────────────────────────────
    if debug:
        dbg_dir = Path(output_path).parent / "lane_debug"
        dbg_dir.mkdir(exist_ok=True)
        stem = Path(output_path).stem

        cv2.imwrite(str(dbg_dir / f"{stem}_01_edges.jpg"),      edges)
        cv2.imwrite(str(dbg_dir / f"{stem}_02_masked.jpg"),     masked_edges)
        cv2.imwrite(str(dbg_dir / f"{stem}_03_color_mask.jpg"), full_color_mask)
        cv2.imwrite(str(dbg_dir / f"{stem}_04_white_mask.jpg"), white_mask)
        cv2.imwrite(str(dbg_dir / f"{stem}_05_yellow_mask.jpg"),yellow_mask)

        # BEV 시각화
        bev_vis = cv2.cvtColor(bev_bin, cv2.COLOR_GRAY2BGR)
        if left_fit  is not None:
            lx = np.polyval(left_fit,  ys).clip(0, BEV_W - 1).astype(np.int32)
            for i in range(len(ys) - 1):
                cv2.line(bev_vis, (lx[i], int(ys[i])), (lx[i+1], int(ys[i+1])),
                         (255, 220, 0), 3)
        if right_fit is not None:
            rx = np.polyval(right_fit, ys).clip(0, BEV_W - 1).astype(np.int32)
            for i in range(len(ys) - 1):
                cv2.line(bev_vis, (rx[i], int(ys[i])), (rx[i+1], int(ys[i+1])),
                         (0, 200, 255), 3)
        cv2.imwrite(str(dbg_dir / f"{stem}_06_bev_fit.jpg"), bev_vis)

        print(f"  디버그 이미지: {dbg_dir}/")

    return result


# ═══════════════════════════════════════════════════════════════════
# 9. 폴더 일괄 처리
# ═══════════════════════════════════════════════════════════════════

def process_folder(
    folder_path: str,
    output_dir: str = None,
    debug: bool = False,
    ext: list = None,
):
    """폴더 내 모든 이미지에 차선 검출 수행."""
    if ext is None:
        ext = [".jpg", ".jpeg", ".png", ".bmp"]

    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"폴더가 없습니다: {folder_path}")

    out_dir = Path(output_dir) if output_dir else folder.parent / (folder.name + "_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in ext]
    print(f"\n📁 대상 폴더: {folder}  ({len(images)}장)")
    print(f"📂 출력 폴더: {out_dir}\n")

    ok_count = 0
    for i, img_path in enumerate(images, 1):
        out_path = out_dir / img_path.name
        try:
            detect_lanes(str(img_path), str(out_path), debug=debug)
            ok_count += 1
        except Exception as e:
            print(f"  [오류] {img_path.name}: {e}")

        if i % 20 == 0:
            print(f"  진행: {i}/{len(images)}")

    print(f"\n✅ 완료: {ok_count}/{len(images)}장 처리")
    print(f"결과 위치: {out_dir}")


# ═══════════════════════════════════════════════════════════════════
# 10. CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="lane_detector_v0 – 교차로 특화 강건 차선 인식",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예:
  # 단일 이미지
  python lane_detector_v0.py --image v0/image/08_103700_220615_01.jpg

  # 단일 이미지 + 출력 경로 지정
  python lane_detector_v0.py --image v0/image/08_103700_220615_01.jpg --output result.jpg

  # 폴더 일괄 처리
  python lane_detector_v0.py --folder v0/image --output_dir v0/results

  # 디버그 모드 (중간 단계 이미지 저장)
  python lane_detector_v0.py --image v0/image/08_103700_220615_01.jpg --debug
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="단일 입력 이미지 경로")
    group.add_argument("--folder", type=str, help="입력 이미지 폴더 경로")

    parser.add_argument("--output",     type=str, default=None,
                        help="출력 파일 경로 (--image 사용 시)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 폴더 경로 (--folder 사용 시)")
    parser.add_argument("--debug",      action="store_true",
                        help="디버그 중간 이미지 저장")
    parser.add_argument("--no_temporal", action="store_true",
                        help="Temporal smoothing 비활성화")

    args = parser.parse_args()

    if args.image:
        out = args.output or "lane_result_v0.jpg"
        detect_lanes(args.image, out, debug=args.debug,
                     use_temporal=not args.no_temporal)
    else:
        process_folder(args.folder, args.output_dir, debug=args.debug)


if __name__ == "__main__":
    main()
