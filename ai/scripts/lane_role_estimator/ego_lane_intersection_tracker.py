import cv2
import numpy as np
import math
from enum import Enum
from collections import deque
import argparse

class State(Enum):
    LANE_TRACKING = 1
    CROSSWALK_APPROACH = 2
    ENTERING_INTERSECTION = 3
    INTERSECTION_TRACKING = 4
    RELOCK_LANE = 5

class EgoLaneIntersectionTracker:
    def __init__(self):
        self.state = State.LANE_TRACKING
        self.frame_width = 640
        self.frame_height = 480
        
        # Lane State
        self.left_boundary = None
        self.right_boundary = None
        self.lane_centerline = None
        self.lane_width_estimate = 300  # Initial guess in pixels near bottom
        self.lane_confidence = 0.0
        
        # Stability/Memory
        self.left_boundary_ema = None
        self.right_boundary_ema = None
        self.lane_confidence_history = deque(maxlen=5)
        
        # Crosswalk State
        self.crosswalk_detected = False
        self.crosswalk_bbox = None
        self.crosswalk_center = None
        self.crosswalk_area_ratio = 0.0
        self.crosswalk_near_ego_lane = False
        self.crosswalk_history = deque(maxlen=10) # Track bools
        self.crosswalk_ratio_history = deque(maxlen=10)

        # Saved Reliable Lane Anchor (for intersection tracking)
        self.last_left_boundary = None
        self.last_right_boundary = None
        self.last_lane_centerline = None
        self.last_lane_width = None
        self.last_heading_direction = None
        
        # State Machine memory
        self.relock_frame_count = 0
        self.mode_text = "LANE_BASED"
        
        # Debug toggles
        self.show_edges = False
        self.show_mask = False
        self.show_roi = False
        self.show_boundaries = True
        self.show_crosswalk_box = True
        self.show_path_corridor = True
        self.show_text = True
        
    def preprocess_frame(self, frame):
        """
        [2] 전처리: 해상도 조정, 색상/에지 마스크 추출 및 ROI 적용.
        """
        resized = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # HSV Color mask for white/yellow lanes
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Yellow lane mask
        lower_yellow = np.array([10, 50, 80])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # White lane mask (Sensitive to brightness)
        # 백색은 흐린 날씨에도 잡힐 수 있도록 V값을 100 정도로 낮추고 S의 허용범위도 약간 늘림
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        color_mask = cv2.bitwise_or(yellow_mask, white_mask)
        
        # Grayscale & Edges extraction
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # ROI generation
        mask = np.zeros_like(edges)
        rows, cols = self.frame_height, self.frame_width
        
        # 좌우의 인도/수풀(Curb)을 처음부터 어느 정도 배제하기 위해 하단 너비를 조금 줄임
        bottom_left  = [int(cols * 0.15), rows]
        top_left     = [int(cols * 0.4), int(rows * 0.5)]
        top_right    = [int(cols * 0.6), int(rows * 0.5)]
        bottom_right = [int(cols * 0.85), rows]
        pts = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Color Mask는 디버그/시각화 용도로만 남겨두고, 실제 엣지는 순수 Canny 엣지를 ROI로만 잘라 사용함.
        # "색상 필터가 진짜 차선을 날려버리는 현상"을 원천 차단하기 위함.
        final_edges = cv2.bitwise_and(edges, mask)
        
        # Save for debugging
        self._debug_edges = final_edges
        self._debug_mask = cv2.bitwise_and(color_mask, mask)
        self._debug_roi = cv2.bitwise_and(resized, resized, mask=mask)
        
        return final_edges
        
    def detect_lane_boundaries(self, edges):
        """
        [3] 차선/도로 경계 검출: Hough 변환 및 좌우 구분, Confidence 계산.
        """
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=30, maxLineGap=20)
        
        left_lines = []
        right_lines = []
        center_x = self.frame_width / 2

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    slope = 999.0 if x1 > center_x else -999.0
                else:
                    slope = (y2 - y1) / (x2 - x1)
                
                # Filter horizontal lines out
                if abs(slope) < 0.3:
                    continue
                
                # 선분의 아랫부분 x좌표 도출
                y_bottom = max(y1, y2)
                x_bottom = x1 if y_bottom == y1 else x2
                
                # 왼쪽 차선: 기울기가 음수이고 아랫부분이 좌측에 존재할 것 (+ 너무 먼 연석 배제)
                if slope < -0.3 and (50 < x_bottom < center_x + 80):
                    left_lines.append((x1, y1, x2, y2, slope))
                # 오른쪽 차선: 기울기가 양수이고 아랫부분이 우측에 존재할 것 (+ 너무 먼 연석 배제)
                elif slope > 0.3 and (center_x - 80 < x_bottom < self.frame_width - 50):
                    right_lines.append((x1, y1, x2, y2, slope))
                    
        # Select dominant lines based on length
        left_boundary = self._select_dominant_line(left_lines)
        right_boundary = self._select_dominant_line(right_lines)
        
        # Calculate single frame confidence
        conf = 0.0
        if left_boundary is not None: conf += 0.5
        if right_boundary is not None: conf += 0.5
        
        # Time-based smoothing with EMA (Exponential Moving Average)
        alpha = 0.2
        if left_boundary:
            self.left_boundary_ema = self._blend_lines(self.left_boundary_ema, left_boundary, alpha)
        if right_boundary:
            self.right_boundary_ema = self._blend_lines(self.right_boundary_ema, right_boundary, alpha)
            
        self.lane_confidence_history.append(conf)
        self.lane_confidence = sum(self.lane_confidence_history) / len(self.lane_confidence_history)
        
        return self.left_boundary_ema, self.right_boundary_ema
        
    def build_ego_lane_polygon(self, left, right):
        """
        [4] ego lane polygon 생성: 좌우 경계 및 이전 폭을 이용해 polygon 형성.
        """
        y_bottom = self.frame_height
        y_top = int(self.frame_height * 0.6)
        
        left_b = left
        right_b = right
        
        # Missing side fallback mechanism (Lane width memory)
        if left_b and not right_b:
            right_b = (left_b[0] + int(self.lane_width_estimate), y_bottom, 
                       left_b[2] + int(self.lane_width_estimate*0.6), y_top)
        elif right_b and not left_b:
            left_b = (right_b[0] - int(self.lane_width_estimate), y_bottom, 
                      right_b[2] - int(self.lane_width_estimate*0.6), y_top)
                      
        self.left_boundary = left_b
        self.right_boundary = right_b
        
        if left_b and right_b:
            current_width = right_b[0] - left_b[0]
            # 안정적인 폭일 때만 EMA 업데이트
            if 100 < current_width < 600:
                self.lane_width_estimate = self.lane_width_estimate * 0.9 + current_width * 0.1
                
            self.lane_centerline = (
                int((left_b[0] + right_b[0]) / 2), y_bottom,
                int((left_b[2] + right_b[2]) / 2), y_top
            )
        else:
            self.lane_centerline = None
            
        pts = None
        if left_b and right_b:
            pts = np.array([
                [left_b[0], left_b[1]],
                [left_b[2], left_b[3]],
                [right_b[2], right_b[3]],
                [right_b[0], right_b[1]]
            ], np.int32)
            
        return pts
        
    def update_crosswalk_state(self, mock_bbox):
        """
        [5] 횡단보도 bbox 계산: 접근 확인 및 면적 비율 갱신.
        """
        if mock_bbox and len(mock_bbox) == 4:
            x, y, w, h = mock_bbox
            self.crosswalk_bbox = (x, y, w, h)
            self.crosswalk_center = (x + w//2, y + h//2)
            
            frame_area = self.frame_width * self.frame_height
            area = w * h
            self.crosswalk_area_ratio = area / frame_area
            
            # Check if bbox is centered near the ego lane
            if self.lane_centerline:
                x1, y1, x2, y2 = self.lane_centerline
                if y1 != y2:
                    slope = (y2 - y1) / (x2 - x1) if x2!=x1 else 9999
                    if slope != 0:
                        center_x_at_cw = x1 + (self.crosswalk_center[1] - y1) / slope
                        dist = abs(center_x_at_cw - self.crosswalk_center[0])
                        self.crosswalk_near_ego_lane = dist < self.lane_width_estimate
                    else:
                        self.crosswalk_near_ego_lane = True
            else:
                self.crosswalk_near_ego_lane = abs(self.crosswalk_center[0] - self.frame_width//2) < 200
                
            self.crosswalk_detected = True
            self.crosswalk_history.append(1)
        else:
            self.crosswalk_bbox = None
            self.crosswalk_center = None
            self.crosswalk_area_ratio = 0.0
            self.crosswalk_detected = False
            self.crosswalk_history.append(0)
            
        self.crosswalk_ratio_history.append(self.crosswalk_area_ratio)

    def save_last_reliable_lane(self):
        """
        [6] 마지막 reliable lane 저장: 교차로 내부 추적의 Anchor 저장
        """
        if self.left_boundary and self.right_boundary and self.lane_centerline:
            self.last_left_boundary = self.left_boundary
            self.last_right_boundary = self.right_boundary
            self.last_lane_centerline = self.lane_centerline
            self.last_lane_width = self.lane_width_estimate
            
            x1, y1, x2, y2 = self.lane_centerline
            self.last_heading_direction = x2 - x1
            
    def estimate_path_corridor(self):
        """
        [7] path corridor 추정: 마지막 차선 정보를 기반으로 가상의 교차로 주행 경로 설정
        """
        if self.last_lane_centerline:
            x1, y1, x2, y2 = self.last_lane_centerline
            half_w_bottom = int(self.last_lane_width / 2)
            half_w_top = int(self.last_lane_width * 0.3)
            
            pts = np.array([
                [x1 - half_w_bottom, y1],
                [x2 - half_w_top, y2],
                [x2 + half_w_top, y2],
                [x1 + half_w_bottom, y1]
            ], np.int32)
            return pts
        return None

    def update_state_machine(self):
        """
        [8] 상태 전이 규칙 업데이트
        """
        cw_count = sum(self.crosswalk_history)
        recent_cw_detected = cw_count >= 3  # N=3 frames threshold
        
        cw_ratio_increasing = False
        if len(self.crosswalk_ratio_history) > 3:
            cw_ratio_increasing = self.crosswalk_ratio_history[-1] > self.crosswalk_ratio_history[0]
            
        if self.state == State.LANE_TRACKING:
            self.mode_text = "LANE_BASED"
            if recent_cw_detected and self.crosswalk_near_ego_lane and self.lane_confidence >= 0.5:
                self.state = State.CROSSWALK_APPROACH
                self.save_last_reliable_lane()
                
        elif self.state == State.CROSSWALK_APPROACH:
            self.mode_text = "LANE_BASED"
            self.save_last_reliable_lane() # Update anchor right up to entry
            
            if cw_ratio_increasing and self.lane_confidence < 0.7:
                self.state = State.ENTERING_INTERSECTION
            elif not recent_cw_detected:
                # false positive recovery
                self.state = State.LANE_TRACKING
                
        elif self.state == State.ENTERING_INTERSECTION:
            self.mode_text = "CORRIDOR_TRACKING"
            if self.crosswalk_area_ratio > 0.15 or self.lane_confidence < 0.3:
                self.state = State.INTERSECTION_TRACKING
                
        elif self.state == State.INTERSECTION_TRACKING:
            self.mode_text = "CORRIDOR_TRACKING"
            # Passing intersection implies crosswalk is passed and lanes are coming back
            if not recent_cw_detected and self.lane_confidence > 0.6:
                self.relock_frame_count += 1
                if self.relock_frame_count >= 5:
                    self.state = State.RELOCK_LANE
                    self.relock_frame_count = 0
            else:
                self.relock_frame_count = max(0, self.relock_frame_count - 1)
                
        elif self.state == State.RELOCK_LANE:
            self.mode_text = "LANE_BASED"
            if self.lane_confidence > 0.8:
                self.relock_frame_count += 1
                if self.relock_frame_count >= 10:
                    self.state = State.LANE_TRACKING
                    self.relock_frame_count = 0
            elif self.lane_confidence < 0.4:
                # Failed relock, still in intersection zone logic
                self.state = State.INTERSECTION_TRACKING
                self.relock_frame_count = 0

    def draw_overlay(self, frame, lane_pts, corridor_pts):
        """
        [10] 최종 화면 오버레이: 정보 시각화
        """
        overlay = frame.copy()
        
        if self.mode_text == "LANE_BASED" and lane_pts is not None:
            cv2.fillPoly(overlay, [lane_pts], (0, 255, 0)) # Green: Lane Based
            if self.lane_centerline:
                x1, y1, x2, y2 = self.lane_centerline
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        elif self.mode_text == "CORRIDOR_TRACKING" and self.show_path_corridor and corridor_pts is not None:
            cv2.fillPoly(overlay, [corridor_pts], (0, 255, 255)) # Yellow: Corridor Based
            if self.last_lane_centerline:
                x1, y1, x2, y2 = self.last_lane_centerline
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        if self.show_boundaries:
            if self.left_boundary:
                x1, y1, x2, y2 = self.left_boundary
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            if self.right_boundary:
                x1, y1, x2, y2 = self.right_boundary
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
        if self.show_crosswalk_box and self.crosswalk_bbox:
            x, y, w, h = self.crosswalk_bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 128, 0), 2)
            cv2.putText(frame, "CROSSWALK", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
            
        if self.show_text:
            hud_texts = [
                f"State: {self.state.name}",
                f"Lane Conf: {self.lane_confidence:.2f}",
                f"Crosswalk Ratio: {self.crosswalk_area_ratio:.3f}",
                f"Mode: {self.mode_text}"
            ]
            
            for i, text in enumerate(hud_texts):
                y_pos = 30 + i * 30
                cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        return frame

    def process_frame(self, frame, mock_crosswalk_bbox=None):
        """
        [1] 파이프라인 흐름 & [12] 함수 분리
        """
        # 해상도를 미리 줄여놓아야 생성된 차선 폴리곤 좌표와 출력 영상 사이즈가 일치함
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        edges = self.preprocess_frame(frame)
        left_b, right_b = self.detect_lane_boundaries(edges)
        lane_pts = self.build_ego_lane_polygon(left_b, right_b)
        
        self.update_crosswalk_state(mock_crosswalk_bbox)
        corridor_pts = self.estimate_path_corridor()
        
        self.update_state_machine()
        
        out_frame = self.draw_overlay(frame, lane_pts, corridor_pts)
        return out_frame

    # Internal Utilities
    def _select_dominant_line(self, lines):
        if not lines: return None
        lines.sort(key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2), reverse=True)
        x1, y1, x2, y2, slope = lines[0]
        
        y_bottom = self.frame_height
        y_top = int(self.frame_height * 0.6)
        x_bottom = int(x1 + (y_bottom - y1) / slope)
        x_top = int(x1 + (y_top - y1) / slope)
        
        return (x_bottom, y_bottom, x_top, y_top)
        
    def _blend_lines(self, old_line, new_line, alpha):
        if old_line is None: return new_line
        if new_line is None: return old_line
        return tuple(int(old * (1 - alpha) + new * alpha) for old, new in zip(old_line, new_line))

# ====================================================================
# [5] Mock Detector Function
# ====================================================================
def build_mock_crosswalk_detector(total_frames):
    def mock_detector(frame_idx):
        # 교차로 횡단보도가 화면에 접근하고 커지는 과정을 시뮬레이션
        mid = total_frames // 2
        if mid - 40 < frame_idx < mid + 30:
            scale = (frame_idx - (mid - 40)) / 70.0 
            w = int(200 + 400 * scale)
            h = int(50 + 150 * scale)
            x = 320 - w//2
            y = 240 + int(100 * scale)
            return (x, y, w, h)
        return None
    return mock_detector

# ====================================================================
# [1] Main Execution
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="Real-time Ego Lane & Intersection Tracker")
    parser.add_argument("--source", type=str, default="", help="Path to video or image file")
    parser.add_argument("--video", type=str, default="", help="Legacy option, use --source instead")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    args = parser.parse_args()

    tracker = EgoLaneIntersectionTracker()

    input_source = args.source if args.source else args.video
    is_image = input_source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')) if input_source else False

    static_frame = None
    if is_image:
        static_frame = cv2.imread(input_source)
        if static_frame is None:
            print("Error: Could not open static image file.")
            return
        cap = None
    else:
        if input_source:
            cap = cv2.VideoCapture(input_source)
        else:
            cap = cv2.VideoCapture(args.cam)
            
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

    # Total frames to configure mock tracking
    if static_frame is not None:
        total_frames = 1
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 500
    
    detector = build_mock_crosswalk_detector(total_frames)
    frame_idx = 0

    print("====================================")
    print("Starting EgoLaneIntersectionTracker...")
    print("Toggle Debugs:")
    print(" 'e': Toggle Edge Detection View")
    print(" 'm': Toggle Color Mask View")
    print(" 'q': Quit")
    print("====================================")

    while True:
        if static_frame is not None:
            frame = static_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break
            
        # [5] Inject Mock Bbox. In actual usage, pass the inference result here.
        cw_bbox = detector(frame_idx)
        
        out_frame = tracker.process_frame(frame, mock_crosswalk_bbox=cw_bbox)
        
        cv2.imshow("Intersection Tracker", out_frame)
        
        # [11] 디버깅용 시각화 옵션
        if tracker.show_edges and hasattr(tracker, '_debug_edges'):
            cv2.imshow("Debug Edges", tracker._debug_edges)
        if tracker.show_mask and hasattr(tracker, '_debug_mask'):
            cv2.imshow("Debug Color Mask", tracker._debug_mask)

        # 이미지일 경우 0 (무한대기), 비디오일 경우 30ms 대기
        wait_time = 0 if static_frame is not None else 30
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            tracker.show_edges = not tracker.show_edges
            if not tracker.show_edges: cv2.destroyWindow("Debug Edges")
        elif key == ord('m'):
            tracker.show_mask = not tracker.show_mask
            if not tracker.show_mask: cv2.destroyWindow("Debug Color Mask")
            
        frame_idx += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
