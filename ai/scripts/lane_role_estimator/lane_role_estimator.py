import cv2
import numpy as np
import argparse
import time
from collections import deque
import math

# ============================================================================
# Parameters & Thresholds Configuration
# ============================================================================
class Config:
    # Resolution
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480
    
    # ROI: Trapezoid for forward road view
    ROI_VERTICES_RATIO = [
        (0.0, 1.0),    # Bottom-left
        (0.2, 0.45),   # Top-left (wider to catch curves)
        (0.8, 0.45),   # Top-right
        (1.0, 1.0),    # Bottom-right
    ]
    
    # Canny Edge Parameters
    CANNY_LOW = 30
    CANNY_HIGH = 100
    
    # Lane Detection (Hough)
    HOUGH_RHO = 1
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESH = 30
    HOUGH_MIN_LEN = 15
    HOUGH_MAX_GAP = 50
    MIN_LANE_SLOPE = 0.25  # Allow slightly flatter lines
    MAX_LANE_SLOPE = 5.0   # Suppress vertical objects like poles
    
    # Stop Line
    STOP_LINE_ROI_MIN_Y = 0.6  # Scan bottom 40% of image
    MAX_STOP_LINE_SLOPE = 0.2  # Threshold for horizontal lines
    
    # Crosswalk
    CROSSWALK_HSV_MIN = np.array([0, 0, 180]) # White-ish
    CROSSWALK_HSV_MAX = np.array([179, 50, 255])
    CROSSWALK_MIN_ASPECT = 2.0  # Stripe width/height ratio
    MIN_STRIPES_FOR_CROSSWALK = 3
    
    # Temporal Smoothing
    HISTORY_FRAMES = 15
    LANE_WIDTH_EMA_ALPHA = 0.1
    DEFAULT_LANE_WIDTH = 350 # Fits the exact width of the single straight lane in this camera angle

# ============================================================================
# Enums
# ============================================================================
class LaneRole:
    STRAIGHT = "STRAIGHT_LANE"
    RIGHT = "RIGHT_TURN_LANE"
    LEFT = "LEFT_TURN_LANE"
    STRAIGHT_RIGHT = "STRAIGHT_OR_RIGHT"
    STRAIGHT_LEFT = "STRAIGHT_OR_LEFT"
    UNKNOWN = "UNKNOWN"

class IntersectionStage:
    APPROACH = "APPROACH"
    STOP_LINE_WAIT = "STOP_LINE_WAIT"
    ENTERING = "ENTERING_INTERSECTION"
    EXITING = "EXITING"
    UNKNOWN = "UNKNOWN"

# ============================================================================
# Main Estimator Class
# ============================================================================
class LaneRoleEstimator:
    def __init__(self, debug=False):
        self.debug = debug
        self.w = Config.TARGET_WIDTH
        self.h = Config.TARGET_HEIGHT
        
        # Temporal State
        self.lane_width_mem = Config.DEFAULT_LANE_WIDTH
        self.role_history = deque(maxlen=Config.HISTORY_FRAMES)
        self.stage_history = deque(maxlen=Config.HISTORY_FRAMES)
        self.score_history_s = deque(maxlen=Config.HISTORY_FRAMES)
        self.score_history_r = deque(maxlen=Config.HISTORY_FRAMES)
        self.score_history_l = deque(maxlen=Config.HISTORY_FRAMES)
        self.last_ego_polygon = None
        
        # Current Frame State
        self.left_boundary = None
        self.right_boundary = None
        self.lane_centerline = None
        self.lane_confidence = 0.0
        self.stop_line_y = None
        self.crosswalk_detected = False

    def _get_roi_vertices(self):
        return np.array([[(int(x * self.w), int(y * self.h)) for x, y in Config.ROI_VERTICES_RATIO]], dtype=np.int32)

    def preprocess_frame(self, frame):
        """Image preprocessing: resize, color mask (white/yellow), Canny, ROI."""
        resized = cv2.resize(frame, (self.w, self.h))
        
        # Color masking for Yellow + White lines
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        # White (lower V for dark scenes)
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        # Yellow (lower S and V for faint shadowed yellow lines)
        lower_yellow = np.array([15, 40, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        color_mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Apply Canny on grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, Config.CANNY_LOW, Config.CANNY_HIGH)
        
        # Apply Canny directly on color mask to force faint colored lines into edges
        color_edges = cv2.Canny(color_mask, 50, 150)
        
        # Combine Canny and Color Edges
        combined_edges = cv2.bitwise_or(edges, color_edges)
        
        # Apply ROI first so we only count pixels inside the road area
        mask = np.zeros_like(combined_edges)
        cv2.fillPoly(mask, self._get_roi_vertices(), 255)
        combined_roi = cv2.bitwise_and(combined_edges, mask)
        
        return resized, combined_roi, gray

    def detect_lane_boundaries(self, edges):
        """Extract Left and Right lane boundaries using Hough + line fitting."""
        lines = cv2.HoughLinesP(edges, Config.HOUGH_RHO, Config.HOUGH_THETA, Config.HOUGH_THRESH,
                                minLineLength=Config.HOUGH_MIN_LEN, maxLineGap=Config.HOUGH_MAX_GAP)
        
        left_segments = []
        right_segments = []
        
        cx = self.w // 2
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                
                # Filter by line length (Optional, Hough already filters but just in case)
                if np.hypot(x2 - x1, y2 - y1) < Config.HOUGH_MIN_LEN: continue
                
                if x1 == x2: continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter by slope (stricter to avoid arrows and horizontal markings)
                if abs(slope) < 0.4 or abs(slope) > Config.MAX_LANE_SLOPE:
                    continue
                    
                # Separate left and right (screen coords: y goes down)
                mid_x = (x1 + x2) / 2
                
                # Project segment to the top and bottom of ROI
                y_top_proj = self.h * 0.45
                x_top_proj = x1 + (y_top_proj - y1) / slope
                x_bottom_proj = x1 + (self.h - y1) / slope
                
                # Exclusion Zone: Reject segments that start right in the middle of the ego lane (painted ground arrows)
                if 150 < x_bottom_proj < 350:
                    continue
                    
                # Vanishing Point Filter: A real lane line must point towards the horizon center.
                if x_top_proj < cx - 80 or x_top_proj > cx + 150:
                    continue
                    
                # Sort left/right purely by horizontal location, since the camera perspective might curve making the slope sign unreliable
                if mid_x < cx:
                    left_segments.append((x1, y1, x2, y2, slope))
                else:
                    right_segments.append((x1, y1, x2, y2, slope))

        def fit_line_points(segments, is_left):
            if not segments: return None
            # Extract points to fit a robust 1D line using weighted least squares
            x_pts = []
            y_pts = []
            weights = []
            for x1, y1, x2, y2, _ in segments:
                length = np.hypot(x2 - x1, y2 - y1)
                x_pts.extend([x1, x2])
                y_pts.extend([y1, y2])
                weights.extend([length, length]) # Weight by segment length
            
            try:
                poly = np.polyfit(y_pts, x_pts, 1, w=weights) # x = a*y + b
            except:
                return None
                
            # Line endpoints based on ROI top and bottom
            y_bottom = self.h
            y_top = int(self.h * Config.ROI_VERTICES_RATIO[1][1])
            
            x_bottom = int(np.polyval(poly, y_bottom))
            x_top = int(np.polyval(poly, y_top))
            
            # Sanity Checks to prevent "X" crossing and prevent snapping to distant curbs
            if is_left and x_bottom > cx + 40:
                return None
            if not is_left and x_bottom < cx - 40:
                return None
            if not is_left and x_bottom > cx + 160: # Prevent snapping to the far curb so the ego lane width is bounded
                return None
                
            # Prevent extreme crossing at the top
            if is_left and x_top > cx + 150:
                return None
            if not is_left and x_top < cx - 60:
                return None
                
            return ((x_bottom, y_bottom), (x_top, y_top))

        self.left_boundary = fit_line_points(left_segments, True)
        self.right_boundary = fit_line_points(right_segments, False)
        
        # DEBUG: store raw segments for visualization
        self.raw_left_segments = left_segments
        self.raw_right_segments = right_segments

    def estimate_lane_polygon(self):
        """Constructs the Ego Lane Polygon and handles missing boundaries via memory."""
        y_bot = self.h
        y_top = int(self.h * Config.ROI_VERTICES_RATIO[1][1])
        
        # 1. Both detected
        if self.left_boundary and self.right_boundary:
            self.lane_confidence = 1.0
            # Update width memory (distance between bottoms)
            w = self.right_boundary[0][0] - self.left_boundary[0][0]
            if w > 100: # sanity check
                self.lane_width_mem = (1.0 - Config.LANE_WIDTH_EMA_ALPHA) * self.lane_width_mem + Config.LANE_WIDTH_EMA_ALPHA * w
                
        # 2. Only Left detected
        elif self.left_boundary and not self.right_boundary:
            self.lane_confidence = 0.6
            xb, yb = self.left_boundary[0]
            xt, yt = self.left_boundary[1]
            self.right_boundary = ((int(xb + self.lane_width_mem), yb), (int(xt + self.lane_width_mem * 0.3), yt)) # perspective shrink
            
        # 3. Only Right detected
        elif self.right_boundary and not self.left_boundary:
            self.lane_confidence = 0.6
            xb, yb = self.right_boundary[0]
            xt, yt = self.right_boundary[1]
            self.left_boundary = ((int(xb - self.lane_width_mem), yb), (int(xt - self.lane_width_mem * 0.3), yt))
            
        # 4. Neither detected (Fallback)
        else:
            self.lane_confidence = 0.2
            # Use last polygon if exists, else default dummy
            if self.last_ego_polygon is not None:
                (self.left_boundary, self.right_boundary) = self.last_ego_polygon
            else:
                cx = self.w // 2
                self.left_boundary = ((int(cx - self.lane_width_mem/2), y_bot), (int(cx - 50), y_top))
                self.right_boundary = ((int(cx + self.lane_width_mem/2), y_bot), (int(cx + 50), y_top))
        
        self.last_ego_polygon = (self.left_boundary, self.right_boundary)
        
        # Calculate centerline
        self.lane_centerline = (
            (int((self.left_boundary[0][0] + self.right_boundary[0][0])/2), y_bot),
            (int((self.left_boundary[1][0] + self.right_boundary[1][0])/2), y_top)
        )

    def detect_stop_line(self, edges):
        """Looking for horizontal bright lines in the lower part of the ROI."""
        self.stop_line_y = None
        min_y = int(self.h * Config.STOP_LINE_ROI_MIN_Y)
        roi_bottom = edges[min_y:self.h, :]
        
        lines = cv2.HoughLinesP(roi_bottom, 1, np.pi/180, 40, minLineLength=50, maxLineGap=20)
        
        best_y = -1
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 1e-3: continue
                slope = abs(y2 - y1) / abs(x2 - x1)
                
                # Check if roughly horizontal
                if slope < Config.MAX_STOP_LINE_SLOPE:
                    global_y = min_y + (y1 + y2) // 2
                    # Must be somehow inside/crossing ego lane
                    cx = self.lane_centerline[0][0]
                    if min(x1, x2) < cx < max(x1, x2):
                        if global_y > best_y:
                            best_y = global_y
                            
        if best_y != -1:
            self.stop_line_y = best_y

    def detect_crosswalk(self, frame):
        """Detect crosswalk stripes via morphology and connected components."""
        self.crosswalk_detected = False
        
        # Consider lower part of the image
        min_y = int(self.h * 0.5)
        roi = frame[min_y:, :, :]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, Config.CROSSWALK_HSV_MIN, Config.CROSSWALK_HSV_MAX)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stripe_count = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > w else float(h) / w
            
            # Crosswalk stripes are elongated rectangles
            if aspect_ratio > Config.CROSSWALK_MIN_ASPECT:
                stripe_count += 1
                
        if stripe_count >= Config.MIN_STRIPES_FOR_CROSSWALK:
            self.crosswalk_detected = True

    def score_lane_role(self):
        """Heuristic scoring for Lane Role."""
        s_score, r_score, l_score = 0.5, 0.2, 0.2
        
        # 1. Curvature / Geometric Flow
        cx_bottom = self.lane_centerline[0][0]
        cx_top = self.lane_centerline[1][0]
        dx = cx_top - cx_bottom
        
        # If centerline severely drifts right or left
        flow_weight = 0.6
        # In 1D linear fit, dx is mostly just vanishing point offset from camera position.
        # A true turn lane would have extreme slant or we need 2D curve fitting.
        if dx > 300: # Extremely slanted, meaning road physically goes right
            r_score += flow_weight
            s_score -= flow_weight/2
        elif dx < -300: # Extremely slanted left
            l_score += flow_weight
            s_score -= flow_weight/2
        else: # Normal vanishing point behavior means straight lane
            s_score += flow_weight
            
        # 2. Stop Line & Crosswalk Influence
        # Approaching intersection clues
        if self.stop_line_y is not None or self.crosswalk_detected:
            # At intersection, if the lane is still straight, reinforce straight.
            if dx > 300:
                r_score += 0.3
            elif dx < -300:
                l_score += 0.3
            else:
                s_score += 0.3
                
        # Normalize
        total = s_score + r_score + l_score
        s_score = max(0, s_score / total)
        r_score = max(0, r_score / total)
        l_score = max(0, l_score / total)
        
        self.score_history_s.append(s_score)
        self.score_history_r.append(r_score)
        self.score_history_l.append(l_score)

        avg_s = sum(self.score_history_s) / len(self.score_history_s)
        avg_r = sum(self.score_history_r) / len(self.score_history_r)
        avg_l = sum(self.score_history_l) / len(self.score_history_l)
        
        # Classification
        margin = 0.15
        if avg_s > avg_r + margin and avg_s > avg_l + margin:
            role = LaneRole.STRAIGHT
        elif avg_r > avg_s + margin and avg_r > avg_l + margin:
            role = LaneRole.RIGHT
        elif avg_l > avg_s + margin and avg_l > avg_r + margin:
            role = LaneRole.LEFT
        elif abs(avg_s - avg_r) <= margin and avg_s > avg_l:
            role = LaneRole.STRAIGHT_RIGHT
        elif abs(avg_s - avg_l) <= margin and avg_s > avg_r:
            role = LaneRole.STRAIGHT_LEFT
        else:
            role = LaneRole.UNKNOWN
            
        if self.lane_confidence < 0.3:
            role = LaneRole.UNKNOWN
            
        self.role_history.append(role)
        return role, avg_s, avg_r, avg_l

    def estimate_intersection_stage(self):
        """Rule-based stage estimation."""
        stage = IntersectionStage.APPROACH
        
        if self.stop_line_y is not None:
            dist_to_bottom = self.h - self.stop_line_y
            if dist_to_bottom < 50:
                stage = IntersectionStage.ENTERING
            elif dist_to_bottom < 150:
                stage = IntersectionStage.STOP_LINE_WAIT
            else:
                stage = IntersectionStage.APPROACH
        elif self.crosswalk_detected:
             stage = IntersectionStage.STOP_LINE_WAIT
        else:
             # If no stop line but low confidence, might be exiting
             if self.lane_confidence < 0.4:
                 stage = IntersectionStage.EXITING
             else:
                 stage = IntersectionStage.APPROACH
                 
        self.stage_history.append(stage)
        # return mode of history
        return max(set(self.stage_history), key=self.stage_history.count)

    def draw_overlay(self, img, final_role, act_s, s_s, r_s, l_s):
        """Visualize all results."""
        out = img.copy()
        
        # 1. Draw Ego Polygon (green translucent)
        if self.left_boundary and self.right_boundary:
            pts = np.array([
                self.left_boundary[0],  # LB
                self.left_boundary[1],  # LT
                self.right_boundary[1], # RT
                self.right_boundary[0]  # RB
            ], np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
            
            # Centerline (dashed)
            c1, c2 = self.lane_centerline
            cv2.line(out, c1, c2, (0, 255, 255), 2)
            
            # Left/Right solid borders
            cv2.line(out, self.left_boundary[0], self.left_boundary[1], (255, 0, 0), 3) # Blue for Left
            cv2.line(out, self.right_boundary[0], self.right_boundary[1], (0, 0, 255), 3) # Red for Right

        # 2. Draw Stop Line
        if self.stop_line_y is not None:
            cv2.line(out, (0, self.stop_line_y), (self.w, self.stop_line_y), (255, 255, 0), 4) # Cyan
            
        # 3. Draw Crosswalk status
        if self.crosswalk_detected:
            cv2.putText(out, "[ CROSSWALK DETECTED ]", (self.w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
        # 4. Text HUD
        final_role_text = max(set(self.role_history), key=self.role_history.count) if self.role_history else final_role
        

        hud_lines = [
            f"Lane Role: {final_role_text}",
            f"Stage: {act_s}",
            f"Lane Conf: {self.lane_confidence:.2f}",
            f"Scores: S={s_s:.2f} R={r_s:.2f} L={l_s:.2f}"
        ]
        
        # Draw HUD Box
        cv2.rectangle(out, (10, 10), (320, 140), (0, 0, 0), -1)
        for i, text in enumerate(hud_lines):
            color = (0, 255, 0) if "STRAIGHT" in text else (0, 255, 255)
            cv2.putText(out, text, (20, 35 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return out

    def process_frame(self, frame):
        """Execute the entire pipeline for a single frame."""
        # Preprocess
        resized, edges, gray = self.preprocess_frame(frame)
        
        # Detect Features
        self.detect_lane_boundaries(edges)
        self.estimate_lane_polygon()
        self.detect_stop_line(edges)
        self.detect_crosswalk(resized)
        
        # High level logic
        role, s, r, l = self.score_lane_role()
        stage = self.estimate_intersection_stage()
        
        # Visualization
        result_img = self.draw_overlay(resized, role, stage, s, r, l)
        
        # Debug views
        # Always show candidates for debugging
        debug_img = np.zeros_like(resized)
        if hasattr(self, 'raw_left_segments'):
            for x1, y1, x2, y2, _ in self.raw_left_segments:
                cv2.line(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan for left candidates
        if hasattr(self, 'raw_right_segments'):
            for x1, y1, x2, y2, _ in self.raw_right_segments:
                cv2.line(debug_img, (x1, y1), (x2, y2), (255, 100, 255), 2) # Magenta for right candidates
                
        # Draw ROI
        cv2.polylines(debug_img, [self._get_roi_vertices()], True, (255, 255, 255), 1)
        
        cv2.imshow("Debug: Lane Candidates", debug_img)
        if self.debug:
            cv2.imshow("Debug: Combined Edges ROI", edges)
            
        return result_img


def main():
    parser = argparse.ArgumentParser(description="Lane Role Estimator (OpenCV Traditional)")
    parser.add_argument("--source", default="0", help="Webcam ID (e.g. 0) or Video File Path")
    parser.add_argument("--debug", action="store_true", help="Show debug windows")
    args = parser.parse_args()

    # Handle numeric vs string source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source {src}")
        return
        
    estimator = LaneRoleEstimator(debug=args.debug)
    print("Starting Lane Role Estimation. Press 'q' to quit.")
    
    result = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or error reading frame.")
            if result is None:
                print("Could not read any frames from the source.")
            else:
                print("Processing finished. Press any key in the OpenCV window to exit.")
                cv2.waitKey(0)
            break
            
        start_t = time.time()
        
        result = estimator.process_frame(frame)
        
        fps = 1.0 / (time.time() - start_t + 1e-6)
        cv2.putText(result, f"FPS: {fps:.1f}", (Config.TARGET_WIDTH - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Lane Role Estimation", result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
