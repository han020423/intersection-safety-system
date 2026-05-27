import cv2
import numpy as np
import itertools

def get_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-5:
        return None
        
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    return (int(x), int(y))

img = cv2.imread('t2.jpg')
img = cv2.resize(img, (640, 480))
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_white = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 50, 255]))
mask_yellow = cv2.inRange(hsv, np.array([15, 40, 50]), np.array([35, 255, 255]))
color_mask = cv2.bitwise_or(mask_white, mask_yellow)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 30, 100)
color_edges = cv2.Canny(color_mask, 50, 150)
combined_edges = cv2.bitwise_or(edges, color_edges)

pts = np.array([[(0, h), (0, int(h*0.3)), (w, int(h*0.3)), (w, h)]])
mask = np.zeros_like(combined_edges)
cv2.fillPoly(mask, pts, 255)
roi_edges = cv2.bitwise_and(combined_edges, mask)

lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=40)

valid_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        if x1 == x2: continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.25 or abs(slope) > 10.0:
            continue
        valid_lines.append((x1, y1, x2, y2, slope))

pts_x = []
pts_y = []
for l1, l2 in itertools.combinations(valid_lines, 2):
    if l1[4] * l2[4] >= 0: continue
    pt = get_intersection(l1[:4], l2[:4])
    if pt is not None:
        px, py = pt
        if 0 < px < w and 0 < py < h:
            pts_x.append(px)
            pts_y.append(py)

if pts_x:
    hist, xedges, yedges = np.histogram2d(pts_x, pts_y, bins=[w//20, h//20], range=[[0, w], [0, h]])
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)
    vp_x = int((xedges[max_idx[0]] + xedges[max_idx[0]+1]) / 2)
    vp_y = int((yedges[max_idx[1]] + yedges[max_idx[1]+1]) / 2)
    print(f"VP Found: {vp_x}, {vp_y}")
    
    out = img.copy()
    for l in valid_lines:
        cv2.line(out, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1)
    for px, py in zip(pts_x, pts_y):
        cv2.circle(out, (px, py), 2, (0, 0, 255), -1)
    
    cv2.circle(out, (vp_x, vp_y), 10, (255, 0, 255), -1)
    cv2.imwrite('vp_test.jpg', out)
