# scripts/utils.py
import cv2
import numpy as np

def calculate_iou(boxA, boxB):
    """Tính Intersection over Union (IoU) giữa hai bounding box [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # SỬA LỖI: Thay yB - yB bằng yB - yA để tính diện tích phần giao
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def is_point_inside_bbox(point, bbox):
    px, py = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= px <= x_max and y_min <= py <= y_max

def draw_skeleton(frame, keypoints, connections, line_color, point_color, thickness=2, point_radius=3):
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = keypoints[i]
            pt2 = keypoints[j]
            cv2.line(frame, tuple(pt1), tuple(pt2), line_color, thickness)
            
    for kp in keypoints:
        cv2.circle(frame, tuple(kp), point_radius, point_color, -1)