# scripts/tracker.py

import cv2
import numpy as np
from typing import List, Tuple, Any

# --- CÁC HẰNG SỐ CHỈ SỐ KEYPOINT CỦA POSE MODEL ---
# NOTE: Cần trùng khớp với chỉ số Keypoint của mô hình YOLOv8 Pose
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (3, 1), (4, 2)
]


class KalmanFilterBox:
    """Bộ lọc Kalman cho Bounding Box [x, y, w, h] - Dùng để làm mịn và dự đoán Bbox."""
    def __init__(self):
        # 8 trạng thái (x, y, w, h, vx, vy, vw, vh), 4 đo lường (x, y, w, h)
        self.kf = cv2.KalmanFilter(8, 4) 
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0], 
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0], 
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        # Thiết lập ma trận nhiễu quá trình và đo lường
        self.kf.processNoiseCov = np.diag([1e-2] * 8, k=0).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([1e-1] * 4, k=0).astype(np.float32)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1

    def initiate(self, bbox: List[int]):
        """Khởi tạo trạng thái ban đầu của Kalman Filter bằng Bbox."""
        x, y, w, h = self._xyxy_to_xywh(bbox)
        self.kf.statePost = np.array([x, y, w, h, 0., 0., 0., 0.], np.float32).reshape(-1, 1)

    def predict(self) -> List[int]:
        """Dự đoán trạng thái Bbox tiếp theo."""
        predicted = self.kf.predict()
        return self._xywh_to_xyxy(predicted[:4].flatten())

    def update(self, bbox: List[int]) -> List[int]:
        """Cập nhật trạng thái Bbox dựa trên đo lường mới."""
        measurement = np.array(self._xyxy_to_xywh(bbox), np.float32).reshape(-1, 1)
        corrected = self.kf.correct(measurement)
        return self._xywh_to_xyxy(corrected[:4].flatten())

    def _xyxy_to_xywh(self, bbox: List[int]) -> List[float]:
        """Chuyển đổi từ [x1, y1, x2, y2] sang [x_center, y_center, width, height]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2
        y_c = y1 + h / 2
        return [x_c, y_c, w, h]

    def _xywh_to_xyxy(self, xywh: np.ndarray) -> List[int]:
        """Chuyển đổi từ [x_center, y_center, width, height] sang [x1, y1, x2, y2]."""
        x_c, y_c, w, h = xywh
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return [int(round(val)) for val in [x1, y1, x2, y2]]