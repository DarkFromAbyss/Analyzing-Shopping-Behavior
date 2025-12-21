import numpy as np
from filterpy.kalman import KalmanFilter

# Định nghĩa các mối nối xương (Skeleton connections)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), 
    (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

class KalmanFilterBox:
    def __init__(self):
        # 8 state variables (cx, cy, aspect_ratio, height, v_cx, v_cy, v_a, v_h)
        # 4 measurement variables (cx, cy, aspect_ratio, height)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i + 4] = 1.0
            
        self.kf.H = np.eye(4, 8)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        # Trạng thái theo dõi
        self.time_since_update = 0 
        self.history = [] 
        self.hits = 0 
        self.hit_streak = 0 
        self.age = 0

    def initiate(self, measurement_xyxy):
        cx, cy, a, h = self._xyxy_to_cxcyah(measurement_xyxy)
        self.kf.x = np.zeros(8)
        self.kf.x[:4] = [cx, cy, a, h]
        
        std = [
            2 * self._std_weight_position * h, 2 * self._std_weight_position * h, 1e-2, 2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h, 10 * self._std_weight_velocity * h, 1e-5, 10 * self._std_weight_velocity * h
        ]
        self.kf.P = np.diag(np.square(std))
        self.kf.Q = np.eye(8)
        self.kf.R = np.eye(4)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def predict(self):
        # Update process noise based on height
        h = self.kf.x[3]
        std_q = [
            self._std_weight_position * h, self._std_weight_position * h, 1e-2, self._std_weight_position * h,
            self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5, self._std_weight_velocity * h
        ]
        self.kf.Q = np.diag(np.square(std_q))
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        
        self.kf.predict()
        return self._cxcyah_to_xyxy(self.kf.x[:4])

    def update(self, measurement_xyxy):
        self.time_since_update = 0
        self.history.append(self.kf.x)
        self.hits += 1
        self.hit_streak += 1
        
        cx, cy, a, h = self._xyxy_to_cxcyah(measurement_xyxy)
        z = np.array([cx, cy, a, h])
        
        h_pred = self.kf.x[3]
        std_r = [
            self._std_weight_position * h_pred, 
            self._std_weight_position * h_pred, 
            1e-1, 
            self._std_weight_position * h_pred
        ]
        self.kf.R = np.diag(np.square(std_r))
        self.kf.update(z)

    def get_current_state(self):
        """
        Lấy trạng thái hiện tại (đã được hiệu chỉnh bởi update).
        Dùng cái này để vẽ sẽ chính xác hơn predict() khi đang track tốt.
        """
        return self._cxcyah_to_xyxy(self.kf.x[:4])

    def get_rect(self):
        return self._cxcyah_to_xyxy(self.kf.x[:4])

    def _xyxy_to_cxcyah(self, xyxy):
        w, h = max(0, xyxy[2]-xyxy[0]), max(0, xyxy[3]-xyxy[1])
        return xyxy[0]+w/2, xyxy[1]+h/2, w/(h+1e-6), h

    def _cxcyah_to_xyxy(self, cxcyah):
        cx, cy, a, h = cxcyah
        w = a * h
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]