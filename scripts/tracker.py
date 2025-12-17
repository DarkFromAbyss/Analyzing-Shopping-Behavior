import numpy as np

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), 
    (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

class KalmanFilterBox:
    def __init__(self):
        # State: [cx, cy, a, h, vx, vy, va, vh]
        self._motion_mat = np.eye(8)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1
        
        self._update_mat = np.eye(4, 8)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
        self.mean = None
        self.covariance = None

    def initiate(self, measurement_xyxy):
        cx, cy, a, h = self._xyxy_to_cxcyah(measurement_xyxy)
        self.mean = np.zeros(8)
        self.mean[:4] = [cx, cy, a, h]
        
        std = [
            2 * self._std_weight_position * h, 2 * self._std_weight_position * h, 1e-2, 2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h, 10 * self._std_weight_velocity * h, 1e-5, 10 * self._std_weight_velocity * h
        ]
        self.covariance = np.diag(np.square(std))

    def predict(self):
        h = self.mean[3]
        std_q = [
            self._std_weight_position * h, self._std_weight_position * h, 1e-2, self._std_weight_position * h,
            self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5, self._std_weight_velocity * h
        ]
        Q = np.diag(np.square(std_q))
        self.mean = np.dot(self._motion_mat, self.mean)
        self.covariance = np.linalg.multi_dot([self._motion_mat, self.covariance, self._motion_mat.T]) + Q
        return self._cxcyah_to_xyxy(self.mean[:4])

    def project(self):
        h = self.mean[3]
        std_r = [self._std_weight_position * h, self._std_weight_position * h, 1e-1, self._std_weight_position * h]
        R = np.diag(np.square(std_r))
        projected_mean = np.dot(self._update_mat, self.mean)
        projected_cov = np.linalg.multi_dot([self._update_mat, self.covariance, self._update_mat.T]) + R
        return projected_mean, projected_cov

    def gating_distance(self, measurement_xyxy):
        cx, cy, a, h = self._xyxy_to_cxcyah(measurement_xyxy)
        z = np.array([cx, cy, a, h])
        projected_mean, projected_cov = self.project()
        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = z - projected_mean
        z_score = np.linalg.solve(cholesky_factor, d)
        return np.sum(z_score**2)

    def update(self, measurement_xyxy):
        cx, cy, a, h = self._xyxy_to_cxcyah(measurement_xyxy)
        z = np.array([cx, cy, a, h])
        projected_mean, projected_cov = self.project()
        L = np.linalg.cholesky(projected_cov)
        K = np.linalg.multi_dot([self.covariance, self._update_mat.T, np.linalg.inv(L).T, np.linalg.inv(L)])
        innovation = z - projected_mean
        self.mean = self.mean + np.dot(K, innovation)
        self.covariance = self.covariance - np.linalg.multi_dot([K, self._update_mat, self.covariance])

    def get_rect(self):
        if self.mean is None: return [0, 0, 0, 0]
        return self._cxcyah_to_xyxy(self.mean[:4])

    def _xyxy_to_cxcyah(self, xyxy):
        w, h = max(0, xyxy[2]-xyxy[0]), max(0, xyxy[3]-xyxy[1])
        return xyxy[0]+w/2, xyxy[1]+h/2, w/(h+1e-6), h

    def _cxcyah_to_xyxy(self, cxcyah):
        cx, cy, a, h = cxcyah
        w = a * h
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]