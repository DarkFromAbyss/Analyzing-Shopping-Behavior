# pose_tracker.py (CẬP NHẬT: Đọc cấu hình từ Dictionary YAML)

import cv2
import numpy as np
from ultralytics import YOLO
import time 
import math 
# Không cần import config.py nữa

# ====================================================================
# --- CHỈ SỐ KEYPOINT VÀ THAO TÁC CƠ BẢN (KHÔNG PHỤ THUỘC CẤU HÌNH NGOÀI) ---
# ====================================================================

NOSE_INDEX = 0          
LEFT_EYE_INDEX = 1      
RIGHT_EYE_INDEX = 2     
LEFT_HIP_INDEX = 11     
RIGHT_HIP_INDEX = 12    
LEFT_WRIST_INDEX = 9   
RIGHT_WRIST_INDEX = 10  

RISK_LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}

SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (3, 1), (4, 2)
]

# ====================================================================
## Lớp Kalman Filter (Giữ nguyên)
# ====================================================================
class KalmanFilterBox:
    # ... (Giữ nguyên nội dung lớp KalmanFilterBox) ...
    """Bộ lọc Kalman cho Bounding Box [x, y, w, h]"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4) 
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0], [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0], [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.kf.processNoiseCov = np.diag([1e-2] * 8, k=0).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([1e-1] * 4, k=0).astype(np.float32)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1

    def initiate(self, bbox):
        x, y, w, h = self.xyxy_to_xywh(bbox)
        self.kf.statePost = np.array([x, y, w, h, 0., 0., 0., 0.], np.float32).reshape(-1, 1)

    def predict(self):
        predicted = self.kf.predict()
        return self.xywh_to_xyxy(predicted[:4].flatten())

    def update(self, bbox):
        measurement = np.array(self.xyxy_to_xywh(bbox), np.float32).reshape(-1, 1)
        corrected = self.kf.correct(measurement)
        return self.xywh_to_xyxy(corrected[:4].flatten())

    @staticmethod
    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2
        y_c = y1 + h / 2
        return [x_c, y_c, w, h]

    @staticmethod
    def xywh_to_xyxy(xywh):
        x_c, y_c, w, h = xywh
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return [int(round(val)) for val in [x1, y1, x2, y2]]

# ====================================================================
## HÀM TÍNH TOÁN VÀ PHÂN LOẠI (Nhận config_data)
# ====================================================================

def calculate_iou(boxA, boxB):
    # ... (Giữ nguyên nội dung hàm calculate_iou) ...
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0: return 0.0
    return interArea / unionArea

def is_point_inside_bbox(point, bbox):
    # ... (Giữ nguyên nội dung hàm is_point_inside_bbox) ...
    px, py = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= px <= x_max and y_min <= py <= y_max

def classify_interactions(tracked_keypoints, tracked_bboxes, object_results, config_data):
    """Gán nhãn TAKING, PUSHING (Keypoint) hoặc HOLDING (IoU)."""
    
    # Lấy các ID và ngưỡng từ config
    IOU_HOLDING_THRESHOLD = config_data['THRESHOLDS']['IOU_HOLDING_THRESHOLD']
    CABINET_SHELF_ID = config_data['BEHAVIOR']['CABINET_SHELF_ID']
    TROLLEY_ID = config_data['BEHAVIOR']['TROLLEY_ID']
    HANDBAG_SATCHEL_ID = config_data['BEHAVIOR']['HANDBAG_SATCHEL_ID']
    
    interaction_labels = {}
    target_bboxes = {"TAKING": [], "PUSHING": [], "HOLDING": []}
    
    for obj_box in object_results:
        if len(obj_box) < 5: continue
        class_id = obj_box[4]
        bbox = [int(x) for x in obj_box[:4]]
        
        if class_id == CABINET_SHELF_ID: target_bboxes["TAKING"].append(bbox)
        elif class_id == TROLLEY_ID: target_bboxes["PUSHING"].append(bbox)
        elif class_id == HANDBAG_SATCHEL_ID: target_bboxes["HOLDING"].append(bbox)

    for track_id, keypoints in tracked_keypoints.items():
        current_interaction = None
        person_bbox = tracked_bboxes.get(track_id)
        
        if person_bbox is None or len(keypoints) < max(LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX) + 1:
            interaction_labels[track_id] = None
            continue
            
        left_wrist = keypoints[LEFT_WRIST_INDEX]
        right_wrist = keypoints[RIGHT_WRIST_INDEX]
        
        # 3. Kiểm tra TAKING (Cổ tay)
        for cabinet_bbox in target_bboxes["TAKING"]:
            if is_point_inside_bbox(left_wrist, cabinet_bbox) or is_point_inside_bbox(right_wrist, cabinet_bbox):
                current_interaction = "TAKING"; break
        
        # 4. Kiểm tra PUSHING (Cổ tay)
        if current_interaction is None: 
            for trolley_bbox in target_bboxes["PUSHING"]:
                if is_point_inside_bbox(left_wrist, trolley_bbox) or is_point_inside_bbox(right_wrist, trolley_bbox):
                    current_interaction = "PUSHING"; break

        # 5. Kiểm tra HOLDING (IoU)
        if current_interaction is None:
            for handbag_bbox in target_bboxes["HOLDING"]:
                iou = calculate_iou(person_bbox, handbag_bbox)
                if iou >= IOU_HOLDING_THRESHOLD:
                    current_interaction = "HOLDING"; break

        interaction_labels[track_id] = current_interaction

    return interaction_labels

def get_hip_centroid(keypoints):
    # ... (Giữ nguyên nội dung hàm get_hip_centroid) ...
    if len(keypoints) < RIGHT_HIP_INDEX + 1: return None
    left_hip = keypoints[LEFT_HIP_INDEX]; right_hip = keypoints[RIGHT_HIP_INDEX]
    cx = (left_hip[0] + right_hip[0]) // 2; cy = (left_hip[1] + right_hip[1]) // 2
    return cx, cy

def get_face_centroid(keypoints):
    # ... (Giữ nguyên nội dung hàm get_face_centroid) ...
    if len(keypoints) < RIGHT_EYE_INDEX + 1: return None
    nose = keypoints[NOSE_INDEX]; left_eye = keypoints[LEFT_EYE_INDEX]; right_eye = keypoints[RIGHT_EYE_INDEX]
    avg_x = (nose[0] + left_eye[0] + right_eye[0]) // 3; avg_y = (nose[1] + left_eye[1] + right_eye[1]) // 3
    return avg_x, avg_y

def get_grid_position(cx, cy, frame_w, frame_h, cols):
    # ... (Giữ nguyên nội dung hàm get_grid_position) ...
    rows = cols; cell_w = frame_w / cols; cell_h = frame_h / rows
    col_idx = max(0, min(int(cx // cell_w), cols - 1)); row_idx = max(0, min(int(cy // cell_h), rows - 1))
    return col_idx, row_idx

def get_movement_label(track_id, current_centroid, frame_w, frame_h, history_key, movement_history, config_data):
    
    # Lấy các tham số di chuyển từ config
    GRID_COLS = config_data['BEHAVIOR']['GRID_COLS']
    MOVEMENT_HISTORY_LENGTH = config_data['BEHAVIOR']['MOVEMENT_HISTORY_LENGTH']
    MOVEMENT_THRESHOLD_CELLS = config_data['BEHAVIOR']['MOVEMENT_THRESHOLD_CELLS']
    
    cx, cy = current_centroid
    current_grid = get_grid_position(cx, cy, frame_w, frame_h, GRID_COLS)
    
    if track_id not in movement_history:
        movement_history[track_id] = {'hip_grids': [], 'face_grids': []}
        
    history = movement_history[track_id]
    history[history_key].append(current_grid)
    
    if len(history[history_key]) > MOVEMENT_HISTORY_LENGTH:
        history[history_key] = history[history_key][-MOVEMENT_HISTORY_LENGTH:]

    is_body = (history_key == 'hip_grids')
    default_stable_label = "BODY_STABLE" if is_body else "HEAD_STABLE"
    
    if len(history[history_key]) >= MOVEMENT_HISTORY_LENGTH:
        start_grid = history[history_key][0]; end_grid = history[history_key][-1]
        total_moved_cells = abs(end_grid[0] - start_grid[0]) + abs(end_grid[1] - start_grid[1])
        
        if total_moved_cells >= MOVEMENT_THRESHOLD_CELLS:
            return "MOVING" if is_body else "HEAD_MOVING"
        else:
            return default_stable_label
            
    return default_stable_label

def draw_skeleton(frame, keypoints, connections, line_color, point_color, thickness=2, point_radius=3):
    # ... (Giữ nguyên nội dung hàm draw_skeleton) ...
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = keypoints[i]; pt2 = keypoints[j]
            cv2.line(frame, tuple(pt1), tuple(pt2), line_color, thickness)
            
    for kp in keypoints:
        cv2.circle(frame, tuple(kp), point_radius, point_color, -1) 

# ====================================================================
## Hàm process_video_stream (HÀM TẠO ĐỂ STREAMING - CẬP NHẬT THAM SỐ CONFIG)
# ====================================================================

def process_video_stream(video_file_path, settings, cap, config_data):
    """
    Hàm tạo (generator) để xử lý video và trả về frame đã xử lý.
    Sử dụng settings (từ frontend) và config_data (từ YAML)
    """
    
    # --- LẤY CÁC THAM SỐ TỪ CONFIG DICTIONARY ---
    IOU_RISK_ASSIGN_THRESHOLD = config_data['THRESHOLDS']['IOU_RISK_ASSIGN_THRESHOLD']
    RISK_NORMAL_ID = config_data['BEHAVIOR']['RISK_NORMAL_ID']
    RISK_SUSPICIOUS_ID = config_data['BEHAVIOR']['RISK_SUSPICIOUS_ID']
    CABINET_SHELF_ID = config_data['BEHAVIOR']['CABINET_SHELF_ID']
    TROLLEY_ID = config_data['BEHAVIOR']['TROLLEY_ID']
    HANDBAG_SATCHEL_ID = config_data['BEHAVIOR']['HANDBAG_SATCHEL_ID']
    
    LINE_COLOR = tuple(config_data['DRAWING']['LINE_COLOR'])
    POINT_COLOR = tuple(config_data['DRAWING']['POINT_COLOR'])
    RISK_NORMAL_COLOR = tuple(config_data['DRAWING']['RISK_NORMAL_COLOR'])
    RISK_SUSPICIOUS_COLOR = tuple(config_data['DRAWING']['RISK_SUSPICIOUS_COLOR'])
    
    # --- 1. LOAD 3 MODEL ---
    try:
        pose_model = YOLO(config_data['MODELS']['POSE_MODEL_NAME'])
        object_model = YOLO(config_data['MODELS']['OBJECT_MODEL_NAME'])
        risk_model = YOLO(config_data['MODELS']['RISK_MODEL_NAME']) 
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}"); return

    # --- 2. CẤU HÌNH TỪ FRONTEND ---
    pose_conf_thresh = settings.get('pose_conf_thresh')
    object_conf_thresh = settings.get('object_conf_thresh')
    risk_conf_thresh = settings.get('risk_conf_thresh')
    pose_skip_frames = settings.get('pose_skip_frames')
    object_skip_frames = settings.get('object_skip_frames')
    risk_skip_frames = settings.get('risk_skip_frames')
    
    keypoint_draw = settings.get('keypoint_draw')
    risk_draw_bbox = settings.get('risk_draw_bbox')
    risk_draw_keypoint = settings.get('risk_draw_keypoint')
    objects_draw = settings.get('objects_draw')
    
    model_pose_enabled = settings.get('model_pose_enabled')
    model_object_enabled = settings.get('model_object_enabled')
    model_risk_enabled = settings.get('model_risk_enabled')
    
    # --- 3. BIẾN LƯU TẠM ---
    kalman_filters = {}; movement_history = {}; tracked_keypoints = {}; last_object_boxes = [] 
    last_tracked_bboxes = {} 
    last_risk_labels = {} 

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_h, frame_w = frame.shape[:2]
        current_tracked_bboxes = {} 
        current_risk_labels = last_risk_labels.copy() 

        # ================================================================
        # --- A. LOGIC INFERENCE VÀ CẬP NHẬT BIẾN TẠM ---
        # ================================================================
        
        # 1. INFERENCE POSE & TRACKING (Behavior/Tracking)
        is_pose_inference_frame = (frame_count % pose_skip_frames == 0) and model_pose_enabled
        
        if is_pose_inference_frame:
            
            pose_results = pose_model.track(source=frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=pose_conf_thresh)
            current_ids = set()
            
            if pose_results and pose_results[0].boxes.id is not None:
                # ... (Giữ nguyên logic xử lý pose results)
                boxes = pose_results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = pose_results[0].boxes.id.cpu().numpy().astype(int)
                keypoints_norm = pose_results[0].keypoints.xyn.cpu().numpy()
                
                new_tracked_keypoints = {}
                for box, track_id, kp_norm in zip(boxes, track_ids, keypoints_norm):
                    current_ids.add(track_id)
                    
                    if track_id not in kalman_filters:
                        kalman = KalmanFilterBox()
                        kalman.initiate(box.tolist())
                        kalman_filters[track_id] = kalman
                    else:
                        kalman_filters[track_id].update(box.tolist()) 
                    
                    kp_pixels = (kp_norm * [frame_w, frame_h]).astype(int).reshape(-1, 2).tolist()
                    new_tracked_keypoints[track_id] = kp_pixels 
                    current_tracked_bboxes[track_id] = box.tolist() 
                
                tracked_keypoints = new_tracked_keypoints
                
            ids_to_remove = set(kalman_filters.keys()) - current_ids
            for track_id in ids_to_remove:
                del kalman_filters[track_id]
                if track_id in movement_history: del movement_history[track_id]
                if track_id in current_risk_labels: del current_risk_labels[track_id] 
        
        
        # 2. INFERENCE OBJECT DETECTION (Interaction)
        if (frame_count % object_skip_frames == 0) and model_object_enabled:
            
            object_results = object_model(source=frame, verbose=False, conf=object_conf_thresh)
            
            if object_results and len(object_results[0].boxes) > 0:
                boxes_data = object_results[0].boxes.data.cpu().numpy().tolist()
                last_object_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[5]), b[4]] for b in boxes_data]
            else:
                last_object_boxes = [] 
        
        elif not model_object_enabled and (frame_count % object_skip_frames != 0):
             pass

        # 3. INFERENCE RISK POSE MODEL (Risk Classification)
        if (frame_count % risk_skip_frames == 0) and model_risk_enabled:
            
            risk_results = risk_model(source=frame, verbose=False, conf=risk_conf_thresh)
            
            if risk_results and len(risk_results[0].boxes) > 0:
                boxes = risk_results[0].boxes.xyxy.cpu().numpy().astype(int)
                class_ids = risk_results[0].boxes.cls.cpu().numpy().astype(int)
                keypoints_norm = risk_results[0].keypoints.xyn.cpu().numpy()
                
                for box, class_id, kp_norm in zip(boxes, class_ids, keypoints_norm):
                    risk_label = RISK_LABELS.get(class_id, "UNKNOWN")
                    
                    max_iou = 0.0
                    best_id = None
                    
                    for track_id, tracked_box in last_tracked_bboxes.items():
                        iou = calculate_iou(box.tolist(), tracked_box)
                        if iou > max_iou:
                            max_iou = iou
                            best_id = track_id
                            
                    if best_id is not None and max_iou > IOU_RISK_ASSIGN_THRESHOLD:
                        current_risk_labels[best_id] = risk_label
                        
                        if risk_draw_bbox or risk_draw_keypoint:
                            risk_color = RISK_SUSPICIOUS_COLOR if class_id == RISK_SUSPICIOUS_ID else RISK_NORMAL_COLOR
                            
                            if risk_draw_bbox:
                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), risk_color, 1, lineType=cv2.LINE_AA)
                                cv2.putText(frame, f'Risk:{risk_label}', (box[0], box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 1)

                            if risk_draw_keypoint:
                                kp_pixels = (kp_norm * [frame_w, frame_h]).astype(int).reshape(-1, 2).tolist()
                                draw_skeleton(frame, kp_pixels, SKELETON_CONNECTIONS, risk_color, risk_color, thickness=1, point_radius=2)
                        
            last_risk_labels = current_risk_labels.copy()

        # ================================================================
        # --- B. DỰ ĐOÁN, PHÂN LOẠI VÀ VẼ KẾT QUẢ ---
        # ================================================================
        
        # 1. Dự đoán vị trí cho tất cả ID đang theo dõi (sử dụng Kalman)
        for track_id in list(kalman_filters.keys()):
            predicted_box = kalman_filters[track_id].predict() 
            
            if not is_pose_inference_frame:
                current_tracked_bboxes[track_id] = predicted_box 
            
        last_tracked_bboxes = current_tracked_bboxes.copy()
        
        # 2. Hậu xử lý tương tác (Sử dụng Bbox của người) - Truyền config_data
        interaction_labels = classify_interactions(tracked_keypoints, last_tracked_bboxes, last_object_boxes, config_data)
        
        # 3. Vẽ kết quả Behavior/Risk lên Bounding Box của người
        for track_id in list(kalman_filters.keys()): 
            
            predicted_box = last_tracked_bboxes.get(track_id)
            if predicted_box is None: continue

            x1, y1, x2, y2 = predicted_box
            
            if track_id not in tracked_keypoints: continue 

            keypoints = tracked_keypoints[track_id] 

            # A. Phân loại Di chuyển Cơ thể và Đầu (Truyền config_data)
            body_state = "BODY_STABLE"
            head_state = "HEAD_STABLE"
            hip_centroid = None
            face_centroid = None
            
            if model_pose_enabled:
                hip_centroid = get_hip_centroid(keypoints); face_centroid = get_face_centroid(keypoints)
                if hip_centroid is None or face_centroid is None: continue

                body_state = get_movement_label(track_id, hip_centroid, frame_w, frame_h, 'hip_grids', movement_history, config_data)
                head_state = get_movement_label(track_id, face_centroid, frame_w, frame_h, 'face_grids', movement_history, config_data)

            # B. Gán nhãn Hành vi
            display_labels = []
            current_interaction = interaction_labels.get(track_id)
            risk_label = last_risk_labels.get(track_id, "N/A")

            # ----------------- TẠO NHÃN VÀ MÀU SẮC CUỐI CÙNG -----------------
            
            if risk_label == "SUSPICIOUS":
                display_color = RISK_SUSPICIOUS_COLOR 
            elif current_interaction is not None:
                if current_interaction == "HOLDING": display_color = (128, 0, 128) 
                else: display_color = (255, 0, 0) 
            elif body_state == "MOVING": 
                display_color = (0, 165, 255) 
            elif head_state == "HEAD_MOVING":
                display_color = (0, 0, 255) 
            else:
                 display_color = (0, 255, 0) 
            
            # Xây dựng danh sách nhãn hành vi
            if model_object_enabled and current_interaction is not None: display_labels.append(current_interaction)
            
            if model_pose_enabled:
                if body_state == "MOVING": display_labels.append("MOVING") 
                if head_state == "HEAD_MOVING": display_labels.append("HEAD_MOVING")
                if body_state == "BODY_STABLE" and head_state == "HEAD_STABLE": display_labels.append("WATCHING_STILL")
            
            final_behavior_string = " + ".join(sorted(list(set(display_labels))))
            if not final_behavior_string: final_behavior_string = "IDLE/STABLE"
            
            # Cấu trúc nhãn cuối cùng
            risk_display = f'Risk: {risk_label}' if model_risk_enabled else 'Risk: N/A'
            final_label = f'ID:{track_id} | {risk_display} | Behavior: {final_behavior_string}'
            
            # VẼ Bounding Box và Nhãn
            cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
            cv2.putText(frame, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
            
            # ĐIỀU KIỆN VẼ KEYPOINT/KHUNG XƯƠNG (Mô hình Pose Behavior)
            if keypoint_draw and model_pose_enabled:
                if hip_centroid and face_centroid:
                    cv2.circle(frame, hip_centroid, 5, (255, 0, 0), -1)       
                    cv2.circle(frame, face_centroid, 5, (0, 255, 255), -1)    
                draw_skeleton(frame, keypoints, SKELETON_CONNECTIONS, LINE_COLOR, POINT_COLOR)

        # 4. VẼ KẾT QUẢ OBJECT DETECTION
        if objects_draw and model_object_enabled:
            for box in last_object_boxes:
                x1, y1, x2, y2, class_id, conf = box
                
                # SỬ DỤNG ID TỪ CONFIG
                if class_id == CABINET_SHELF_ID:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    cv2.putText(frame, f'Cabinet/Shelf', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                
                elif class_id == TROLLEY_ID:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f'Trolley', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                elif class_id == HANDBAG_SATCHEL_ID:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2) 
                    cv2.putText(frame, f'Handbag/Satchel', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

        # TRẢ VỀ FRAME ĐÃ XỬ LÝ (Không dùng cv2.imshow)
        yield frame
            
        frame_count += 1
    
    cap.release()