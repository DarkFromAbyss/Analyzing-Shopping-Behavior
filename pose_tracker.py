# pose_tracker.py (File Entry Point mới)

import cv2
from ultralytics import YOLO
from typing import Dict, Any, Generator

# Import các hàm và lớp từ module con
from scripts.tracker import KalmanFilterBox, SKELETON_CONNECTIONS
from scripts.utils import calculate_iou, draw_skeleton
from scripts.behavior import (
    get_hip_centroid, get_face_centroid, get_movement_label, classify_interactions
)

# --- CÁC HẰNG SỐ KHÔNG THAY ĐỔI / CHÚ THÍCH ---
# ID Lớp của mô hình Rủi ro (Chỉ cần trong file này để map nhãn)
RISK_LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}


def process_video_stream(video_file_path: str, settings: Dict[str, Any], cap: cv2.VideoCapture, config_data: Dict[str, Any]) -> Generator[cv2.Mat, None, None]:
    """
    Hàm generator xử lý video frame-by-frame và yield ra frame đã xử lý.
    Sử dụng cấu hình từ Frontend (settings) và YAML (config_data).
    """
    
    # --- 0. LẤY CÁC THAM SỐ TỪ CONFIG DICTIONARY ---
    
    # Parameters từ YAML (cho Vision logic)
    IOU_RISK_ASSIGN_THRESHOLD = config_data['THRESHOLDS']['IOU_RISK_ASSIGN_THRESHOLD']
    RISK_NORMAL_ID = config_data['BEHAVIOR']['RISK_NORMAL_ID']
    RISK_SUSPICIOUS_ID = config_data['BEHAVIOR']['RISK_SUSPICIOUS_ID']
    
    # Drawing Colors (Chuyển List BGR sang Tuple BGR)
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
    # (Được giữ nguyên như trước, vì settings chỉ là Dict[str, Any])
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
    
    # --- 3. BIẾN LƯU TRỮ TRẠNG THÁI ---
    kalman_filters = {}; movement_history = {}; tracked_keypoints = {}; last_object_boxes = [] 
    last_tracked_bboxes = {} 
    last_risk_labels = {} 

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_h, frame_w = frame.shape[:2]
        current_tracked_bboxes = {}; current_risk_labels = last_risk_labels.copy() 

        # ================================================================
        # --- A. LOGIC INFERENCE VÀ CẬP NHẬT BIẾN TẠM (Cần Import và Gọi) ---
        # ================================================================
        
        # 1. INFERENCE POSE & TRACKING
        is_pose_inference_frame = (frame_count % pose_skip_frames == 0) and model_pose_enabled
        
        if is_pose_inference_frame:
            # ... (Logic YOLOv8 tracking giữ nguyên) ...
            pose_results = pose_model.track(source=frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=pose_conf_thresh)
            current_ids = set()
            
            if pose_results and pose_results[0].boxes.id is not None:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = pose_results[0].boxes.id.cpu().numpy().astype(int)
                keypoints_norm = pose_results[0].keypoints.xyn.cpu().numpy()
                
                new_tracked_keypoints = {}
                for box, track_id, kp_norm in zip(boxes, track_ids, keypoints_norm):
                    current_ids.add(track_id)
                    
                    if track_id not in kalman_filters:
                        kalman_filters[track_id] = KalmanFilterBox() # Sử dụng LỚP TỪ MODULE
                        kalman_filters[track_id].initiate(box.tolist())
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
        
        # 2. INFERENCE OBJECT DETECTION 
        if (frame_count % object_skip_frames == 0) and model_object_enabled:
            object_results = object_model(source=frame, verbose=False, conf=object_conf_thresh)
            # ... (Logic xử lý object results giữ nguyên)
            if object_results and len(object_results[0].boxes) > 0:
                boxes_data = object_results[0].boxes.data.cpu().numpy().tolist()
                last_object_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[5]), b[4]] for b in boxes_data]
            else:
                last_object_boxes = [] 
        
        elif not model_object_enabled and (frame_count % object_skip_frames != 0):
             pass

        # 3. INFERENCE RISK POSE MODEL
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
                        iou = calculate_iou(box.tolist(), tracked_box) # SỬ DỤNG HÀM TỪ MODULE
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
        
        # 1. Dự đoán vị trí cho tất cả ID đang theo dõi
        for track_id in list(kalman_filters.keys()):
            predicted_box = kalman_filters[track_id].predict() 
            
            if not is_pose_inference_frame:
                current_tracked_bboxes[track_id] = predicted_box 
            
        last_tracked_bboxes = current_tracked_bboxes.copy()
        
        # 2. Hậu xử lý tương tác
        interaction_labels = classify_interactions(tracked_keypoints, last_tracked_bboxes, last_object_boxes, config_data) # SỬ DỤNG HÀM TỪ MODULE
        
        # 3. Vẽ kết quả Behavior/Risk lên Bounding Box của người
        for track_id in list(kalman_filters.keys()): 
            
            predicted_box = last_tracked_bboxes.get(track_id)
            if predicted_box is None: continue

            x1, y1, x2, y2 = predicted_box
            if track_id not in tracked_keypoints: continue 

            keypoints = tracked_keypoints[track_id] 

            # A. Phân loại Di chuyển Cơ thể và Đầu
            body_state = "BODY_STABLE"; head_state = "HEAD_STABLE"
            hip_centroid = None; face_centroid = None
            
            if model_pose_enabled:
                hip_centroid = get_hip_centroid(keypoints) # SỬ DỤNG HÀM TỪ MODULE
                face_centroid = get_face_centroid(keypoints) # SỬ DỤNG HÀM TỪ MODULE
                if hip_centroid and face_centroid:
                    body_state = get_movement_label(track_id, hip_centroid, frame_w, frame_h, 'hip_grids', movement_history, config_data) # SỬ DỤNG HÀM TỪ MODULE
                    head_state = get_movement_label(track_id, face_centroid, frame_w, frame_h, 'face_grids', movement_history, config_data) # SỬ DỤNG HÀM TỪ MODULE
                
            # B. Gán nhãn Hành vi
            display_labels = []; current_interaction = interaction_labels.get(track_id)
            risk_label = last_risk_labels.get(track_id, "N/A")

            # ----------------- TẠO NHÃN VÀ MÀU SẮC CUỐI CÙNG -----------------
            if risk_label == "SUSPICIOUS": display_color = RISK_SUSPICIOUS_COLOR 
            elif current_interaction is not None:
                if current_interaction == "HOLDING": display_color = (128, 0, 128) 
                else: display_color = (255, 0, 0) 
            elif body_state == "MOVING": display_color = (0, 165, 255) 
            elif head_state == "HEAD_MOVING": display_color = (0, 0, 255) 
            else: display_color = (0, 255, 0) 
            
            if model_object_enabled and current_interaction is not None: display_labels.append(current_interaction)
            if model_pose_enabled:
                if body_state == "MOVING": display_labels.append("MOVING") 
                if head_state == "HEAD_MOVING": display_labels.append("HEAD_MOVING")
                if body_state == "BODY_STABLE" and head_state == "HEAD_STABLE": display_labels.append("WATCHING_STILL")
            
            final_behavior_string = " + ".join(sorted(list(set(display_labels))))
            if not final_behavior_string: final_behavior_string = "IDLE/STABLE"
            
            risk_display = f'Risk: {risk_label}' if model_risk_enabled else 'Risk: N/A'
            final_label = f'ID:{track_id} | {risk_display} | Behavior: {final_behavior_string}'
            
            # VẼ Bounding Box và Nhãn
            cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
            cv2.putText(frame, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
            
            # ĐIỀU KIỆN VẼ KEYPOINT/KHUNG XƯƠNG
            if keypoint_draw and model_pose_enabled:
                if hip_centroid and face_centroid:
                    cv2.circle(frame, hip_centroid, 5, (255, 0, 0), -1)       
                    cv2.circle(frame, face_centroid, 5, (0, 255, 255), -1)    
                draw_skeleton(frame, keypoints, SKELETON_CONNECTIONS, LINE_COLOR, POINT_COLOR) # SỬ DỤNG HÀM TỪ MODULE

        # 4. VẼ KẾT QUẢ OBJECT DETECTION
        if objects_draw and model_object_enabled:
            # ... (Logic vẽ object giữ nguyên)
            cfg_behavior = config_data['BEHAVIOR']
            CABINET_SHELF_ID = cfg_behavior['CABINET_SHELF_ID']
            TROLLEY_ID = cfg_behavior['TROLLEY_ID']
            HANDBAG_SATCHEL_ID = cfg_behavior['HANDBAG_SATCHEL_ID']
            
            for box in last_object_boxes:
                x1, y1, x2, y2, class_id, conf = box
                
                if class_id == CABINET_SHELF_ID:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    cv2.putText(frame, f'Cabinet/Shelf', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                
                elif class_id == TROLLEY_ID:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f'Trolley', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                elif class_id == HANDBAG_SATCHEL_ID:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2) 
                    cv2.putText(frame, f'Handbag/Satchel', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

        yield frame
            
        frame_count += 1
    
    cap.release()