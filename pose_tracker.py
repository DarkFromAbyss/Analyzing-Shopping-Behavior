<<<<<<< HEAD
# pose_tracker.py (File Entry Point ĐÃ SỬA LỖI KIỂU DỮ LIỆU VÀ LOGIC HIỂN THỊ RISK)

=======
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
import cv2
import os
import numpy as np
import traceback
from ultralytics import YOLO
<<<<<<< HEAD
import numpy as np
import traceback
try:
    import torch
except Exception:
    torch = None
=======
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
from typing import Dict, Any, Generator, List, Tuple

# Import các modules nội bộ
from scripts.tracker import KalmanFilterBox, SKELETON_CONNECTIONS
from scripts.utils import calculate_iou, draw_skeleton
from scripts.behavior import (
    get_hip_centroid, get_face_centroid, get_movement_label, classify_interactions
)

<<<<<<< HEAD
# --- CÁC HẰNG SỐ CỤC BỘ ---
=======
# Hằng số nhãn cho mô hình Risk
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
RISK_LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}

# Ngưỡng Gating (Mahalanobis Distance) 
# Giá trị 9.488 tương ứng với mức tin cậy 95% cho 4 bậc tự do (cx, cy, a, h)
GATING_THRESHOLD = 9.488

def process_video_stream(video_file_path: str, settings: Dict[str, Any], cap: cv2.VideoCapture, config_data: Dict[str, Any]) -> Generator[cv2.Mat, None, None]:
    """
<<<<<<< HEAD
    Hàm generator xử lý video frame-by-frame và yield ra frame đã xử lý.
    """
    
    # --- 0. LẤY CÁC THAM SỐ TỪ CONFIG DICTIONARY ---
    
=======
    Hệ thống xử lý video tích hợp Adaptive Kalman Filter với cơ chế Gating chống tráo ID.
    """
    
    # --- 1. KHỞI TẠO CẤU HÌNH ---
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
    try:
        IOU_RISK_ASSIGN_THRESHOLD = config_data['THRESHOLDS']['IOU_RISK_ASSIGN_THRESHOLD']
        RISK_SUSPICIOUS_ID = config_data['BEHAVIOR']['RISK_SUSPICIOUS_ID']
        
<<<<<<< HEAD
        # Đảm bảo chuyển đổi sang tuple các số nguyên (int) để tránh lỗi TypeError khi vẽ
=======
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
        LINE_COLOR = tuple(map(int, config_data['DRAWING']['LINE_COLOR']))
        POINT_COLOR = tuple(map(int, config_data['DRAWING']['POINT_COLOR']))
        RISK_NORMAL_COLOR = tuple(map(int, config_data['DRAWING']['RISK_NORMAL_COLOR']))
        RISK_SUSPICIOUS_COLOR = tuple(map(int, config_data['DRAWING']['RISK_SUSPICIOUS_COLOR']))
<<<<<<< HEAD
    except Exception as e:
        print(f"LỖI CẤU HÌNH (YAML): {e}")
        return
    
    # --- 1. LOAD 3 MODEL ---
    # Determine device: prefer explicit `settings['device']`, else use CUDA if available
    requested_device = settings.get('device') if isinstance(settings, dict) else None
    if requested_device:
        device = requested_device
    else:
        if torch is not None and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

=======
        
        tracker_config_name = config_data.get('TRACKER', {}).get('DEFAULT_TRACKER_CONFIG', 'bytetrack.yaml')
        tracker_config_path = os.path.join(os.getcwd(), 'trackers', tracker_config_name)
    except Exception as e:
        print(f"Lỗi cấu hình: {e}")
        return

    # --- 2. LOAD MODELS ---
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
    try:
        pose_model = YOLO(config_data['MODELS']['POSE_MODEL_NAME'])
        object_model = YOLO(config_data['MODELS']['OBJECT_MODEL_NAME'])
        risk_model = YOLO(config_data['MODELS']['RISK_MODEL_NAME'])
<<<<<<< HEAD
        # Move models to desired device if supported by ultralytics wrapper
        try:
            pose_model.to(device)
            object_model.to(device)
            risk_model.to(device)
        except Exception:
            # Some ultralytics versions ignore .to() or manage device internally
            pass
        print(f"Models loaded on device: {device}")
=======
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        return

<<<<<<< HEAD
    # --- 2. CẤU HÌNH TỪ FRONTEND ---
    pose_conf_thresh = settings.get('pose_conf_thresh')
    object_conf_thresh = settings.get('object_conf_thresh')
    risk_conf_thresh = settings.get('risk_conf_thresh')
    pose_skip_frames = settings.get('pose_skip_frames')
    object_skip_frames = settings.get('object_skip_frames')
    risk_skip_frames = settings.get('risk_skip_frames')
=======
    # --- 3. BIẾN TRẠNG THÁI ---
    kalman_filters: Dict[int, KalmanFilterBox] = {}
    movement_history: Dict[int, Dict[str, List[Tuple[int, int]]]] = {}
    tracked_keypoints: Dict[int, List[List[int]]] = {}
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
    
    last_object_boxes = []
    last_tracked_bboxes: Dict[int, List[float]] = {}
    last_risk_labels: Dict[int, str] = {}
    
<<<<<<< HEAD
    model_pose_enabled = settings.get('model_pose_enabled')
    model_object_enabled = settings.get('model_object_enabled')
    model_risk_enabled = settings.get('model_risk_enabled')
    
    # --- 3. BIẾN LƯU TRỮ TRẠNG THÁI ---
    kalman_filters: Dict[int, KalmanFilterBox] = {} 
    movement_history: Dict[int, Dict[str, List[Tuple[int, int]]]] = {} 
    tracked_keypoints: Dict[int, List[List[int]]] = {} 
    last_object_boxes: List[List[Any]] = [] 
    last_tracked_bboxes: Dict[int, List[float]] = {} 
    last_risk_labels: Dict[int, str] = {} 

=======
>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
    frame_count = 0
    p_skip = settings.get('pose_skip_frames', 1)
    o_skip = settings.get('object_skip_frames', 5)
    r_skip = settings.get('risk_skip_frames', 3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        try:
<<<<<<< HEAD
            frame_h, frame_w = frame.shape[:2]
            current_tracked_bboxes: Dict[int, List[float]] = {} 
            current_risk_labels: Dict[int, str] = last_risk_labels.copy() 
            current_ids = set()

            # ================================================================
            # --- A. LOGIC INFERENCE VÀ CẬP NHẬT BIẾN TẠM ---
            # ================================================================
            
            is_pose_inference_frame = (frame_count % pose_skip_frames == 0) and model_pose_enabled
            
            if is_pose_inference_frame:
                
                pose_results = pose_model.track(source=frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=pose_conf_thresh)
                
                if pose_results and pose_results[0].boxes.id is not None:
                    boxes_float: List[List[float]] = pose_results[0].boxes.xyxy.cpu().numpy().tolist()
                    track_ids: List[int] = pose_results[0].boxes.id.cpu().numpy().astype(int).tolist()
                    keypoints_norm = pose_results[0].keypoints.xyn.cpu().numpy()
                    
                    new_tracked_keypoints = {}
                    for box_float, track_id, kp_norm in zip(boxes_float, track_ids, keypoints_norm):
                        current_ids.add(track_id)
                        
                        if track_id not in kalman_filters:
                            kalman_filters[track_id] = KalmanFilterBox() 
                            kalman_filters[track_id].initiate(box_float)
                        else:
                            # Cập nhật bằng đo lường mới (có Gating)
                            kalman_filters[track_id].update(box_float) 
                        
                        # Cập nhật BBox bằng kết quả Kalman (Predict/Update)
                        current_tracked_bboxes[track_id] = kalman_filters[track_id].predict() 
                        
                        # Tính Keypoint pixel 
                        frame_dims = np.array([frame_w, frame_h], dtype=np.float32) 
                        kp_pixels = (kp_norm * frame_dims).astype(int).reshape(-1, 2).tolist()
                        new_tracked_keypoints[track_id] = kp_pixels 
                    
                    tracked_keypoints = new_tracked_keypoints
                    
                # Xóa các track ID bị mất
                ids_to_remove = set(kalman_filters.keys()) - current_ids
                for track_id in ids_to_remove:
                    # Dữ liệu sẽ được lọc ở các vòng lặp tiếp theo
                    if track_id in movement_history: del movement_history[track_id]
            
            # 2. INFERENCE OBJECT DETECTION 
            if (frame_count % object_skip_frames == 0) and model_object_enabled:
                object_results = object_model(source=frame, verbose=False, conf=object_conf_thresh)
                
                if object_results and len(object_results[0].boxes) > 0:
                    boxes_data = object_results[0].boxes.data.cpu().numpy().tolist()
                    # Lưu Bbox dưới dạng int cho object detection
                    last_object_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[5]), b[4]] for b in boxes_data]
                else:
                    last_object_boxes = [] 
            
            # 3. INFERENCE RISK POSE MODEL
            if (frame_count % risk_skip_frames == 0) and model_risk_enabled:
                risk_results = risk_model(source=frame, verbose=False, conf=risk_conf_thresh)
                
                if risk_results and len(risk_results[0].boxes) > 0:
                    boxes_risk_float: List[List[float]] = risk_results[0].boxes.xyxy.cpu().numpy().tolist()
                    class_ids_risk: List[int] = risk_results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                    keypoints_norm_risk = risk_results[0].keypoints.xyn.cpu().numpy()
                    
                    for box_risk_float, class_id_risk, kp_norm_risk in zip(boxes_risk_float, class_ids_risk, keypoints_norm_risk):
                        risk_label = RISK_LABELS.get(class_id_risk, "UNKNOWN")
                        
                        max_iou = 0.0
                        best_id = None
                        box_risk_int = [int(round(val)) for val in box_risk_float]
                        
                        for track_id, tracked_box_float in last_tracked_bboxes.items():
                            tracked_box_int = [int(round(val)) for val in tracked_box_float]
                            iou = calculate_iou(box_risk_int, tracked_box_int) 
                            if iou > max_iou:
                                max_iou = iou
                                best_id = track_id
                                
                        if best_id is not None and max_iou > IOU_RISK_ASSIGN_THRESHOLD:
                            current_risk_labels[best_id] = risk_label
                            
                            if risk_draw_bbox or risk_draw_keypoint:
                                risk_color = RISK_SUSPICIOUS_COLOR if class_id_risk == RISK_SUSPICIOUS_ID else RISK_NORMAL_COLOR
                                
                                if risk_draw_bbox:
                                    cv2.rectangle(frame, (box_risk_int[0], box_risk_int[1]), (box_risk_int[2], box_risk_int[3]), risk_color, 1, lineType=cv2.LINE_AA)
                                    cv2.putText(frame, f'Risk:{risk_label}', (box_risk_int[0], box_risk_int[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 1)

                                if risk_draw_keypoint:
                                    frame_dims = np.array([frame_w, frame_h], dtype=np.float32) 
                                    kp_pixels = (kp_norm_risk * frame_dims).astype(int).reshape(-1, 2).tolist()
                                    draw_skeleton(frame, kp_pixels, SKELETON_CONNECTIONS, risk_color, risk_color, thickness=1, point_radius=2)
            else:
                # Nếu model_risk_enabled=False, reset nhãn risk cho frame này
                current_risk_labels = {}


            # ================================================================
            # --- B. DỰ ĐOÁN, PHÂN LOẠI VÀ VẼ KẾT QUẢ ---
            # ================================================================
            
            # 1. Dự đoán vị trí cho các ID không có inference Pose/Tracking trong frame này
            temp_tracked_bboxes: Dict[int, List[float]] = last_tracked_bboxes.copy()
            if not is_pose_inference_frame:
                 for track_id in list(kalman_filters.keys()):
                    predicted_box_float = kalman_filters[track_id].predict() 
                    temp_tracked_bboxes[track_id] = predicted_box_float
            else:
                 # Nếu là frame inference, current_tracked_bboxes đã được cập nhật ở bước A
                 temp_tracked_bboxes.update(current_tracked_bboxes)
                 
            last_tracked_bboxes = temp_tracked_bboxes
            last_risk_labels = current_risk_labels.copy()
            
            # 2. Hậu xử lý tương tác
            interaction_labels = classify_interactions(tracked_keypoints, last_tracked_bboxes, last_object_boxes, config_data) 
            
            # 3. Vẽ kết quả Behavior/Risk lên Bounding Box của người
            for track_id in list(kalman_filters.keys()): 
                
                predicted_box_float = last_tracked_bboxes.get(track_id)
                if predicted_box_float is None or track_id not in tracked_keypoints: continue
                
                # *** ÉP KIỂU SANG INT CHỈ TRƯỚC KHI VẼ ***
                x1, y1, x2, y2 = [int(round(val)) for val in predicted_box_float]
                keypoints = tracked_keypoints[track_id] 

                # A. Phân loại Di chuyển Cơ thể và Đầu
                body_state = "BODY_STABLE"; head_state = "HEAD_STABLE"
                hip_centroid = None; face_centroid = None
                
                if model_pose_enabled:
                    hip_centroid = get_hip_centroid(keypoints)
                    face_centroid = get_face_centroid(keypoints)
                    if hip_centroid and face_centroid:
                        body_state = get_movement_label(track_id, hip_centroid, frame_w, frame_h, 'hip_grids', movement_history, config_data)
                        head_state = get_movement_label(track_id, face_centroid, frame_w, frame_h, 'face_grids', movement_history, config_data)
                    
                # B. Gán nhãn Hành vi
                display_labels = []; current_interaction = interaction_labels.get(track_id)
                risk_label = last_risk_labels.get(track_id) # Lấy nhãn risk (có thể là None nếu risk disabled)

                # ----------------- TẠO NHÃN VÀ MÀU SẮC CUỐI CÙNG -----------------
                
                # Cập nhật màu sắc dựa trên Risk và Interaction
                display_color = (0, 255, 0) # Mặc định: Xanh lá (IDLE)
                if model_risk_enabled and risk_label == "SUSPICIOUS": 
                    display_color = RISK_SUSPICIOUS_COLOR 
                elif current_interaction is not None:
                    if current_interaction == "HOLDING": display_color = (128, 0, 128) # Tím
                    else: display_color = (255, 0, 0) # Xanh dương
                elif body_state == "MOVING": display_color = (0, 165, 255) # Cam
                elif head_state == "HEAD_MOVING": display_color = (0, 0, 255) # Đỏ

                # Cập nhật Nhãn hiển thị
                if model_object_enabled and current_interaction is not None: display_labels.append(current_interaction)
                if model_pose_enabled:
                    if body_state == "MOVING": display_labels.append("MOVING") 
                    if head_state == "HEAD_MOVING": display_labels.append("HEAD_MOVING")
                    if body_state == "BODY_STABLE" and head_state == "HEAD_STABLE": display_labels.append("WATCHING_STILL")
                
                final_behavior_string = " + ".join(sorted(list(set(display_labels))))
                if not final_behavior_string: final_behavior_string = "IDLE/STABLE"
                
                # LOGIC SỬA: CHỈ HIỂN THỊ NHÃN RISK NẾU MODEL ĐƯỢC KÍCH HOẠT
                if model_risk_enabled:
                    risk_display = f'Risk: {risk_label if risk_label else "N/A"}'
                    final_label = f'ID:{track_id} | {risk_display} | Behavior: {final_behavior_string}'
                else:
                    final_label = f'ID:{track_id} | Behavior: {final_behavior_string}'
                
                # VẼ Bounding Box và Nhãn
                cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
                cv2.putText(frame, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
                
                # ĐIỀU KIỆN VẼ KEYPOINT/KHUNG XƯƠNG
                if keypoint_draw and model_pose_enabled:
                    if hip_centroid and face_centroid:
                        cv2.circle(frame, hip_centroid, 5, (255, 0, 0), -1)       
                        cv2.circle(frame, face_centroid, 5, (0, 255, 255), -1)    
                    draw_skeleton(frame, keypoints, SKELETON_CONNECTIONS, LINE_COLOR, POINT_COLOR)

            # 4. VẼ KẾT QUẢ OBJECT DETECTION
            if objects_draw and model_object_enabled:
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
            
        except Exception as e:
            # Ghi lại lỗi chi tiết để debug
            print(f"LỖI XỬ LÝ FRAME: {e}")
            traceback.print_exc()
            break
    
=======
            h_img, w_img = frame.shape[:2]
            current_ids = set()
            m_pose_en = settings.get('model_pose_enabled', True)
            m_obj_en = settings.get('model_object_enabled', True)
            m_risk_en = settings.get('model_risk_enabled', True)

            # ==========================================
            # GIAI ĐOẠN 1: INFERENCE & GATING
            # ==========================================

            if (frame_count % p_skip == 0) and m_pose_en:
                results = pose_model.track(
                    source=frame, tracker=tracker_config_path, 
                    persist=True, verbose=False, conf=settings.get('pose_conf_thresh', 0.3)
                )
                
                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    kpts_norm = results[0].keypoints.xyn.cpu().numpy()
                    
                    new_kpts = {}
                    for box, tid, kn in zip(boxes, ids, kpts_norm):
                        current_ids.add(tid)
                        
                        if tid not in kalman_filters:
                            # Khởi tạo mới nếu ID lần đầu xuất hiện
                            kalman_filters[tid] = KalmanFilterBox()
                            kalman_filters[tid].initiate(box)
                        else:
                            # KIỂM TRA GATING TRƯỚC KHI CẬP NHẬT
                            # Tính khoảng cách Mahalanobis giữa BBox mới và dự đoán Kalman
                            m_distance = kalman_filters[tid].gating_distance(box)
                            
                            if m_distance < GATING_THRESHOLD:
                                # Nếu nằm trong ngưỡng tin cậy -> Cập nhật đo lường mới
                                kalman_filters[tid].predict()
                                kalman_filters[tid].update(box)
                            else:
                                # Nếu nằm ngoài (nghi ngờ tráo ID) -> Chỉ tin vào vận tốc dự đoán
                                kalman_filters[tid].predict()
                        
                        last_tracked_bboxes[tid] = kalman_filters[tid].get_rect()
                        new_kpts[tid] = (kn * [w_img, h_img]).astype(int).tolist()
                    
                    tracked_keypoints = new_kpts
            else:
                # Frame bị skip: Kalman tự dự đoán dựa trên vector vận tốc [vx, vy]
                for tid in list(kalman_filters.keys()):
                    last_tracked_bboxes[tid] = kalman_filters[tid].predict()

            # --- Object Detection ---
            if (frame_count % o_skip == 0) and m_obj_en:
                obj_res = object_model(frame, verbose=False, conf=settings.get('object_conf_thresh', 0.3))
                if obj_res and len(obj_res[0].boxes) > 0:
                    b_data = obj_res[0].boxes.data.cpu().numpy()
                    last_object_boxes = [[int(x1), int(y1), int(x2), int(y2), int(cls), conf] for x1, y1, x2, y2, conf, cls in b_data]
            
            # --- Risk Inference (Chỉ hiện Risk nếu model enabled) ---
            if (frame_count % r_skip == 0) and m_risk_en:
                risk_res = risk_model(frame, verbose=False, conf=settings.get('risk_conf_thresh', 0.3))
                new_risks = {}
                if risk_res and len(risk_res[0].boxes) > 0:
                    r_boxes = risk_res[0].boxes.xyxy.cpu().numpy()
                    r_clss = risk_res[0].boxes.cls.cpu().numpy().astype(int)
                    for rb, rc in zip(r_boxes, r_clss):
                        best_iou, best_id = IOU_RISK_ASSIGN_THRESHOLD, None
                        for tid, tbox in last_tracked_bboxes.items():
                            iou = calculate_iou(rb.astype(int), [int(i) for i in tbox])
                            if iou > best_iou:
                                best_iou, best_id = iou, tid
                        if best_id is not None:
                            new_risks[best_id] = RISK_LABELS.get(rc, "UNKNOWN")
                last_risk_labels = new_risks

            # ==========================================
            # GIAI ĐOẠN 2: PHÂN TÍCH HÀNH VI & VẼ
            # ==========================================
            
            int_labels = classify_interactions(tracked_keypoints, last_tracked_bboxes, last_object_boxes, config_data)
            
            for tid in list(kalman_filters.keys()):
                if tid not in last_tracked_bboxes or tid not in tracked_keypoints: continue
                
                bbox = [int(i) for i in last_tracked_bboxes[tid]]
                kpts = tracked_keypoints[tid]
                
                # Tính di chuyển lưới (Grid)
                body_st = "BODY_STABLE"
                hip = get_hip_centroid(kpts)
                if hip:
                    body_st = get_movement_label(tid, hip, w_img, h_img, 'hip_grids', movement_history, config_data)

                # Quyết định màu sắc
                risk_val = last_risk_labels.get(tid) if m_risk_en else None
                inter_val = int_labels.get(tid) if m_obj_en else None
                
                color = (0, 255, 0) # Mặc định xanh lá
                if risk_val == "SUSPICIOUS": color = RISK_SUSPICIOUS_COLOR
                elif inter_val: color = (255, 0, 255) # Tím cho tương tác
                elif body_st == "MOVING": color = (0, 165, 255) # Cam

                # Tạo nhãn text
                risk_txt = f"Risk:{risk_val} | " if m_risk_en and risk_val else ""
                inter_txt = f"{inter_val} | " if inter_val else ""
                move_txt = "MOVING" if body_st == "MOVING" else "STABLE"
                
                final_label = f"ID:{tid} | {risk_txt}{inter_txt}{move_txt}"

                # Vẽ BBox và Label
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, final_label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if settings.get('keypoint_draw') and m_pose_en:
                    draw_skeleton(frame, kpts, SKELETON_CONNECTIONS, LINE_COLOR, POINT_COLOR)

            # Cleanup IDs
            if (frame_count % p_skip == 0):
                for tid in list(kalman_filters.keys()):
                    if tid not in current_ids:
                        del kalman_filters[tid]

            yield frame
            frame_count += 1

        except Exception as e:
            traceback.print_exc()
            break

>>>>>>> 4d0a98cefd45f90e0179c169742c2868f2e4d758
    cap.release()