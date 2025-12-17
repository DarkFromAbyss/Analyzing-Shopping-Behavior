import cv2
import os
import numpy as np
import traceback
from ultralytics import YOLO
from typing import Dict, Any, Generator, List, Tuple

# Import các modules nội bộ
from scripts.tracker import KalmanFilterBox, SKELETON_CONNECTIONS
from scripts.utils import calculate_iou, draw_skeleton
from scripts.behavior import (
    get_hip_centroid, get_face_centroid, get_movement_label, classify_interactions
)

# Hằng số nhãn cho mô hình Risk
RISK_LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}

# Ngưỡng Gating (Mahalanobis Distance) 
# Giá trị 9.488 tương ứng với mức tin cậy 95% cho 4 bậc tự do (cx, cy, a, h)
GATING_THRESHOLD = 9.488

def process_video_stream(video_file_path: str, settings: Dict[str, Any], cap: cv2.VideoCapture, config_data: Dict[str, Any]) -> Generator[cv2.Mat, None, None]:
    """
    Hệ thống xử lý video tích hợp Adaptive Kalman Filter với cơ chế Gating chống tráo ID.
    """
    
    # --- 1. KHỞI TẠO CẤU HÌNH ---
    try:
        IOU_RISK_ASSIGN_THRESHOLD = config_data['THRESHOLDS']['IOU_RISK_ASSIGN_THRESHOLD']
        RISK_SUSPICIOUS_ID = config_data['BEHAVIOR']['RISK_SUSPICIOUS_ID']
        
        LINE_COLOR = tuple(map(int, config_data['DRAWING']['LINE_COLOR']))
        POINT_COLOR = tuple(map(int, config_data['DRAWING']['POINT_COLOR']))
        RISK_NORMAL_COLOR = tuple(map(int, config_data['DRAWING']['RISK_NORMAL_COLOR']))
        RISK_SUSPICIOUS_COLOR = tuple(map(int, config_data['DRAWING']['RISK_SUSPICIOUS_COLOR']))
        
        tracker_config_name = config_data.get('TRACKER', {}).get('DEFAULT_TRACKER_CONFIG', 'bytetrack.yaml')
        tracker_config_path = os.path.join(os.getcwd(), 'trackers', tracker_config_name)
    except Exception as e:
        print(f"Lỗi cấu hình: {e}")
        return

    # --- 2. LOAD MODELS ---
    try:
        pose_model = YOLO(config_data['MODELS']['POSE_MODEL_NAME'])
        object_model = YOLO(config_data['MODELS']['OBJECT_MODEL_NAME'])
        risk_model = YOLO(config_data['MODELS']['RISK_MODEL_NAME'])
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        return

    # --- 3. BIẾN TRẠNG THÁI ---
    kalman_filters: Dict[int, KalmanFilterBox] = {}
    movement_history: Dict[int, Dict[str, List[Tuple[int, int]]]] = {}
    tracked_keypoints: Dict[int, List[List[int]]] = {}
    
    last_object_boxes = []
    last_tracked_bboxes: Dict[int, List[float]] = {}
    last_risk_labels: Dict[int, str] = {}
    
    frame_count = 0
    p_skip = settings.get('pose_skip_frames', 1)
    o_skip = settings.get('object_skip_frames', 5)
    r_skip = settings.get('risk_skip_frames', 3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        try:
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

    cap.release()