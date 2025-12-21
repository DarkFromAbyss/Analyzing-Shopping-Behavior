import cv2
from ultralytics import YOLO
import numpy as np
import traceback
import time
try:
    import torch
except Exception:
    torch = None
from typing import Dict, Any, Generator, List, Tuple

# Import module con
from scripts.tracker import KalmanFilterBox, SKELETON_CONNECTIONS
from scripts.utils import calculate_iou, draw_skeleton
from scripts.behavior import (
    get_hip_centroid, get_face_centroid, get_movement_label, classify_interactions
)

# --- CONFIG CONSTANTS ---
RISK_LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}
MAX_LOST_FRAMES = 30    
IOU_RECOVERY_THRESH = 0.3 
EMA_ALPHA = 0.3 

# --- METRIC THRESHOLDS ---
THRESH_ML_FRAMES = 15   
THRESH_MT_FRAMES = 90   

def draw_grid_overlay(frame, grid_cols, color=(50, 50, 50), thickness=1):
    """Vẽ lưới chia ô lên frame để trực quan hóa vùng di chuyển."""
    h, w = frame.shape[:2]
    step_x = w / grid_cols
    for i in range(1, grid_cols):
        x = int(i * step_x)
        cv2.line(frame, (x, 0), (x, h), color, thickness)
    
    grid_rows = int((h / w) * grid_cols)
    if grid_rows < 1: grid_rows = 1
    
    step_y = h / grid_rows
    for i in range(1, grid_rows):
        y = int(i * step_y)
        cv2.line(frame, (0, y), (w, y), color, thickness)

def draw_dashboard_overlay(frame, metrics: Dict[str, Any]):
    """Vẽ Dashboard hiển thị metrics."""
    h, w = frame.shape[:2]
    # [THÊM] Tăng chiều cao panel để chứa thêm thông tin Latency
    panel_w, panel_h = 280, 245 
    x_start, y_start = w - panel_w - 20, 20
    
    cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (100, 100, 100), 1)
    
    cv2.putText(frame, "SYSTEM HEALTH", (x_start + 10, y_start + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(frame, (x_start + 10, y_start + 35), (x_start + panel_w - 10, y_start + 35), (100, 100, 100), 1)

    col1_x = x_start + 10
    y_row = y_start + 60
    gap = 25
    
    cv2.putText(frame, f"FPS: {metrics.get('fps', 0):.1f}", (col1_x, y_row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Active: {metrics.get('active', 0)}", (col1_x, y_row + gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Total IDs: {metrics.get('total', 0)}", (col1_x, y_row + gap*2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Risk: {metrics.get('risk', 0)}", (col1_x, y_row + gap*3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    col2_x = x_start + 140
    
    fn_val = metrics.get('coasting', 0)
    fn_color = (0, 255, 255) if fn_val > 0 else (100, 100, 100)
    cv2.putText(frame, f"FN(Coast): {fn_val}", (col2_x, y_row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fn_color, 1)

    rec_val = metrics.get('recovered', 0)
    cv2.putText(frame, f"Recovered: {rec_val}", (col2_x, y_row + gap), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
    
    ml_val = metrics.get('ml', 0)
    cv2.putText(frame, f"ML(Noise): {ml_val}", (col2_x, y_row + gap*2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    mt_val = metrics.get('mt', 0)
    cv2.putText(frame, f"MT(Stable): {mt_val}", (col2_x, y_row + gap*3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 155, 0), 1)

    # [THÊM] Phần hiển thị tốc độ xử lý (Latency)
    y_lat_sep = y_row + gap * 4 + 5
    cv2.line(frame, (x_start + 10, y_lat_sep), (x_start + panel_w - 10, y_lat_sep), (100, 100, 100), 1)
    
    cv2.putText(frame, "LATENCY (ms)", (x_start + 10, y_lat_sep + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    lat_pose = metrics.get('lat_pose', 0)
    lat_obj = metrics.get('lat_obj', 0)
    lat_risk = metrics.get('lat_risk', 0)

    # Hiển thị 3 cột nhỏ hoặc dòng
    cv2.putText(frame, f"Pose: {lat_pose:.1f}", (x_start + 10, y_lat_sep + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Obj: {lat_obj:.1f}", (x_start + 100, y_lat_sep + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Risk: {lat_risk:.1f}", (x_start + 190, y_lat_sep + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def process_video_stream(video_file_path: str, settings: Dict[str, Any], cap: cv2.VideoCapture, config_data: Dict[str, Any]) -> Generator[cv2.Mat, None, None]:
    try:
        IOU_RISK = config_data['THRESHOLDS']['IOU_RISK_ASSIGN_THRESHOLD']
        GRID_COLS = config_data['BEHAVIOR']['GRID_COLS']
        
        LINE_COLOR = tuple(map(int, config_data['DRAWING']['LINE_COLOR']))
        POINT_COLOR = tuple(map(int, config_data['DRAWING']['POINT_COLOR']))
        RISK_SUS_CLR = tuple(map(int, config_data['DRAWING']['RISK_SUSPICIOUS_COLOR']))
    except Exception as e:
        print(f"Config Error: {e}"); return
    
    device = settings.get('device', 'cuda' if torch and torch.cuda.is_available() else 'cpu')
    try:
        pose_model = YOLO(config_data['MODELS']['POSE_MODEL_NAME']).to(device)
        object_model = YOLO(config_data['MODELS']['OBJECT_MODEL_NAME']).to(device)
        risk_model = YOLO(config_data['MODELS']['RISK_MODEL_NAME']).to(device)
    except Exception as e:
        print(f"Model Load Error: {e}"); return

    # Settings from Frontend/Config
    pose_conf = settings.get('pose_conf_thresh', 0.5)
    obj_conf = settings.get('object_conf_thresh', 0.5)
    risk_conf = settings.get('risk_conf_thresh', 0.5)
    pose_skip = settings.get('pose_skip_frames', 2)
    obj_skip = settings.get('object_skip_frames', 10)
    risk_skip = settings.get('risk_skip_frames', 10)
    
    draw_kp = settings.get('keypoint_draw')
    draw_grid = settings.get('draw_grid', False)
    draw_risk_raw = settings.get('risk_draw_bbox', True)
    draw_obj = settings.get('objects_draw')
    
    en_pose = settings.get('model_pose_enabled')
    en_obj = settings.get('model_object_enabled')
    en_risk = settings.get('model_risk_enabled')
    
    # --- GLOBAL STATE VARIABLES ---
    kalman_filters: Dict[int, KalmanFilterBox] = {} 
    movement_history = {} 
    tracked_keypoints = {} 
    last_object_boxes = [] 
    last_tracked_bboxes = {} 
    last_risk_labels = {} 
    last_raw_risk_boxes = [] 

    # --- METRICS COUNTERS ---
    total_ids_seen = set()
    metric_recovered_count = 0  
    metric_ml_count = 0         
    metric_mt_count = 0         
    
    # [THÊM] Biến lưu thời gian xử lý (Latency)
    lat_pose_ms = 0.0
    lat_obj_ms = 0.0
    lat_risk_ms = 0.0

    frame_count = 0
    fps_avg = 0
    
    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        try:
            h_frame, w_frame = frame.shape[:2]
            
            if draw_grid:
                draw_grid_overlay(frame, GRID_COLS)

            current_tracked_bboxes = {} 
            current_risk_labels = last_risk_labels.copy() 
            active_ids_this_frame = set() 

            # --- POSE & TRACKING LOGIC ---
            is_pose_frame = (frame_count % pose_skip == 0) and en_pose
            
            if is_pose_frame:
                # [THÊM] Đo thời gian Pose
                t_start = time.time()
                results = pose_model.track(frame, tracker="botsort.yaml", persist=True, verbose=False, conf=pose_conf)
                lat_pose_ms = (time.time() - t_start) * 1000 # ms

                id_map = {} 

                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
                    ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
                    kps = results[0].keypoints.xyn.cpu().numpy()
                    
                    # 1. Recovery Logic 
                    detected_ids = set(ids)
                    existing_ids = set(kalman_filters.keys())
                    new_ids = detected_ids - existing_ids
                    lost_ids = existing_ids - detected_ids

                    for nid in list(new_ids):
                        idx = ids.index(nid)
                        nbox = [int(x) for x in boxes[idx]]
                        best_match, best_iou = None, -1.0

                        for lid in lost_ids:
                            kf = kalman_filters[lid]
                            if kf.time_since_update < MAX_LOST_FRAMES:
                                pbox = [int(x) for x in kf.get_rect()]
                                iou = calculate_iou(nbox, pbox)
                                if iou > best_iou: best_iou, best_match = iou, lid
                        
                        if best_match is not None and best_iou > IOU_RECOVERY_THRESH:
                            id_map[nid] = best_match
                            new_ids.remove(nid)
                            lost_ids.remove(best_match)
                            metric_recovered_count += 1 
                    
                    # 2. Update Kalman & Smooth Box
                    new_tracked_kps = {}
                    for box, rid, kp in zip(boxes, ids, kps):
                        fid = id_map.get(rid, rid)
                        active_ids_this_frame.add(fid)
                        total_ids_seen.add(fid)
                        
                        if fid not in kalman_filters:
                            kalman_filters[fid] = KalmanFilterBox() 
                            kalman_filters[fid].initiate(box)
                            current_tracked_bboxes[fid] = box
                        else:
                            kalman_filters[fid].update(box)
                            kf_box = kalman_filters[fid].get_current_state()
                            if fid in last_tracked_bboxes:
                                last_box = last_tracked_bboxes[fid]
                                kf_w, kf_h = kf_box[2]-kf_box[0], kf_box[3]-kf_box[1]
                                kf_cx, kf_cy = (kf_box[0]+kf_box[2])/2, (kf_box[1]+kf_box[3])/2
                                last_w, last_h = last_box[2]-last_box[0], last_box[3]-last_box[1]
                                new_w = last_w*(1-EMA_ALPHA) + kf_w*EMA_ALPHA
                                new_h = last_h*(1-EMA_ALPHA) + kf_h*EMA_ALPHA
                                current_tracked_bboxes[fid] = [kf_cx-new_w/2, kf_cy-new_h/2, kf_cx+new_w/2, kf_cy+new_h/2]
                            else:
                                current_tracked_bboxes[fid] = kf_box
                        
                        kp_px = (kp * np.array([w_frame, h_frame])).astype(int).reshape(-1, 2).tolist()
                        new_tracked_kps[fid] = kp_px
                    
                    tracked_keypoints = new_tracked_kps
                
                # 3. Clean Lost IDs
                del_ids = []
                for tid, kf in kalman_filters.items():
                    if tid not in active_ids_this_frame:
                        pred_box = kf.predict() # Coasting
                        current_tracked_bboxes[tid] = pred_box
                        
                        if kf.time_since_update > MAX_LOST_FRAMES: 
                            del_ids.append(tid)
                            if getattr(kf, 'age', 0) < THRESH_ML_FRAMES:
                                metric_ml_count += 1

                for tid in del_ids:
                    del kalman_filters[tid]
                    if tid in movement_history: del movement_history[tid]
            
            # --- OBJECT DETECTION ---
            if (frame_count % obj_skip == 0) and en_obj:
                # [THÊM] Đo thời gian Object
                t_start = time.time()
                res = object_model(frame, verbose=False, conf=obj_conf)
                lat_obj_ms = (time.time() - t_start) * 1000 # ms

                if res and len(res[0].boxes) > 0:
                    last_object_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[5]), b[4]] for b in res[0].boxes.data.cpu().tolist()]
                else: last_object_boxes = []
            
            # --- RISK DETECTION (LOGIC CONTAINMENT) ---
            if (frame_count % risk_skip == 0) and en_risk:
                # [THÊM] Đo thời gian Risk
                t_start = time.time()
                res = risk_model(frame, verbose=False, conf=risk_conf)
                lat_risk_ms = (time.time() - t_start) * 1000 # ms

                detected_risks = []
                
                if res and len(res[0].boxes) > 0:
                    for box, cls in zip(res[0].boxes.xyxy, res[0].boxes.cls):
                        r_box = [int(x) for x in box]
                        label = RISK_LABELS.get(int(cls), "UNK")
                        detected_risks.append((r_box, label))
                        
                        # Logic Containment
                        if en_pose and last_tracked_bboxes:
                            best_overlap_ratio, best_id = 0.0, None
                            risk_area = (r_box[2] - r_box[0]) * (r_box[3] - r_box[1])
                            
                            for tid, p_box in last_tracked_bboxes.items():
                                p_box = [int(x) for x in p_box]
                                xA, yA = max(r_box[0], p_box[0]), max(r_box[1], p_box[1])
                                xB, yB = min(r_box[2], p_box[2]), min(r_box[3], p_box[3])
                                interArea = max(0, xB - xA) * max(0, yB - yA)
                                
                                if risk_area > 0:
                                    overlap_ratio = interArea / risk_area
                                else: overlap_ratio = 0
                                
                                if overlap_ratio > best_overlap_ratio:
                                    best_overlap_ratio, best_id = overlap_ratio, tid
                            
                            if best_id is not None and best_overlap_ratio > 0.3:
                                current_risk_labels[best_id] = label

                if not en_pose: last_raw_risk_boxes = detected_risks
                else: last_raw_risk_boxes = []
            
            if not en_risk:
                current_risk_labels = {}
                last_raw_risk_boxes = []

            # --- PREDICTION & DRAWING ---
            temp_bboxes = last_tracked_bboxes.copy()
            if is_pose_frame: temp_bboxes = current_tracked_bboxes
            else:
                for tid, kf in kalman_filters.items(): temp_bboxes[tid] = kf.predict()
            
            last_tracked_bboxes = temp_bboxes
            last_risk_labels = current_risk_labels.copy()
            
            interactions = classify_interactions(tracked_keypoints, last_tracked_bboxes, last_object_boxes, config_data)
            
            # Count Metrics
            current_mt_count = 0
            current_coasting_count = 0
            for tid, kf in kalman_filters.items():
                if getattr(kf, 'age', 0) > THRESH_MT_FRAMES: current_mt_count += 1
                if tid not in active_ids_this_frame and is_pose_frame: current_coasting_count += 1

            # --- VẼ THÔNG TIN LÊN NGƯỜI ---
            for tid in list(kalman_filters.keys()):
                if tid not in last_tracked_bboxes: continue
                x1, y1, x2, y2 = [int(x) for x in last_tracked_bboxes[tid]]
                
                kf = kalman_filters[tid]
                is_lost = kf.time_since_update > (pose_skip + 2)
                kps = tracked_keypoints.get(tid)
                
                body, head = "STABLE", "STABLE"
                hc, fc = None, None
                if en_pose and kps:
                    hc, fc = get_hip_centroid(kps), get_face_centroid(kps)
                    if hc and fc:
                        body = get_movement_label(tid, hc, w_frame, h_frame, 'hip_grids', movement_history, config_data)
                        head = get_movement_label(tid, fc, w_frame, h_frame, 'face_grids', movement_history, config_data)
                
                inter, risk = interactions.get(tid), last_risk_labels.get(tid)
                
                # Màu sắc
                color = (0, 255, 0)
                if is_lost: color = (192, 192, 192)
                elif risk == "SUSPICIOUS": color = RISK_SUS_CLR
                elif inter == "HOLDING": color = (128, 0, 128)
                elif inter in ["TAKING", "PUSHING"]: color = (255, 0, 0)
                elif body == "MOVING": color = (0, 165, 255)
                
                lbls = []
                if en_obj and inter: lbls.append(inter)
                
                if en_pose and not is_lost:
                    if body == "MOVING": 
                        lbls.append("MOVING")
                    if head == "HEAD_MOVING": 
                        lbls.append("HEAD_MOVING")
                    elif head == "HEAD_STABLE" and body == "BODY_STABLE":
                        lbls.append("WATCHING")
                
                if is_lost: lbls.append("LOST")
                if not lbls: lbls.append("IDLE")

                risk_display = risk if risk else "None"
                final_txt = f"ID:{tid}|Risk:{risk_display}|{'+'.join(lbls)}"
                
                # Vẽ Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ Nhãn Nền Đen
                (txt_w, txt_h), _ = cv2.getTextSize(final_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - txt_h - 10), (x1 + txt_w, y1), (0, 0, 0), -1) 
                cv2.rectangle(frame, (x1, y1 - txt_h - 10), (x1 + txt_w, y1), color, 1) 
                cv2.putText(frame, final_txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if draw_kp and en_pose and kps and not is_lost:
                    draw_skeleton(frame, kps, SKELETON_CONNECTIONS, LINE_COLOR, POINT_COLOR)

            # --- VẼ OBJECT ---
            if draw_obj and en_obj:
                for b in last_object_boxes:
                    is_cabinet = (b[4] == config_data['BEHAVIOR']['CABINET_SHELF_ID'])
                    c = (255, 100, 0) if is_cabinet else (255, 255, 255)
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), c, 2)
                    if is_cabinet:
                        (tw, th), _ = cv2.getTextSize("CABINET", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (b[0], b[1]-th-5), (b[0]+tw, b[1]), (0,0,0), -1)
                        cv2.putText(frame, "CABINET", (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

            # --- VẼ RISK (RAW) ---
            if en_risk and (not en_pose) and draw_risk_raw:
                for (rbox, rlabel) in last_raw_risk_boxes:
                    c = RISK_SUS_CLR if rlabel == "SUSPICIOUS" else (0, 255, 0)
                    cv2.rectangle(frame, (rbox[0], rbox[1]), (rbox[2], rbox[3]), c, 2)
                    lbl = f"Risk: {rlabel}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (rbox[0], rbox[1]-th-5), (rbox[0]+tw, rbox[1]), (0,0,0), -1)
                    cv2.putText(frame, lbl, (rbox[0], rbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

            # --- METRICS OVERLAY ---
            fps_curr = 1 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
            fps_avg = 0.9 * fps_avg + 0.1 * fps_curr if fps_avg > 0 else fps_curr
            
            metrics_data = {
                'fps': fps_avg,
                'active': len(active_ids_this_frame),
                'total': len(total_ids_seen),
                'risk': list(last_risk_labels.values()).count("SUSPICIOUS") + len([r for r in last_raw_risk_boxes if r[1] == "SUSPICIOUS"]),
                'coasting': current_coasting_count,
                'recovered': metric_recovered_count,
                'ml': metric_ml_count,
                'mt': current_mt_count,
                # [THÊM] Truyền dữ liệu Latency vào dashboard
                'lat_pose': lat_pose_ms,
                'lat_obj': lat_obj_ms,
                'lat_risk': lat_risk_ms
            }
            draw_dashboard_overlay(frame, metrics_data)

            yield frame
            frame_count += 1
            
        except Exception as e:
            print(f"Frame Error: {e}"); traceback.print_exc(); break
    
    cap.release()