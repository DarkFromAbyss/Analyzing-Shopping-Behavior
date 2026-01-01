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

from scripts.tracker import KalmanFilterBox, SKELETON_CONNECTIONS
from scripts.utils import calculate_iou, draw_skeleton
from scripts.behavior import (
    get_hip_centroid, get_face_centroid, get_movement_label, classify_interactions
)

RISK_LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}
MAX_LOST_FRAMES = 30    
IOU_RECOVERY_THRESH = 0.3 
EMA_ALPHA = 0.3 
THRESH_ML_FRAMES = 15   
THRESH_MT_FRAMES = 90   

def draw_grid_overlay(frame, grid_cols, color=(50, 50, 50), thickness=1):
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
    """Vẽ Dashboard hiển thị metrics bao gồm cả thông tin mAP trung bình."""
    h, w = frame.shape[:2]
    # Tăng chiều cao panel để chứa thêm thông tin mAP
    panel_w, panel_h = 280, 280 
    x_start, y_start = w - panel_w - 20, 20
    
    cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (100, 100, 100), 1)
    
    cv2.putText(frame, "SYSTEM HEALTH", (x_start + 10, y_start + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(frame, (x_start + 10, y_start + 35), (x_start + panel_w - 10, y_start + 35), (100, 100, 100), 1)

    y_row = y_start + 55
    gap = 22
    
    cv2.putText(frame, f"FPS: {metrics.get('fps', 0):.1f}", (x_start + 10, y_row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(frame, f"Active: {metrics.get('active', 0)}", (x_start + 10, y_row + gap), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"Risk: {metrics.get('risk', 0)}", (x_start + 10, y_row + gap*2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # --- PHẦN mAP TRUNG BÌNH (AVERAGE CONFIDENCE) ---
    y_map_sep = y_row + gap*3 
    cv2.line(frame, (x_start + 10, y_map_sep), (x_start + panel_w - 10, y_map_sep), (100, 100, 100), 1)
    cv2.putText(frame, "AVG mAP (CONF)", (x_start + 10, y_map_sep + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    
    cv2.putText(frame, f"Pose: {metrics.get('map_pose', 0):.2f}", (x_start + 10, y_map_sep + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Obj:  {metrics.get('map_obj', 0):.2f}", (x_start + 10, y_map_sep + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Risk: {metrics.get('map_risk', 0):.2f}", (x_start + 10, y_map_sep + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # --- PHẦN LATENCY ---
    y_lat_sep = y_map_sep + 85
    cv2.line(frame, (x_start + 10, y_lat_sep), (x_start + panel_w - 10, y_lat_sep), (100, 100, 100), 1)
    cv2.putText(frame, "LATENCY (ms)", (x_start + 10, y_lat_sep + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    l_pose, l_obj, l_risk = metrics.get('lat_pose', 0), metrics.get('lat_obj', 0), metrics.get('lat_risk', 0)
    cv2.putText(frame, f"P:{l_pose:.1f} O:{l_obj:.1f} R:{l_risk:.1f}", (x_start + 10, y_lat_sep + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

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

    pose_conf = settings.get('pose_conf_thresh', 0.5)
    obj_conf = settings.get('object_conf_thresh', 0.5)
    risk_conf = settings.get('risk_conf_thresh', 0.5)
    pose_skip, obj_skip, risk_skip = settings.get('pose_skip_frames', 2), settings.get('object_skip_frames', 10), settings.get('risk_skip_frames', 10)
    draw_kp, draw_grid, draw_risk_raw, draw_obj = settings.get('keypoint_draw'), settings.get('draw_grid', False), settings.get('risk_draw_bbox', True), settings.get('objects_draw')
    en_pose, en_obj, en_risk = settings.get('model_pose_enabled'), settings.get('model_object_enabled'), settings.get('model_risk_enabled')
    
    kalman_filters, movement_history, tracked_keypoints, last_object_boxes = {}, {}, {}, []
    last_tracked_bboxes, last_risk_labels, last_raw_risk_boxes = {}, {}, []
    total_ids_seen, metric_recovered_count, metric_ml_count, metric_mt_count = set(), 0, 0, 0
    
    # --- BIẾN MỚI CHO mAP (AVG CONFIDENCE) ---
    sum_pose_conf, cnt_pose = 0.0, 0
    sum_obj_conf, cnt_obj = 0.0, 0
    sum_risk_conf, cnt_risk = 0.0, 0
    last_pose_confs = {} # Lưu confidence hiện tại của từng ID

    lat_pose_ms, lat_obj_ms, lat_risk_ms = 0.0, 0.0, 0.0
    frame_count, fps_avg = 0, 0
    
    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret: break
        try:
            h_frame, w_frame = frame.shape[:2]
            if draw_grid: draw_grid_overlay(frame, GRID_COLS)
            current_tracked_bboxes, active_ids_this_frame = {}, set()
            current_risk_labels = last_risk_labels.copy()

            if (frame_count % pose_skip == 0) and en_pose:
                t_start = time.time()
                results = pose_model.track(frame, tracker="botsort.yaml", persist=True, verbose=False, conf=pose_conf)
                lat_pose_ms = (time.time() - t_start) * 1000
                id_map = {}
                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
                    ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
                    kps = results[0].keypoints.xyn.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy().tolist() # Lấy confidence
                    
                    for c in confs:
                        sum_pose_conf += c
                        cnt_pose += 1

                    detected_ids, existing_ids = set(ids), set(kalman_filters.keys())
                    new_ids, lost_ids = detected_ids - existing_ids, existing_ids - detected_ids
                    for nid in list(new_ids):
                        idx = ids.index(nid)
                        nbox = [int(x) for x in boxes[idx]]
                        best_match, best_iou = None, -1.0
                        for lid in lost_ids:
                            kf = kalman_filters[lid]
                            if kf.time_since_update < MAX_LOST_FRAMES:
                                pbox = [int(x) for x in kf.get_rect()]; iou = calculate_iou(nbox, pbox)
                                if iou > best_iou: best_iou, best_match = iou, lid
                        if best_match is not None and best_iou > IOU_RECOVERY_THRESH:
                            id_map[nid] = best_match; new_ids.remove(nid); lost_ids.remove(best_match); metric_recovered_count += 1
                    
                    new_tracked_kps = {}
                    for box, rid, kp, conf in zip(boxes, ids, kps, confs):
                        fid = id_map.get(rid, rid)
                        active_ids_this_frame.add(fid); total_ids_seen.add(fid); last_pose_confs[fid] = conf
                        if fid not in kalman_filters:
                            kalman_filters[fid] = KalmanFilterBox(); kalman_filters[fid].initiate(box)
                            current_tracked_bboxes[fid] = box
                        else:
                            kalman_filters[fid].update(box); kf_box = kalman_filters[fid].get_current_state()
                            if fid in last_tracked_bboxes:
                                last_box = last_tracked_bboxes[fid]
                                kf_w, kf_h = kf_box[2]-kf_box[0], kf_box[3]-kf_box[1]
                                kf_cx, kf_cy = (kf_box[0]+kf_box[2])/2, (kf_box[1]+kf_box[3])/2
                                last_w, last_h = last_box[2]-last_box[0], last_box[3]-last_box[1]
                                new_w, new_h = last_w*(1-EMA_ALPHA) + kf_w*EMA_ALPHA, last_h*(1-EMA_ALPHA) + kf_h*EMA_ALPHA
                                current_tracked_bboxes[fid] = [kf_cx-new_w/2, kf_cy-new_h/2, kf_cx+new_w/2, kf_cy+new_h/2]
                            else: current_tracked_bboxes[fid] = kf_box
                        new_tracked_kps[fid] = (kp * np.array([w_frame, h_frame])).astype(int).reshape(-1, 2).tolist()
                    tracked_keypoints = new_tracked_kps
                del_ids = []
                for tid, kf in kalman_filters.items():
                    if tid not in active_ids_this_frame:
                        current_tracked_bboxes[tid] = kf.predict()
                        if kf.time_since_update > MAX_LOST_FRAMES: 
                            del_ids.append(tid)
                            if getattr(kf, 'age', 0) < THRESH_ML_FRAMES: metric_ml_count += 1
                for tid in del_ids:
                    del kalman_filters[tid]
                    if tid in movement_history: del movement_history[tid]

            if (frame_count % obj_skip == 0) and en_obj:
                t_start = time.time()
                res = object_model(frame, verbose=False, conf=obj_conf)
                lat_obj_ms = (time.time() - t_start) * 1000
                if res and len(res[0].boxes) > 0:
                    last_object_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4], int(b[5])] for b in res[0].boxes.data.cpu().tolist()]
                    for b in last_object_boxes:
                        sum_obj_conf += b[4] # confidence index 4
                        cnt_obj += 1
                else: last_object_boxes = []

            if (frame_count % risk_skip == 0) and en_risk:
                t_start = time.time()
                res = risk_model(frame, verbose=False, conf=risk_conf)
                lat_risk_ms = (time.time() - t_start) * 1000
                detected_risks = []
                if res and len(res[0].boxes) > 0:
                    confs = res[0].boxes.conf.cpu().numpy().tolist()
                    for box, cls, c_val in zip(res[0].boxes.xyxy, res[0].boxes.cls, confs):
                        sum_risk_conf += c_val; cnt_risk += 1
                        r_box, label = [int(x) for x in box], RISK_LABELS.get(int(cls), "UNK")
                        detected_risks.append((r_box, label, c_val))
                        if en_pose and last_tracked_bboxes:
                            best_overlap, best_id, risk_area = 0.0, None, (r_box[2]-r_box[0])*(r_box[3]-r_box[1])
                            for tid, p_box in last_tracked_bboxes.items():
                                xA, yA, xB, yB = max(r_box[0], p_box[0]), max(r_box[1], p_box[1]), min(r_box[2], p_box[2]), min(r_box[3], p_box[3])
                                inter = max(0, xB-xA)*max(0, yB-yA)
                                overlap = inter/risk_area if risk_area > 0 else 0
                                if overlap > best_overlap: best_overlap, best_id = overlap, tid
                            if best_id is not None and best_overlap > 0.3: current_risk_labels[best_id] = label
                last_raw_risk_boxes = detected_risks if not en_pose else []
            if not en_risk: current_risk_labels, last_raw_risk_boxes = {}, []

            temp_bboxes = current_tracked_bboxes if (frame_count % pose_skip == 0) and en_pose else {tid: kf.predict() for tid, kf in kalman_filters.items()}
            last_tracked_bboxes, last_risk_labels = temp_bboxes, current_risk_labels.copy()
            interactions = classify_interactions(tracked_keypoints, last_tracked_bboxes, last_object_boxes, config_data)
            
            # --- VẼ THÔNG TIN LÊN NGƯỜI ---
            for tid in list(kalman_filters.keys()):
                if tid not in last_tracked_bboxes: continue
                x1, y1, x2, y2 = [int(x) for x in last_tracked_bboxes[tid]]
                kf = kalman_filters[tid]
                if kf.time_since_update > (pose_skip + 2): continue
                kps = tracked_keypoints.get(tid)
                body, head = "STABLE", "STABLE"
                if en_pose and kps:
                    hc, fc = get_hip_centroid(kps), get_face_centroid(kps)
                    if hc and fc:
                        body = get_movement_label(tid, hc, w_frame, h_frame, 'hip_grids', movement_history, config_data)
                        head = get_movement_label(tid, fc, w_frame, h_frame, 'face_grids', movement_history, config_data)
                inter, risk = interactions.get(tid), last_risk_labels.get(tid)
                color = (0, 255, 0)
                if risk == "SUSPICIOUS": color = RISK_SUS_CLR
                elif inter == "HOLDING": color = (128, 0, 128)
                elif inter in ["TAKING", "PUSHING"]: color = (255, 0, 0)
                elif body == "MOVING": color = (0, 165, 255)
                lbls = []
                if en_obj and inter: lbls.append(inter)
                if en_pose:
                    if body == "MOVING": lbls.append("MOVING")
                    if head == "HEAD_MOVING": lbls.append("HEAD_MOVING")
                    elif head == "HEAD_STABLE" and body == "BODY_STABLE": lbls.append("WATCHING")
                if not lbls: lbls.append("IDLE")
                
                # CẬP NHẬT NHÃN: Thêm Pose Confidence vào nhãn
                p_conf = last_pose_confs.get(tid, 0.0)
                final_txt = f"ID:{tid}({p_conf:.2f})|{'+'.join(lbls)}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (txt_w, txt_h), _ = cv2.getTextSize(final_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - txt_h - 10), (x1 + txt_w, y1), (0, 0, 0), -1) 
                cv2.putText(frame, final_txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if draw_kp and en_pose and kps: draw_skeleton(frame, kps, SKELETON_CONNECTIONS, LINE_COLOR, POINT_COLOR)

            if draw_obj and en_obj:
                for b in last_object_boxes:
                    is_cab = (b[5] == config_data['BEHAVIOR']['CABINET_SHELF_ID'])
                    c = (255, 100, 0) if is_cab else (255, 255, 255)
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), c, 2)
                    # CẬP NHẬT NHÃN OBJECT: Thêm Object Confidence
                    obj_lbl = f"{'CABINET' if is_cab else 'ITEM'} {b[4]:.2f}"
                    cv2.putText(frame, obj_lbl, (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

            fps_curr = 1 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
            fps_avg = 0.9 * fps_avg + 0.1 * fps_curr if fps_avg > 0 else fps_curr
            
            # TÍNH TOÁN mAP (AVG CONFIDENCE)
            m_pose = sum_pose_conf/cnt_pose if cnt_pose > 0 else 0
            m_obj = sum_obj_conf/cnt_obj if cnt_obj > 0 else 0
            m_risk = sum_risk_conf/cnt_risk if cnt_risk > 0 else 0

            metrics_data = {
                'fps': fps_avg, 'active': len(active_ids_this_frame), 'risk': list(last_risk_labels.values()).count("SUSPICIOUS"),
                'map_pose': m_pose, 'map_obj': m_obj, 'map_risk': m_risk,
                'lat_pose': lat_pose_ms, 'lat_obj': lat_obj_ms, 'lat_risk': lat_risk_ms
            }
            draw_dashboard_overlay(frame, metrics_data)
            yield frame
            frame_count += 1
        except Exception:
            traceback.print_exc(); break
    cap.release()