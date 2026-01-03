import cv2
from ultralytics import YOLO
import numpy as np
import traceback
import time
try:
    import torch
except Exception:
    torch = None
from typing import Dict, Any, Generator

# Imports
from scripts.tracker import KalmanFilterBox, SKELETON_CONNECTIONS
from scripts.utils import calculate_iou, draw_skeleton
from scripts.behavior import (
    get_hip_centroid, get_face_centroid, get_movement_label, 
    classify_interactions, ActionFilter
)

# Constants
RISK_LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}
MAX_LOST_FRAMES = 30    
IOU_RECOVERY_THRESH = 0.3 
EMA_ALPHA = 0.3 
THRESH_ML_FRAMES = 15   

# [CẬP NHẬT] Ngưỡng Mahalanobis (Chi-squared distribution, df=4, p=0.05 => 9.488)
THRESH_MAHALANOBIS = 9.488

class BehaviorTracker:
    def __init__(self, config_data: Dict[str, Any]):
        self.config = config_data
        self.device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
        self.output_scale = config_data.get('VIDEO_OUTPUT', {}).get('OUTPUT_SCALE', 1.0)
        # Load Models
        try:
            self.pose_model = YOLO(config_data['MODELS']['POSE_MODEL_NAME']).to(self.device)
            self.object_model = YOLO(config_data['MODELS']['OBJECT_MODEL_NAME']).to(self.device)
            self.risk_model = YOLO(config_data['MODELS']['RISK_MODEL_NAME']).to(self.device)
        except Exception as e:
            print(f"Model Load Error: {e}")
            raise e

        # Constants from Config
        self.GRID_COLS = config_data['BEHAVIOR']['GRID_COLS']
        self.LINE_COLOR = tuple(map(int, config_data['DRAWING']['LINE_COLOR']))
        self.POINT_COLOR = tuple(map(int, config_data['DRAWING']['POINT_COLOR']))
        self.RISK_SUS_CLR = tuple(map(int, config_data['DRAWING']['RISK_SUSPICIOUS_COLOR']))

        # Action Filter (Patience)
        patience_val = config_data['BEHAVIOR'].get('ACTION_PATIENCE', 3)
        self.action_filter = ActionFilter(patience=patience_val)

        # State Variables
        self.kalman_filters = {} 
        self.movement_history = {}
        self.tracked_keypoints = {}
        self.last_object_boxes = []
        self.last_tracked_bboxes = {}
        self.last_risk_labels = {}
        self.last_raw_risk_boxes = []
        
        # Metrics & mAP Accumulators
        self.total_ids_seen = set()
        self.metric_recovered = 0
        self.metric_ml = 0
        self.sum_pose_conf, self.cnt_pose = 0.0, 0
        self.sum_obj_conf, self.cnt_obj = 0.0, 0
        self.sum_risk_conf, self.cnt_risk = 0.0, 0
        self.last_pose_confs = {}
        
        self.frame_count = 0
        self.fps_avg = 0

    def _draw_dashboard(self, frame, metrics):
        h, w = frame.shape[:2]
        panel_w, panel_h = 280, 280 
        x_start, y_start = w - panel_w - 20, 20
        
        cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (100, 100, 100), 1)
        
        cv2.putText(frame, "SYSTEM HEALTH", (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.line(frame, (x_start + 10, y_start + 35), (x_start + panel_w - 10, y_start + 35), (100, 100, 100), 1)

        y_row = y_start + 55
        gap = 22
        
        cv2.putText(frame, f"FPS: {metrics.get('fps', 0):.1f}", (x_start + 10, y_row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, f"Active: {metrics.get('active', 0)}", (x_start + 10, y_row + gap), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Risk: {metrics.get('risk', 0)}", (x_start + 10, y_row + gap*2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        y_map_sep = y_row + gap*3 
        cv2.line(frame, (x_start + 10, y_map_sep), (x_start + panel_w - 10, y_map_sep), (100, 100, 100), 1)
        cv2.putText(frame, "AVG mAP (CONF)", (x_start + 10, y_map_sep + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, f"Pose: {metrics.get('map_pose', 0):.2f}", (x_start + 10, y_map_sep + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Obj:  {metrics.get('map_obj', 0):.2f}", (x_start + 10, y_map_sep + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Risk: {metrics.get('map_risk', 0):.2f}", (x_start + 10, y_map_sep + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        y_lat_sep = y_map_sep + 85
        cv2.line(frame, (x_start + 10, y_lat_sep), (x_start + panel_w - 10, y_lat_sep), (100, 100, 100), 1)
        cv2.putText(frame, "LATENCY (ms)", (x_start + 10, y_lat_sep + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        l_pose, l_obj, l_risk = metrics.get('lat_pose', 0), metrics.get('lat_obj', 0), metrics.get('lat_risk', 0)
        cv2.putText(frame, f"P:{l_pose:.1f} O:{l_obj:.1f} R:{l_risk:.1f}", (x_start + 10, y_lat_sep + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_grid(self, frame):
        h, w = frame.shape[:2]
        step_x = w / self.GRID_COLS
        for i in range(1, self.GRID_COLS):
            x = int(i * step_x)
            cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
        grid_rows = int((h / w) * self.GRID_COLS)
        if grid_rows < 1: grid_rows = 1
        step_y = h / grid_rows
        for i in range(1, grid_rows):
            y = int(i * step_y)
            cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)

    def process_stream(self, cap, settings):
        pose_conf = settings.get('pose_conf_thresh', 0.5)
        obj_conf = settings.get('object_conf_thresh', 0.5)
        risk_conf = settings.get('risk_conf_thresh', 0.5)
        pose_skip = settings.get('pose_skip_frames', 2)
        obj_skip = settings.get('object_skip_frames', 10)
        risk_skip = settings.get('risk_skip_frames', 10)
        
        en_pose, en_obj, en_risk = settings.get('model_pose_enabled'), settings.get('model_object_enabled'), settings.get('model_risk_enabled')
        draw_kp, draw_obj = settings.get('keypoint_draw'), settings.get('objects_draw')
        draw_grid_flag = settings.get('draw_grid', False)

        lat_pose, lat_obj, lat_risk = 0.0, 0.0, 0.0

        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            try:
                h_frame, w_frame = frame.shape[:2]
                if draw_grid_flag: self._draw_grid(frame)
                
                current_tracked_bboxes, active_ids_this_frame = {}, set()
                current_risk_labels = self.last_risk_labels.copy()

                # --- 1. POSE TRACKING ---
                if (self.frame_count % pose_skip == 0) and en_pose:
                    t0 = time.time()
                    results = self.pose_model.track(frame, tracker="botsort.yaml", persist=True, verbose=False, conf=pose_conf)
                    lat_pose = (time.time() - t0) * 1000

                    id_map = {}
                    if results and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
                        ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
                        kps = results[0].keypoints.xyn.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy().tolist()

                        for c in confs: self.sum_pose_conf += c; self.cnt_pose += 1

                        # --- LOGIC KHÔI PHỤC ID (RECOVERY) CẬP NHẬT ---
                        # Sử dụng Khoảng cách Mahalanobis thay vì Euclidean
                        det_ids, ext_ids = set(ids), set(self.kalman_filters.keys())
                        new_ids, lost_ids = det_ids - ext_ids, ext_ids - det_ids
                        
                        for nid in list(new_ids):
                            idx = ids.index(nid)
                            nbox = [int(x) for x in boxes[idx]]
                            
                            best_match, min_dist = None, float('inf')
                            
                            for lid in lost_ids:
                                kf = self.kalman_filters[lid]
                                if kf.time_since_update < MAX_LOST_FRAMES:
                                    # [MỚI] Sử dụng Mahalanobis Distance
                                    dist = kf.get_mahalanobis_distance(nbox)
                                    
                                    if dist < min_dist:
                                        min_dist, best_match = dist, lid
                                        
                            # Kiểm tra ngưỡng Chi-squared
                            if best_match is not None and min_dist < THRESH_MAHALANOBIS:
                                id_map[nid] = best_match
                                new_ids.remove(nid)
                                lost_ids.remove(best_match)
                                self.metric_recovered += 1

                        # Update Kalman & State
                        new_tracked_kps = {}
                        for box, rid, kp, conf in zip(boxes, ids, kps, confs):
                            fid = id_map.get(rid, rid)
                            active_ids_this_frame.add(fid); self.total_ids_seen.add(fid)
                            self.last_pose_confs[fid] = conf

                            if fid not in self.kalman_filters:
                                self.kalman_filters[fid] = KalmanFilterBox()
                                self.kalman_filters[fid].initiate(box)
                                current_tracked_bboxes[fid] = box
                            else:
                                self.kalman_filters[fid].update(box)
                                kf_box = self.kalman_filters[fid].get_current_state()
                                if fid in self.last_tracked_bboxes:
                                    lb = self.last_tracked_bboxes[fid]
                                    kw, kh = kf_box[2]-kf_box[0], kf_box[3]-kf_box[1]
                                    kcx, kcy = (kf_box[0]+kf_box[2])/2, (kf_box[1]+kf_box[3])/2
                                    lw, lh = lb[2]-lb[0], lb[3]-lb[1]
                                    nw, nh = lw*(1-EMA_ALPHA) + kw*EMA_ALPHA, lh*(1-EMA_ALPHA) + kh*EMA_ALPHA
                                    current_tracked_bboxes[fid] = [kcx-nw/2, kcy-nh/2, kcx+nw/2, kcy+nh/2]
                                else: current_tracked_bboxes[fid] = kf_box
                            
                            new_tracked_kps[fid] = (kp * np.array([w_frame, h_frame])).astype(int).reshape(-1, 2).tolist()
                        self.tracked_keypoints = new_tracked_kps

                    # Clean Lost IDs
                    del_ids = []
                    for tid, kf in self.kalman_filters.items():
                        if tid not in active_ids_this_frame:
                            current_tracked_bboxes[tid] = kf.predict()
                            if kf.time_since_update > MAX_LOST_FRAMES:
                                del_ids.append(tid)
                                if getattr(kf, 'age', 0) < THRESH_ML_FRAMES: self.metric_ml += 1
                    for tid in del_ids:
                        del self.kalman_filters[tid]
                        if tid in self.movement_history: del self.movement_history[tid]

                # --- 2. OBJECT DETECTION ---
                if (self.frame_count % obj_skip == 0) and en_obj:
                    t0 = time.time()
                    res = self.object_model(frame, verbose=False, conf=obj_conf)
                    lat_obj = (time.time() - t0) * 1000
                    if res and len(res[0].boxes) > 0:
                        self.last_object_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4], int(b[5])] for b in res[0].boxes.data.cpu().tolist()]
                        for b in self.last_object_boxes: self.sum_obj_conf += b[4]; self.cnt_obj += 1
                    else: self.last_object_boxes = []

                # --- 3. RISK DETECTION ---
                if (self.frame_count % risk_skip == 0) and en_risk:
                    t0 = time.time()
                    res = self.risk_model(frame, verbose=False, conf=risk_conf)
                    lat_risk = (time.time() - t0) * 1000
                    detected_risks = []
                    if res and len(res[0].boxes) > 0:
                        confs = res[0].boxes.conf.cpu().numpy().tolist()
                        for box, cls, c_val in zip(res[0].boxes.xyxy, res[0].boxes.cls, confs):
                            self.sum_risk_conf += c_val; self.cnt_risk += 1
                            r_box, label = [int(x) for x in box], RISK_LABELS.get(int(cls), "UNK")
                            detected_risks.append((r_box, label, c_val))
                            if en_pose and self.last_tracked_bboxes:
                                best_ov, best_id, r_area = 0.0, None, (r_box[2]-r_box[0])*(r_box[3]-r_box[1])
                                for tid, p_box in self.last_tracked_bboxes.items():
                                    xA, yA, xB, yB = max(r_box[0], p_box[0]), max(r_box[1], p_box[1]), min(r_box[2], p_box[2]), min(r_box[3], p_box[3])
                                    inter = max(0, xB-xA)*max(0, yB-yA)
                                    if r_area > 0 and (inter/r_area) > best_ov: best_ov, best_id = inter/r_area, tid
                                if best_id is not None and best_ov > 0.3: current_risk_labels[best_id] = label
                    self.last_raw_risk_boxes = detected_risks if not en_pose else []
                if not en_risk: current_risk_labels, self.last_raw_risk_boxes = {}, []

                # --- 4. BEHAVIOR ANALYSIS ---
                temp_bboxes = current_tracked_bboxes if (self.frame_count % pose_skip == 0) and en_pose else {tid: kf.predict() for tid, kf in self.kalman_filters.items()}
                self.last_tracked_bboxes, self.last_risk_labels = temp_bboxes, current_risk_labels.copy()
                
                raw_interactions = classify_interactions(self.tracked_keypoints, self.last_tracked_bboxes, self.last_object_boxes, self.config)

                # --- 5. VISUALIZATION ---
                for tid in list(self.kalman_filters.keys()):
                    if tid not in self.last_tracked_bboxes: continue
                    x1, y1, x2, y2 = [int(x) for x in self.last_tracked_bboxes[tid]]
                    if self.kalman_filters[tid].time_since_update > (pose_skip + 2): continue
                    
                    kps = self.tracked_keypoints.get(tid)
                    body, head = "STABLE", "STABLE"
                    if en_pose and kps:
                        hc, fc = get_hip_centroid(kps), get_face_centroid(kps)
                        if hc and fc:
                            body = get_movement_label(tid, hc, w_frame, h_frame, 'hip_grids', self.movement_history, self.config)
                            head = get_movement_label(tid, fc, w_frame, h_frame, 'face_grids', self.movement_history, self.config)
                    
                    raw_inter = raw_interactions.get(tid)
                    inter = self.action_filter.update(tid, raw_inter)
                    risk = self.last_risk_labels.get(tid)
                    
                    color = (0, 255, 0)
                    if risk == "SUSPICIOUS": color = self.RISK_SUS_CLR
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
                    
                    p_conf = self.last_pose_confs.get(tid, 0.0)
                    final_txt = f"ID:{tid}({p_conf:.2f})|{'+'.join(lbls)}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (txt_w, txt_h), _ = cv2.getTextSize(final_txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - txt_h - 10), (x1 + txt_w, y1), (0, 0, 0), -1) 
                    cv2.putText(frame, final_txt, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    if draw_kp and en_pose and kps: draw_skeleton(frame, kps, SKELETON_CONNECTIONS, self.LINE_COLOR, self.POINT_COLOR)

                if draw_obj and en_obj:
                    for b in self.last_object_boxes:
                        is_cab = (b[5] == self.config['BEHAVIOR']['CABINET_SHELF_ID'])
                        c = (255, 100, 0) if is_cab else (255, 255, 255)
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), c, 2)
                        cv2.putText(frame, f"{'CABINET' if is_cab else 'ITEM'} {b[4]:.2f}", (b[0], b[1]-5), cv2.FONT_HERSHEY_DUPLEX, 0.8, c, 1)

                fps_curr = 1 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
                self.fps_avg = 0.9 * self.fps_avg + 0.1 * fps_curr if self.fps_avg > 0 else fps_curr
                
                m_pose = self.sum_pose_conf/self.cnt_pose if self.cnt_pose > 0 else 0
                m_obj = self.sum_obj_conf/self.cnt_obj if self.cnt_obj > 0 else 0
                m_risk = self.sum_risk_conf/self.cnt_risk if self.cnt_risk > 0 else 0

                metrics_data = {
                    'fps': self.fps_avg, 'active': len(active_ids_this_frame), 
                    'risk': list(self.last_risk_labels.values()).count("SUSPICIOUS"),
                    'map_pose': m_pose, 'map_obj': m_obj, 'map_risk': m_risk,
                    'lat_pose': lat_pose, 'lat_obj': lat_obj, 'lat_risk': lat_risk
                }
                self._draw_dashboard(frame, metrics_data)
                
                
                if self.output_scale != 1.0:    
                    frame = cv2.resize(frame, (int(w_frame * self.output_scale), int(h_frame * self.output_scale))) 
                    
                yield frame
                self.frame_count += 1
                
            except Exception:
                traceback.print_exc(); break