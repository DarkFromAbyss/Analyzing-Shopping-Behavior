# scripts/behavior.py
from . import utils 
from typing import Dict, List, Tuple, Any

# --- CONSTANTS ---
LEFT_WRIST_INDEX = 9   
RIGHT_WRIST_INDEX = 10  
LEFT_HIP_INDEX = 11     
RIGHT_HIP_INDEX = 12    
NOSE_INDEX = 0          
RIGHT_EYE_INDEX = 2     
LEFT_EYE_INDEX = 1      

# --- CLASS: Action Filter (NEW) ---
class ActionFilter:
    """Bộ lọc giúp loại bỏ hiện tượng nhấp nháy trạng thái (flickering)."""
    def __init__(self, patience=5):
        self.patience = patience
        # history: {track_id: {'count': int, 'current_raw': str|None, 'confirmed': str|None}}
        self.history = {}

    def update(self, track_id, raw_action):
        if track_id not in self.history:
            self.history[track_id] = {'count': 0, 'current_raw': None, 'confirmed': None}
        
        state = self.history[track_id]
        
        # Nếu raw action giống frame trước
        if raw_action == state['current_raw']:
            state['count'] += 1
        else:
            state['current_raw'] = raw_action
            state['count'] = 1  # Reset đếm lại
            
        # Nếu đủ độ bền (patience), cập nhật hành động chính thức
        if state['count'] >= self.patience:
            state['confirmed'] = raw_action
            
        return state['confirmed']

# --- HELPER FUNCTIONS ---
def get_hip_centroid(keypoints: List[List[int]]) -> Tuple[int, int] | None:
    if len(keypoints) < RIGHT_HIP_INDEX + 1: return None
    left_hip = keypoints[LEFT_HIP_INDEX]
    right_hip = keypoints[RIGHT_HIP_INDEX]
    return (left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2

def get_face_centroid(keypoints: List[List[int]]) -> Tuple[int, int] | None:
    if len(keypoints) < RIGHT_EYE_INDEX + 1: return None
    nose = keypoints[NOSE_INDEX]
    left_eye = keypoints[LEFT_EYE_INDEX]
    right_eye = keypoints[RIGHT_EYE_INDEX]
    return (nose[0] + left_eye[0] + right_eye[0]) // 3, (nose[1] + left_eye[1] + right_eye[1]) // 3

def get_movement_label(track_id, current_centroid, frame_w, frame_h, history_key, movement_history, config_data):
    cfg = config_data['BEHAVIOR']
    GRID_COLS = cfg['GRID_COLS']
    MOVEMENT_HISTORY_LENGTH = cfg['MOVEMENT_HISTORY_LENGTH']
    MOVEMENT_THRESHOLD_CELLS = cfg['MOVEMENT_THRESHOLD_CELLS']
    
    cx, cy = current_centroid
    col_idx = max(0, min(int(cx // (frame_w / GRID_COLS)), GRID_COLS - 1))
    row_idx = max(0, min(int(cy // (frame_h / GRID_COLS)), GRID_COLS - 1))
    current_grid = (col_idx, row_idx)
    
    if track_id not in movement_history:
        movement_history[track_id] = {'hip_grids': [], 'face_grids': []}
    
    history = movement_history[track_id]
    history[history_key].append(current_grid)
    if len(history[history_key]) > MOVEMENT_HISTORY_LENGTH:
        history[history_key] = history[history_key][-MOVEMENT_HISTORY_LENGTH:]

    is_body = (history_key == 'hip_grids')
    default_stable = "BODY_STABLE" if is_body else "HEAD_STABLE"
    
    if len(history[history_key]) >= MOVEMENT_HISTORY_LENGTH:
        start, end = history[history_key][0], history[history_key][-1]
        dist = abs(end[0] - start[0]) + abs(end[1] - start[1])
        return ("MOVING" if is_body else "HEAD_MOVING") if dist >= MOVEMENT_THRESHOLD_CELLS else default_stable
            
    return default_stable

def boxes_intersect(boxA, boxB):
    return not (boxA[2] < boxB[0] or boxA[0] > boxB[2] or boxA[3] < boxB[1] or boxA[1] > boxB[3])

def classify_interactions(tracked_keypoints, tracked_bboxes, object_results, config_data):
    cfg_thresh = config_data['THRESHOLDS']
    cfg_behavior = config_data['BEHAVIOR']
    IOU_HOLDING_THRESHOLD = cfg_thresh['IOU_HOLDING_THRESHOLD']
    
    ID_TAKING = cfg_behavior['CABINET_SHELF_ID']
    ID_PUSHING = cfg_behavior['TROLLEY_ID']
    ID_HOLDING = cfg_behavior['HANDBAG_SATCHEL_ID']
    
    interaction_labels = {}
    targets = {"TAKING": [], "PUSHING": [], "HOLDING": []}
    
    for obj in object_results:
        if len(obj) < 5: continue
        cid, bbox = obj[4], [int(x) for x in obj[:4]]
        if cid == ID_TAKING: targets["TAKING"].append(bbox)
        elif cid == ID_PUSHING: targets["PUSHING"].append(bbox)
        elif cid == ID_HOLDING: targets["HOLDING"].append(bbox)

    for tid, kps in tracked_keypoints.items():
        person_box = tracked_bboxes.get(tid)
        if person_box is None or len(kps) < 11: 
            interaction_labels[tid] = None
            continue
            
        p_box = [int(x) for x in person_box]
        action = None
        lw, rw = kps[LEFT_WRIST_INDEX], kps[RIGHT_WRIST_INDEX]

        if not action:
            for box in targets["TAKING"]:
                if boxes_intersect(p_box, box): 
                    if utils.is_point_inside_bbox(lw, box) or utils.is_point_inside_bbox(rw, box):
                        action = "TAKING"; break
        if not action:
            for box in targets["PUSHING"]:
                if boxes_intersect(p_box, box): 
                    if utils.is_point_inside_bbox(lw, box) or utils.is_point_inside_bbox(rw, box):
                        action = "PUSHING"; break
        if not action:
            for box in targets["HOLDING"]:
                if boxes_intersect(p_box, box): 
                    if utils.calculate_iou(p_box, box) >= IOU_HOLDING_THRESHOLD:
                        action = "HOLDING"; break

        interaction_labels[tid] = action

    return interaction_labels