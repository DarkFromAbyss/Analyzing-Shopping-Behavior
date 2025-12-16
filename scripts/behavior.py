# scripts/behavior.py

from . import utils # Import utils từ package scripts
from typing import Dict, List, Tuple, Any

# --- CÁC HẰNG SỐ CHỈ SỐ KEYPOINT (Cần giữ lại trong module này) ---
LEFT_WRIST_INDEX = 9   
RIGHT_WRIST_INDEX = 10  
LEFT_HIP_INDEX = 11     
RIGHT_HIP_INDEX = 12    
NOSE_INDEX = 0          
RIGHT_EYE_INDEX = 2     
LEFT_EYE_INDEX = 1      


# --- 1. Tính toán Centroid ---
def get_hip_centroid(keypoints: List[List[int]]) -> Tuple[int, int] | None:
    """Tính toán tâm (centroid) giữa hai hông."""
    if len(keypoints) < RIGHT_HIP_INDEX + 1: return None
    left_hip = keypoints[LEFT_HIP_INDEX]
    right_hip = keypoints[RIGHT_HIP_INDEX]
    cx = (left_hip[0] + right_hip[0]) // 2
    cy = (left_hip[1] + right_hip[1]) // 2
    return cx, cy

def get_face_centroid(keypoints: List[List[int]]) -> Tuple[int, int] | None:
    """Tính toán tâm (centroid) của khuôn mặt (mũi và mắt)."""
    if len(keypoints) < RIGHT_EYE_INDEX + 1: return None
    nose = keypoints[NOSE_INDEX]
    left_eye = keypoints[LEFT_EYE_INDEX]
    right_eye = keypoints[RIGHT_EYE_INDEX]
    avg_x = (nose[0] + left_eye[0] + right_eye[0]) // 3
    avg_y = (nose[1] + left_eye[1] + right_eye[1]) // 3
    return avg_x, avg_y

# --- 2. Phân loại Di chuyển (Movement) ---
def get_grid_position(cx: int, cy: int, frame_w: int, frame_h: int, cols: int) -> Tuple[int, int]:
    """Chuyển đổi tọa độ pixel thành tọa độ lưới (grid)."""
    rows = cols
    cell_w = frame_w / cols
    cell_h = frame_h / rows
    col_idx = max(0, min(int(cx // cell_w), cols - 1))
    row_idx = max(0, min(int(cy // cell_h), rows - 1))
    return col_idx, row_idx

def get_movement_label(
    track_id: int, 
    current_centroid: Tuple[int, int], 
    frame_w: int, 
    frame_h: int, 
    history_key: str, 
    movement_history: Dict[int, Dict[str, List[Tuple[int, int]]]], 
    config_data: Dict[str, Any]
) -> str:
    """Xác định trạng thái di chuyển (MOVING/STABLE) dựa trên lịch sử lưới."""
    
    cfg = config_data['BEHAVIOR']
    GRID_COLS = cfg['GRID_COLS']
    MOVEMENT_HISTORY_LENGTH = cfg['MOVEMENT_HISTORY_LENGTH']
    MOVEMENT_THRESHOLD_CELLS = cfg['MOVEMENT_THRESHOLD_CELLS']
    
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
        start_grid = history[history_key][0]
        end_grid = history[history_key][-1]
        total_moved_cells = abs(end_grid[0] - start_grid[0]) + abs(end_grid[1] - start_grid[1])
        
        if total_moved_cells >= MOVEMENT_THRESHOLD_CELLS:
            return "MOVING" if is_body else "HEAD_MOVING"
        else:
            return default_stable_label
            
    return default_stable_label

# --- 3. Phân loại Tương tác (Interaction) ---
def classify_interactions(
    tracked_keypoints: Dict[int, List[List[int]]], 
    tracked_bboxes: Dict[int, List[int]], 
    object_results: List[List[Any]], 
    config_data: Dict[str, Any]
) -> Dict[int, str | None]:
    """Phân loại tương tác dựa trên vị trí cổ tay (Taking/Pushing) và IoU (Holding)."""
    
    cfg_thresh = config_data['THRESHOLDS']
    cfg_behavior = config_data['BEHAVIOR']
    
    IOU_HOLDING_THRESHOLD = cfg_thresh['IOU_HOLDING_THRESHOLD']
    CABINET_SHELF_ID = cfg_behavior['CABINET_SHELF_ID']
    TROLLEY_ID = cfg_behavior['TROLLEY_ID']
    HANDBAG_SATCHEL_ID = cfg_behavior['HANDBAG_SATCHEL_ID']
    
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
        
        # Kiểm tra TAKING, PUSHING (Keypoint) và HOLDING (IoU)
        for cabinet_bbox in target_bboxes["TAKING"]:
            if utils.is_point_inside_bbox(left_wrist, cabinet_bbox) or utils.is_point_inside_bbox(right_wrist, cabinet_bbox):
                current_interaction = "TAKING"; break
        
        if current_interaction is None: 
            for trolley_bbox in target_bboxes["PUSHING"]:
                if utils.is_point_inside_bbox(left_wrist, trolley_bbox) or utils.is_point_inside_bbox(right_wrist, trolley_bbox):
                    current_interaction = "PUSHING"; break

        if current_interaction is None:
            for handbag_bbox in target_bboxes["HOLDING"]:
                iou = utils.calculate_iou(person_bbox, handbag_bbox)
                if iou >= IOU_HOLDING_THRESHOLD:
                    current_interaction = "HOLDING"; break

        interaction_labels[track_id] = current_interaction

    return interaction_labels