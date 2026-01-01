# app.py

from flask import Flask, render_template, request, Response, jsonify
import cv2
import threading
import time
import os
import yaml

# Import các logic cần thiết
from pose_tracker import process_video_stream 

# --- LOAD CẤU HÌNH TỪ YAML ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("LỖI: Không tìm thấy file config.yaml. Đảm bảo file tồn tại!")
    exit(1)
# ------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Đảm bảo thư mục upload tồn tại
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- CÁC BIẾN QUẢN LÝ TRẠNG THÁI TOÀN CỤC ---
video_processor = None
processing_thread = None
is_running = False
current_frame = None
video_path = None
# ----------------------------------------------


class VideoProcessor:
    def __init__(self, video_file_path):
        self.video_file_path = video_file_path
        self.cap = None 
        self.running = threading.Event() 
        self.running.clear()
        self.stop_requested = False
        self.frame_data = None 

    def run(self, settings):
        global current_frame, is_running
        
        self.cap = cv2.VideoCapture(self.video_file_path)
        
        try:
            for processed_frame in process_video_stream(self.video_file_path, settings, self.cap, config):
                # 1. Cập nhật frame hiện tại để streaming
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    self.frame_data = b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                    current_frame = self.frame_data
                
                # 2. Kiểm tra trạng thái dừng
                if self.stop_requested:
                    self.stop_requested = False 
                    self.running.wait() 
                
                # 3. Kiểm tra trạng thái kết thúc hoàn toàn
                if not self.running.is_set():
                    break 
        except Exception as e:
            print(f"LỖI XỬ LÝ VIDEO: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.running.clear()
            is_running = False
            print("Video Processor finished/released.")

    def start(self):
        self.running.set()
        self.stop_requested = False
    
    def pause(self):
        self.stop_requested = True
    
    def stop_and_reset(self):
        self.running.clear() 

    def generate_frames(self):
        """Streaming frame đã xử lý ra trình duyệt"""
        while self.running.is_set() or self.stop_requested: 
            if self.frame_data is not None:
                yield self.frame_data
            time.sleep(1/30) 

@app.route('/')
def index():
    # [CẬP NHẬT] Lấy giá trị mặc định từ config.yaml để gửi sang HTML
    defaults = {
        'pose_conf': config['THRESHOLDS'].get('DEFAULT_POSE_CONF', 0.5),
        'object_conf': config['THRESHOLDS'].get('DEFAULT_OBJECT_CONF', 0.5),
        'risk_conf': config['THRESHOLDS'].get('DEFAULT_RISK_CONF', 0.5),
        'pose_skip': config['SKIP_FRAMES'].get('DEFAULT_POSE_SKIP', 2),
        'object_skip': config['SKIP_FRAMES'].get('DEFAULT_OBJECT_SKIP', 10),
        'risk_skip': config['SKIP_FRAMES'].get('DEFAULT_RISK_SKIP', 10)
    }
    return render_template('index.html', defaults=defaults)

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_processor, processing_thread, video_path
    
    if video_processor:
        video_processor.stop_and_reset()
        if processing_thread and processing_thread.is_alive():
            processing_thread.join()
        
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'Không tìm thấy file video'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Không có video được chọn'}), 400
    
    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        
        video_processor = VideoProcessor(video_path)
        
        return jsonify({'success': True, 'filename': file.filename, 'message': 'Upload thành công. Sẵn sàng Start.'})

@app.route('/control', methods=['POST'])
def control():
    global video_processor, processing_thread, is_running
    
    if not video_processor:
        return jsonify({'success': False, 'message': 'Chưa upload video.'}), 400

    action = request.json.get('action')
    settings = request.json.get('settings', {})
    
    # Lấy các giá trị mặc định từ YAML (Backup nếu frontend gửi thiếu)
    yaml_conf = config['THRESHOLDS']
    yaml_skip = config['SKIP_FRAMES']

    if action == 'start':
        if not is_running:
            current_settings = {
                'pose_conf_thresh': float(settings.get('pose_conf_thresh', yaml_conf['DEFAULT_POSE_CONF'])),
                'object_conf_thresh': float(settings.get('object_conf_thresh', yaml_conf['DEFAULT_OBJECT_CONF'])),
                'risk_conf_thresh': float(settings.get('risk_conf_thresh', yaml_conf['DEFAULT_RISK_CONF'])),
                'pose_skip_frames': int(settings.get('pose_skip_frames', yaml_skip['DEFAULT_POSE_SKIP'])),
                'object_skip_frames': int(settings.get('object_skip_frames', yaml_skip['DEFAULT_OBJECT_SKIP'])),
                'risk_skip_frames': int(settings.get('risk_skip_frames', yaml_skip['DEFAULT_RISK_SKIP'])),
                
                'keypoint_draw': settings.get('keypoint_draw', config['DRAWING']['DEFAULT_KEYPOINT_DRAW']),
                'risk_draw_bbox': settings.get('risk_draw_bbox', config['DRAWING']['DEFAULT_RISK_DRAW_BBOX']),
                'risk_draw_keypoint': settings.get('risk_draw_keypoint', config['DRAWING']['DEFAULT_RISK_DRAW_KEYPOINT']),
                'objects_draw': settings.get('objects_draw', config['DRAWING']['DEFAULT_OBJECTS_DRAW']),
                'model_pose_enabled': settings.get('model_pose_enabled', True),
                'model_object_enabled': settings.get('model_object_enabled', True),
                'model_risk_enabled': settings.get('model_risk_enabled', True),
                'draw_grid': settings.get('draw_grid', False)
            }

            video_processor.start()
            processing_thread = threading.Thread(target=video_processor.run, args=(current_settings,))
            processing_thread.start()
            is_running = True
            return jsonify({'success': True, 'message': 'Đang xử lý video...'}) 
        
        return jsonify({'success': False, 'message': 'Đã chạy rồi.'})

    elif action == 'stop':
        video_processor.pause()
        return jsonify({'success': True, 'message': 'Đã dừng tại frame hiện tại.'})
    
    elif action == 'continue':
        video_processor.start()
        return jsonify({'success': True, 'message': 'Tiếp tục xử lý.'})
    
    elif action == 'reset':
        video_processor.stop_and_reset()
        if processing_thread and processing_thread.is_alive():
            processing_thread.join()
        is_running = False
        return jsonify({'success': True, 'message': 'Đã reset. Sẵn sàng Start lại.'})

    return jsonify({'success': False, 'message': 'Hành động không hợp lệ.'}), 400

@app.route('/video_feed')
def video_feed():
    global video_processor
    if not video_processor:
        return Response('', mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(video_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)