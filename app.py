from flask import Flask, render_template, request, Response, jsonify
import cv2
import threading
import time
import os
import yaml
from pose_tracker import BehaviorTracker

try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("ERROR: config.yaml not found!")
    exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- SYSTEM MANAGER (SINGLETON) ---
class SystemManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemManager, cls).__new__(cls)
            cls._instance.tracker = None
            cls._instance.thread = None
            cls._instance.stop_event = threading.Event()
            cls._instance.pause_event = threading.Event()
            cls._instance.pause_event.set() 
            cls._instance.frame_data = None
            cls._instance.lock = threading.Lock()
            cls._instance.is_active = False
            # [FIX] Biến lưu video path chính xác
            cls._instance.current_video_path = None
        return cls._instance

    def start_processing(self, video_path, settings):
        with self.lock:
            self.stop_processing() 
            
            self.stop_event.clear()
            self.pause_event.set()
            self.is_active = True
            
            self.thread = threading.Thread(target=self._run_loop, args=(video_path, settings))
            self.thread.daemon = True
            self.thread.start()

    def _run_loop(self, video_path, settings):
        print(f"Starting tracking on: {video_path}")
        cap = cv2.VideoCapture(video_path)
        tracker = BehaviorTracker(config) 
        
        try:
            for processed_frame in tracker.process_stream(cap, settings):
                if self.stop_event.is_set(): break
                self.pause_event.wait() 
                
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    self.frame_data = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                
                time.sleep(0.001) 
        except Exception as e:
            print(f"Thread Error: {e}")
        finally:
            cap.release()
            self.is_active = False
            print("Tracking Finished.")

    def stop_processing(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.is_active = False
        self.frame_data = None

    def pause(self): self.pause_event.clear()
    def resume(self): self.pause_event.set()

    def get_frame(self):
        while self.is_active:
            if self.frame_data: yield self.frame_data
            time.sleep(0.03)

system = SystemManager()

@app.route('/')
def index():
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
    if 'video' not in request.files: return jsonify({'success': False}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({'success': False}), 400
    
    # Reset system cũ
    system.stop_processing()
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    
    # [FIX] Cập nhật đường dẫn file mới vào biến quản lý
    system.current_video_path = path
    
    return jsonify({'success': True, 'filename': file.filename, 'path': path})

@app.route('/control', methods=['POST'])
def control():
    data = request.json
    action = data.get('action')
    settings = data.get('settings', {})
    
    full_settings = {
        'pose_conf_thresh': float(settings.get('pose_conf_thresh', 0.5)),
        'object_conf_thresh': float(settings.get('object_conf_thresh', 0.5)),
        'risk_conf_thresh': float(settings.get('risk_conf_thresh', 0.5)),
        'pose_skip_frames': int(settings.get('pose_skip_frames', 2)),
        'object_skip_frames': int(settings.get('object_skip_frames', 10)),
        'risk_skip_frames': int(settings.get('risk_skip_frames', 10)),
        'keypoint_draw': settings.get('keypoint_draw', True),
        'objects_draw': settings.get('objects_draw', True),
        'model_pose_enabled': settings.get('model_pose_enabled', True),
        'model_object_enabled': settings.get('model_object_enabled', True),
        'model_risk_enabled': settings.get('model_risk_enabled', True),
        'draw_grid': settings.get('draw_grid', False)
    }

    # [FIX] Lấy đường dẫn từ SystemManager thay vì listdir
    video_path = system.current_video_path

    if action == 'start':
        if not video_path or not os.path.exists(video_path):
             return jsonify({'success': False, 'message': 'Vui lòng upload video trước!'}), 400
             
        if not system.is_active:
            system.start_processing(video_path, full_settings)
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Already running'})
    
    elif action == 'stop': 
        system.pause()
        return jsonify({'success': True})
    
    elif action == 'continue':
        system.resume()
        return jsonify({'success': True})
    
    elif action == 'reset':
        system.stop_processing()
        return jsonify({'success': True})
        
    return jsonify({'success': False}), 400

@app.route('/video_feed')
def video_feed():
    return Response(system.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)