from flask import Flask, render_template, request, Response, jsonify
import cv2
import threading
import time
import os
import yaml
import redis
from pose_tracker import BehaviorTracker

# --- LOAD CONFIG ---
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

# --- REDIS CONNECTION ---
# Lưu ý: Đảm bảo Redis Server đang chạy (redis-cli ping -> PONG)
try:
    redis_client = redis.Redis(
        host=config['REDIS']['HOST'],
        port=config['REDIS']['PORT'],
        db=config['REDIS']['DB'],
        decode_responses=False # False để lưu bytes (ảnh)
    )
    redis_client.ping()
    print(f"--- KẾT NỐI REDIS THÀNH CÔNG TẠI {config['REDIS']['HOST']}:{config['REDIS']['PORT']} ---")
except redis.ConnectionError:
    print("!!! KHÔNG THỂ KẾT NỐI REDIS. VUI LÒNG BẬT REDIS SERVER !!!")
    # Không exit ở đây để app vẫn chạy được giao diện, dù chức năng sẽ lỗi
    
# --- WORKER CLASS (Đã tối ưu hóa ảnh) ---
class RedisWorker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisWorker, cls).__new__(cls)
            cls._instance.thread = None
            cls._instance.stop_event = threading.Event()
            cls._instance.pause_event = threading.Event()
            cls._instance.pause_event.set()
            cls._instance.lock = threading.Lock()
            cls._instance.is_active = False
        return cls._instance

    def start_processing(self, video_path, settings):
        with self.lock:
            self.stop_processing() 
            self.stop_event.clear()
            self.pause_event.set()
            self.is_active = True
            
            # Lưu path vào Redis để persistence
            redis_client.set(config['REDIS']['VIDEO_PATH_KEY'], video_path)
            
            self.thread = threading.Thread(target=self._run_loop, args=(video_path, settings))
            self.thread.daemon = True
            self.thread.start()

    def _run_loop(self, video_path, settings):
        print(f"RedisWorker: Bắt đầu xử lý {video_path}")
        cap = cv2.VideoCapture(video_path)
        tracker = BehaviorTracker(config) 
        
        frame_key = config['REDIS']['FRAME_KEY']

        # Cấu hình nén ảnh JPEG (Chất lượng 70/100 - Cân bằng tốt nhất giữa tốc độ và độ nét)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        
        # Kích thước resize mục tiêu (480p - Đủ nét cho Dashboard)
        # target_width = 854
        # target_height = 480

        try:
            for processed_frame in tracker.process_stream(cap, settings):
                if self.stop_event.is_set(): break
                self.pause_event.wait() 
                
                # --- [BƯỚC TỐI ƯU HÓA QUAN TRỌNG] ---
                # 1. Resize ảnh xuống 480p để giảm tải băng thông Redis & Mạng
                # processed_frame thường là 1920x1080 -> resize xuống 854x480
                # try:
                #     frame_resized = cv2.resize(processed_frame, (target_width, target_height))
                # except Exception:
                #     frame_resized = processed_frame # Fallback nếu lỗi resize

                # 2. Mã hóa sang JPEG với chất lượng 70% (Giảm dung lượng file ~5 lần)
                ret, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                
                if ret:
                    # 3. Đẩy vào Redis (Giờ đây gói tin rất nhẹ)
                    # ex=1: Tự xóa sau 1 giây
                    redis_client.set(frame_key, buffer.tobytes(), ex=1)
                
                # Sleep cực ngắn để nhường CPU cho Flask
                time.sleep(0.005) 

        except Exception as e:
            print(f"Worker Error: {e}")
        finally:
            cap.release()
            self.is_active = False
            # Dọn dẹp frame cuối
            redis_client.delete(frame_key)
            print("RedisWorker: Đã dừng.")

    def stop_processing(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.is_active = False

    def pause(self): self.pause_event.clear()
    def resume(self): self.pause_event.set()

worker = RedisWorker()

# --- ROUTES ---

@app.route('/')
def index():
    # Truyền giá trị mặc định xuống giao diện
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
    
    worker.stop_processing()
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    
    # Lưu path mới vào Redis
    redis_client.set(config['REDIS']['VIDEO_PATH_KEY'], path)
    
    return jsonify({'success': True, 'filename': file.filename, 'path': path})

@app.route('/control', methods=['POST'])
def control():
    data = request.json
    action = data.get('action')
    settings = data.get('settings', {})
    
    # Mapping settings
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

    # Lấy path từ Redis
    path_bytes = redis_client.get(config['REDIS']['VIDEO_PATH_KEY'])
    video_path = path_bytes.decode('utf-8') if path_bytes else None

    if action == 'start':
        if not video_path or not os.path.exists(video_path):
             return jsonify({'success': False, 'message': 'Vui lòng upload video trước!'}), 400
        
        if not worker.is_active:
            worker.start_processing(video_path, full_settings)
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Hệ thống đang chạy.'})
    
    elif action == 'stop': 
        worker.pause(); return jsonify({'success': True})
    elif action == 'continue':
        worker.resume(); return jsonify({'success': True})
    elif action == 'reset':
        worker.stop_processing(); return jsonify({'success': True})
        
    return jsonify({'success': False}), 400

def generate_frames_from_redis():
    """Đọc ảnh đã nén từ Redis và stream ra trình duyệt"""
    frame_key = config['REDIS']['FRAME_KEY']
    while True:
        # Lấy ảnh bytes từ Redis
        frame_bytes = redis_client.get(frame_key)
        
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # Giới hạn tốc độ đọc để trình duyệt không bị quá tải (khoảng 30 FPS)
            time.sleep(0.03) 
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_from_redis(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)