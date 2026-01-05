vieets # FINAL — Pose, Object & Behavior Tracking

**Project name:** FINAL

## Giới thiệu

`FINAL` là một ứng dụng xử lý video (real-time hoặc offline) dựa trên Flask, kết hợp ba thành phần chính:
- Ước lượng pose (pose estimation) để lấy keypoints người.
- Phát hiện và phân loại vật thể (object detection) để nhận diện các đối tượng quan tâm.
- Phát hiện hành vi / đánh giá rủi ro (behavior/risk classification) để suy luận tương tác người–vật.

Hệ thống dùng Ultralytics (YOLO) cho inference, Kalman filter + tracker để theo dõi ID xuyên khung, và logic hậu xử lý để gán nhãn hành vi.

## Business value

- Giảm chi phí giám sát thủ công bằng cách tự động phát hiện hành vi nguy cơ (ví dụ shoplifting, va chạm).
- Hỗ trợ phân tích lưu lượng và hành vi trong cửa hàng, kho bãi, hay môi trường an ninh.
- Có thể tích hợp vào hệ thống báo động thời gian thực hoặc dashboard phân tích.

## Performance (ước lượng / tham chiếu)

- Inference latency: ~15–60 ms / frame trên GPU (tùy mô hình và kích thước ảnh).
- CPU throughput: ~1–5 fps trên CPU consumer (tùy CPU và kích thước ảnh).
- Tracking stability: giữ ID ổn định trong đa số trường hợp chuyển động chậm/ trung bình; cạnh trường hợp occlusion dài có thể mất ID.

Lưu ý: các con số trên là ước lượng; hiệu năng thực tế phụ thuộc vào mô hình (`best.pt`, `yolo11n_object365.pt`, `yolo11n-pose.pt`), độ phân giải, và thiết bị (CPU/GPU).

## Công nghệ sử dụng

- Backend: `Flask`
- Detection / Pose: Ultralytics YOLO (PyTorch)
- Tracking: Kalman filter + custom tracker (scripts/tracker.py)
- Dependencies: Python, OpenCV, NumPy, PyTorch và các gói trong `requirements.txt`
- Container: `Docker` (kèm tùy chọn image hỗ trợ CUDA cho GPU)

## Cấu trúc chính

- `app.py` — Flask server và REST/web endpoints
- `pose_tracker.py` — pipeline inference, tracking, drawing, streaming
- `scripts/` — `tracker.py`, `behavior.py`, `utils.py`
- `models/` — chứa trọng số mô hình
- `uploads/` — video input

## Cài đặt (không dùng Docker)

1. Tạo virtualenv và cài dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Chuẩn bị mô hình: sao chép file trọng số vào `models/` (ví dụ `best.pt`, `yolo11n_object365.pt`, `yolo11n-pose.pt`).

3. Cấu hình:

- Mở `config.yaml` để thiết lập `REDIS.HOST/PORT/DB` và các `FRAME_KEY` nếu cần.
- Nếu chạy Redis trên máy local dùng WSL hoặc native, để `REDIS.HOST: 127.0.0.1`; nếu dùng container, đổi thành `redis`.

4. Kiểm tra môi trường:

```powershell
python -c "import sys, torch; print('Python', sys.version); print('Torch', getattr(torch, '__version__', 'not installed'))"
```

## Chạy (không dùng Docker)

1. Bật Redis (nếu sử dụng Redis): theo hướng dẫn phần Redis ở dưới hoặc chạy `redis-server`.

2. Khởi chạy server Flask:

```powershell
set FLASK_ENV=production
python app.py
# Mở http://localhost:5000
```

3. Upload video qua UI hoặc copy file vào `uploads/` rồi dùng API `/control` để start.

## Cài đặt (Docker)

1. Cài Docker (và Docker Compose nếu chưa có). Nếu muốn GPU, cài NVIDIA Container Toolkit và driver tương thích.

2. (Tuỳ chọn) Chỉnh `docker-compose.yaml`:

- Mình đã thêm service `redis` và volume `redis-data` sẵn vào `docker-compose.yaml`.
- Nếu muốn khởi động với hỗ trợ GPU, cấu hình service `pose_tracker` để dùng runtime NVIDIA (tham khảo comment trong file).

3. Build image (không dùng compose):

```bash
docker build -t final-app .
```

## Chạy (Docker)

- Với Docker Compose (khuyến nghị, sẽ start cả Redis + app):

```bash
docker compose up -d
```

- Kiểm tra trạng thái:

```bash
docker ps
docker logs final_redis
docker exec -it final_redis redis-cli ping
```

- Chạy chỉ container app (không dùng compose):

```bash
docker run --rm -p 5000:5000 -v %cd%/uploads:/app/uploads final-app
```

- GPU (nếu image build cho GPU và host có NVIDIA):

```bash
docker run --gpus all --rm -p 5000:5000 -v %cd%/uploads:/app/uploads final-app-gpu
```

## Ghi chú thêm

- Nếu dùng Docker Compose và Redis service, đặt `REDIS.HOST` trong `config.yaml` là `redis` (tên service), không phải `127.0.0.1`.
- Luôn kiểm tra logs (`docker logs pose_tracker_app`) nếu không thấy UI hoạt động.


## API & UI

- Mở web UI: `http://localhost:5000`
- MJPEG stream: `/video_feed`
- Control endpoint: POST `/control` nhận payload settings (tham khảo `config.yaml` và ví dụ JSON).

## Cấu hình mẫu (ví dụ payload)

```json
{
  "pose_conf_thresh": 0.3,
  "object_conf_thresh": 0.35,
  "risk_conf_thresh": 0.4,
  "pose_skip_frames": 2,
  "object_skip_frames": 5,
  "risk_skip_frames": 5,
  "keypoint_draw": true,
  "objects_draw": true,
  "model_pose_enabled": true,
  "model_object_enabled": true,
  "model_risk_enabled": false,
  "device": "cuda"
}
```

## Troubleshooting

- Nếu model không load: kiểm tra đường dẫn file trong `models/` và `config.yaml`.
- Nếu Docker build lỗi liên quan thư viện native (ví dụ `psycopg2`): cài thêm hệ thống packages tương ứng hoặc dùng phiên bản binary.
- Nếu chậm: giảm độ phân giải input, tăng `skip_frames`, hoặc chạy trên GPU.

## Gợi ý phát triển

- Pin phiên bản `torch` phù hợp với CUDA khi build Docker để tránh lỗi runtime.
- Thêm test tự động cho pipeline inference/tracking.
- Cung cấp sample videos trong `uploads/sample/` để dễ QA.

## License

- Thêm license và credits tại đây (ví dụ MIT).

---

See the main server at [app.py](app.py) and configuration at [config.yaml](config.yaml).

## Redis — Cài đặt & Hướng dẫn sử dụng chi tiết

Ứng dụng dùng Redis làm nơi lưu frame nén tạm thời (`FRAME_KEY`), lưu `VIDEO_PATH_KEY` để persistence và một vài key trạng thái khác (`STATUS_KEY`).

Keys mặc định (xem `config.yaml`):
- `REDIS.FRAME_KEY` — `system:live_frame` (frame JPEG, cài TTL ngắn ex=1 trong code)
- `REDIS.VIDEO_PATH_KEY` — `system:video_path` (path tới file video được upload)
- `REDIS.STATUS_KEY` — `system:status`

1) Cài đặt Redis

- Linux (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install redis-server -y
sudo systemctl enable --now redis-server
redis-cli ping  # -> PONG
```

- macOS (Homebrew):

```bash
brew install redis
brew services start redis
redis-cli ping
```

- Windows:
  - Khuyến nghị: cài đặt WSL (Windows Subsystem for Linux) và cài Redis trong WSL theo hướng dẫn Linux ở trên.
  - Hoặc dùng Docker (dưới) để chạy Redis trên Windows.

- Docker (mọi nền tảng):

```bash
docker run -d --name final-redis -p 6379:6379 redis:7
docker exec -it final-redis redis-cli ping
```

2a) Windows + WSL (Ubuntu) — cài WSL và Redis

Nếu bạn đang dùng Windows, khuyến nghị cài WSL2 và chạy Redis trong một distro Ubuntu để môi trường giống Linux:

```powershell
# 1) Mở PowerShell (Admin) và cài WSL + Ubuntu (Windows 10/11, build mới):
wsl --install -d ubuntu
# Nếu lệnh trên không có sẵn, tham khảo: https://learn.microsoft.com/windows/wsl/install

# 2) Mở Ubuntu (từ Start menu) và cập nhật hệ thống:
sudo apt update && sudo apt upgrade -y

# 3) Cài Redis:
sudo apt install redis-server -y

# 4) Bật Redis và kiểm tra:
sudo systemctl enable --now redis-server
redis-cli ping  # -> PONG
```

Lưu ý:
- Trên hệ thống Windows + WSL2 mới, `localhost:6379` từ Windows thường được forward tới dịch vụ trong WSL — bạn có thể để `REDIS.HOST: 127.0.0.1` trong `config.yaml`.
- Nếu không thể kết nối từ Windows, bạn có thể kiểm tra IP của WSL: `wsl hostname -I` và dùng IP đó trong `config.yaml`, hoặc chạy `redis-cli` bên trong WSL để xác thực.
- Để cho phép truy cập từ host (không chỉ localhost), chỉnh `/etc/redis/redis.conf` và đổi `bind 127.0.0.1` thành `bind 0.0.0.0` rồi `sudo systemctl restart redis-server` (chú ý bảo mật — nên cấu hình `requirepass` nếu mở cổng).

2) Docker Compose mẫu

Thêm service Redis vào `docker-compose.yaml` (nếu chạy toàn bộ stack):

```yaml
services:
  redis:
    image: redis:7
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

3) Bảo mật & cấu hình cơ bản

- Để yêu cầu mật khẩu (khuyến nghị khi mở cổng): sửa `redis.conf` hoặc chạy container với lệnh `redis-server --requirepass "yourpassword"`.
- Ví dụ docker-compose có mật khẩu (không lưu mật khẩu trong repo):

```yaml
  redis:
    image: redis:7
    command: ["redis-server","--requirepass","yourpassword"]
```

- Trong ứng dụng, nếu bật mật khẩu thì khởi tạo `redis.Redis(..., password='yourpassword')`.

4) Cấu hình ứng dụng (`config.yaml`)

- `config.yaml` đã có mục `REDIS` với `HOST`, `PORT`, `DB`, `FRAME_KEY`, `VIDEO_PATH_KEY`.
- Nếu chạy Redis trên cùng máy: giữ `127.0.0.1:6379`; trên container network đổi `HOST` thành `redis` (tên service trong docker-compose).

5) Ví dụ Python (phù hợp với mã nguồn hiện tại ở `app.py`)

```python
import redis

cfg = {
  'HOST': '127.0.0.1',
  'PORT': 6379,
  'DB': 0,
}

client = redis.Redis(host=cfg['HOST'], port=cfg['PORT'], db=cfg['DB'], decode_responses=False)
client.ping()

# Ghi frame JPEG (bytes) với TTL ngắn (app dùng ex=1)
client.set('system:live_frame', jpeg_bytes, ex=1)

# Lấy path video đã upload
path_bytes = client.get('system:video_path')
video_path = path_bytes.decode('utf-8') if path_bytes else None
```

6) Thử nghiệm & kiểm tra

- Kiểm tra kết nối từ máy host: `redis-cli -h 127.0.0.1 -p 6379 ping` -> `PONG`.
- Kiểm tra key frame: `redis-cli get system:live_frame` (nếu là bytes sẽ trả chuỗi thô).
- Dùng `redis-cli monitor` để theo dõi realtime các thao tác.

7) Tối ưu & lưu ý

- Ứng dụng nén frame JPEG và lưu vào Redis với `ex=1` (xóa tự động sau 1s) để giảm dung lượng lưu trữ.
- Nếu muốn hỗ trợ nhiều client đồng thời, hãy tăng bộ nhớ Redis hoặc dùng eviction policy phù hợp (`maxmemory`, `maxmemory-policy`).
- Tránh lưu frame lớn hoặc giữ TTL quá lâu — Redis không phải object storage dài hạn.

8) Khắc phục sự cố

- Nếu `redis_client.ping()` lỗi: kiểm tra service status, firewall, và `HOST`/`PORT` trong `config.yaml`.
- Nếu thấy lỗi `ConnectionRefused`: đảm bảo Redis đang chạy trong cùng network/container và không bị bound tới localhost nếu truy cập qua network.

---

Xem thêm: [app.py](app.py) (mẫu kết nối/ghi đọc), [config.yaml](config.yaml) (key mặc định).
