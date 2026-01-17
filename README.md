# FINAL — Pose, Object & Behavior Tracking

## Giới thiệu

`FINAL` là một ứng dụng Flask để phân tích video (real-time hoặc offline) kết hợp ba thành phần chính: ước lượng pose, phát hiện vật thể và phân loại hành vi/rủi ro. Hệ thống sử dụng Ultralytics (YOLO) cho inference, Kalman filter cho tracking và logic hậu xử lý để suy luận tương tác giữa người và vật.

## Người sáng tác

- Author: Your Name (thay bằng tên thực tế)
- Liên hệ / Repo: (thêm URL hoặc email nếu cần)

## Cấu trúc thư mục

- `app.py` — Flask server và các endpoint
- `pose_tracker.py` — pipeline chính: inference, tracking, vẽ, streaming
- `scripts/` — helper modules: `tracker.py`, `behavior.py`, `utils.py`
- `models/` — lưu trọng số mô hình (không bao gồm)
- `uploads/` — nơi lưu video được upload
- `requirements.txt` — danh sách package Python
- `Dockerfile` — file dựng image
- `config.yaml` — cấu hình runtime (thresholds, skip frames, draw options)

## Hướng dẫn cài đặt

### Chạy local (virtualenv)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate     # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

Chạy server:

```bash
python app.py
# mở http://localhost:5000
```

### Chạy bằng Docker

CPU build:

```bash
docker build -t final-app .
docker run --rm -p 5000:5000 -v $(pwd)/uploads:/app/uploads final-app
```

GPU build (ví dụ):

```bash
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.2.1-cudnn8-runtime-ubuntu22.04 -t final-app-gpu .
docker run --gpus all --rm -p 5000:5000 -v $(pwd)/uploads:/app/uploads final-app-gpu
```

> Lưu ý: với GPU hãy cài `torch` phù hợp với CUDA host; pin wheel trong Dockerfile để reproducible build.

## Hướng dẫn truy cập

- Mở trình duyệt: http://localhost:5000
- Stream MJPEG: `/video_feed`

## Hướng dẫn sử dụng

1. Upload video bằng UI (hay copy file vào `uploads/`).
2. Cấu hình (tuỳ chọn) các tham số confidence, skip frames và bật/tắt mô hình trong UI.
3. Nhấn **Start** để xử lý; có thể **Stop**, **Continue**, **Reset**.
4. Xem luồng đã xử lý: bounding boxes, nhãn hành vi, lưới bên phải hiện trajectory và `FPS(avg)`.

`settings` ví dụ gửi tới `/control`:

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
  "device": "cuda"   // or "cpu"
}
```

## Cấu hình

- Sửa `config.yaml` để trỏ tới mô hình và điều chỉnh thresholds/skip frames.
- Thêm `device` trong payload `settings` để ép dùng `cpu` hoặc `cuda`.

