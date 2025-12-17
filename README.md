# FINAL (Pose + Object + Behavior Tracking)

This repository contains a Flask-based video processing app that runs Ultralytics (YOLO) models for pose/object/risk detection, Kalman-based tracking and behavior classification.

Contents
- `app.py` - Flask web server and controller
- `pose_tracker.py` - main pipeline for inference, tracking, drawing and streaming
- `scripts/` - helper modules (`tracker.py`, `behavior.py`, `utils.py`)
- `uploads/` - place to upload input videos
- `models/` - (optional) pretrained model files (not included)
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container build file (supports custom base image via build-arg)

Quick start (CPU)
1. Build the image (CPU default):

```bash
docker build -t final-app .
```

2. Run the container (exposes port 5000):

```bash
docker run --rm -p 5000:5000 -v $(pwd)/uploads:/app/uploads final-app
```

Quick start (GPU)
- Choose an appropriate CUDA base image and build with `--build-arg`. Example (adjust tag for your GPU/CUDA):

```bash
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.2.1-cudnn8-runtime-ubuntu22.04 -t final-app-gpu .
```

- Run with NVIDIA runtime (nvidia-container-toolkit must be installed):

```bash
docker run --gpus all --rm -p 5000:5000 -v $(pwd)/uploads:/app/uploads final-app-gpu
```

Notes on dependencies and GPU
- `requirements.txt` lists main Python packages. For GPU builds you must ensure `torch` (if used by your models) is installed with CUDA support matching your base image. The `ultralytics` package will select device automatically if CUDA is available.
- If you need a specific `torch` wheel, install it prior to other packages in the Dockerfile or replace the `pip install -r requirements.txt` step with explicit `pip install` commands for `torch` + `torchvision`.

Running locally (no Docker)
1. Create and activate a virtualenv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate     # Windows PowerShell
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python app.py
```

Configuration
- Edit `config.yaml` to set model paths, thresholds, drawing options and skip frames. The server reads it at startup.
- You can pass `device` in the frontend `settings` payload to prefer `'cpu'` or `'cuda'`.

Project layout and runtime behavior
- The Flask UI uploads a video to `uploads/` and starts a `VideoProcessor` thread which calls `process_video_stream` in `pose_tracker.py`.
- `pose_tracker.py` loads three models (pose/object/risk), runs them according to `skip_frames` settings, updates Kalman trackers, computes interactions and draws results.
- A side grid is rendered alongside the video to show recent hip-centroid trajectories and an average FPS is overlayed.

Recommendations for production
- Use GPU for inference-heavy workloads; pin correct `torch` / CUDA versions.
- Reduce model input resolution or increase skip frames to improve throughput.
- Use a process manager (systemd / docker-compose) and mount persistent `models/` and `uploads/` directories.

Troubleshooting
- If models fail to load, check `config.yaml` model paths and ensure model files are present under `models/` or reachable via the paths.
- For GPU issues, confirm `nvidia-smi` works on the host and the container is run with `--gpus all`.

License / Credits
- (Add your project license and credits here)
# FINAL