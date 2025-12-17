# Sử dụng Python 3.10 slim làm base image
# Slim images nhỏ gọn và phù hợp cho deployment
FROM python:3.12.9-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Tạo thư mục làm việc và chuyển vào đó
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

# Cài đặt các dependencies hệ thống cần thiết cho OpenCV
# Cần thiết để xử lý video và hình ảnh
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    libssl-dev \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    # Dọn dẹp cache
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt và cài đặt các thư viện Python
COPY requirements.txt $APP_HOME/
# Upgrade pip first to ensure latest resolver and wheels support
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy toàn bộ mã nguồn vào thư mục làm việc
# Bao gồm app.py, pose_tracker.py, scripts/, config.yaml, models/
COPY . $APP_HOME/

# Mở cổng 5000 (cổng mặc định của Flask)
EXPOSE 5000

# Lệnh chạy ứng dụng khi container được khởi động
CMD ["python", "app.py"]