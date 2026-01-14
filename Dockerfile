# Sử dụng Python 3.12.9 slim (Debian Bookworm)
FROM python:3.12.9-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Tạo thư mục làm việc
WORKDIR $APP_HOME

# Cài đặt các thư viện hệ thống
# Đã xóa chữ "xử" thừa và sắp xếp lại cho gọn gàng
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    libssl-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt $APP_HOME/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY . $APP_HOME/

# Mở cổng
EXPOSE 5000

# Chạy ứng dụng
CMD ["python", "app.py"]