# [cite_start]Sử dụng Python 3.12.9 slim làm base image (như file gốc của bạn) [cite: 1]
FROM python:3.12.9-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Tạo thư mục làm việc
WORKDIR $APP_HOME

# Cài đặt các dependencies hệ thống cần thiết
# Python 3.12 + Slim cần build-essential và python3-dev để biên dịch một số thư viện C++
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    libssl-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    [cite_start]&& rm -rf /var/lib/apt/lists/* [cite: 1, 2]

# Copy requirements và cài đặt thư viện
COPY requirements.txt $APP_HOME/

# Upgrade pip trước khi cài đặt để đảm bảo tương thích tốt nhất với Python 3.12
# --no-cache-dir giúp giảm dung lượng image
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# [cite_start]Copy toàn bộ mã nguồn vào image [cite: 2]
COPY . $APP_HOME/

# [cite_start]Mở cổng 5000 (Flask) [cite: 3]
EXPOSE 5000

# [cite_start]Lệnh chạy ứng dụng [cite: 3]
CMD ["python", "app.py"]