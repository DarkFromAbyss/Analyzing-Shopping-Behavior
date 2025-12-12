import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 

# --- 1. Thiết lập các thông số mô phỏng ---

# Hai đầu mút của đoạn thẳng
DIEM_A = -5.0
DIEM_B = 5.0

# Biên độ (Amplitude) của dao động
BIEN_DO = (DIEM_B - DIEM_A) / 2
# Vị trí cân bằng (Trung điểm)
TRUNG_DIEM = (DIEM_A + DIEM_B) / 2

# Thời gian mô phỏng (giây)
THOI_GIAN_MAX = 8

# Số chu kỳ dao động trong THOI_GIAN_MAX
SO_CHU_KY = 2 
# Tần số góc (Omega)
TOC_DO_GOC_OMEGA = SO_CHU_KY * 2 * np.pi / THOI_GIAN_MAX

# Số khung hình (frames) cho animation
SO_KHUNG_HINH = 200
thoi_gian = np.linspace(0, THOI_GIAN_MAX, SO_KHUNG_HINH) 

# --- 2. Tính toán Vị trí của chất điểm theo thời gian ---

# Sử dụng hàm cos để bắt đầu từ biên độ dương (DIEM_B)
# Vị trí X(t) = Trung_điểm + Biên_độ * cos(omega * t)
X_pos = TRUNG_DIEM + BIEN_DO * np.cos(TOC_DO_GOC_OMEGA * thoi_gian)

# Giữ trục Y cố định để vẽ trên đường thẳng nằm ngang
Y_pos = np.zeros_like(thoi_gian) 

# --- 3. Thiết lập Hình vẽ 2D và Animation ---

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_title(f'Animation Chất điểm Di chuyển Qua Lại giữa X={DIEM_A} và X={DIEM_B}')
ax.set_xlabel('Trục X')
ax.set_yticks([]) # Ẩn trục Y vì chuyển động chỉ là 1 chiều
# ax.grid(True, axis='x', linestyle='--', alpha=0.7) # Chỉ hiển thị grid trên trục X

# Vẽ đoạn thẳng cố định (đường tham chiếu)
ax.hlines(0, DIEM_A, DIEM_B, color='black', linestyle='-', linewidth=2, label='Đoạn Thẳng')
# Vẽ hai đầu mút
ax.plot(DIEM_A, 0, 'go', markersize=10, label=f'Đầu A ({DIEM_A})')
ax.plot(DIEM_B, 0, 'go', markersize=10, label=f'Đầu B ({DIEM_B})')

# Cài đặt giới hạn trục X
limit = BIEN_DO + 1.0 # Thêm khoảng trống hai bên
ax.set_xlim([TRUNG_DIEM - limit, TRUNG_DIEM + limit])
ax.set_ylim([-0.5, 0.5]) # Giới hạn trục Y nhỏ

# Khởi tạo đường đi đã vẽ (Màu xanh lam, chỉ là đường thẳng 1 chiều)
# Trong chuyển động 1 chiều, chúng ta thường chỉ vẽ điểm chuyển động
# diem_vat_the, = ax.plot([], [], 'o', color='red', markersize=12, label='Chất điểm')

# Khởi tạo điểm vật thể (màu đỏ)
diem_vat_the, = ax.plot([], [], 'o', color='red', markersize=12, label='Chất điểm')

# Thêm chú thích cho rõ ràng
plt.legend(loc='upper right')

# Hàm cập nhật khung hình
def update(frame):
    # Cập nhật vị trí X của chất điểm
    x_current = X_pos[frame]
    y_current = Y_pos[frame]
    
    # Cập nhật điểm vật thể
    diem_vat_the.set_data([x_current], [y_current])
    
    return diem_vat_the,

# Tạo animation
ani = FuncAnimation(
    fig, 
    update, 
    frames=SO_KHUNG_HINH, 
    blit=True, # Dùng blit=True để tăng tốc độ vẽ
    interval=(THOI_GIAN_MAX * 1000 / SO_KHUNG_HINH), # Thời gian (ms) giữa các khung
    repeat=False # Không lặp lại
)

# --- 4. Lưu Animation thành tệp GIF ---
TEN_FILE_GIF = 'dao_dong_qua_lai.gif'

print(f"Bắt đầu lưu animation vào: {TEN_FILE_GIF}...")

ani.save(
    TEN_FILE_GIF, 
    writer='pillow',
    fps=SO_KHUNG_HINH / THOI_GIAN_MAX 
)

print(f"Lưu tệp GIF hoàn tất tại: {TEN_FILE_GIF}")