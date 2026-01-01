import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- CẤU HÌNH THÔNG SỐ ---
GOC_NGHIENG_DO = 30
THETA = np.radians(GOC_NGHIENG_DO)
BAN_KINH_GIOI_HAN = 5.0
SO_BUOC = 200
KICH_THUOC_BUOC = 0.5  # Tăng bước nhảy một chút để thấy rõ chuyển động
FPS = 20
INTERVAL = 50

# --- PHẦN 1: SINH DỮ LIỆU ---

# Sửa lỗi tên hàm: dùng dấu gạch dưới
def sinh_quy_dao_ngau_nhien(r_gioi_han, n_buoc, buoc_max):
    xp_list = [0.0]
    yp_list = [0.0]
    curr_x, curr_y = 0.0, 0.0
    
    np.random.seed(42) # Giữ seed để kết quả giống nhau mỗi lần chạy

    for _ in range(n_buoc):
        dx = np.random.randn() * buoc_max
        dy = np.random.randn() * buoc_max
        
        next_x = curr_x + dx
        next_y = curr_y + dy
        
        if np.sqrt(next_x**2 + next_y**2) < r_gioi_han:
            curr_x, curr_y = next_x, next_y
            
        xp_list.append(curr_x)
        yp_list.append(curr_y)
        
    return np.array(xp_list), np.array(yp_list)

# Tạo dữ liệu
print("1. Đang tính toán quỹ đạo...")
xp_data, yp_data = sinh_quy_dao_ngau_nhien(BAN_KINH_GIOI_HAN, SO_BUOC, KICH_THUOC_BUOC)

# Chuyển đổi sang 3D
X_3d = xp_data
Y_3d = yp_data * np.cos(THETA)
Z_3d = yp_data * np.sin(THETA)

# Tạo mặt phẳng nghiêng (để vẽ nền)
xx, yy = np.meshgrid(np.linspace(-6, 6, 10), np.linspace(-6, 6, 10))
zz = yy * np.tan(THETA)

# Tạo vòng tròn biên trong 3D
theta_circ = np.linspace(0, 2*np.pi, 100)
xc = BAN_KINH_GIOI_HAN * np.cos(theta_circ)
yc = BAN_KINH_GIOI_HAN * np.sin(theta_circ)
Yc_3d = yc * np.cos(THETA)
Zc_3d = yc * np.sin(THETA)

# --- PHẦN 2: HÀM TẠO GIF ---

def tao_gif_3d(ten_file, co_luoi):
    print(f"-> Đang tạo GIF 3D: {ten_file}...")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Góc nhìn
    ax.view_init(elev=20, azim=-45)
    ax.set_box_aspect([1, 1, 0.5]) # Tỷ lệ khung hình để không bị méo

    # Vẽ tĩnh (Mặt phẳng & Vòng tròn)
    ax.plot(xc, Yc_3d, Zc_3d, 'k--', alpha=0.3, lw=1)
    
    if co_luoi:
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='blue')
        ax.set_title(f"3D: Mặt phẳng nghiêng {GOC_NGHIENG_DO}° (Có lưới)")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    else:
        # Xóa trục và nền
        ax.set_axis_off()
        ax.set_title(f"3D: Mặt phẳng nghiêng {GOC_NGHIENG_DO}° (Không lưới)")
        # Vẽ mặt phẳng cực mờ để tạo cảm giác không gian
        ax.plot_surface(xx, yy, zz, alpha=0.05, color='gray')

    # Khởi tạo đối tượng động
    # Dùng plot thay vì scatter để update dễ hơn
    duong_dan, = ax.plot([], [], [], 'b-', lw=1, alpha=0.6)
    chat_diem, = ax.plot([], [], [], 'ro', markersize=8) # Vật thể là dấu chấm đỏ

    # Giới hạn khung hình cố định
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_zlim(-4, 4)

    def update(frame):
        # Cập nhật đường dẫn
        duong_dan.set_data(X_3d[:frame], Y_3d[:frame])
        duong_dan.set_3d_properties(Z_3d[:frame])
        
        # Cập nhật vị trí điểm
        chat_diem.set_data([X_3d[frame]], [Y_3d[frame]]) # Lưu ý: phải để trong list []
        chat_diem.set_3d_properties([Z_3d[frame]])
        return duong_dan, chat_diem

    ani = animation.FuncAnimation(fig, update, frames=len(X_3d), interval=INTERVAL, blit=False)
    ani.save(ten_file, writer='pillow', fps=FPS)
    plt.close(fig)

def tao_gif_2d(ten_file, co_luoi):
    print(f"-> Đang tạo GIF 2D: {ten_file}...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Vẽ vòng tròn giới hạn
    circle = plt.Circle((0, 0), BAN_KINH_GIOI_HAN, color='gray', fill=False, ls='--')
    ax.add_artist(circle)
    
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')

    if co_luoi:
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title("2D: Tham chiếu (Có lưới)")
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)
    else:
        ax.axis('off')
        ax.set_title("2D: Tham chiếu (Không lưới)")

    # Đối tượng động
    line, = ax.plot([], [], 'b-', lw=1, alpha=0.5)
    point, = ax.plot([], [], 'ro', markersize=8)

    def update(frame):
        line.set_data(xp_data[:frame], yp_data[:frame])
        point.set_data([xp_data[frame]], [yp_data[frame]])
        return line, point

    ani = animation.FuncAnimation(fig, update, frames=len(xp_data), interval=INTERVAL, blit=True)
    ani.save(ten_file, writer='pillow', fps=FPS)
    plt.close(fig)

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    try:
        # Tạo 4 video
        tao_gif_3d('1_3D_CoLuoi.gif', True)
        tao_gif_3d('2_3D_KhongLuoi.gif', False)
        tao_gif_2d('3_2D_CoLuoi.gif', True)
        tao_gif_2d('4_2D_KhongLuoi.gif', False)
        print("\nHOÀN TẤT! Đã tạo xong 4 file GIF.")
    except Exception as e:
        print(f"\nCÓ LỖI XẢY RA: {e}")