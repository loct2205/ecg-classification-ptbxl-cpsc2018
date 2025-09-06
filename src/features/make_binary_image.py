import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage.transform import resize

# Đường dẫn tới thư mục chứa các tệp .mat
input_folder = 'H:/Training_WFDB'
output_folder = 'H:/Output/binary_images_output'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Kích thước cố định của ảnh
M, N = 900, 600

# Hàm tạo ảnh nhị phân từ tín hiệu
def create_binary_images(ecg_signal, r_peaks, start_idx, width, fs):
    images = []
    for j in range(start_idx, start_idx + width):
        # Lấy tín hiệu xung quanh đỉnh R
        r_pos = r_peaks[j]
        segment_start = max(0, r_pos - int(0.4 * fs))  # Bắt đầu tại -0.4 giây
        segment_end = min(len(ecg_signal), r_pos + int(0.6 * fs))  # Kết thúc tại +0.6 giây
        segment = ecg_signal[segment_start:segment_end]

        # Xây dựng vector thời gian cho đoạn tín hiệu
        time_segment = np.arange(segment_start, segment_end) / fs

        # Vẽ đoạn tín hiệu
        fig, ax = plt.subplots(figsize=(6, 9))  # Tỉ lệ phù hợp với M x N
        ax.plot(time_segment, segment, color='black')
        ax.axis('off')  # Ẩn trục
        plt.tight_layout()

        # Lưu ảnh nhị phân
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Chuyển ảnh sang định dạng nhị phân và resize
        binary_img = (img[..., 0] < 128).astype(np.uint8)  # Chuyển ảnh sang nhị phân
        binary_img_resized = resize(binary_img, (M, N), anti_aliasing=False)
        images.append(binary_img_resized)

    return images
# Định nghĩa chỉ số i và độ rộng w
i = 1  # Chỉ số nhịp bắt đầu
w = 5  # Số nhịp cần lấy
# Duyệt qua từng tệp .mat trong thư mục
for filename in os.listdir(input_folder):
    if filename.endswith(".mat"):
        filepath = os.path.join(input_folder, filename)

        try:
            # Đọc dữ liệu từ tệp .mat
            data = scipy.io.loadmat(filepath)

            # Kiểm tra dữ liệu đầu vào
            if 'val' not in data:
                print(f"Tệp {filename} không chứa tín hiệu hợp lệ.")
                continue

            signals = np.array(data['val'])

            # Xử lý từng đạo trình trong tệp
            for signal_idx, ecg_signal in enumerate(signals):
                ecg_signal = ecg_signal.squeeze()

                # Chỉ lấy đủ 75000 mẫu
                ecg_signal = ecg_signal[:75000]
                fs = 500  # Tần số lấy mẫu

                # Phát hiện các đỉnh R
                peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.6))

                # Giới hạn giá trị i
                max_i = len(peaks) - w
                if max_i < 1:
                    print(f"Tệp {filename}, đạo trình {signal_idx + 1} không đủ số đỉnh R.")
                    continue

                # Duyệt qua các giá trị của i
                for i in range(1, max_i + 1):
                    # Tạo 12 ảnh nhị phân cho 12 đạo trình
                    all_images = []
                    for lead_idx, lead_signal in enumerate(signals):
                        lead_signal = lead_signal.squeeze()
                        images = create_binary_images(lead_signal, peaks, i - 1, w, fs)  # i - 1 vì Python index từ 0
                        all_images.append(images)

                    # Lưu kết quả dưới dạng ma trận
                    output_filename = os.path.join(
                        output_folder,
                        f"{filename.replace('.mat', '')}_i{i}_w{w}.npy"
                    )
                    np.save(output_filename, np.array(all_images))
                    print(f"Lưu ma trận nhị phân tại: {output_filename}")

        except Exception as e:
            print(f"Lỗi khi xử lý tệp {filename}: {e}")
            continue
