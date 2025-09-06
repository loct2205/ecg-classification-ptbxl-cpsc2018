import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Đường dẫn tới thư mục chứa các tệp .mat
input_folder = 'H:/Training_WFDB'
output_folder = 'H:/Image_new'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Hàm chia các nhịp tim thành nhóm 5 nhịp để vẽ
def group_heartbeats(peaks, num_images=9, group_size=5):
    num_peaks = len(peaks)
    groups = []
    if num_peaks >= 13:
        # Với >= 13 nhịp tim, lấy theo nhóm liên tiếp
        for i in range(num_images):
            groups.append(peaks[i:i + group_size])
    else:
        # Với < 13 nhịp tim, lấy theo kiểu xoay vòng
        for i in range(num_images):
            group = [(j % num_peaks) for j in range(i, i + group_size)]
            groups.append([peaks[idx] for idx in group])
    return groups

# Định nghĩa số lượng nhịp tim cho từng tín hiệu trong các tệp ngoại lệ
exceptions = {
    'A0718.mat', 'A1140.mat', 'A1798.mat', 'A2663.mat', 'A2758.mat', 'A2905.mat', 'A3049.mat', 'A3263.mat', 'A3329.mat', 'A3545.mat', 'A3736.mat', 'A3875.mat',
    'A3762.mat', 'A4151.mat', 'A4181.mat', 'A4591.mat', 'A4680.mat', 'A5421.mat', 'A5556.mat', 'A5936.mat', 'A6316.mat', 'A6837.mat', 'A3146.mat', 'A5277.mat'
}

# Duyệt qua từng tệp .mat trong thư mục
for filename in os.listdir(input_folder):
    if filename.endswith(".mat"):
        # Bỏ qua tệp nếu nằm trong danh sách ngoại lệ
        if filename in exceptions:
            print(f"Bỏ qua tệp ngoại lệ: {filename}")
            continue  # Chuyển sang tệp tiếp theo

        else:
            filepath = os.path.join(input_folder, filename)

            try:
                # Đọc dữ liệu từ tệp .mat
                data = scipy.io.loadmat(filepath)

                # Kiểm tra dữ liệu đầu vào
                if 'val' not in data:
                    print(f"Tệp {filename} không chứa tín hiệu hợp lệ.")
                    continue

                signals = np.array(data['val'])

                # Tạo thư mục theo tên tệp .mat để lưu ảnh
                signal_folder = os.path.join(output_folder, filename.replace('.mat', ''))
                os.makedirs(signal_folder, exist_ok=True)

                # Xử lý từng tín hiệu trong tệp
                for signal_idx, ecg_signal in enumerate(signals):
                    ecg_signal = ecg_signal.squeeze()

                    # Chỉ lấy đủ 75000 mẫu
                    ecg_signal = ecg_signal[:75000]
                    fs = 500  # Tần số lấy mẫu

                    # Xác định vector thời gian
                    time_vector = np.arange(len(ecg_signal)) / fs

                    # Phát hiện các đỉnh R (R-peaks)
                    peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.6))

                    # Nhóm các nhịp tim
                    groups = group_heartbeats(peaks)

                    # Vẽ mỗi nhóm nhịp tim trong một ảnh
                    for img_idx, group in enumerate(groups):
                        plt.figure(figsize=(10, 6))

                        # Vẽ từng nhịp tim trong nhóm
                        for peak in group:
                            # Xác định các đoạn xung quanh đỉnh R cho nhịp tim hiện tại
                            p_wave_start = int(peak - 0.2 * fs)
                            pr_interval_end = int(peak - 0.04 * fs)
                            qrs_end = int(peak + 0.04 * fs)
                            st_end = int(peak + 0.2 * fs)
                            t_wave_end = int(peak + 0.4 * fs)

                            # Đảm bảo các chỉ số trong giới hạn của tín hiệu
                            p_wave_start = max(0, p_wave_start)
                            t_wave_end = min(len(ecg_signal), t_wave_end)

                            # Màu cho từng đoạn
                            colors = ["blue", "orange", "green", "red", "purple"]
                            segments = [
                                (p_wave_start, pr_interval_end),
                                (pr_interval_end, peak),
                                (peak, qrs_end),
                                (qrs_end, st_end),
                                (st_end, t_wave_end),
                            ]

                            # Vẽ các đoạn cho nhịp tim hiện tại
                            for (start, end), color in zip(segments, colors):
                                plt.plot(time_vector[start:end], ecg_signal[start:end], color=color)

                        # Ẩn trục
                        plt.axis('off')

                        # Lưu ảnh
                        output_path = os.path.join(signal_folder, f"{signal_idx * 9 + img_idx + 1}.png")
                        plt.tight_layout()
                        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
            except Exception as e:
                print(f"Lỗi khi xử lý tệp {filename}: {e}")
                continue