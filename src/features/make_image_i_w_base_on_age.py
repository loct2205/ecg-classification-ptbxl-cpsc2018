# -*- coding: utf-8 -*-
import re
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Đường dẫn tới thư mục chứa các tệp .mat
input_folder = 'H:/Data - Copy'
output_folder = 'H:/Output/image_base_on_age'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Hàm chia các nhịp tim thành nhóm 5 nhịp liên tiếp để vẽ
def group_heartbeats(peaks, group_size, k):
    num_peaks = len(peaks)
    groups = []

    # Chia các đỉnh thành các nhóm 5 nhịp liên tiếp
    for i in range(k, num_peaks - group_size + 1):
        group = peaks[i:i + group_size]
        groups.append(group)

    return groups

# Định nghĩa số lượng nhịp tim cho từng tín hiệu trong các tệp ngoại lệ
exceptions = {
    'A0718.mat', 'A1140.mat', 'A1798.mat', 'A2663.mat', 'A2758.mat', 'A2905.mat', 'A3049.mat', 'A3263.mat', 'A3329.mat', 'A3545.mat', 'A3736.mat', 'A3875.mat',
    'A3762.mat', 'A4151.mat', 'A4181.mat', 'A4591.mat', 'A4680.mat', 'A5421.mat', 'A5556.mat', 'A5936.mat', 'A6316.mat', 'A6837.mat', 'A3146.mat', 'A5277.mat'
}
k = 0
w = 5
# Duyệt qua từng tệp .mat trong thư mục
for filename in os.listdir(input_folder):
    if filename.endswith(".mat"):
        name = (filename[:filename.find('.mat')])
        # Bỏ qua tệp nếu nằm trong danh sách ngoại lệ
        if filename in exceptions:
            print(f"Bỏ qua tệp ngoại lệ: {filename}")
            continue  # Chuyển sang tệp tiếp theo

        else:
            # Tên tệp .hea tương ứng
            hea_file = os.path.join(input_folder, f"{name}.hea")
            # Kiểm tra xem tệp .hea có tồn tại không
            if os.path.exists(hea_file):
                with open(hea_file, 'r') as file:
                    lines = file.readlines()

                # Tìm dòng chứa tuổi
                age = None
                alpha = 0.6
                for line in lines:
                    if line.startswith("#Age:"):
                        age = line.split(":")[1].strip()
                        break
                if age is None or age.lower() == 'nan':
                    print(f"Giá trị 'age' không hợp lệ trong file: {hea_file}, giá trị: {age}. Gán alpha = 0.6")
                    alpha = 0.6  # Giá trị mặc định
                else:
                    age = int(age)
                    alpha = 60 / (220 - age)
                print(f"alpha: {alpha}")

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
                    peaks, _ = find_peaks(ecg_signal, distance=int(fs * alpha))

                    # Nhóm các nhịp tim thành các nhóm 5 đỉnh liên tiếp
                    groups = group_heartbeats(peaks, w, k)

                    # Vẽ mỗi nhóm nhịp tim trong một ảnh
                    for img_idx, group in enumerate(groups):
                        plt.figure(figsize=(9, 6))

                        # Lấy thời gian của đỉnh đầu tiên và đỉnh thứ 5
                        start_time = time_vector[group[0]]  # Thời gian của đỉnh đầu tiên
                        end_time = time_vector[group[-1]]   # Thời gian của đỉnh thứ 5

                        # Tìm các chỉ số tương ứng với thời gian
                        start_idx = max(0, group[0] - int(0.5 * fs))  # 500 ms trước đỉnh đầu tiên
                        end_idx = min(len(ecg_signal), group[-1] + int(0.5 * fs))  # 500 ms sau đỉnh thứ 5

                        # Vẽ tín hiệu ECG từ đỉnh 1 đến đỉnh 5
                        plt.plot(time_vector[start_idx:end_idx], ecg_signal[start_idx:end_idx], color='black')

                        # Đánh dấu các đỉnh R trong nhóm
                        for peak in group:
                            plt.plot(time_vector[peak], ecg_signal[peak], 'ko')  # Đánh dấu các đỉnh bằng chấm đỏ

                        # Ẩn trục
                        plt.axis('off')

                        # Lưu ảnh với kích thước 900x600 và đảm bảo màu trắng đen
                        output_path = os.path.join(signal_folder, f"{name}_{signal_idx + 1}_{img_idx + 1}.png")
                        plt.tight_layout()
                        plt.savefig(output_path, dpi=100, bbox_inches=None, pad_inches=0, transparent=False)
                        plt.close()
            except Exception as e:
                print(f"Lỗi khi xử lý tệp {filename}: {e}")
                continue



# import re
# import os
# import scipy.io
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
#
# # Đường dẫn tới thư mục chứa các tệp .mat
# input_folder = 'H:/Training_WFDB'
# output_folder = 'H:/Output/images_i_w_output'
#
# # Tạo thư mục đích nếu chưa tồn tại
# os.makedirs(output_folder, exist_ok=True)
#
# # Hàm chia các nhịp tim thành nhóm 5 nhịp liên tiếp để vẽ
# def group_heartbeats(peaks, group_size, k):
#     num_peaks = len(peaks)
#     groups = []
#
#     # Chia các đỉnh thành các nhóm 5 nhịp liên tiếp
#     for i in range(k, num_peaks - group_size + 1):
#         group = peaks[i:i + group_size]
#         groups.append(group)
#
#     return groups
#
# # Định nghĩa số lượng nhịp tim cho từng tín hiệu trong các tệp ngoại lệ
# exceptions = {
#     'A0718.mat', 'A1140.mat', 'A1798.mat', 'A2663.mat', 'A2758.mat', 'A2905.mat', 'A3049.mat', 'A3263.mat', 'A3329.mat', 'A3545.mat', 'A3736.mat', 'A3875.mat',
#     'A3762.mat', 'A4151.mat', 'A4181.mat', 'A4591.mat', 'A4680.mat', 'A5421.mat', 'A5556.mat', 'A5936.mat', 'A6316.mat', 'A6837.mat', 'A3146.mat', 'A5277.mat'
# }
# k = 0
# w = 5
# # Duyệt qua từng tệp .mat trong thư mục
# for filename in os.listdir(input_folder):
#     if filename.endswith(".mat"):
#         name = int(filename[1:filename.find('.mat')])
#         # Bỏ qua tệp nếu nằm trong danh sách ngoại lệ
#         if filename in exceptions:
#             print(f"Bỏ qua tệp ngoại lệ: {filename}")
#             continue  # Chuyển sang tệp tiếp theo
#
#         else:
#             filepath = os.path.join(input_folder, filename)
#
#             try:
#                 # Đọc dữ liệu từ tệp .mat
#                 data = scipy.io.loadmat(filepath)
#
#                 # Kiểm tra dữ liệu đầu vào
#                 if 'val' not in data:
#                     print(f"Tệp {filename} không chứa tín hiệu hợp lệ.")
#                     continue
#
#                 signals = np.array(data['val'])
#
#                 # Tạo thư mục theo tên tệp .mat để lưu ảnh
#                 signal_folder = os.path.join(output_folder, filename.replace('.mat', ''))
#                 os.makedirs(signal_folder, exist_ok=True)
#
#                 # Xử lý từng tín hiệu trong tệp
#                 for signal_idx, ecg_signal in enumerate(signals):
#                     ecg_signal = ecg_signal.squeeze()
#
#                     # Chỉ lấy đủ 75000 mẫu
#                     ecg_signal = ecg_signal[:75000]
#                     fs = 500  # Tần số lấy mẫu
#
#                     # Xác định vector thời gian
#                     time_vector = np.arange(len(ecg_signal)) / fs
#
#                     # Phát hiện các đỉnh R (R-peaks)
#                     peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.6))
#
#                     # Nhóm các nhịp tim thành các nhóm 5 đỉnh liên tiếp
#                     groups = group_heartbeats(peaks, w, k)
#
#                     # Vẽ mỗi nhóm nhịp tim trong một ảnh
#                     for img_idx, group in enumerate(groups):
#                         plt.figure(figsize=(9, 6))
#
#                         # Lấy thời gian của đỉnh đầu tiên và đỉnh thứ 5
#                         start_time = time_vector[group[0]]  # Thời gian của đỉnh đầu tiên
#                         end_time = time_vector[group[-1]]   # Thời gian của đỉnh thứ 5
#
#                         # Tìm các chỉ số tương ứng với thời gian
#                         start_idx = max(0, group[0] - int(0.5 * fs))  # 500 ms trước đỉnh đầu tiên
#                         end_idx = min(len(ecg_signal), group[-1] + int(0.5 * fs))  # 500 ms sau đỉnh thứ 5
#
#                         # Vẽ tín hiệu ECG từ đỉnh 1 đến đỉnh 5
#                         plt.plot(time_vector[start_idx:end_idx], ecg_signal[start_idx:end_idx], color='black')
#
#                         # Đánh dấu các đỉnh R trong nhóm
#                         for peak in group:
#                             plt.plot(time_vector[peak], ecg_signal[peak], 'ro')  # Đánh dấu các đỉnh bằng chấm đỏ
#
#                         # Ẩn trục
#                         plt.axis('off')
#
#                         # Lưu ảnh
#                         output_path = os.path.join(signal_folder, f"{name}_{signal_idx + 1}_{img_idx + 1}.png")
#                         plt.tight_layout()
#                         plt.savefig(output_path, bbox_inches=None, pad_inches=0)
#                         plt.close()
#             except Exception as e:
#                 print(f"Lỗi khi xử lý tệp {filename}: {e}")
#                 continue
