import os
import scipy.io
import numpy as np
import csv
from scipy.signal import find_peaks

# Đường dẫn tới thư mục chứa các tệp .mat
input_folder = 'H:/Training_WFDB'
output_folder = 'H:/Output/count_find_peaks_output'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Định nghĩa danh sách ngoại lệ (nếu có)
exceptions = {
    'A0718.mat', 'A1140.mat', 'A1798.mat', 'A2663.mat', 'A2758.mat', 'A2905.mat', 'A3049.mat', 'A3263.mat', 'A3329.mat', 'A3545.mat', 'A3736.mat', 'A3875.mat',
    'A3762.mat', 'A4151.mat', 'A4181.mat', 'A4591.mat', 'A4680.mat', 'A5421.mat', 'A5556.mat', 'A5936.mat', 'A6316.mat', 'A6837.mat', 'A3146.mat', 'A5277.mat'
}

# Duyệt qua từng tệp .mat trong thư mục
for filename in os.listdir(input_folder):
    if filename.endswith(".mat"):
        filepath = os.path.join(input_folder, filename)

        # Bỏ qua tệp nếu nằm trong danh sách ngoại lệ
        if filename in exceptions:
            print(f"Bỏ qua tệp ngoại lệ: {filename}")
            continue

        try:
            # Đọc dữ liệu từ tệp .mat
            data = scipy.io.loadmat(filepath)

            # Kiểm tra dữ liệu đầu vào
            if 'val' not in data:
                print(f"Tệp {filename} không chứa tín hiệu hợp lệ.")
                continue

            signals = np.array(data['val'])

            # File CSV sẽ được lưu trong thư mục output
            csv_filename = os.path.join(output_folder, filename.replace('.mat', '.csv'))

            # Mở file CSV để ghi dữ liệu
            with open(csv_filename, mode='w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)

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
                    r_positions = time_vector[peaks]

                    # Dòng đầu tiên: Số lượng đỉnh R và vị trí của chúng
                    row = [len(r_positions)] + list(r_positions)
                    csvwriter.writerow(row)

        except Exception as e:
            print(f"Lỗi khi xử lý tệp {filename}: {e}")
            continue
