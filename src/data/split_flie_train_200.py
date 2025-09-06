import pandas as pd
import shutil
import os

# Đọc file CSV chứa nhãn bệnh
labels_df = pd.read_csv('C:/Users/Admin/Downloads/LABELS.csv')

# Dữ liệu về số lượng bệnh (sửa theo dữ liệu của bạn)
disease_map = {
    1: "NORM",
    2: "AF",
    3: "I-AVB",
    4: "LBBB",
    5: "RBBB",
    6: "PAC",
    7: "PVC",
    8: "STD",
    9: "STE"
}

# Thư mục nguồn chứa các file dữ liệu ECG (bao gồm .hea và .mat)
source_dir = "H:/Training_WFDB"

# Thư mục đích để sao chép các file
destination_dir = "H:/Data"

# Tạo một thư mục cho mỗi bệnh (nếu chưa có)
for disease in disease_map.values():
    os.makedirs(os.path.join(destination_dir, disease), exist_ok=True)

# Lọc và sao chép 200 file cho mỗi bệnh
for disease_label, disease_name in disease_map.items():
    disease_files = labels_df[labels_df['First_label'] == disease_label]['Recording'].head(200)

    for file in disease_files:
        # Đường dẫn của các file .hea và .mat
        source_hea = os.path.join(source_dir, f"{file}.hea")
        source_mat = os.path.join(source_dir, f"{file}.mat")

        # Đường dẫn đích cho các file .hea và .mat
        dest_hea = os.path.join(destination_dir, disease_name, f"{file}.hea")
        dest_mat = os.path.join(destination_dir, disease_name, f"{file}.mat")

        # Sao chép file .hea nếu tồn tại
        if os.path.exists(source_hea):
            shutil.copy(source_hea, dest_hea)
            print(f"Sao chép {file}.hea vào {disease_name}")
        else:
            print(f"File {file}.hea không tồn tại.")

        # Sao chép file .mat nếu tồn tại
        if os.path.exists(source_mat):
            shutil.copy(source_mat, dest_mat)
            print(f"Sao chép {file}.mat vào {disease_name}")
        else:
            print(f"File {file}.mat không tồn tại.")
