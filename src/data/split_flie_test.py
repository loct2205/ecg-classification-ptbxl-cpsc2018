import os
import shutil

# Thư mục nguồn chứa các file dữ liệu ECG (bao gồm .hea và .mat)
source_dir = "H:/Training_WFDB"

# Thư mục đích chứa các file bệnh
destination_dir = "H:/Data"

# Thư mục để lưu các file không có trong thư mục đích
missing_files_dir = "H:/Data_Train"
os.makedirs(missing_files_dir, exist_ok=True)

# Dữ liệu về các loại bệnh
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

# Lặp qua các thư mục bệnh trong thư mục đích để kiểm tra các file
existing_files = set()

# Lấy tất cả các file đã sao chép vào thư mục đích (bao gồm .hea và .mat)
for disease in disease_map.values():
    disease_folder = os.path.join(destination_dir, disease)
    if os.path.exists(disease_folder):
        existing_files.update([f[:-4] for f in os.listdir(disease_folder) if f.endswith('.hea') or f.endswith('.mat')])

# Lặp qua các file trong thư mục nguồn và kiểm tra xem có nằm trong thư mục đích không
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.hea') or file.endswith('.mat'):
            base_name = file[:-4]  # Lấy tên file không có phần mở rộng
            if base_name not in existing_files:
                # Nếu file không tồn tại trong thư mục đích, sao chép vào thư mục missing_files
                source_file = os.path.join(root, file)
                dest_file = os.path.join(missing_files_dir, file)
                shutil.copy(source_file, dest_file)
                print(f"File {file} không có trong thư mục đích và đã được sao chép vào thư mục Data_Train.")
