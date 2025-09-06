import os
import csv

# Thư mục chứa các tệp .hea
folder_path = "H:/Training_WFDB"  # Thay đổi đường dẫn nếu cần

# Danh sách để lưu kết quả
age_data = []

# Duyệt qua các tệp trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith(".hea"):
        file_path = os.path.join(folder_path, filename)

        # Đọc nội dung file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Tìm dòng chứa tuổi
        age = None
        for line in lines:
            if line.startswith("#Age:"):
                age = line.split(":")[1].strip()
                break

        # Thêm vào danh sách kết quả
        record_id = filename.split(".")[0]  # Lấy ID từ tên tệp
        age_data.append({"ID": record_id, "Age": age})

# Xuất kết quả ra file CSV
output_csv = "H:/Output/ages.csv"
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["ID", "Age"])
    writer.writeheader()
    writer.writerows(age_data)

print(f"Dữ liệu tuổi đã được lưu vào file {output_csv}")
