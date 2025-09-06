import os
import pandas as pd

# Đường dẫn tới file .csv ban đầu
input_csv_path = "C:/Users/Admin/Downloads/LABELS.csv"

# Đường dẫn tới thư mục chứa các file .mat
mat_directory = "H:/Data - Copy"

# Đường dẫn để lưu file .csv đầu ra
output_csv_path = "H:/filtered_output.csv"

# Đọc file .csv ban đầu
df = pd.read_csv(input_csv_path)

# Lấy danh sách các file .mat trong thư mục (không kèm phần mở rộng)
mat_files = {os.path.splitext(file)[0] for file in os.listdir(mat_directory) if file.endswith(".mat")}

# Lọc các dòng trong dataframe dựa trên danh sách các file .mat
filtered_df = df[df["Recording"].isin(mat_files)]

# Ghi file .csv đã lọc
filtered_df.to_csv(output_csv_path, index=False)

print(f"File đã được tạo: {output_csv_path}")
