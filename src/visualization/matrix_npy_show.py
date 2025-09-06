import numpy as np
import os

# Đường dẫn đến thư mục gốc chứa các thư mục con A001, A002, ...
base_folder = 'H:/Output/binary_images_i_w_base_on_age_output'

# Duyệt qua các tệp trong thư mục gốc (không đọc thư mục con)
for filename in os.listdir(base_folder):
    file_path = os.path.join(base_folder, filename)

    # Kiểm tra xem có phải là tệp (không phải thư mục con)
    if os.path.isfile(file_path) and filename.endswith(".npy"):  # Tìm các tệp có đuôi _matrix.npy
        # Tải ma trận ảnh từ tệp .npy
        binary_matrix = np.load(file_path)

        # In kích thước của ma trận
        print(f"Kích thước của ma trận ảnh {filename}: {binary_matrix.shape}")

        # In nội dung của ma trận
        print(f"Nội dung của ma trận ảnh {filename}:")
        print(binary_matrix)

# import numpy as np
# import os
#
# # Đường dẫn đến thư mục gốc chứa các thư mục con A001, A002, ...
# base_folder = 'H:/Output/binary_images_i_w_output'
#
# # Duyệt qua từng thư mục con (A001, A002, ...)
# for folder_name in os.listdir(base_folder):
#     folder_path = os.path.join(base_folder, folder_name)
#
#     # Kiểm tra xem thư mục có phải là thư mục con
#     if os.path.isdir(folder_path):
#         # Duyệt qua các tệp trong thư mục con
#         for filename in os.listdir(folder_path):
#             if filename.endswith("_matrix.npy"):  # Tìm các tệp có đuôi _matrix.npy
#                 matrix_path = os.path.join(folder_path, filename)
#
#                 # Tải ma trận ảnh từ tệp .npy
#                 binary_matrix = np.load(matrix_path)
#
#                 # In kích thước của ma trận
#                 print(f"Kích thước của ma trận ảnh {filename} trong thư mục {folder_name}: {binary_matrix.shape}")
#
#                 # In nội dung của ma trận
#                 print(f"Nội dung của ma trận ảnh {filename}:")
#                 print(binary_matrix)
