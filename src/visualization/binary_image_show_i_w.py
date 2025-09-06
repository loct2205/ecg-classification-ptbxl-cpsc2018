import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image  # Import thư viện PIL để thay đổi kích thước ảnh

# Đường dẫn đến thư mục gốc chứa các tệp .npy
base_folder = 'H:/Output/binary_images_i_w_output'

# Thư mục đích nơi lưu các ảnh
output_folder = 'H:/Output/Processed_Binary_Images_i_w'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua các tệp trong thư mục gốc
for filename in os.listdir(base_folder):
    if filename.endswith(".npy"):  # Tìm các tệp có đuôi .npy
        matrix_path = os.path.join(base_folder, filename)

        # Tải ma trận ảnh từ tệp .npy
        binary_matrix = np.load(matrix_path)
        print(binary_matrix.shape)

        # Kiểm tra kích thước của ma trận
        if binary_matrix.shape == (12, 900, 600, 4):
            print(f"Ma trận ảnh {filename} có kích thước chính xác 12 x 900 x 600.")

            # Lưu các ảnh trong ma trận vào thư mục đích
            for i in range(binary_matrix.shape[0]):  # Duyệt qua từng ảnh trong ma trận
                img = binary_matrix[i]

                # Chuyển ma trận numpy thành ảnh và thay đổi kích thước
                img_pil = Image.fromarray(img.astype(np.uint8))  # Chuyển numpy array thành Image
                img_resized = img_pil.resize((900, 600))  # Thay đổi kích thước ảnh

                # Tạo tên tệp cho ảnh
                #img_filename = f"{filename[:-4]}_image_{i + 1}.png"
                img_filename = f"{filename[:-4]}_.png"

                # Đường dẫn lưu ảnh
                img_output_path = os.path.join(output_folder, img_filename)

                # Lưu ảnh đã thay đổi kích thước
                img_resized.save(img_output_path)

                print(f"Lưu ảnh {img_filename} tại: {img_output_path}")
        else:
            print(f"Ma trận ảnh {filename} có kích thước không chính xác!")





# check file con
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# # Đường dẫn đến thư mục gốc chứa các thư mục con A001, A002, ...
# base_folder = 'H:/Output/binary_images_i_w_output'
#
# # Thư mục đích nơi lưu các ảnh
# output_folder = 'H:/Output/Processed_Binary_Images_i_w'
#
# # Tạo thư mục đích nếu chưa tồn tại
# os.makedirs(output_folder, exist_ok=True)
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
#                 # Kiểm tra kích thước của ma trận
#                 if binary_matrix.shape == (12, 900, 600):
#                     print(f"Ma trận ảnh {filename} trong thư mục {folder_name} có kích thước chính xác 12 x 900 x 600.")
#
#                     # Lưu các ảnh trong ma trận vào thư mục đích
#                     for i in range(binary_matrix.shape[0]):  # Duyệt qua từng ảnh trong ma trận
#                         img = binary_matrix[i]
#
#                         # Tạo tên tệp cho ảnh
#                         img_filename = f"{folder_name}_{filename[:-4]}_image_{i + 1}.png"
#
#                         # Đường dẫn lưu ảnh
#                         img_output_path = os.path.join(output_folder, img_filename)
#
#                         # Lưu ảnh
#                         plt.imsave(img_output_path, img, cmap='gray')
#
#                         print(f"Lưu ảnh {img_filename} tại: {img_output_path}")
#                 else:
#                     print(f"Ma trận ảnh {filename} trong thư mục {folder_name} có kích thước không chính xác!")
