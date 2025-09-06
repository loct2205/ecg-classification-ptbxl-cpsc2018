import os
import numpy as np
from PIL import Image

# Đường dẫn tới thư mục chứa các thư mục con (A0001, A0002...)
base_folder = 'H:/Output/images_i_w_output'
output_folder = 'H:/Output/binary_images_i_w_output'
os.makedirs(output_folder, exist_ok=True)

# Lấy danh sách các thư mục con A0001, A0002, ...
subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

# Duyệt qua từng thư mục con
for subfolder in subfolders:
    image_folder = os.path.join(base_folder, subfolder)
    print(f"Đang xử lý thư mục: {image_folder}")

    # Lấy danh sách ảnh trong thư mục con
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Phân tích các ảnh theo định dạng {name}_{signal_idx + 1}_{img_idx + 1}
    images_by_idx = {}
    for file in image_files:
        parts = file.split('_')
        if len(parts) == 3 and parts[2].endswith('.png'):
            name = parts[0]
            signal_idx = int(parts[1])
            img_idx = int(parts[2].replace('.png', ''))

            if img_idx not in images_by_idx:
                images_by_idx[img_idx] = []
            images_by_idx[img_idx].append((signal_idx, file))

    # Duyệt qua từng img_idx và xử lý nếu có đủ 12 ảnh
    for img_idx, files in images_by_idx.items():
        if len(files) == 12:
            # Sắp xếp ảnh theo signal_idx để đảm bảo thứ tự
            files = sorted(files, key=lambda x: x[0])

            # Mở từng ảnh và lưu vào danh sách
            images = []
            for file in files:
                img = Image.open(os.path.join(image_folder, file[1]))

                # Kiểm tra và thay đổi kích thước ảnh về 900x600 nếu cần
                img = img.resize((900, 600))  # Chỉnh kích thước ảnh về 900x600

                # Chuyển ảnh thành mảng numpy
                images.append(np.array(img))

            # Chuyển danh sách ảnh thành ma trận numpy (12 x 900 x 600)
            images_array = np.array(images)  # Kích thước sẽ là (12, 900, 600)

            # Lưu ma trận numpy
            output_path = os.path.join(output_folder, f"{subfolder}_{img_idx}.npy")
            np.save(output_path, images_array)
            print(f"Ma trận đã được lưu tại: {output_path}")

print("Hoàn thành chuyển đổi và lưu ma trận!")
