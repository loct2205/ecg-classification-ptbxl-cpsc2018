import os
from PIL import Image

# Đường dẫn tới thư mục chứa các thư mục con (A0001, A0002...)
base_folder = 'H:/Image'
output_folder = 'H:/Image_output_'
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

    # Duyệt qua từng img_idx và ghép các ảnh nếu có đủ 12 ảnh
    for img_idx, files in images_by_idx.items():
        if len(files) == 12:
            # Sắp xếp ảnh theo signal_idx để đảm bảo thứ tự
            files = sorted(files, key=lambda x: x[0])

            # Mở từng ảnh và lưu vào danh sách
            images = [Image.open(os.path.join(image_folder, file[1])) for file in files]

            # Ghép ảnh theo chiều ngang
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)

            merged_image = Image.new('RGB', (total_width, max_height))

            # Thêm từng ảnh vào vị trí thích hợp
            x_offset = 0
            for img in images:
                merged_image.paste(img, (x_offset, 0))
                x_offset += img.width

            # Lưu ảnh đã ghép
            output_path = os.path.join(output_folder, f"{subfolder}_{img_idx}.png")
            merged_image.save(output_path)
            print(f"Ảnh đã được ghép và lưu tại: {output_path}")

print("Hoàn thành ghép ảnh!")
