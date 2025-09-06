import os
import numpy as np
import matplotlib.pyplot as plt

# Thư mục chứa các file ma trận nhị phân
input_folder = r"H:\Output\binary_images_output"
output_folder = r"H:\Processed_Binary_Images"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua từng file trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):  # Kiểm tra file có định dạng .npy không
        input_path = os.path.join(input_folder, filename)

        # Đọc dữ liệu ma trận từ file .npy
        try:
            binary_images = np.load(input_path)  # Kích thước: (12, 5, 900, 600)
            print(f"Đang xử lý file: {filename}, kích thước: {binary_images.shape}")

            num_leads, num_beats, height, width = binary_images.shape

            # Duyệt qua từng đạo trình và từng nhịp
            for lead_idx in range(num_leads):  # 12 đạo trình
                for beat_idx in range(num_beats):  # 5 nhịp
                    binary_image = binary_images[lead_idx, beat_idx]  # Lấy từng ảnh nhị phân

                    # Vẽ ảnh
                    plt.figure(figsize=(6, 9))  # Kích thước khung ảnh
                    plt.imshow(binary_image, cmap='gray')  # Hiển thị ảnh nhị phân
                    plt.axis('off')  # Tắt trục

                    # Tên file lưu
                    output_file = os.path.join(
                        output_folder, f"{os.path.splitext(filename)[0]}_lead_{lead_idx + 1}_beat_{beat_idx + 1}.png"
                    )

                    # Lưu ảnh
                    plt.savefig(output_file, bbox_inches=None, pad_inches=0)
                    plt.close()
                    print(f"Đã lưu: {output_file}")
        except Exception as e:
            print(f"Lỗi khi xử lý file {filename}: {e}")
