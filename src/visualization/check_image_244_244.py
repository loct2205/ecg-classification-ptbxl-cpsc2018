

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Định nghĩa các phép biến đổi
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
])

# Tải ảnh
image_path = 'C:/Users/Admin/Desktop/A0001_1_4.png'
image = Image.open(image_path)

# Chuyển ảnh thành RGB nếu chưa phải là RGB
if image.mode != 'RGB':
    image = image.convert('RGB')

# Áp dụng các phép biến đổi
transformed_image = transform(image)

# Hiển thị ảnh gốc và ảnh đã biến đổi
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Hiển thị ảnh gốc
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

# Chuyển tensor về ảnh để hiển thị
transformed_image_show = transformed_image.permute(1, 2, 0).numpy()  # Chuyển từ (C, H, W) sang (H, W, C)
transformed_image_show = transformed_image_show * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Inverse normalize

# Hiển thị ảnh đã biến đổi
ax[1].imshow(transformed_image_show)
ax[1].set_title("Transformed Image")
ax[1].axis('off')

plt.show()

