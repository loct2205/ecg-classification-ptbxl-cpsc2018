import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torchvision.io import read_image
from tqdm import tqdm
from joblib import dump

# Đường dẫn dữ liệu
data_path = 'H:/Image_new'
csv_path = 'C:/Users/Admin/Downloads/LABELS - Copy.csv'

# Đọc nhãn từ file CSV
print("Đọc dữ liệu từ CSV...")
labels_df = pd.read_csv(csv_path)
labels_df = labels_df[['Recording', 'First_label']]  # Lấy nhãn chính, bỏ qua nhãn phụ
print(f"Số lượng mẫu: {len(labels_df)}")

# Dataset class để tải ảnh
class CardiacImageDataset(Dataset):
    def __init__(self, patient_ids, labels, img_dir, transform=None):
        self.patient_ids = patient_ids.tolist()
        self.labels = labels.tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]
        img_features = []

        # Lấy tất cả các tệp ảnh trong thư mục của bệnh nhân
        patient_folder = os.path.join(self.img_dir, f'{patient_id}')
        if not os.path.exists(patient_folder):
            print(f"Thư mục không tồn tại: {patient_folder}")
            raise FileNotFoundError(f"Không tìm thấy thư mục của bệnh nhân {patient_id}")

        # Lọc tất cả các tệp PNG trong thư mục bệnh nhân
        img_files = [f for f in os.listdir(patient_folder) if f.endswith('.png')]
        if not img_files:
            print(f"Không tìm thấy ảnh nào trong thư mục: {patient_folder}")
            raise FileNotFoundError(f"Không có ảnh trong thư mục {patient_folder}")

        img_files.sort()  # Nếu muốn sắp xếp theo tên tệp (ví dụ: từ 1.png đến n.png)
        #print(f"Bệnh nhân {patient_id} có {len(img_files)} ảnh")

        for img_file in img_files:
            img_path = os.path.join(patient_folder, img_file)
            image = read_image(img_path).float() / 255.0  # Chuẩn hóa ảnh

            if image.shape[0] == 4:
                image = image[:3, :, :]  # Nếu ảnh có 4 kênh, giữ lại 3 kênh (RGB)

            if self.transform:
                image = self.transform(image)

            img_features.append(image)

        return torch.stack(img_features), label

# Chuẩn bị ResNet-50 để trích xuất đặc trưng
print("Khởi tạo mô hình ResNet-50...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = resnet50(pretrained=True)
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp phân loại cuối
model.to(device)
model.eval()
print("Mô hình đã được khởi tạo thành công!")

# Hàm trích xuất đặc trưng từ ảnh
def extract_features(model, dataloader):
    print("Bắt đầu trích xuất đặc trưng...")
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Trich xuat dac trung"):
            images, label = batch
            batch_features = []

            # Lặp qua từng ảnh trong batch và trích xuất đặc trưng
            for seq_images in images:
                seq_features = []
                for img in seq_images:
                    img = img.unsqueeze(0).to(device)  # Thêm batch dimension
                    feature = model(img)
                    seq_features.append(feature.squeeze().cpu().numpy())

                batch_features.append(np.array(seq_features).flatten())

            features.extend(batch_features)
            labels.extend(label.tolist())


    print("Hoàn tất trích xuất đặc trưng.")
    return np.array(features), np.array(labels)

# Chia dữ liệu
print("Chia dữ liệu thành tập train, val và test...")
train_ids, test_ids, train_labels, test_labels = train_test_split(
    labels_df['Recording'], labels_df['First_label'], test_size=0.2, stratify=labels_df['First_label'], random_state=42
)
val_ids, test_ids, val_labels, test_labels = train_test_split(
    test_ids, test_labels, test_size=0.5, stratify=test_labels, random_state=42
)
print(f"Tập train: {len(train_ids)} mẫu, tập val: {len(val_ids)} mẫu, tập test: {len(test_ids)} mẫu.")

# Biến đổi ảnh
print("Chuẩn bị transform cho ảnh...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tạo DataLoader
print("Tạo DataLoader cho tập train...")
train_dataset = CardiacImageDataset(train_ids, train_labels, data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
print("Tạo DataLoader thành công!")

# Trích xuất đặc trưng và lưu file
print("Bat dau trich xuat dac trung tu tap train...")
train_features, train_labels = extract_features(model, train_loader)
print("Trich xuat thanh cong. Luu đac trung vao file...")
np.save('H:/_Project III/feature_extract/train_features.npy', train_features)
np.save('H:/_Project III/feature_extract/train_labels.npy', train_labels)
print("Dac trung da duoc luu.")

print("Tao DataLoader cho tap val...")
val_dataset = CardiacImageDataset(val_ids, val_labels, data_path, transform=transform)
val_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
print("Tao DataLoader thanh cong!")
val_features, val_labels = extract_features(model, train_loader)
print("Trich xuat thanh cong. Luu đac trung vao file...")
np.save('H:/_Project III/feature_extract/val_features.npy', val_features)
np.save('H:/_Project III/feature_extract/val_labels.npy', val_labels)
print("Dac trung da duoc luu.")

# Huấn luyện mô hình đơn giản trên đặc trưng
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000, verbose=1)  # Mặc định solver='lbfgs'
clf.fit(train_features, train_labels)

# Lưu mô hình sau khi huấn luyện
dump(clf, 'logistic_regression_model.joblib')
print("Mô hình đã được lưu tại 'logistic_regression_model.joblib'")

# Đánh giá trên tập kiểm thử
test_dataset = CardiacImageDataset(test_ids, test_labels, data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("Bat dau trich xuat dac trung tu tap test...")
test_features, test_labels = extract_features(model, test_loader)
np.save('H:/_Project III/feature_extract/test_features.npy', test_features)
np.save('H:/_Project III/feature_extract/test_labels.npy', test_labels)
print("Đac trung da duoc luu.")
predictions = clf.predict(test_features)

# Tính các chỉ số và vẽ confusion matrix
accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
roc_auc = roc_auc_score(test_labels, clf.predict_proba(test_features), multi_class='ovr')

print(f'Accuracy: {accuracy}')
print(f'AUC: {roc_auc}')
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_heatmap.png')
plt.show()

# Tính các chỉ số khác
print(classification_report(test_labels, predictions))