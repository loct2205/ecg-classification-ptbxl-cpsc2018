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
from sklearn.linear_model import LogisticRegression



data_path = 'H:/Image_new'
csv_path = 'C:/Users/Admin/Downloads/LABELS - Copy.csv'


print("Doc du lieu tu CSV...")
labels_df = pd.read_csv(csv_path)
labels_df = labels_df[['Recording', 'First_label']]
print(f"So luong mau: {len(labels_df)}")

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


        patient_folder = os.path.join(self.img_dir, f'{patient_id}')
        print(patient_folder)
        if not os.path.exists(patient_folder):
            print(f"Thu muc khong ton tai: {patient_folder}")
            raise FileNotFoundError(f"Khong tim thay thu muc cua benh nhan {patient_id}")

        img_files = [f for f in os.listdir(patient_folder) if f.endswith('.png')]
        if not img_files:
            print(f"Khong tim thay anh trong thu muc: {patient_folder}")
            raise FileNotFoundError(f"Khong co anh trong thu muc {patient_folder}")

        img_files.sort()


        for img_file in img_files:
            img_path = os.path.join(patient_folder, img_file)
            image = read_image(img_path).float() / 255.0

            if image.shape[0] == 4:
                image = image[:3, :, :]

            if self.transform:
                image = self.transform(image)

            img_features.append(image)

            # Dem tensor dam bao so luong anh co dinh
            max_images = 7818 # lam tran bo nho vi batch
            if len(img_features) < max_images:
                pad = max_images - len(img_features)
                padding = [torch.zeros_like(img_features[0]) for _ in range(pad)]
                img_features.extend(padding)

            # Chi lay toi da max_images
            img_features = img_features[:max_images]

        return torch.stack(img_features), label


print("Khoi tao mo hinh ResNet-50...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()
print("Mo hinh da duoc khoi tao thanh cong!")

def extract_features(model, dataloader):
    print("Bat dau trich xuat dac trung...")
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Trich xuat dac trung"):
            images, label = batch
            batch_features = []

            for seq_images in images:
                seq_features = []
                for img in seq_images:
                    img = img.unsqueeze(0).to(device)
                    feature = model(img)
                    seq_features.append(feature.squeeze().cpu().numpy())

                batch_features.append(np.array(seq_features).flatten())

            features.extend(batch_features)
            labels.extend(label.tolist())


    print("Hoan tat trich xuat dac trung.")
    return np.array(features), np.array(labels)

print("Chia du lieu thanh train, val va test...")
train_ids, test_ids, train_labels, test_labels = train_test_split(
    labels_df['Recording'], labels_df['First_label'], test_size=0.2, stratify=labels_df['First_label'], random_state=42
)
val_ids, test_ids, val_labels, test_labels = train_test_split(
    test_ids, test_labels, test_size=0.5, stratify=test_labels, random_state=42
)
print(f"Tap train: {len(train_ids)} mau, tap val: {len(val_ids)} mau, tap test: {len(test_ids)} mau.")
print(train_ids)


print("Chuan bi transform cho anh...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("Tao DataLoader cho tap train...")
train_dataset = CardiacImageDataset(train_ids, train_labels, data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
print("Tao DataLoader thanh cong!")


print("Bat dau trich xuat dac trung tu tap train...")
train_features, train_labels = extract_features(model, train_loader)
print("Trich xuat thanh cong. Luu đac trung vao file...")
np.save('./train_features.npy', train_features)
np.save('./train_labels.npy', train_labels)
print("Dac trung da duoc luu.")

print("Tao DataLoader cho tap val...")
val_dataset = CardiacImageDataset(val_ids, val_labels, data_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
print("Tao DataLoader thanh cong!")
val_features, val_labels = extract_features(model, train_loader)
print("Trich xuat thanh cong. Luu đac trung vao file...")
np.save('./val_features.npy', val_features)
np.save('./val_labels.npy', val_labels)
print("Dac trung da duoc luu.")

clf = LogisticRegression(max_iter=1000, verbose=1)
clf.fit(train_features, train_labels)


dump(clf, './logistic_regression_model.joblib')
print("Mo hinh da duoc luu tai 'logistic_regression_model.joblib'")


test_dataset = CardiacImageDataset(test_ids, test_labels, data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("Bat dau trich xuat dac trung tu tap test...")
test_features, test_labels = extract_features(model, test_loader)
np.save('./test_features.npy', test_features)
np.save('./test_labels.npy', test_labels)
print("Dac trung da duoc luu.")
predictions = clf.predict(test_features)


accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
roc_auc = roc_auc_score(test_labels, clf.predict_proba(test_features), multi_class='ovr')

print(f'Accuracy: {accuracy}')
print(f'AUC: {roc_auc}')
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('./confusion_matrix_heatmap.png')


print(classification_report(test_labels, predictions))