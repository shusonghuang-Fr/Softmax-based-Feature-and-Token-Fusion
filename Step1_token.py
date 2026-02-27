# ======  code smooth by ChatGpt


import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_model_data_config, create_transform
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)

createFolder(
    "./test3_BEIT_DFetAFFetSDC_finT_NoDecalage_5clas_train21-23_DF/")  #########for instance

# Define the root input and output directories
input_root_dir = "./test2/"
output_root_dir = "./test3_BEIT_DFetAFFetSDC_finT_NoDecalage_5clas_train21-23_DF/"

EPOCHS = 100


# === 1. Dataset ===
class NPZImageDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))  # if float32 in [0,1]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.y_data[idx], dtype=torch.float32)  # one-hot
        return image, label

# === 2. Load .npz data ===
###############RGB
# Save random multiple arrays into a single .npz file
Dir_x_train_RGB = os.path.join(output_root_dir, "x_train_RGB.npz")
Dir_x_test_RGB = os.path.join(output_root_dir, "x_test_RGB.npz")
Dir_x_test_2024_RGB = os.path.join(output_root_dir, "x_test_2024_RGB.npz")



# Load the random .npz file
x_train_RGB = ( np.load(Dir_x_train_RGB, mmap_mode="r"))["arr_0"]
x_test_RGB = ( np.load(Dir_x_test_RGB, mmap_mode="r"))["arr_0"]
x_test_2024_RGB = ( np.load(Dir_x_test_2024_RGB, mmap_mode="r"))["arr_0"]

# x_train_RGB = x_train_RGB[0:20 :, :, :]
# x_test_RGB = x_test_RGB[0:20 :, :, :]
# x_test_2024_RGB = x_test_2024_RGB[0:20 :, :, :]

###############RGB
# Save random multiple arrays into a single .npz file
Dir_y_train_RGB = os.path.join(output_root_dir, "y_train_RGB.npz")
Dir_y_test_RGB = os.path.join(output_root_dir, "y_test_RGB.npz")
Dir_y_test_2024_RGB = os.path.join(output_root_dir, "y_test_2024_RGB.npz")


# Load the random .npz file
y_train_RGB = ( np.load(Dir_y_train_RGB, mmap_mode="r"))["arr_0"]
y_test_RGB = ( np.load(Dir_y_test_RGB, mmap_mode="r"))["arr_0"]
y_test_2024_RGB = ( np.load(Dir_y_test_2024_RGB, mmap_mode="r"))["arr_0"]

# y_train_RGB = y_train_RGB[0:20 :]
# y_test_RGB = y_test_RGB[0:20 :]
# y_test_2024_RGB = y_test_2024_RGB[0:20 :]

# === 3. Model and transform ===
model = timm.create_model('beitv2_large_patch16_224.in1k_ft_in22k', pretrained=True)
model.head = nn.Linear(model.head.in_features, 5)
data_config = resolve_model_data_config(model)
transform_train = create_transform(**data_config, is_training=True)
transform_test = create_transform(**data_config, is_training=False)


# === 4. Dataloaders ===
train_dataset = NPZImageDataset(x_train_RGB, y_train_RGB, transform=transform_train)
test_dataset = NPZImageDataset(x_test_RGB, y_test_RGB, transform=transform_test)
test_2024_dataset = NPZImageDataset(x_test_2024_RGB, y_test_2024_RGB, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_2024_loader = DataLoader(test_2024_dataset, batch_size=32, shuffle=False)

# === 5. Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()  #
optimizer = optim.AdamW(model.parameters(), lr=2e-6)


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        labels = torch.argmax(labels, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # ===  Save model ===
        RGB_model_dir = os.path.join(output_root_dir, "model_5cla_RGB_BEITv2timm_21et22_transform_lr2e-6.pth")
        torch.save(model.state_dict(), RGB_model_dir)

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {running_loss / len(train_loader):.4f}", flush=True)

# === 6. Evaluation ===
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        # If your labels are one-hot encoded, convert to class indices
        if labels.ndim > 1:
            labels = torch.argmax(labels, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---------- Accuracy ----------
acc = accuracy_score(all_labels, all_preds)
print(f"\n Test Accuracy: {acc:.4f}", flush=True)

# ---------- Classification Report ----------
target_names = [f"Class {i}" for i in range(5)]
report = classification_report(all_labels, all_preds, target_names=target_names, digits=5)
print("\n Classification Report:\n", report, flush=True)

# ---------- Confusion Matrix ----------
conf_matrix = confusion_matrix(all_labels, all_preds)
conf_matrix = np.round(conf_matrix, 5)
print("\n Confusion Matrix:\n", conf_matrix, flush=True)

all_preds, all_labels = [], []

#################################test 2024
all_preds_2024, all_labels_2024 = [], []
with torch.no_grad():
    for images, labels in test_2024_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        # If your labels are one-hot encoded, convert to class indices
        if labels.ndim > 1:
            labels = torch.argmax(labels, dim=1)

        all_preds_2024.extend(preds.cpu().numpy())
        all_labels_2024.extend(labels.cpu().numpy())

# ---------- Accuracy ----------
acc_2024 = accuracy_score(all_labels_2024, all_preds_2024)
print(f"\n Test Accuracy: {acc_2024:.4f}", flush=True)

# ---------- Classification Report ----------
target_names = [f"Class {i}" for i in range(5)]
report_2024 = classification_report(all_labels_2024, all_preds_2024, target_names=target_names, digits=5)
print("\n Classification Report_2024:\n", report_2024, flush=True)

# ---------- Confusion Matrix ----------
conf_matrix_2024 = confusion_matrix(all_labels_2024, all_preds_2024)
conf_matrix_2024 = np.round(conf_matrix_2024, 5)
print("\n Confusion Matrix_2024:\n", conf_matrix_2024, flush=True)
all_preds_2024, all_labels_2024 = [], []

