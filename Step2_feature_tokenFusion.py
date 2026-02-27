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
import time
start_time = time.time()
import torch.nn.functional as F
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

# image_shape1 = 224
# image_shape2 = 224
# EPOCHS = 100
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cut_layer = 23  # Choose a layer to cut at 6-32
num_epochs = 100

############################################################################
# --- Dataset ---
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

class NPZFeatureDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y = torch.tensor(self.y_data[idx], dtype=torch.float32)  # one-hot or scalar
        return x, y

#########load dataset
###############RGB
# Save random multiple arrays into a single .npz file
Dir_x_train_RGB = os.path.join(output_root_dir, "x_train_RGB.npz")
Dir_x_test_RGB = os.path.join(output_root_dir, "x_test_RGB.npz")
# Dir_x_test_2023_RGB = os.path.join(output_root_dir, "x_test_2023_RGB.npz")
Dir_x_test_2024_RGB = os.path.join(output_root_dir, "x_test_2024_RGB.npz")

# Load the random .npz file
x_train_RGB = ( np.load(Dir_x_train_RGB, mmap_mode='r'))["arr_0"]
x_test_RGB = ( np.load(Dir_x_test_RGB, mmap_mode='r'))["arr_0"]
# x_test_2023_RGB = ( np.load(Dir_x_test_2023_RGB, mmap_mode='r'))["arr_0"]
x_test_2024_RGB = ( np.load(Dir_x_test_2024_RGB, mmap_mode='r'))["arr_0"]

###############RGB
# Save random multiple arrays into a single .npz file
Dir_y_train_RGB = os.path.join(output_root_dir, "y_train_RGB.npz")
Dir_y_test_RGB = os.path.join(output_root_dir, "y_test_RGB.npz")
# Dir_y_test_2023_RGB = os.path.join(output_root_dir, "y_test_2023_RGB.npz")
Dir_y_test_2024_RGB = os.path.join(output_root_dir, "y_test_2024_RGB.npz")

# Load the random .npz file
y_train_RGB = ( np.load(Dir_y_train_RGB, mmap_mode='r'))["arr_0"]
y_test_RGB = ( np.load(Dir_y_test_RGB, mmap_mode='r'))["arr_0"]
# y_test_2023_RGB = ( np.load(Dir_y_test_2023_RGB, mmap_mode='r'))["arr_0"]
y_test_2024_RGB = ( np.load(Dir_y_test_2024_RGB, mmap_mode='r'))["arr_0"]




######### load feature RestNet50
Dir_feature_maps_pré_RGB = os.path.join(output_root_dir, "feature_maps_pré_RGB_32.npz")
feature_maps_pré_RGB1 = (np.load(Dir_feature_maps_pré_RGB, mmap_mode='r')).keys()
num_batches = int(len(feature_maps_pré_RGB1))
feature_maps_pré_RGB1 = []

feature_maps_train_RGB = []

for i in range(num_batches):
    Dir_feature_maps_pré_RGB = os.path.join(output_root_dir, "feature_maps_pré_RGB_32.npz")
    key = "feature_maps_batch_" + str(i)
    feature_maps_pré_RGB = (np.load(Dir_feature_maps_pré_RGB, mmap_mode='r'))[key]

    np_maps_RGB = np.array(feature_maps_pré_RGB)
    np_maps_RGB = np.array(np_maps_RGB).reshape(len(np_maps_RGB), 56, 56, 256)

    list_maps_multi_ini = np_maps_RGB
    feature_maps_train_RGB.append(list_maps_multi_ini)
    np_maps_RGB = []
    list_maps_multi_ini = []

x_feature_maps_train_RGB = np.concatenate(feature_maps_train_RGB, axis=0)
feature_maps_train_RGB = []

###load 21-23 features
reshape_array_test = []
Dir_feature_maps_test_RGB = os.path.join(output_root_dir, "feature_maps_test_RGB_32.npz")
feature = np.load(Dir_feature_maps_test_RGB)
#############################################################test donnnes
for key in feature.keys():

    Dir_feature_maps_test_RGB = os.path.join(output_root_dir, "feature_maps_test_RGB_32.npz")

    feature_maps_test_RGB = (np.load(Dir_feature_maps_test_RGB, mmap_mode='r'))[key]
    list_maps_RGB = (feature_maps_test_RGB)
    feature_maps_test_RGB = []

    np_maps_RGB = np.array(list_maps_RGB)
    list_maps_RGB = []

    reshape_array_test_1 = np.array(np_maps_RGB).reshape(len(np_maps_RGB), 56, 56, 256)  # 256 is 32 compo*3  #for example len(x_test_RGBdedge)=len(feature_maps_test_RGB)=
    reshape_array_test.append(reshape_array_test_1)
    reshape_array_test_1 = []

    # pred = cut_model_multi_champs.predict(reshape_array_test)
x_feature_maps_test_RGB = np.concatenate(reshape_array_test, axis=0)
reshape_array_test = []

###load 24 features
reshape_array_test_2024 = []
Dir_feature_maps_test_2024_RGB = os.path.join(output_root_dir, "feature_maps_test_2024_RGB_32.npz")
feature = np.load(Dir_feature_maps_test_2024_RGB)
#############################################################test donnnes
for key in feature.keys():

    Dir_feature_maps_test_2024_RGB = os.path.join(output_root_dir, "feature_maps_test_2024_RGB_32.npz")

    feature_maps_test_2024_RGB = (np.load(Dir_feature_maps_test_2024_RGB, mmap_mode='r'))[key]
    list_maps_2024_RGB = (feature_maps_test_2024_RGB)
    feature_maps_test_2024_RGB = []

    np_maps_2024_RGB = np.array(list_maps_2024_RGB)
    list_maps_2024_RGB = []

    reshape_array_test_2024_1 = np.array(np_maps_2024_RGB).reshape(len(np_maps_2024_RGB), 56, 56, 256)  # 256 is 32 compo*3  #for example len(x_test_RGBdedge)=len(feature_maps_test_2024_RGB)=
    reshape_array_test_2024.append(reshape_array_test_2024_1)
    reshape_array_test_2024_1 = []

    # pred = cut_model_multi_champs.predict(reshape_array_test)
x_feature_maps_test_2024_RGB = np.concatenate(reshape_array_test_2024, axis=0)
reshape_array_test_2024 = []


#############use conv to (N, 1024, 14, 14)
# NHWC → NCHW
x_feature_maps_train_RGB = np.transpose(x_feature_maps_train_RGB, (0, 3, 1, 2))  # (N, C, H, W)
# To tensor
x_tensor = torch.tensor(x_feature_maps_train_RGB, dtype=torch.float32)
# Conv to (N, 1024, 14, 14)
conv = nn.Conv2d(256, 1024, kernel_size=4, stride=4)
x_feature_maps_train_RGB = conv(x_tensor)  # output shape: (N, 1024, 14, 14)

# NHWC → NCHW
x_feature_maps_test_RGB = np.transpose(x_feature_maps_test_RGB, (0, 3, 1, 2))  # (N, C, H, W)
# To tensor
x_tensor = torch.tensor(x_feature_maps_test_RGB, dtype=torch.float32)
# Conv to (N, 1024, 14, 14)
conv = nn.Conv2d(256, 1024, kernel_size=4, stride=4)
x_feature_maps_test_RGB = conv(x_tensor)  # output shape: (N, 1024, 14, 14)

# NHWC → NCHW
x_feature_maps_test_2024_RGB = np.transpose(x_feature_maps_test_2024_RGB, (0, 3, 1, 2))  # (N, C, H, W)
# To tensor
x_tensor = torch.tensor(x_feature_maps_test_2024_RGB, dtype=torch.float32)
# Conv to (N, 1024, 14, 14)
conv = nn.Conv2d(256, 1024, kernel_size=4, stride=4)
x_feature_maps_test_2024_RGB = conv(x_tensor)  # output shape: (N, 1024, 14, 14)



###########################RGB
# ===  load transform for dataset ===
model_ini = timm.create_model('beitv2_large_patch16_224.in1k_ft_in22k', pretrained=True)
model_ini.head = nn.Linear(model_ini.head.in_features, 5)
data_config = resolve_model_data_config(model_ini)
transform_train = create_transform(**data_config, is_training=True)
transform_test = create_transform(**data_config, is_training=False)

# --- Dataloaders ---
train_dataset_RGB = NPZImageDataset(x_train_RGB, y_train_RGB, transform=transform_train)
test_dataset_RGB = NPZImageDataset(x_test_RGB, y_test_RGB, transform=transform_test)
# train_loader_RGB = DataLoader(train_dataset_RGB, batch_size=batch_size, shuffle=False)
train_loader_RGB = DataLoader(train_dataset_RGB, batch_size=batch_size, shuffle=False) #Without shuffling: The data is loaded in the exact same order every time. This can lead to overfitting, especially if your data has some order or grouping (e.g., all class 0 samples, then class 1).
test_loader_RGB = DataLoader(test_dataset_RGB, batch_size=batch_size, shuffle=False)
test_2024_dataset_RGB = NPZImageDataset(x_test_2024_RGB, y_test_2024_RGB, transform=transform_test)
test_2024_loader_RGB = DataLoader(test_2024_dataset_RGB, batch_size=batch_size, shuffle=False)

train_feature_RGB = NPZFeatureDataset(x_feature_maps_train_RGB, y_train_RGB)
test_feature_RGB = NPZFeatureDataset(x_feature_maps_test_RGB, y_test_RGB)
# train_loader_RGB = DataLoader(train_feature_RGB, batch_size=batch_size, shuffle=False)
train_loader_feature_RGB = DataLoader(train_feature_RGB, batch_size=batch_size, shuffle=False)
test_loader_feature_RGB = DataLoader(test_feature_RGB, batch_size=batch_size, shuffle=False)
test_2024_feature_RGB = NPZFeatureDataset(x_feature_maps_test_2024_RGB, y_test_2024_RGB)
test_2024_loader_feature_RGB = DataLoader(test_2024_feature_RGB, batch_size=batch_size, shuffle=False)





# 1. Load Pretrained Models

model_RGB = timm.create_model('beitv2_large_patch16_224.in1k_ft_in22k', pretrained=True)
model_RGB.head = nn.Linear(model_RGB.head.in_features, 5)
RGB_model_dir = os.path.join(output_root_dir, "model_5cla_RGB_BEITv2timm_21et22_transform_lr2e-5.pth")
state_dict = torch.load(RGB_model_dir, map_location='cuda')  # or 'cuda'
model_RGB.load_state_dict(state_dict)


model_RGB = model_RGB.to(device)

###############################




# 2. Freeze full models
for model in [model_RGB]:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False



# Define the cut model (the remaining part of BEiT v2 after the decoder)
class CutBEiTModel(nn.Module):
    def __init__(self, input_channels, num_classes, scale=10.0):
        super(CutBEiTModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.scale = scale  # fixed scale; you can also make this learnable with nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Normalize and scale features before classification
        x = F.normalize(x, p=2, dim=1) * self.scale  # [B, 512]

        x = self.fc(x)  # [B, num_classes]
        return x

# Initialize the cut model
cut_model = CutBEiTModel(input_channels=1024, num_classes=5)  # Adjust input_channels as needed
cut_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cut_model.parameters(), lr=1e-3)

# Scheduler: decrease LR by 0.5 every 30 epochs or others
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


def softmax_fusion_group(tensor_group):
    """
    Softmax fusion of a list of tensors.
    Each tensor must have shape [B, C, H, W].
    """
    stacked = torch.stack(tensor_group, dim=1)  # [B, N, C, H, W]
    scores = stacked.mean(dim=[2, 3, 4])  # [B, N]
    weights = F.softmax(scores, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1, 1]
    fused = (stacked * weights).sum(dim=1)  # [B, C, H, W]
    return fused


def generate_combinations(features, depth, start=0, current=None, result=None):
    """
    Recursively generate combinations of `depth` elements from `features`.
    """
    if current is None: current = []
    if result is None: result = []

    if len(current) == depth:
        result.append(current[:])
        return

    for i in range(start, len(features)):
        current.append(features[i])
        generate_combinations(features, depth, i + 1, current, result)
        current.pop()

    return result

# Training loop
# num_epochs = 20  # Set the number of epochs as needed
for epoch in range(num_epochs):
    cut_model.train()
    total_loss = 0
    total_samples = 0

    for (batch9, batch10) in zip( train_loader_RGB, train_loader_feature_RGB):

        x9, y9 = batch9              # From model_Green_Red_Rededge input
        feature10, y10 = batch10        # Precomputed feature10 and label

        x9, y9 = x9.to(device), y9.to(device)
        feature10, y10 = feature10.to(device), y10.to(device)


        # assert all(torch.equal(y1, y) for y in [y2, y3, y4, y5, y6, y7, y8, y9, y10]), "Mismatch in labels"

        with torch.no_grad():
            # tokens = model_Green_Red_Rededge.forward_features(x1)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token1 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Green.forward_features(x3)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token3 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Red.forward_features(x5)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token5 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Rededge.forward_features(x7)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token7 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            tokens = model_RGB.forward_features(x9)  # [B, seq_len, hidden_dim]

            # Remove CLS token if present
            if tokens.size(1) in [197, 577]:
                tokens = tokens[:, 1:, :]

            B, seq_len, hidden_dim = tokens.shape
            H = W = int(seq_len ** 0.5)
            assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # [B, C, H, W]
            token9 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

        # === Normalize input tensors ===
        features = [
            # F.normalize(feature2, p=2, dim=1),
            # F.normalize(feature4, p=2, dim=1),
            # F.normalize(feature6, p=2, dim=1),
            # F.normalize(feature8, p=2, dim=1),
            F.normalize(feature10, p=2, dim=1),
            # F.normalize(token1, p=2, dim=1),
            # F.normalize(token3, p=2, dim=1),
            # F.normalize(token5, p=2, dim=1),
            # F.normalize(token7, p=2, dim=1),
            F.normalize(token9, p=2, dim=1),
        ]

        # === Generate fused combinations and compute logits ===
        logits_list = []

        N = len(features)  # Assume len(features) == 10
        # Group size 2 to 10
        for i in range(N):
            for j in range(i + 1, N):
                # group size 2
                group = [features[i], features[j]]
                fused = softmax_fusion_group(group)
                logits = cut_model(fused)
                logits_list.append(logits)

                # for k in range(j + 1, N):
                #     # group size 3
                #     group = [features[i], features[j], features[k]]
                #     fused = softmax_fusion_group(group)
                #     logits = cut_model(fused)
                #     logits_list.append(logits)

                #     for l in range(k + 1, N):
                #         # group size 4
                #         group = [features[i], features[j], features[k], features[l]]
                #         fused = softmax_fusion_group(group)
                #         logits = cut_model(fused)
                #         logits_list.append(logits)

                #         for m in range(l + 1, N):
                #             # group size 5
                #             group = [features[i], features[j], features[k], features[l], features[m]]
                #             fused = softmax_fusion_group(group)
                #             logits = cut_model(fused)
                #             logits_list.append(logits)

                #             for n in range(m + 1, N):
                #                 # group size 6
                #                 group = [features[i], features[j], features[k], features[l], features[m], features[n]]
                #                 fused = softmax_fusion_group(group)
                #                 logits = cut_model(fused)
                #                 logits_list.append(logits)

                #                 for o in range(n + 1, N):
                #                     # group size 7
                #                     group = [features[i], features[j], features[k], features[l], features[m],
                #                               features[n], features[o]]
                #                     fused = softmax_fusion_group(group)
                #                     logits = cut_model(fused)
                #                     logits_list.append(logits)

                #                     for p in range(o + 1, N):
                #                         # group size 8
                #                         group = [features[i], features[j], features[k], features[l], features[m],
                #                                   features[n], features[o], features[p]]
                #                         fused = softmax_fusion_group(group)
                #                         logits = cut_model(fused)
                #                         logits_list.append(logits)

                #                         for q in range(p + 1, N):
                #                             # group size 9
                #                             group = [features[i], features[j], features[k], features[l], features[m],
                #                                       features[n], features[o], features[p], features[q]]
                #                             fused = softmax_fusion_group(group)
                #                             logits = cut_model(fused)
                #                             logits_list.append(logits)

                #                             for r in range(q + 1, N):
                #                                 # group size 10
                #                                 group = [features[i], features[j], features[k], features[l],
                #                                           features[m], features[n], features[o], features[p],
                #                                           features[q], features[r]]
                #                                 fused = softmax_fusion_group(group)
                #                                 logits = cut_model(fused)
                #                                 logits_list.append(logits)

        # Average probabilities (soft voting)
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # [B, num_classes]

        # Convert to "pseudo-logits" using log
        final_logits = torch.log(avg_probs + 1e-8)  # Add epsilon to avoid log(0)

        # === Compute loss and update ===
        loss = criterion(final_logits, y9)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x9.size(0)
        total_samples += x9.size(0)

        FeatureFusion_CUT_dir = os.path.join(output_root_dir, "Head_5cla_5banVRREetRGB_FeaFus_BEITv2etResN_45SoftetDeciLevelSoftVote_Nor_Sca_2123_TF_lrDec_PerCom.pth")
        torch.save(cut_model.state_dict(), FeatureFusion_CUT_dir)

        state_dict1 = torch.load(FeatureFusion_CUT_dir, map_location='cuda')  # or 'cuda'
        cut_model.load_state_dict(state_dict, strict=False)



    avg_loss = total_loss / total_samples
    # Update LR at end of epoch
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}", flush=True)

# Evaluation
cut_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    # for batches in zip(test_loader_Green_Red_Rededge, test_loader_Red, test_loader_Rededge):
    for (batch9, batch10) in zip(
        test_loader_RGB, test_loader_feature_RGB):
        # x1, y1 = batch1  # From model_Green_Red_Rededge input
        # feature2, y2 = batch2  # Precomputed feature2 and label

        # x1, y1 = x1.to(device), y1.to(device)
        # feature2, y2 = feature2.to(device), y2.to(device)

        # x3, y3 = batch3  # From model_Green_Red_Rededge input
        # feature4, y4 = batch4  # Precomputed feature4 and label

        # x3, y3 = x3.to(device), y3.to(device)
        # feature4, y4 = feature4.to(device), y4.to(device)

        # x5, y5 = batch5  # From model_Green_Red_Rededge input
        # feature6, y6 = batch6  # Precomputed feature6 and label

        # x5, y5 = x5.to(device), y5.to(device)
        # feature6, y6 = feature6.to(device), y6.to(device)

        # x7, y7 = batch7  # From model_Green_Red_Rededge input
        # feature8, y8 = batch8  # Precomputed feature8 and label

        # x7, y7 = x7.to(device), y7.to(device)
        # feature8, y8 = feature8.to(device), y8.to(device)

        x9, y9 = batch9  # From model_Green_Red_Rededge input
        feature10, y10 = batch10  # Precomputed feature10 and label

        x9, y9 = x9.to(device), y9.to(device)
        feature10, y10 = feature10.to(device), y10.to(device)

        # assert all(torch.equal(y1, y) for y in [y2, y3, y4, y5, y6, y7, y8, y9, y10]), "Mismatch in labels"


        with torch.no_grad():
            # tokens = model_Green_Red_Rededge.forward_features(x1)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token1 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Green.forward_features(x3)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token3 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Red.forward_features(x5)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token5 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Rededge.forward_features(x7)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token7 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            tokens = model_RGB.forward_features(x9)  # [B, seq_len, hidden_dim]

            # Remove CLS token if present
            if tokens.size(1) in [197, 577]:
                tokens = tokens[:, 1:, :]

            B, seq_len, hidden_dim = tokens.shape
            H = W = int(seq_len ** 0.5)
            assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # [B, C, H, W]
            token9 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

        # === Normalize input tensors ===
        features = [
            # F.normalize(feature2, p=2, dim=1),
            # F.normalize(feature4, p=2, dim=1),
            # F.normalize(feature6, p=2, dim=1),
            # F.normalize(feature8, p=2, dim=1),
            F.normalize(feature10, p=2, dim=1),
            # F.normalize(token1, p=2, dim=1),
            # F.normalize(token3, p=2, dim=1),
            # F.normalize(token5, p=2, dim=1),
            # F.normalize(token7, p=2, dim=1),
            F.normalize(token9, p=2, dim=1),
        ]

        # === Generate fused combinations and compute logits ===
        logits_list = []

        N = len(features)  # Assume len(features) == 10
        # Group size 2 to 10
        for i in range(N):
            for j in range(i + 1, N):
                # group size 2
                group = [features[i], features[j]]
                fused = softmax_fusion_group(group)
                logits = cut_model(fused)
                logits_list.append(logits)

                # for k in range(j + 1, N):
                #     # group size 3
                #     group = [features[i], features[j], features[k]]
                #     fused = softmax_fusion_group(group)
                #     logits = cut_model(fused)
                #     logits_list.append(logits)

                #     for l in range(k + 1, N):
                #         # group size 4
                #         group = [features[i], features[j], features[k], features[l]]
                #         fused = softmax_fusion_group(group)
                #         logits = cut_model(fused)
                #         logits_list.append(logits)

                #         for m in range(l + 1, N):
                #             # group size 5
                #             group = [features[i], features[j], features[k], features[l], features[m]]
                #             fused = softmax_fusion_group(group)
                #             logits = cut_model(fused)
                #             logits_list.append(logits)

                #             for n in range(m + 1, N):
                #                 # group size 6
                #                 group = [features[i], features[j], features[k], features[l], features[m], features[n]]
                #                 fused = softmax_fusion_group(group)
                #                 logits = cut_model(fused)
                #                 logits_list.append(logits)

                #                 for o in range(n + 1, N):
                #                     # group size 7
                #                     group = [features[i], features[j], features[k], features[l], features[m],
                #                              features[n], features[o]]
                #                     fused = softmax_fusion_group(group)
                #                     logits = cut_model(fused)
                #                     logits_list.append(logits)

                #                     for p in range(o + 1, N):
                #                         # group size 8
                #                         group = [features[i], features[j], features[k], features[l], features[m],
                #                                  features[n], features[o], features[p]]
                #                         fused = softmax_fusion_group(group)
                #                         logits = cut_model(fused)
                #                         logits_list.append(logits)

                #                         for q in range(p + 1, N):
                #                             # group size 9
                #                             group = [features[i], features[j], features[k], features[l], features[m],
                #                                      features[n], features[o], features[p], features[q]]
                #                             fused = softmax_fusion_group(group)
                #                             logits = cut_model(fused)
                #                             logits_list.append(logits)

                #                             for r in range(q + 1, N):
                #                                 # group size 10
                #                                 group = [features[i], features[j], features[k], features[l],
                #                                          features[m], features[n], features[o], features[p],
                #                                          features[q], features[r]]
                #                                 fused = softmax_fusion_group(group)
                #                                 logits = cut_model(fused)
                #                                 logits_list.append(logits)

        # Average probabilities (soft voting)
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # [B, num_classes]

        # Convert to "pseudo-logits" using log
        final_logits = torch.log(avg_probs + 1e-8)  # Add epsilon to avoid log(0)
        _, preds = torch.max(final_logits, 1)

        y9 = y9.cpu().numpy()
        y9 = np.argmax(y9, axis=1) if len(y9.shape) > 1 else y9

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y9)

# Compute evaluation metrics
acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc:.4f}", flush=True)

target_names = [f"Class {i}" for i in range(5)]
report = classification_report(all_labels, all_preds, target_names=target_names, digits=5)
print("\nClassification Report:\n", report, flush=True)

conf_matrix = confusion_matrix(all_labels, all_preds)
conf_matrix = np.round(conf_matrix, 5)
print("\nConfusion Matrix:\n", conf_matrix, flush=True)


# === 6. Evaluation ===
# Evaluation 2 #
all_preds = []
all_labels = []

with torch.no_grad():
    # for batches in zip(test_loader_Green_Red_Rededge, test_loader_Red, test_loader_Rededge):
    for (batch9, batch10) in zip(
        test_2024_loader_RGB, test_2024_loader_feature_RGB):
        # x1, y1 = batch1  # From model_Green_Red_Rededge input
        # feature2, y2 = batch2  # Precomputed feature2 and label

        # x1, y1 = x1.to(device), y1.to(device)
        # feature2, y2 = feature2.to(device), y2.to(device)

        # x3, y3 = batch3  # From model_Green_Red_Rededge input
        # feature4, y4 = batch4  # Precomputed feature4 and label

        # x3, y3 = x3.to(device), y3.to(device)
        # feature4, y4 = feature4.to(device), y4.to(device)

        # x5, y5 = batch5  # From model_Green_Red_Rededge input
        # feature6, y6 = batch6  # Precomputed feature6 and label

        # x5, y5 = x5.to(device), y5.to(device)
        # feature6, y6 = feature6.to(device), y6.to(device)

        # x7, y7 = batch7  # From model_Green_Red_Rededge input
        # feature8, y8 = batch8  # Precomputed feature8 and label

        # x7, y7 = x7.to(device), y7.to(device)
        # feature8, y8 = feature8.to(device), y8.to(device)

        x9, y9 = batch9  # From model_Green_Red_Rededge input
        feature10, y10 = batch10  # Precomputed feature10 and label

        x9, y9 = x9.to(device), y9.to(device)
        feature10, y10 = feature10.to(device), y10.to(device)

        # assert all(torch.equal(y1, y) for y in [y2, y3, y4, y5, y6, y7, y8, y9, y10]), "Mismatch in labels"


        with torch.no_grad():
            # tokens = model_Green_Red_Rededge.forward_features(x1)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token1 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Green.forward_features(x3)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token3 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Red.forward_features(x5)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token5 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            # tokens = model_Rededge.forward_features(x7)  # [B, seq_len, hidden_dim]

            # # Remove CLS token if present
            # if tokens.size(1) in [197, 577]:
            #     tokens = tokens[:, 1:, :]

            # B, seq_len, hidden_dim = tokens.shape
            # H = W = int(seq_len ** 0.5)
            # assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # # [B, C, H, W]
            # token7 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

            tokens = model_RGB.forward_features(x9)  # [B, seq_len, hidden_dim]

            # Remove CLS token if present
            if tokens.size(1) in [197, 577]:
                tokens = tokens[:, 1:, :]

            B, seq_len, hidden_dim = tokens.shape
            H = W = int(seq_len ** 0.5)
            assert H * W == seq_len, f"Cannot reshape sequence length {seq_len} into square."

            # [B, C, H, W]
            token9 = tokens.permute(0, 2, 1).contiguous().view(B, hidden_dim, H, W)

        # === Normalize input tensors ===
        features = [
            # F.normalize(feature2, p=2, dim=1),
            # F.normalize(feature4, p=2, dim=1),
            # F.normalize(feature6, p=2, dim=1),
            # F.normalize(feature8, p=2, dim=1),
            F.normalize(feature10, p=2, dim=1),
            # F.normalize(token1, p=2, dim=1),
            # F.normalize(token3, p=2, dim=1),
            # F.normalize(token5, p=2, dim=1),
            # F.normalize(token7, p=2, dim=1),
            F.normalize(token9, p=2, dim=1),
        ]

        # === Generate fused combinations and compute logits ===
        logits_list = []

        N = len(features)  # Assume len(features) == 10
        # Group size 2 to 10
        for i in range(N):
            for j in range(i + 1, N):
                # group size 2
                group = [features[i], features[j]]
                fused = softmax_fusion_group(group)
                logits = cut_model(fused)
                logits_list.append(logits)

                # for k in range(j + 1, N):
                #     # group size 3
                #     group = [features[i], features[j], features[k]]
                #     fused = softmax_fusion_group(group)
                #     logits = cut_model(fused)
                #     logits_list.append(logits)

                #     for l in range(k + 1, N):
                #         # group size 4
                #         group = [features[i], features[j], features[k], features[l]]
                #         fused = softmax_fusion_group(group)
                #         logits = cut_model(fused)
                #         logits_list.append(logits)

                #         for m in range(l + 1, N):
                #             # group size 5
                #             group = [features[i], features[j], features[k], features[l], features[m]]
                #             fused = softmax_fusion_group(group)
                #             logits = cut_model(fused)
                #             logits_list.append(logits)

                #             for n in range(m + 1, N):
                #                 # group size 6
                #                 group = [features[i], features[j], features[k], features[l], features[m], features[n]]
                #                 fused = softmax_fusion_group(group)
                #                 logits = cut_model(fused)
                #                 logits_list.append(logits)

                #                 for o in range(n + 1, N):
                #                     # group size 7
                #                     group = [features[i], features[j], features[k], features[l], features[m],
                #                              features[n], features[o]]
                #                     fused = softmax_fusion_group(group)
                #                     logits = cut_model(fused)
                #                     logits_list.append(logits)

                #                     for p in range(o + 1, N):
                #                         # group size 8
                #                         group = [features[i], features[j], features[k], features[l], features[m],
                #                                  features[n], features[o], features[p]]
                #                         fused = softmax_fusion_group(group)
                #                         logits = cut_model(fused)
                #                         logits_list.append(logits)

                #                         for q in range(p + 1, N):
                #                             # group size 9
                #                             group = [features[i], features[j], features[k], features[l], features[m],
                #                                      features[n], features[o], features[p], features[q]]
                #                             fused = softmax_fusion_group(group)
                #                             logits = cut_model(fused)
                #                             logits_list.append(logits)

                #                             for r in range(q + 1, N):
                #                                 # group size 10
                #                                 group = [features[i], features[j], features[k], features[l],
                #                                          features[m], features[n], features[o], features[p],
                #                                          features[q], features[r]]
                #                                 fused = softmax_fusion_group(group)
                #                                 logits = cut_model(fused)
                #                                 logits_list.append(logits)

        # Average probabilities (soft voting)
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # [B, num_classes]

        # Convert to "pseudo-logits" using log
        final_logits = torch.log(avg_probs + 1e-8)  # Add epsilon to avoid log(0)
        _, preds = torch.max(final_logits, 1)

        y9 = y9.cpu().numpy()
        y9 = np.argmax(y9, axis=1) if len(y9.shape) > 1 else y9

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y9)

# Compute evaluation metrics
acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy_2024: {acc:.4f}", flush=True)

target_names = [f"Class {i}" for i in range(5)]
report = classification_report(all_labels, all_preds, target_names=target_names, digits=5)
print("\nClassification Report_2024:\n", report, flush=True)

conf_matrix = confusion_matrix(all_labels, all_preds)
conf_matrix = np.round(conf_matrix, 5)
print("\nConfusion Matrix_2024:\n", conf_matrix, flush=True)

