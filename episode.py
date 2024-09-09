import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Definisikan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisikan transformasi untuk dataset training dan validasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
data_dir = r'C:\Users\PH315-53\ProtoNet\split_dataset'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Definisikan model CNN untuk ekstraksi fitur
class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()  # Menghapus fully connected layer terakhir

    def forward(self, x):
        x = self.backbone(x)
        return x

# Load model ResNet-50 pre-trained dan inisialisasi Prototypical Network
backbone = models.resnet50(weights='IMAGENET1K_V1')
model = PrototypicalNetwork(backbone).to(device)

# Definisikan fungsi untuk membuat episode
def create_episode(dataset, n_support):
    unique_classes = torch.unique(torch.tensor(dataset.targets))
    support_indices = []
    query_indices = []
    
    for cls in unique_classes:
        cls_indices = (torch.tensor(dataset.targets) == cls).nonzero(as_tuple=True)[0]
        if len(cls_indices) >= n_support:
            support_indices.append(cls_indices[:n_support])
            query_indices.append(cls_indices[n_support:])
        else:
            print(f"Warning: Class {cls} has less than {n_support} samples.")
    
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices)
    
    support_set = [dataset[i] for i in support_indices]
    query_set = [dataset[i] for i in query_indices]
    
    support_data, support_targets = zip(*support_set)
    query_data, query_targets = zip(*query_set)
    
    support_data = torch.stack(support_data)
    support_targets = torch.tensor(support_targets)
    query_data = torch.stack(query_data)
    query_targets = torch.tensor(query_targets)
    
    return support_data, support_targets, query_data, query_targets

# Buat episode
n_support = 5  # Misalkan mengambil 5 data awal dari setiap kelas
support_set, support_targets, query_set, query_targets = create_episode(train_dataset, n_support)

# Fungsi untuk menampilkan set berdasarkan kelas
def print_set_by_class(data_set, targets_set, set_name):
    unique_classes = targets_set.unique()
    print(f"{set_name}:")
    for cls in unique_classes:
        cls_indices = (targets_set == cls).nonzero(as_tuple=True)[0]
        cls_data = data_set[cls_indices]
        cls_targets = targets_set[cls_indices]
        print(f"  Class {cls.item()}:")
        for i in range(cls_data.size(0)):
            print(f"    Image {i} - Label: {cls_targets[i].item()}")

print_set_by_class(support_set, support_targets, "Support Set")
print_set_by_class(query_set, query_targets, "Query Set")
