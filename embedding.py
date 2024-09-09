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
def create_episode(dataset, n_support, n_query):
    unique_classes = torch.unique(torch.tensor(dataset.targets))
    support_indices = []
    query_indices = []
    
    for cls in unique_classes:
        cls_indices = (torch.tensor(dataset.targets) == cls).nonzero(as_tuple=True)[0]
        if len(cls_indices) >= (n_support + n_query):
            support_indices.append(cls_indices[:n_support])
            query_indices.append(cls_indices[n_support:n_support + n_query])
        else:
            print(f"Warning: Class {cls} has less than {n_support + n_query} samples.")
    
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices)
    
    support_set = [dataset[i] for i in support_indices]
    query_set = [dataset[i] for i in query_indices]
    
    support_data, support_targets = zip(*support_set)
    query_data, query_targets = zip(*query_set)
    
    support_data = torch.stack(support_data).to(device)
    support_targets = torch.tensor(support_targets).to(device)
    query_data = torch.stack(query_data).to(device)
    query_targets = torch.tensor(query_targets).to(device)
    
    return support_data, support_targets, query_data, query_targets

# Fungsi untuk mengekstrak embedding
def get_embedding(model, data):
    model.eval()
    with torch.no_grad():
        embedding = model(data)
    return embedding.view(embedding.size(0), -1)

# Buat episode
n_support = 5  # Misalkan mengambil 5 data awal dari setiap kelas
n_query = 5    # Misalkan mengambil 5 data query dari setiap kelas
support_set, support_targets, query_set, query_targets = create_episode(train_dataset, n_support, n_query)

# Ekstrak embedding untuk support set dan query set
support_embeddings = get_embedding(model, support_set)
query_embeddings = get_embedding(model, query_set)

# Fungsi untuk menampilkan embedding berdasarkan kelas
def print_embeddings(embeddings, targets, set_name):
    unique_classes = targets.unique()
    class_names = {0: 'setengah_matang', 1: 'mentah', 2: 'matang'}
    for cls in unique_classes:
        cls_indices = (targets == cls).nonzero(as_tuple=True)[0]
        cls_embeddings = embeddings[cls_indices]
        for i in range(cls_embeddings.size(0)):
            print(f"Class: {class_names[cls.item()]}")
            print(f"Embedding shape: {cls_embeddings[i].unsqueeze(0).shape}")
            print(f"Embedding: {cls_embeddings[i].unsqueeze(0)}")

print("Support Set Embeddings:")
print_embeddings(support_embeddings, support_targets, "Support Set")

print("\nQuery Set Embeddings:")
print_embeddings(query_embeddings, query_targets, "Query Set")
