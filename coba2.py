import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

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
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# Definisikan fungsi loss prototypical
def prototypical_loss(prototypes, embeddings, targets, n_support):
    def pairwise_distances(x, y):
        return torch.cdist(x, y, p=2)

    n_classes = prototypes.size(0)
    n_query = (targets == 0).sum().item()
    distances = pairwise_distances(embeddings, prototypes)
    log_p_y = torch.log_softmax(-distances, dim=1)
    
    target_inds = torch.arange(n_classes).repeat(n_query).to(device)
    loss_val = -log_p_y[torch.arange(n_query * n_classes), target_inds].mean()
    return loss_val

# Definisikan fungsi training dan evaluasi
def train(model, train_loader, optimizer, epoch, n_support):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        # Buat prototypes dan queries
        prototypes = []
        queries = []
        for cls in target.unique():
            cls_mask = target == cls
            cls_embeddings = embeddings[cls_mask]
            prototypes.append(cls_embeddings[:n_support].mean(0))
            queries.append(cls_embeddings[n_support:])
        prototypes = torch.stack(prototypes)
        queries = torch.cat(queries)
        targets = torch.arange(target.unique().size(0)).repeat_interleave(len(queries) // target.unique().size(0)).to(device)
        # Hitung loss dan update model
        loss = prototypical_loss(prototypes, queries, targets, n_support)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate(model, test_loader, n_support):
    model.eval()
    all_embeddings = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            embeddings = model(data)
            all_embeddings.append(embeddings)
            all_targets.append(target)
        all_embeddings = torch.cat(all_embeddings)
        all_targets = torch.cat(all_targets)
        prototypes = []
        queries = []
        for cls in all_targets.unique():
            cls_mask = all_targets == cls
            cls_embeddings = all_embeddings[cls_mask]
            prototypes.append(cls_embeddings[:n_support].mean(0))
            queries.append(cls_embeddings[n_support:])
        prototypes = torch.stack(prototypes)
        queries = torch.cat(queries)
        query_targets = torch.arange(all_targets.unique().size(0)).repeat_interleave(len(queries) // all_targets.unique().size(0)).to(device)
        distances = torch.cdist(queries, prototypes, p=2)
        log_p_y = torch.log_softmax(-distances, dim=1)
        preds = torch.argmax(log_p_y, dim=1)
        acc = accuracy_score(query_targets.cpu(), preds.cpu())
        print(f'Accuracy: {acc:.4f}')

# Fungsi untuk menyimpan model
def save_model(model, epoch, path='C:\\Users\\PH315-53\\ProtoNet\\best_model'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, f'model_epoch_{epoch}.pth'))

# Pengaturan training
learning_rate = 0.001
epochs = 20
n_support = 7

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, epoch, n_support)
    print('Validation:')
    evaluate(model, val_loader, n_support)
    save_model(model, epoch)  # Simpan model setiap akhir epoch

# Evaluasi akhir pada test set
print('Test:')
evaluate(model, test_loader, n_support)