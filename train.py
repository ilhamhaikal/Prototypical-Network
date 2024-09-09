import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
def prototypical_loss(prototypes, embeddings, targets):
    def pairwise_distances(x, y):
        return torch.cdist(x, y, p=2)
    
    n_classes = prototypes.size(0)
    n_query = targets.size(0) // n_classes
    
    distances = pairwise_distances(embeddings, prototypes)
    log_p_y = torch.log_softmax(-distances, dim=1)
    
    target_inds = torch.arange(n_classes).repeat_interleave(n_query).to(device)
    loss_val = -log_p_y[torch.arange(len(target_inds)), target_inds].mean()
    return loss_val

# Definisikan fungsi untuk membuat episode tanpa pengacakan
def create_episode(data, target, n_support):
    unique_classes = target.unique()
    support_indices = []
    query_indices = []
    
    for cls in unique_classes:
        cls_indices = (target == cls).nonzero(as_tuple=True)[0]
        support_indices.append(cls_indices[:n_support])
        query_indices.append(cls_indices[n_support:])
    
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices)
    
    support_set = data[support_indices]
    query_set = data[query_indices]
    support_targets = target[support_indices]
    query_targets = target[query_indices]
    
    return support_set, support_targets, query_set, query_targets

# Definisikan fungsi untuk menghitung prototipe di awal
def calculate_prototypes(model, loader, n_support, save_path=None):
    model.eval()
    all_prototypes = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            support_set, support_targets, _, _ = create_episode(data, target, n_support)
            
            support_embeddings = model(support_set)
            
            prototypes = []
            for cls in support_targets.unique():
                cls_mask = support_targets == cls
                cls_embeddings = support_embeddings[cls_mask]
                prototypes.append(cls_embeddings.mean(0))
            prototypes = torch.stack(prototypes)
            
            all_prototypes.append(prototypes)
    
    final_prototypes = torch.mean(torch.stack(all_prototypes), dim=0)

    if save_path:
        torch.save(final_prototypes, save_path)
        print(f'Prototypes saved to {save_path}')

    return final_prototypes

# Definisikan fungsi untuk menyimpan prototipe sebagai CSV
def save_prototypes_to_csv(prototypes, path):
    prototypes_np = prototypes.cpu().numpy()  # Convert to numpy array and move to CPU
    df = pd.DataFrame(prototypes_np)
    df.to_csv(path, index=False)
    print(f'Prototypes saved to {path}')

# Definisikan fungsi training dengan episodic training
def train(model, train_loader, optimizer, epoch, prototypes):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, _, query_set, query_targets = create_episode(data, target, n_support)
        
        query_embeddings = model(query_set)
        
        loss = prototypical_loss(prototypes, query_embeddings, query_targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        total += query_targets.size(0)
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        log_p_y = torch.log_softmax(-distances, dim=1)
        preds = torch.argmax(log_p_y, dim=1)
        correct += preds.eq(query_targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_loss /= len(train_loader)
    train_accuracy = correct / total
    
    return train_loss, train_accuracy

def evaluate(model, loader, prototypes):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, _, query_set, query_targets = create_episode(data, target, n_support)
            
            query_embeddings = model(query_set)
            
            loss = prototypical_loss(prototypes, query_embeddings, query_targets)
            val_loss += loss.item()
            
            distances = torch.cdist(query_embeddings, prototypes, p=2)
            log_p_y = torch.log_softmax(-distances, dim=1)
            preds = torch.argmax(log_p_y, dim=1)
            
            correct += preds.eq(query_targets).sum().item()
            total += query_targets.size(0)
    
    val_loss /= len(loader)
    val_accuracy = correct / total
    
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    return val_loss, val_accuracy

# Fungsi untuk menyimpan model dan prototipe
def save_model_and_prototypes(model, prototypes, epoch, path='C:\\Users\\PH315-53\\ProtoNet\\best_model'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'prototypes': prototypes
    }, os.path.join(path, f'model_epoch_{epoch}.pth'))

# Pengaturan training
learning_rate = 0.001
epochs = 20
n_support = 5  # Mengambil 5 data awal dari setiap kelas

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Definisikan path untuk menyimpan prototipe
prototype_save_path = r'C:\Users\PH315-53\ProtoNet\prototypes.pth'
prototype_csv_save_path = r'C:\Users\PH315-53\ProtoNet\prototypes.csv'

# Hitung prototipe di awal dan simpan
print('Calculating prototypes...')
prototypes = calculate_prototypes(model, train_loader, n_support, prototype_save_path)
save_prototypes_to_csv(prototypes, prototype_csv_save_path)
print('Prototypes calculated and saved as CSV.')

# Training loop
for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(model, train_loader, optimizer, epoch, prototypes)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    print('Validation:')
    val_loss, val_accuracy = evaluate(model, val_loader, prototypes)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    save_model_and_prototypes(model, prototypes, epoch)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train Loss vs Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train Accuracy vs Validation Accuracy')

plt.tight_layout()
plt.show()

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Definisikan device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Definisikan transformasi untuk dataset training dan validasi
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Load datasets
# data_dir = r'C:\Users\PH315-53\ProtoNet\split_dataset'
# train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
# val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
# test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Definisikan model CNN untuk ekstraksi fitur
# class PrototypicalNetwork(nn.Module):
#     def __init__(self, backbone):
#         super(PrototypicalNetwork, self).__init__()
#         self.backbone = backbone
#         self.backbone.fc = nn.Identity()  # Menghapus fully connected layer terakhir

#     def forward(self, x):
#         x = self.backbone(x)
#         return x

# # Load model ResNet-50 pre-trained dan inisialisasi Prototypical Network
# backbone = models.resnet50(weights='IMAGENET1K_V1')
# model = PrototypicalNetwork(backbone).to(device)

# # Definisikan fungsi loss prototypical
# def prototypical_loss(prototypes, embeddings, targets, n_support):
#     def pairwise_distances(x, y):
#         return torch.cdist(x, y, p=2)

#     n_classes = prototypes.size(0)
#     n_query = (targets == 0).sum().item()
#     distances = pairwise_distances(embeddings, prototypes)
#     log_p_y = torch.log_softmax(-distances, dim=1)
    
#     target_inds = torch.arange(n_classes).repeat(n_query).to(device)
#     loss_val = -log_p_y[torch.arange(n_query * n_classes), target_inds].mean()
#     return loss_val

# # Definisikan fungsi untuk membuat episode
# def create_episode(data, target, n_support):
#     unique_classes = target.unique()
#     support_indices = []
#     query_indices = []
    
#     for cls in unique_classes:
#         cls_indices = (target == cls).nonzero(as_tuple=True)[0]
#         cls_indices = cls_indices[torch.randperm(len(cls_indices))]  # Acak urutan
#         support_indices.append(cls_indices[:n_support])
#         query_indices.append(cls_indices[n_support:])
    
#     support_indices = torch.cat(support_indices)
#     query_indices = torch.cat(query_indices)
    
#     support_set = data[support_indices]
#     query_set = data[query_indices]
#     support_targets = target[support_indices]
#     query_targets = target[query_indices]
    
#     return support_set, support_targets, query_set, query_targets

# # Definisikan fungsi training dan evaluasi dengan episodic training
# def train(model, train_loader, optimizer, epoch, n_support):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         support_set, support_targets, query_set, query_targets = create_episode(data, target, n_support)
        
#         support_embeddings = model(support_set)
#         query_embeddings = model(query_set)
        
#         # Hitung prototypes
#         prototypes = []
#         for cls in support_targets.unique():
#             cls_mask = support_targets == cls
#             cls_embeddings = support_embeddings[cls_mask]
#             prototypes.append(cls_embeddings.mean(0))
#         prototypes = torch.stack(prototypes)
        
#         # Hitung loss dan update model
#         loss = prototypical_loss(prototypes, query_embeddings, query_targets, n_support)
#         loss.backward()
#         optimizer.step()
        
#         if batch_idx % 10 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# def evaluate(model, test_loader, n_support):
#     model.eval()
#     all_embeddings = []
#     all_targets = []
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             support_set, support_targets, query_set, query_targets = create_episode(data, target, n_support)
            
#             support_embeddings = model(support_set)
#             query_embeddings = model(query_set)
            
#             # Hitung prototypes
#             prototypes = []
#             for cls in support_targets.unique():
#                 cls_mask = support_targets == cls
#                 cls_embeddings = support_embeddings[cls_mask]
#                 prototypes.append(cls_embeddings.mean(0))
#             prototypes = torch.stack(prototypes)
            
#             distances = torch.cdist(query_embeddings, prototypes, p=2)
#             log_p_y = torch.log_softmax(-distances, dim=1)
#             preds = torch.argmax(log_p_y, dim=1)
#             acc = accuracy_score(query_targets.cpu(), preds.cpu())
#             all_embeddings.append(query_embeddings)
#             all_targets.append(query_targets)
            
#         print(f'Accuracy: {acc:.4f}')

# # Fungsi untuk menyimpan model
# def save_model(model, epoch, path='C:\\Users\\PH315-53\\ProtoNet\\best_model'):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     torch.save(model.state_dict(), os.path.join(path, f'model_epoch_{epoch}.pth'))

# # Pengaturan training
# learning_rate = 0.001
# epochs = 20
# n_support = 7

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# for epoch in range(1, epochs + 1):
#     train(model, train_loader, optimizer, epoch, n_support)
#     print('Validation:')
#     evaluate(model, val_loader, n_support)
#     save_model(model, epoch)  # Simpan model setiap akhir epoch

# # Evaluasi akhir pada test set
# print('Test:')
# evaluate(model, test_loader, n_support)
