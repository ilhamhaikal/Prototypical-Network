import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Definisikan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisikan transformasi untuk dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = r'C:\Users\PH315-53\ProtoNet\split_dataset'
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
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
def load_model(model_path):
    backbone = models.resnet50(weights='IMAGENET1K_V1')
    model = PrototypicalNetwork(backbone)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load prototypes from CSV
def load_prototypes(csv_path):
    df = pd.read_csv(csv_path)
    prototypes = []
    target_names = []
    for _, row in df.iterrows():
        prototype = torch.tensor(eval(row['Prototype']), dtype=torch.float32).to(device)
        prototypes.append(prototype)
        target_names.append(row['Class'])
    prototypes = torch.stack(prototypes)
    return prototypes, target_names

# Definisikan fungsi evaluasi
def evaluate(model, test_loader, prototypes, target_names):
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
        
        distances = torch.cdist(all_embeddings, prototypes, p=2)
        log_p_y = torch.log_softmax(-distances, dim=1)
        preds = torch.argmax(log_p_y, dim=1)
        
        # Calculate accuracy, F1, precision, recall, support, and confusion matrix
        acc = accuracy_score(all_targets.cpu(), preds.cpu())
        f1 = f1_score(all_targets.cpu(), preds.cpu(), average='macro')
        precision = precision_score(all_targets.cpu(), preds.cpu(), average='macro')
        recall = recall_score(all_targets.cpu(), preds.cpu(), average='macro')
        cm = confusion_matrix(all_targets.cpu(), preds.cpu())
        cls_report = classification_report(all_targets.cpu(), preds.cpu(), target_names=target_names)

        print(f'Akurasi: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Presisi: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print('Laporan Klasifikasi:\n', cls_report)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Prediksi')
        plt.ylabel('Asli')
        plt.show()

        # Print prototypes
        print("\nPrototypes:")
        for i, prototype in enumerate(prototypes):
            print(f"Prototype {i} (Class: {target_names[i]}): {prototype.cpu().numpy()}")

# Load model dan prototypes
model_path = 'C:\\Users\\PH315-53\\ProtoNet\\best_model\\model_epoch_20.pth'  # Ganti sesuai model yang ingin digunakan
csv_path = 'C:\\Users\\PH315-53\\ProtoNet\\prototypes.csv'  # Path untuk prototipe
model = load_model(model_path)
prototypes, target_names = load_prototypes(csv_path)

# Evaluasi model pada val set
evaluate(model, test_loader, prototypes, target_names)


# import os
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Definisikan device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Definisikan transformasi untuk dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Load dataset
# data_dir = r'C:\Users\PH315-53\ProtoNet\split_dataset'
# test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
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
# def load_model(model_path):
#     backbone = models.resnet50(weights='IMAGENET1K_V1')
#     model = PrototypicalNetwork(backbone)
#     model.load_state_dict(torch.load(model_path))
#     model.to(device)
#     model.eval()
#     return model

# # Definisikan fungsi evaluasi
# def evaluate(model, test_loader, n_support):
#     model.eval()
#     all_embeddings = []
#     all_targets = []
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             embeddings = model(data)
#             all_embeddings.append(embeddings)
#             all_targets.append(target)
#         all_embeddings = torch.cat(all_embeddings)
#         all_targets = torch.cat(all_targets)
#         prototypes = []
#         queries = []
#         query_targets = []
#         for cls in all_targets.unique():
#             cls_mask = all_targets == cls
#             cls_embeddings = all_embeddings[cls_mask]
#             if len(cls_embeddings) >= n_support:
#                 prototypes.append(cls_embeddings[:n_support].mean(0))
#                 queries.append(cls_embeddings[n_support:])
#                 query_targets.append(torch.full((len(cls_embeddings) - n_support,), cls.item(), dtype=torch.long))
#             else:
#                 raise ValueError(f"Class {cls.item()} does not have enough samples for the required support set size.")
#         prototypes = torch.stack(prototypes)
#         queries = torch.cat(queries)
#         query_targets = torch.cat(query_targets).to(device)
#         distances = torch.cdist(queries, prototypes, p=2)
#         log_p_y = torch.log_softmax(-distances, dim=1)
#         preds = torch.argmax(log_p_y, dim=1)
        
#         # Calculate accuracy, F1, precision, recall, support, and confusion matrix
#         acc = accuracy_score(query_targets.cpu(), preds.cpu())
#         f1 = f1_score(query_targets.cpu(), preds.cpu(), average='macro')
#         precision = precision_score(query_targets.cpu(), preds.cpu(), average='macro')
#         recall = recall_score(query_targets.cpu(), preds.cpu(), average='macro')
#         cm = confusion_matrix(query_targets.cpu(), preds.cpu())
#         target_names = ['matang', 'mentah', 'setengah matang']  # Ganti sesuai nama kelas
#         cls_report = classification_report(query_targets.cpu(), preds.cpu(), target_names=target_names)

#         print(f'Akurasi: {acc:.4f}')
#         print(f'F1 Score: {f1:.4f}')
#         print(f'Presisi: {precision:.4f}')
#         print(f'Recall: {recall:.4f}')
#         print('Laporan Klasifikasi:\n', cls_report)

#         # Plot confusion matrix
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
#         plt.xlabel('Prediksi')
#         plt.ylabel('Asli')
#         plt.show()

#         # Prepare data for CSV
#         data_for_csv = []
#         for i, (query, target) in enumerate(zip(queries, query_targets)):
#             dists = distances[i].cpu().numpy()
#             data_for_csv.append({
#                 'Query Index': i,
#                 'Target': target.item(),
#                 'Embedding': query.cpu().numpy().tolist(),
#                 'Distances to Prototypes': dists.tolist()
#             })

#         # # Save to CSV
#         # df = pd.DataFrame(data_for_csv)
#         # df.to_csv('embeddings_and_distances.csv', index=False)

#         # Print embeddings and distances
#         print("\nEmbeddings dan Jarak:")
#         for i, (query, target) in enumerate(zip(queries, query_targets)):
#             dists = distances[i].cpu().numpy()
#             print(f"Query {i} (Target: {target.item()}):")
#             print(f"Embedding: {query.cpu().numpy()[:10]}...")  # Display the first 10 elements of the embedding
#             print(f"Jarak ke Prototipe: {dists}\n")

#         # Print prototypes
#         print("\nPrototypes:")
#         for i, prototype in enumerate(prototypes):
#             print(f"Prototype {i}: {prototype.cpu().numpy()}")

# # Load model yang sudah dilatih
# model_path = 'C:\\Users\\PH315-53\\ProtoNet\\best_model\\model_epoch_20.pth'  # Ganti sesuai model yang ingin digunakan
# model = load_model(model_path)

# # Evaluasi model pada val set
# evaluate(model, test_loader, n_support=3)
