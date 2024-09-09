import os
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Definisikan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisikan transformasi untuk dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Fungsi untuk melakukan klasifikasi pada gambar baru
def classify_image(model, image, prototypes, class_names):
    with torch.no_grad():
        embedding = model(image)
        distances = torch.cdist(embedding, prototypes, p=2)
        log_p_y = torch.log_softmax(-distances, dim=1)
        preds = torch.argmax(log_p_y, dim=1)

    predicted_class = class_names[preds.item()]
    return predicted_class, embedding.cpu().numpy(), distances.cpu().numpy(), log_p_y

# Fungsi untuk menampilkan gambar beserta prediksinya
def show_image(image_path, predicted_class):
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()

# Fungsi untuk menampilkan diagram batang jarak
def show_distance_bar_chart(distances, class_names):
    distances = distances.flatten()
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, distances)
    plt.xlabel('Class Names')
    plt.ylabel('Distance to Prototypes')
    plt.title('Distance from Image Embedding to Prototypes')
    plt.show()

# Fungsi untuk menampilkan diagram batang log-probabilitas
def show_log_probabilities_bar_chart(log_p_y, class_names):
    log_p_y = log_p_y.flatten()
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, log_p_y)
    plt.xlabel('Class Names')
    plt.ylabel('Log-Probabilities')
    plt.title('Log-Probabilities for Each Class')
    plt.show()

# Load model yang sudah dilatih
model_path = r'C:\Users\PH315-53\ProtoNet\best_model\model_epoch_20.pth'  # Ganti sesuai model yang ingin digunakan
model = load_model(model_path)

# Load prototypes from CSV file
prototypes_path = r'C:\Users\PH315-53\ProtoNet\prototypes.csv'
prototypes_df = pd.read_csv(prototypes_path)

# Extract prototypes
prototypes = []
class_names = prototypes_df['Class'].unique()

for cls in class_names:
    prototype = prototypes_df[prototypes_df['Class'] == cls]['Prototype'].values[0]
    prototype = np.array(ast.literal_eval(prototype))
    prototypes.append(torch.tensor(prototype, dtype=torch.float))

prototypes = torch.stack(prototypes).to(device)

# Cetak nilai prototipe
print("\nPrototypes:")
for i, prototype in enumerate(prototypes):
    print(f"Prototype {i}: {prototype.cpu().numpy()}")

# Define class names
class_names = ['Matang', 'Mentah', 'Setengah Matang']  # Ganti dengan nama kelas sebenarnya

# Path gambar yang ingin diklasifikasikan
image_path = r'C:\Users\PH315-53\ProtoNet\split_dataset\Gambar WhatsApp 2024-08-14 pukul 10.29.15_81f51d33.jpg'  # Ganti dengan path gambar sebenarnya

# Klasifikasikan gambar
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)
predicted_class, embedding, distances, log_p_y = classify_image(model, image_tensor, prototypes, class_names)

print(f'Predicted Class: {predicted_class}')
print(f'Embedding: {embedding}')
print(f'Distances to Prototypes: {distances}')
print(f'Log-Probabilities: {log_p_y.cpu().numpy()}')

# Tampilkan gambar beserta prediksinya
show_image(image_path, predicted_class)

# Tampilkan diagram batang jarak
show_distance_bar_chart(distances, class_names)

# Tampilkan diagram batang log-probabilitas
show_log_probabilities_bar_chart(log_p_y.cpu().numpy(), class_names)

# Evaluate model on the test set
data_dir = r'C:\Users\PH315-53\ProtoNet\split_dataset'
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

true_labels = []
predicted_labels = []
criterion = nn.CrossEntropyLoss()

total_loss = 0

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    
    predicted_class, embedding, _, log_p_y = classify_image(model, images, prototypes, class_names)
    
    true_labels.append(labels.item())
    predicted_labels.append(class_names.index(predicted_class))
    
    # Calculate loss
    loss = criterion(log_p_y, labels)
    total_loss += loss.item()

accuracy = accuracy_score(true_labels, predicted_labels)
average_loss = total_loss / len(test_loader)

# print(f'Accuracy: {accuracy}')
# print(f'Average Loss: {average_loss}')


# import os
# import torch
# import torch.nn as nn
# from torchvision import transforms, models, datasets
# from torch.utils.data import DataLoader
# from PIL import Image
# import numpy as np
# import pandas as pd
# import ast
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Definisikan device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Definisikan transformasi untuk dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

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

# # Fungsi untuk melakukan klasifikasi pada gambar baru
# def classify_image(model, image, prototypes, class_names):
#     with torch.no_grad():
#         embedding = model(image)
#         distances = torch.cdist(embedding, prototypes, p=2)
#         log_p_y = torch.log_softmax(-distances, dim=1)
#         preds = torch.argmax(log_p_y, dim=1)

#     predicted_class = class_names[preds.item()]
#     return predicted_class, embedding.cpu().numpy(), distances.cpu().numpy(), log_p_y

# # Fungsi untuk menampilkan gambar beserta prediksinya
# def show_image(image_path, predicted_class):
#     image = Image.open(image_path).convert('RGB')
#     plt.imshow(image)
#     plt.title(f'Predicted Class: {predicted_class}')
#     plt.axis('off')
#     plt.show()

# # Load model yang sudah dilatih
# model_path = r'C:\Users\PH315-53\ProtoNet\best_model\model_epoch_20.pth'  # Ganti sesuai model yang ingin digunakan
# model = load_model(model_path)

# # Load embeddings and distances from CSV file
# csv_path = r'C:\Users\PH315-53\ProtoNet\embeddings_and_distances.csv'
# df = pd.read_csv(csv_path)

# # Extract prototypes from embeddings
# prototypes = []
# class_labels = df['Target'].unique()

# for cls in class_labels:
#     class_embeddings = df[df['Target'] == cls]['Embedding'].apply(ast.literal_eval).values
#     class_embeddings = np.array([np.array(embedding) for embedding in class_embeddings])
#     prototypes.append(torch.tensor(class_embeddings.mean(axis=0), dtype=torch.float))

# prototypes = torch.stack(prototypes).to(device)

# # Cetak nilai prototipe
# print("\nPrototypes:")
# for i, prototype in enumerate(prototypes):
#     print(f"Prototype {i}: {prototype.cpu().numpy()}")

# # Define class names
# class_names = ['Matang', 'Mentah', 'Setengah Matang']  # Ganti dengan nama kelas sebenarnya

# # Path gambar yang ingin diklasifikasikan
# image_path = r'C:\Users\PH315-53\ProtoNet\split_dataset\test\matang\matang_81.jpeg'  # Ganti dengan path gambar sebenarnya

# # Klasifikasikan gambar
# image = Image.open(image_path).convert('RGB')
# image_tensor = transform(image).unsqueeze(0).to(device)
# predicted_class, embedding, distances, log_p_y = classify_image(model, image_tensor, prototypes, class_names)

# print(f'Predicted Class: {predicted_class}')
# print(f'Embedding: {embedding}')
# print(f'Distances to Prototypes: {distances}')
# print(f'Log-Probabilities: {log_p_y.cpu().numpy()}')

# # Tampilkan gambar beserta prediksinya
# show_image(image_path, predicted_class)

# # Evaluate model on the test set
# data_dir = r'C:\Users\PH315-53\ProtoNet\split_dataset'
# test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# true_labels = []
# predicted_labels = []
# criterion = nn.CrossEntropyLoss()

# total_loss = 0

# for images, labels in test_loader:
#     images = images.to(device)
#     labels = labels.to(device)
    
#     predicted_class, embedding, _, log_p_y = classify_image(model, images, prototypes, class_names)
    
#     true_labels.append(labels.item())
#     predicted_labels.append(class_names.index(predicted_class))
    
#     # Calculate loss
#     loss = criterion(log_p_y, labels)
#     total_loss += loss.item()

# accuracy = accuracy_score(true_labels, predicted_labels)
# average_loss = total_loss / len(test_loader)

# print(f'Akurasi: {accuracy:.4f}')
# print(f'Rata-rata Loss: {average_loss:.4f}')
