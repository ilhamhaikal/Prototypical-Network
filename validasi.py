import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ast  # Untuk mengubah string JSON-like menjadi array

# Definisikan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisikan transformasi untuk dataset validasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
data_dir = r'C:\Users\PH315-53\ProtoNet\split_dataset'
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
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

# Load model dan prototipe dari file
def load_model_and_prototypes(model_path, prototypes_path):
    backbone = models.resnet50(weights='IMAGENET1K_V1')
    model = PrototypicalNetwork(backbone)
    # Coba muat model dengan kunci 'model_state_dict'
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        # Jika tidak ada kunci 'model_state_dict', muat model secara langsung
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Muat prototipe dari file CSV
    prototypes_df = pd.read_csv(prototypes_path)
    print("Loaded prototypes dataframe:")
    print(prototypes_df.head())
    
    # Konversi kolom Prototype dari string JSON-like menjadi array numerik
    prototypes = prototypes_df['Prototype'].apply(ast.literal_eval).values
    prototypes = np.array([np.array(p) for p in prototypes])
    print("Prototypes array shape:", prototypes.shape)
    
    prototypes = torch.tensor(prototypes).float().to(device)
    
    return model, prototypes

# Load model dan prototipe
model_path = 'C:\\Users\\PH315-53\\ProtoNet\\best_model\\model_epoch_20.pth'
prototypes_path = 'C:\\Users\\PH315-53\\ProtoNet\\prototypes.csv'
model, prototypes = load_model_and_prototypes(model_path, prototypes_path)

# Definisikan fungsi evaluasi dengan confusion matrix
def evaluate(model, loader, prototypes):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            embeddings = model(data)
            
            distances = torch.cdist(embeddings, prototypes, p=2)
            log_p_y = torch.log_softmax(-distances, dim=1)
            preds = torch.argmax(log_p_y, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    target_names = loader.dataset.classes
    cls_report = classification_report(all_targets, all_preds, target_names=target_names)
    
    print('Classification Report:\n', cls_report)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Evaluasi model
evaluate(model, val_loader, prototypes)
