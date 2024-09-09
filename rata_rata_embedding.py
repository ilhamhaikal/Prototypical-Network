import torch
import os
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisikan transformasi untuk dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = r'C:\Users\PH315-53\ProtoNet\split_dataset'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define CNN model for feature extraction
class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()  # Remove the last fully connected layer

    def forward(self, x):
        x = self.backbone(x)
        return x

# Load pre-trained ResNet-50 model
backbone = models.resnet50(weights='IMAGENET1K_V1')
model = PrototypicalNetwork(backbone).to(device)

# Function to create an episode without shuffling
def create_episode(data, target, n_support):
    unique_classes = target.unique()
    support_indices = []

    for cls in unique_classes:
        cls_indices = (target == cls).nonzero(as_tuple=True)[0]
        support_indices.append(cls_indices[:n_support])

    support_indices = torch.cat(support_indices)
    support_set = data[support_indices]
    support_targets = target[support_indices]

    return support_set, support_targets

# Function to calculate class prototypes
def calculate_prototypes(model, loader, n_support):
    model.eval()
    class_to_prototypes = {}
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            support_set, support_targets = create_episode(data, target, n_support)
            
            # Get the embeddings from the model
            support_embeddings = model(support_set)
            
            for cls in support_targets.unique():
                cls_mask = support_targets == cls
                cls_embeddings = support_embeddings[cls_mask]
                
                # Calculate mean (prototype) for each class
                prototype = cls_embeddings.mean(dim=0)
                
                if cls.item() not in class_to_prototypes:
                    class_to_prototypes[cls.item()] = []
                class_to_prototypes[cls.item()].append(prototype)
    
    # Calculate the mean prototype for each class across batches
    final_prototypes = {}
    for cls, protos in class_to_prototypes.items():
        final_prototypes[cls] = torch.stack(protos).mean(dim=0)

    # Convert to a single tensor for CSV saving
    ordered_classes = sorted(final_prototypes.keys())
    final_prototypes_tensor = torch.stack([final_prototypes[cls] for cls in ordered_classes])

    return final_prototypes_tensor

# Example usage to calculate prototypes
n_support = 5
prototypes = calculate_prototypes(model, train_loader, n_support)

# Save prototypes to CSV
def save_prototypes_to_csv(prototypes, path):
    prototypes_np = prototypes.cpu().numpy()  # Convert to numpy array and move to CPU
    df = pd.DataFrame(prototypes_np)
    df.to_csv(path, index=False)
    print(f'Prototypes saved to {path}')

# prototype_csv_save_path = r'C:\Users\PH315-53\ProtoNet\prototypes.csv'
# save_prototypes_to_csv(prototypes, prototype_csv_save_path)

print("Prototypes shape:", prototypes.shape)
print("First prototype (first 10 features):", prototypes[0][:10])
