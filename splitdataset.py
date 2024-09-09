import os
import random
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Paths
dataset_root = '/content/dataset_kopi'  # Sesuaikan dengan path di Google Colab
output_root = '/content/split_dataset'  # Sesuaikan dengan path di Google Colab

# Create output directories
train_dir = os.path.join(output_root, 'train')
val_dir = os.path.join(output_root, 'val')
test_dir = os.path.join(output_root, 'test')

for dir_path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for class_name in ['matang', 'mentah', 'setengah_matang']:
        os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

# Transformations
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=dataset_root, transform=data_transform)

# Function to split dataset per class
def split_dataset_by_class(dataset, n_train=74, n_test=9, n_val=9):
    class_indices = {class_name: [] for class_name in dataset.classes}
    for idx, (_, label) in enumerate(dataset.samples):
        class_name = dataset.classes[label]
        class_indices[class_name].append(idx)
    
    train_indices, test_indices, val_indices = [], [], []
    
    for class_name, indices in class_indices.items():
        if len(indices) < n_train + n_test + n_val:
            raise ValueError(f"Not enough samples for class {class_name}. Required: {n_train + n_test + n_val}, available: {len(indices)}.")
        
        random.shuffle(indices)
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:n_train + n_test])
        val_indices.extend(indices[n_train + n_test:n_train + n_test + n_val])
    
    return train_indices, test_indices, val_indices

# Get split indices
train_indices, test_indices, val_indices = split_dataset_by_class(dataset)

# Create Subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
val_dataset = Subset(dataset, val_indices)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Print dataset sizes
print(f'Total dataset size: {len(dataset)}')
print(f'Train dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')

# Print directories
print(f'Training data saved to: {train_dir}')
print(f'Validation data saved to: {val_dir}')
print(f'Test data saved to: {test_dir}')
