import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Definisikan transformasi untuk gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load image paths and labels
        for label in ['setengah_matang', 'mentah', 'matang']:
            class_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                if label == 'setengah_matang':
                    self.labels.append(0)
                elif label == 'mentah':
                    self.labels.append(1)
                elif label == 'matang':
                    self.labels.append(2)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Inisialisasi dataset dan dataloader
dataset = CustomDataset(r'C:\Users\PH315-53\ProtoNet\split_dataset\train', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size 1 to process one image at a time

# Load pre-trained ResNet model
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = model.eval()  # Set to evaluation mode

# Remove the fully connected layer to get features
model = torch.nn.Sequential(*list(model.children())[:-2], torch.nn.AdaptiveAvgPool2d((1, 1)))

# Store the outputs of each layer
activations = {}

# Define a hook function to capture the output of each layer
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hook to each layer
for name, layer in model.named_children():
    layer.register_forward_hook(get_activation(name))

# Select one image per class
selected_images = {0: None, 1: None, 2: None}
selected_labels = {0: 'setengah_matang', 1: 'mentah', 2: 'matang'}

for images, labels in dataloader:
    label = labels.item()
    if selected_images[label] is None:
        selected_images[label] = images
    if all(v is not None for v in selected_images.values()):
        break

# Ekstrak fitur untuk satu gambar per kelas
for label, images in selected_images.items():
    with torch.no_grad():
        outputs = model(images)

    print(f"Class: {selected_labels[label]}")
    print(f"Image shape: {images.shape}")
    for layer_name, activation in activations.items():
        print(f"Output of layer {layer_name}:")
        print(activation.shape)
        print(activation)

    activations.clear()
