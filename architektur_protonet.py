import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchsummary import summary

# Definisikan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model ResNet-50 pre-trained dan modifikasi untuk ekstraksi fitur
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()  # Menghapus fully connected layer terakhir

    def forward(self, x):
        x = self.backbone(x)
        return x

backbone = models.resnet50(weights='IMAGENET1K_V1')
model = FeatureExtractor(backbone).to(device)

# Fungsi untuk menampilkan arsitektur model
def display_model_architecture(model, input_size):
    summary(model, input_size=input_size)

# Tampilkan arsitektur model
display_model_architecture(model, (3, 224, 224))