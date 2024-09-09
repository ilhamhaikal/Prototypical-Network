import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from tabulate import tabulate

# Definisikan FeatureExtractor
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()  # Menghapus fully connected layer terakhir

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Meratakan tensor menjadi vektor embedding [batch_size, 2048]
        return x

# Load pre-trained ResNet model dan modifikasi untuk ekstraksi fitur
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = FeatureExtractor(backbone)

# Fungsi untuk mengumpulkan data arsitektur model
def collect_model_data(model):
    data = []
    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential):
            for sub_name, sub_layer in layer.named_children():
                data.append([name, sub_name, sub_layer])
                if isinstance(sub_layer, nn.Sequential):
                    for sub_sub_name, sub_sub_layer in sub_layer.named_children():
                        data.append([f"{name}.{sub_name}", sub_sub_name, sub_sub_layer])
        else:
            data.append([name, '', layer])
    return data

# Mengumpulkan data
model_data = collect_model_data(model)

# Konversi ke DataFrame
df = pd.DataFrame(model_data, columns=['Layer', 'Sub-layer', 'Details'])

# Cetak tabel
print(tabulate(df, headers='keys', tablefmt='grid'))
