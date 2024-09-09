import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
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

        # Hook untuk mengambil output dari setiap block
        self.outputs = {}
        self.backbone.conv1.register_forward_hook(self.save_output('conv1'))
        self.backbone.layer1.register_forward_hook(self.save_output('conv2_x'))
        self.backbone.layer2.register_forward_hook(self.save_output('conv3_x'))
        self.backbone.layer3.register_forward_hook(self.save_output('conv4_x'))
        self.backbone.layer4.register_forward_hook(self.save_output('conv5_x'))
        self.backbone.avgpool.register_forward_hook(self.save_output('avgpool'))

    def save_output(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def forward(self, x):
        self.backbone(x)
        return self.outputs

# Load model ResNet-50 pre-trained dan inisialisasi Prototypical Network
def load_model(model_path):
    backbone = models.resnet50(weights='IMAGENET1K_V1')
    model = PrototypicalNetwork(backbone)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load model yang sudah dilatih
model_path = r'C:\Users\PH315-53\ProtoNet\best_model\model_epoch_20.pth'  # Ganti sesuai model yang ingin digunakan
model = load_model(model_path)

# Path gambar yang ingin diklasifikasikan
image_path = r'C:\Users\PH315-53\ProtoNet\dataset_kopi\matang\matang_1.jpeg'  # Ganti dengan path gambar sebenarnya

# Load and preprocess image
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Menampilkan gambar sebelum dan sesudah preprocessing
plt.figure(figsize=(18, 6))

# Gambar Asli
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Gambar setelah preprocessing
plt.subplot(1, 3, 2)
preprocessed_image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
preprocessed_image = std * preprocessed_image + mean
preprocessed_image = np.clip(preprocessed_image, 0, 1)
plt.imshow(preprocessed_image)
plt.title('Preprocessed Image')
plt.axis('off')

# Ekstraksi fitur
with torch.no_grad():
    outputs = model(image_tensor)

# Menampilkan hasil per convolutional block dan avgpool
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, (name, output) in enumerate(outputs.items()):
    output_np = output.cpu().numpy()[0]  # Ambil batch pertama
    if name == 'avgpool':
        avgpool_output = output_np.flatten()
        axes[i].plot(avgpool_output)
        axes[i].set_xlabel('Feature Index')
        axes[i].set_ylabel('Feature Value')
    else:
        if len(output_np.shape) == 3:
            output_np = np.mean(output_np, axis=0)  # Ambil rata-rata sepanjang channel
            axes[i].imshow(output_np, cmap='viridis')
        elif len(output_np.shape) == 2:
            output_np = np.mean(output_np, axis=0)  # Ambil rata-rata sepanjang channel
            axes[i].imshow(output_np, cmap='viridis')
    axes[i].set_title(f'{name} Output')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print('values avgpool')
print(avgpool_output)

# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image
# import numpy as np
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

#         # Hook untuk mengambil output dari setiap layer
#         self.outputs = {}
#         self.register_hooks()

#     def register_hooks(self):
#         for name, module in self.backbone.named_modules():
#             if 'conv' in name or 'layer' in name or 'bn' in name or 'relu' in name:
#                 module.register_forward_hook(self.save_output(name))

#     def save_output(self, name):
#         def hook(module, input, output):
#             self.outputs[name] = output
#         return hook

#     def forward(self, x):
#         self.backbone(x)
#         return self.outputs

# # Load model ResNet-50 pre-trained dan inisialisasi Prototypical Network
# def load_model(model_path):
#     backbone = models.resnet50(weights='IMAGENET1K_V1')
#     model = PrototypicalNetwork(backbone)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model

# # Load model yang sudah dilatih
# model_path = r'C:\Users\PH315-53\ProtoNet\best_model\model_epoch_20.pth'  # Ganti sesuai model yang ingin digunakan
# model = load_model(model_path)

# # Path gambar yang ingin diklasifikasikan
# image_path = r'C:\Users\PH315-53\ProtoNet\dataset_kopi\matang\matang_1.jpeg'  # Ganti dengan path gambar sebenarnya

# # Load and preprocess image
# image = Image.open(image_path).convert('RGB')
# image_tensor = transform(image).unsqueeze(0).to(device)

# # Menampilkan gambar sebelum dan sesudah preprocessing
# plt.figure(figsize=(18, 6))

# # Gambar Asli
# plt.subplot(1, 3, 1)
# plt.imshow(image)
# plt.title('Original Image')
# plt.axis('off')

# # Gambar setelah preprocessing
# plt.subplot(1, 3, 2)
# preprocessed_image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# preprocessed_image = std * preprocessed_image + mean
# preprocessed_image = np.clip(preprocessed_image, 0, 1)
# plt.imshow(preprocessed_image)
# plt.title('Preprocessed Image')
# plt.axis('off')

# # Ekstraksi fitur
# with torch.no_grad():
#     outputs = model(image_tensor)

# # Menampilkan hasil per layer
# fig, axes = plt.subplots(6, 6, figsize=(24, 24))
# axes = axes.flatten()
# layer_count = min(len(outputs), len(axes))

# for i, (name, output) in enumerate(outputs.items()):
#     if i >= layer_count:
#         break
#     output_np = output.cpu().numpy()[0]  # Ambil batch pertama
#     if len(output_np.shape) == 3:
#         output_np = np.mean(output_np, axis=0)  # Ambil rata-rata sepanjang channel
#         axes[i].imshow(output_np, cmap='viridis')
#     elif len(output_np.shape) == 2:
#         output_np = np.mean(output_np, axis=0)  # Ambil rata-rata sepanjang channel
#         axes[i].imshow(output_np, cmap='viridis')
#     else:
#         output_np = output_np.flatten()
#         axes[i].plot(output_np)
#         axes[i].set_xlabel('Feature Index')
#         axes[i].set_ylabel('Feature Value')
#     axes[i].set_title(f'{name} Output')
#     axes[i].axis('off')

# plt.tight_layout()
# plt.show()
