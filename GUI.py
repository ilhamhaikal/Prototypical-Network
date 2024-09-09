import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import ast
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QTextEdit, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

# Definisikan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisikan transformasi preprocessing untuk dataset
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
    return predicted_class, embedding.cpu().numpy(), prototypes.cpu().numpy(), distances.cpu().numpy(), log_p_y.cpu().numpy()

# Fungsi untuk menampilkan diagram batang dalam bentuk QPixmap
def create_bar_chart(values, class_names, title, xlabel, ylabel):
    plt.figure(figsize=(6, 4))
    plt.bar(class_names, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.tight_layout()
    plt.savefig("chart.png")
    plt.close()

    pixmap = QPixmap("chart.png")
    return pixmap

class ImageClassifierGUI(QWidget):
    def __init__(self, model, prototypes, class_names):
        super().__init__()
        self.model = model
        self.prototypes = prototypes
        self.class_names = class_names
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Classifier')
        self.setGeometry(100, 100, 1200, 800)

        self.layout = QHBoxLayout()

        # tampilkan load image di kiri
        left_layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label)

        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)

        self.layout.addLayout(left_layout)

        # hasil mendapatkan embeddig gambar baru dan grafik jarak
        right_layout = QVBoxLayout()

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.result_label)  # Menambahkan result_label ke layout

        self.prototypes_text = QTextEdit(self)
        self.prototypes_text.setReadOnly(True)
        right_layout.addWidget(QLabel('Prototipe Setiap Kelas'))
        right_layout.addWidget(self.prototypes_text)

        self.embedding_text = QTextEdit(self)
        self.embedding_text.setReadOnly(True)
        right_layout.addWidget(QLabel('Embedding Gambar Baru'))
        right_layout.addWidget(self.embedding_text)

        self.distances_chart = QLabel(self)
        self.distances_chart.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(QLabel('Jarak Euclidean, Probabilitas, dan Log Probabilitas'))
        right_layout.addWidget(self.distances_chart)

        self.log_probs_chart = QLabel(self)
        self.log_probs_chart.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.log_probs_chart)

        self.layout.addLayout(right_layout)

        self.setLayout(self.layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load Image', '', 'Image Files (*.png *.jpg *.jpeg)', options=options)
        if file_path:
            self.show_image(file_path)
            self.predict_image(file_path)

    def show_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(224, 224, Qt.KeepAspectRatio))

    def predict_image(self, file_path):
        image = Image.open(file_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        predicted_class, embedding, prototypes, distances, log_p_y = classify_image(self.model, image_tensor, self.prototypes, self.class_names)
        
        self.result_label.setText(f'Prediction: {predicted_class}')  # Memperbaiki error di sini

        # Tampilkan embedding dalam teks area
        embedding_text = '\n'.join([f"{v:.4f}" for v in embedding.flatten()])
        self.embedding_text.setText(f"Embedding (Gambar Baru):\n{embedding_text}")

        # Tampilkan prototipe setiap kelas
        prototypes_text = ""
        for i, prototype in enumerate(prototypes):
            prototypes_text += f"Prototype ({self.class_names[i]}):\n"
            prototypes_text += '\n'.join([f"{v:.4f}" for v in prototype]) + "\n\n"
        self.prototypes_text.setText(prototypes_text)

        # Buat dan tampilkan diagram batang untuk jarak
        distance_chart_pixmap = create_bar_chart(distances.flatten(), self.class_names, 
                                                 'Distance from Image Embedding to Prototypes', 
                                                 'Class Names', 'Distance')
        self.distances_chart.setPixmap(distance_chart_pixmap.scaled(400, 300, Qt.KeepAspectRatio))

        # Buat dan tampilkan diagram batang untuk log probabilitas
        log_probs_chart_pixmap = create_bar_chart(log_p_y.flatten(), self.class_names, 
                                                  'Log-Probabilities for Each Class', 
                                                  'Class Names', 'Log-Probabilities')
        self.log_probs_chart.setPixmap(log_probs_chart_pixmap.scaled(400, 300, Qt.KeepAspectRatio))

if __name__ == '__main__':
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

    # Define class names
    class_names = ['Matang', 'Mentah', 'Setengah Matang']  # Ganti dengan nama kelas sebenarnya

    app = QApplication(sys.argv)
    ex = ImageClassifierGUI(model, prototypes, class_names)
    ex.show()
    sys.exit(app.exec_())
